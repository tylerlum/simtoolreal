"""Batched inward fingertip contact shaping for the native Gym task.

Adapted from VMZRM 25112a6: positive inward force alignment squared, thumb
weight 0.30 and descending finger weights 0.25/0.20/0.20/0.05. Gym merges
elastomer fixed links into DP bodies, so its net force is NOT partner-filtered.
Object-surface proximity and table clearance reject obvious non-object contact;
simultaneous robot contact near an object cannot be isolated by this API.
"""
import math
import xml.etree.ElementTree as ET

import torch
from scipy.spatial.transform import Rotation


def quat_apply(q, v):
    xyz = q[..., :3]
    uv = torch.cross(xyz, v, dim=-1)
    return v + 2 * (q[..., 3:] * uv + torch.cross(xyz, uv, dim=-1))


def inward_quality(forces, normals, threshold):
    magnitude = forces.norm(dim=-1)
    alignment = (forces * normals).sum(-1) / magnitude.clamp_min(1e-8)
    return (magnitude >= threshold).to(forces.dtype) * alignment.clamp(0, 1).square()


def primitive_sdf(points, half_size, cylinder, axis):
    """Box/capped-cylinder distance; leading dimensions broadcast."""
    delta = points.abs() - half_size
    box = delta.clamp_min(0).norm(dim=-1) + delta.amax(-1).clamp_max(0)
    axial = torch.gather(points, -1, axis).squeeze(-1)
    half_length = torch.gather(half_size.expand_as(points), -1, axis).squeeze(-1)
    # All procedural cylinders are x- or y-aligned; z stores their radius.
    radius = half_size[..., 2]
    radial = (points.square().sum(-1) - axial.square()).clamp_min(0).sqrt()
    d = torch.stack((axial.abs() - half_length, radial - radius), -1)
    cyl = d.clamp_min(0).norm(dim=-1) + d.amax(-1).clamp_max(0)
    return torch.where(cylinder, cyl, box)


class FingertipContactReward(torch.nn.Module):
    def __init__(self, *, pool, num_envs, fingertip_names, robot_urdf,
                 table_urdf, config, step_dt, device):
        super().__init__()
        for name in ('maxRewardPerSecond', 'supportThresholdN', 'surfaceToleranceM', 'tableClearanceM'):
            value = float(config[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Contact reward {} must be finite and positive'.format(name))
        self.max_step_reward = float(config['maxRewardPerSecond']) * step_dt
        self.threshold = float(config['supportThresholdN'])
        self.surface_tolerance = float(config['surfaceToleranceM'])
        self.table_clearance = float(config['tableClearanceM'])
        # VMZRM's independently derived LEFT pad-frame inward normals.
        pad_normals = {'left_thumb_DP': (-0.006576347, 0.437775110, 0.899060457),
                      **{name: (-0.025855984, 0.170824278, 0.984962199)
                         for name in ('left_index_DP', 'left_middle_DP', 'left_ring_DP', 'left_pinky_DP')}}
        if set(fingertip_names) != set(pad_normals):
            raise ValueError('Contact normals require the exact left Sharpa DP names')
        joints = {j.get('name'): j for j in ET.parse(robot_urdf).getroot().findall('joint')}
        normals = []
        for name in fingertip_names:
            joint = joints[name[:-3] + '_elastomer_fix_joint']
            rpy = [float(x) for x in joint.find('origin').get('rpy').split()]
            normals.append(Rotation.from_euler('xyz', rpy).apply(pad_normals[name]).tolist())
        self.register_buffer('normals', torch.tensor(normals, dtype=torch.float32))
        self.thumb = fingertip_names.index('left_thumb_DP')
        self.register_buffer('fingers', torch.tensor([i for i in range(5) if i != self.thumb]))
        self.register_buffer('marginals', torch.tensor([.25, .20, .20, .05]))
        table_box = ET.parse(table_urdf).find('link/collision/geometry/box')
        if table_box is None:
            raise ValueError('Contact table-clearance check requires the stock box table')
        self.table_half_height = float(table_box.get('size').split()[2]) / 2
        centers, half_sizes, cylinders, axes, active = [], [], [], [], []
        for item in pool:
            handle, head = item['handle_scale'], item['head_scale']
            hhalf = [x / 2 for x in handle] if len(handle) == 3 else [handle[0]/2, handle[1]/2, handle[1]/2]
            if head is None:
                phalf, offset = [1., 1., 1.], 0.
            elif len(head) == 3:
                phalf, offset = [x / 2 for x in head], handle[0]/2 + head[0]/2
            else:
                phalf, offset = [head[1]/2, head[0]/2, head[1]/2], handle[0]/2 + head[1]/2
            centers.append([[0., 0., 0.], [offset, 0., 0.]])
            half_sizes.append([hhalf, phalf])
            cylinders.append([len(handle) == 2, head is not None and len(head) == 2])
            axes.append([[0], [1]])
            active.append([True, head is not None])
        # Matches native environment assignment i % len(object_assets).
        indices = torch.arange(num_envs) % len(pool)
        for name, values, dtype in [('centers', centers, torch.float32), ('half_sizes', half_sizes, torch.float32),
                                    ('cylinders', cylinders, torch.bool), ('axes', axes, torch.long),
                                    ('active', active, torch.bool)]:
            self.register_buffer(name, torch.tensor(values, dtype=dtype)[indices])
        self.to(device=device)

    def forward(self, forces, fingertip_quat, fingertip_pos, object_pos, object_quat, table_pos):
        normals_w = quat_apply(fingertip_quat, self.normals.expand_as(forces))
        quality = inward_quality(forces, normals_w, self.threshold)
        inverse_q = torch.cat((-object_quat[:, :3], object_quat[:, 3:]), -1)
        points = quat_apply(inverse_q[:, None].expand(-1, 5, -1), fingertip_pos - object_pos[:, None])
        points = points[:, :, None] - self.centers[:, None]
        sdf = primitive_sdf(points, self.half_sizes[:, None], self.cylinders[:, None],
                            self.axes[:, None].expand(-1, 5, -1, -1))
        sdf = sdf.masked_fill(~self.active[:, None], float('inf')).amin(-1)
        near_object = sdf.abs() <= self.surface_tolerance
        above_table = fingertip_pos[..., 2] > (table_pos[:, None, 2] + self.table_half_height + self.table_clearance)
        quality = quality * (near_object & above_table).to(quality.dtype)
        finger_quality = quality.index_select(1, self.fingers).sort(dim=1, descending=True).values
        reward = (0.30 * quality[:, self.thumb] + (finger_quality * self.marginals).sum(1)) * self.max_step_reward
        return reward, quality
