from __future__ import annotations

from collections import defaultdict

import numpy as np
import yourdfpy
from scipy.spatial.transform import Rotation as R

from isaacgymenvs.utils.robot_info import (
    OBJECT_KEYPOINT_OFFSETS,
    T_W_R,
    RobotSpec,
    get_obs_name_to_names,
    get_robot_spec,
    get_robot_urdf_path,
)


def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0)[..., None]
    b = np.cross(q_vec, v, axis=-1) * q_w[..., None] * 2.0
    c = (
        q_vec
        * (q_vec.reshape(shape[0], 1, 3) @ v.reshape(shape[0], 3, 1))[..., 0]
        * 2.0
    )
    return a + b + c


def tensor_clamp(t, min_t, max_t):
    return np.maximum(np.minimum(t, max_t), min_t)


def matrix_to_quaternion_xyzw_scipy(matrix: np.ndarray) -> np.ndarray:
    return R.from_matrix(matrix).as_quat()


def create_urdf_object(robot_asset_file: str) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(get_robot_urdf_path(robot_asset_file))


def compute_fk_dict(
    urdf: yourdfpy.URDF, q: np.ndarray, link_names: list[str]
) -> dict[str, np.ndarray]:
    n_steps = q.shape[0]
    fk_dict = defaultdict(list)
    for i in range(n_steps):
        urdf.update_cfg(q[i])
        for link_name in link_names:
            fk_dict[link_name].append(urdf.get_transform(frame_to=link_name))
    for link_name in link_names:
        fk_dict[link_name] = np.stack(fk_dict[link_name], axis=0)
    return fk_dict


def compute_observation(
    q: np.ndarray,
    qd: np.ndarray,
    prev_action_targets: np.ndarray,
    object_pose: np.ndarray,
    goal_object_pose: np.ndarray,
    object_scales: np.ndarray,
    urdf: yourdfpy.URDF,
    obs_list: list[str],
    robot_asset_file: str,
) -> np.ndarray:
    spec = get_robot_spec(robot_asset_file)
    n_steps = q.shape[0]
    n_joints = spec.num_hand_arm_dofs

    assert q.shape == (n_steps, n_joints), q.shape
    assert qd.shape == (n_steps, n_joints), qd.shape
    assert prev_action_targets.shape == (n_steps, n_joints), prev_action_targets.shape
    assert object_pose.shape == (n_steps, 7), object_pose.shape
    assert goal_object_pose.shape == (n_steps, 7), goal_object_pose.shape
    assert object_scales.shape == (n_steps, 3), object_scales.shape
    assert list(spec.joint_names) == urdf.actuated_joint_names, (
        spec.joint_names,
        urdf.actuated_joint_names,
    )

    q_unscaled = unscale(q, spec.lower_limits, spec.upper_limits)

    link_names = [spec.palm_body_name, *spec.fingertip_body_names]
    fk_dict = compute_fk_dict(urdf=urdf, q=q, link_names=link_names)
    palm_center_pos, palm_rot = _compute_palm_center_pos_and_rot(fk_dict, spec)
    fingertip_positions_with_offsets = _compute_fingertip_positions_with_offsets(
        fk_dict, spec
    )
    fingertip_rel_pos = fingertip_positions_with_offsets - palm_center_pos[:, None]

    object_keypoint_positions = _compute_keypoint_positions(
        pose=object_pose, scales=object_scales
    )
    goal_keypoint_positions = _compute_keypoint_positions(
        pose=goal_object_pose, scales=object_scales
    )
    keypoints_rel_palm = object_keypoint_positions - palm_center_pos[:, None]
    keypoints_rel_goal = object_keypoint_positions - goal_keypoint_positions

    obs_dict = {
        "joint_pos": q_unscaled,
        "joint_vel": qd,
        "prev_action_targets": prev_action_targets,
        "palm_pos": palm_center_pos,
        "palm_rot": palm_rot,
        "object_rot": object_pose[:, 3:7],
        "keypoints_rel_palm": keypoints_rel_palm.reshape(n_steps, -1),
        "keypoints_rel_goal": keypoints_rel_goal.reshape(n_steps, -1),
        "fingertip_pos_rel_palm": fingertip_rel_pos.reshape(n_steps, -1),
        "object_scales": object_scales,
    }

    obs_name_to_names = get_obs_name_to_names(robot_asset_file)
    obs = np.concatenate([obs_dict[key] for key in obs_list], axis=-1)
    for name, names in obs_name_to_names.items():
        assert obs_dict[name].shape[1] == len(names), (name, obs_dict[name].shape)
    return obs


def compute_joint_pos_targets(
    actions: np.ndarray,
    prev_targets: np.ndarray,
    hand_moving_average: float,
    arm_moving_average: float,
    hand_dof_speed_scale: float,
    dt: float,
    robot_asset_file: str,
) -> np.ndarray:
    spec = get_robot_spec(robot_asset_file)
    n_steps = actions.shape[0]
    n_joints = spec.num_hand_arm_dofs
    n_arm_dofs = spec.num_arm_dofs

    assert actions.shape == (n_steps, n_joints), actions.shape
    assert prev_targets.shape == (n_steps, n_joints), prev_targets.shape

    cur_targets = prev_targets.copy()
    cur_targets[:, n_arm_dofs:] = scale(
        actions[:, n_arm_dofs:],
        spec.lower_limits[n_arm_dofs:],
        spec.upper_limits[n_arm_dofs:],
    )
    cur_targets[:, n_arm_dofs:] = (
        hand_moving_average * cur_targets[:, n_arm_dofs:]
        + (1.0 - hand_moving_average) * prev_targets[:, n_arm_dofs:]
    )
    cur_targets[:, n_arm_dofs:] = tensor_clamp(
        cur_targets[:, n_arm_dofs:],
        spec.lower_limits[n_arm_dofs:],
        spec.upper_limits[n_arm_dofs:],
    )

    cur_targets[:, :n_arm_dofs] = (
        prev_targets[:, :n_arm_dofs] + hand_dof_speed_scale * dt * actions[:, :n_arm_dofs]
    )
    cur_targets[:, :n_arm_dofs] = tensor_clamp(
        cur_targets[:, :n_arm_dofs],
        spec.lower_limits[:n_arm_dofs],
        spec.upper_limits[:n_arm_dofs],
    )
    cur_targets[:, :n_arm_dofs] = (
        arm_moving_average * cur_targets[:, :n_arm_dofs]
        + (1.0 - arm_moving_average) * prev_targets[:, :n_arm_dofs]
    )
    return cur_targets


def _compute_palm_center_pos_and_rot(
    fk_dict: dict[str, np.ndarray],
    spec: RobotSpec,
) -> tuple[np.ndarray, np.ndarray]:
    t_r_ps = fk_dict[spec.palm_body_name]
    n_steps = t_r_ps.shape[0]
    t_w_ps = T_W_R[None] @ t_r_ps
    palm_pos = t_w_ps[:, :3, 3]
    palm_rot = t_w_ps[:, :3, :3]
    palm_quat_xyzw = matrix_to_quaternion_xyzw_scipy(palm_rot)
    palm_offset = spec.palm_offset[None].repeat(n_steps, axis=0)
    palm_center_pos = palm_pos + quat_rotate(palm_quat_xyzw, palm_offset)
    return palm_center_pos, palm_quat_xyzw


def _compute_fingertip_positions_with_offsets(
    fk_dict: dict[str, np.ndarray],
    spec: RobotSpec,
) -> np.ndarray:
    t_r_fs = np.stack([fk_dict[name] for name in spec.fingertip_body_names], axis=1)
    n_steps = t_r_fs.shape[0]
    n_fingertips = spec.num_fingertips
    t_w_fs = T_W_R[None, None] @ t_r_fs
    fingertip_positions = t_w_fs[:, :, :3, 3]
    fingertip_rots = t_w_fs[:, :, :3, :3]
    fingertip_quat_xyzw = matrix_to_quaternion_xyzw_scipy(
        fingertip_rots.reshape(-1, 3, 3)
    ).reshape(n_steps, n_fingertips, 4)

    fingertip_offsets = spec.fingertip_offsets[None].repeat(n_steps, axis=0)
    fingertip_positions_with_offsets = np.zeros(
        (n_steps, n_fingertips, 3), dtype=np.float32
    )
    for i in range(n_fingertips):
        fingertip_positions_with_offsets[:, i] = fingertip_positions[:, i] + quat_rotate(
            fingertip_quat_xyzw[:, i], fingertip_offsets[:, i]
        )
    return fingertip_positions_with_offsets


def _compute_keypoint_positions(
    pose: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    n_steps = pose.shape[0]
    object_base_size = 0.04
    keypoint_scale = 1.5
    object_keypoint_offsets = (
        OBJECT_KEYPOINT_OFFSETS[None]
        * object_base_size
        * keypoint_scale
        / 2
        * scales[:, None]
    )

    pos = pose[:, :3]
    quat_xyzw = pose[:, 3:7]
    keypoint_positions = np.zeros((n_steps, 4, 3), dtype=np.float32)
    for i in range(4):
        keypoint_positions[:, i] = pos + quat_rotate(
            quat_xyzw, object_keypoint_offsets[:, i]
        )
    return keypoint_positions
