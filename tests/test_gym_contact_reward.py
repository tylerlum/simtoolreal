"""Simulator-free contracts for the inward contact shaping formula."""
import importlib.util
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("contact", ROOT / "isaacgymenvs/tasks/simtoolreal/contact_reward.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_force_direction_and_support_threshold():
    force = torch.tensor([[1., 0, 0], [-1., 0, 0], [0., 1, 0], [.01, 0, 0], [1., 1, 0]])
    normal = torch.tensor([1., 0, 0]).expand_as(force)
    torch.testing.assert_close(M.inward_quality(force, normal, .05), torch.tensor([1., 0, 0, 0, .5]))


def test_box_and_cylinder_signed_distances():
    points = torch.tensor([[[0., 0, 0], [2., 0, 0], [0., 2, 0], [2., 2, 0]]])
    half = torch.ones(1, 1, 3)
    axis = torch.zeros(1, 4, 1, dtype=torch.long)
    expected = torch.tensor([[-1., 1., 1., 2**.5]])
    for cylinder in [False, True]:
        torch.testing.assert_close(M.primitive_sdf(points, half, torch.tensor(cylinder), axis), expected)


def make_reward():
    names = ['left_index_DP', 'left_middle_DP', 'left_ring_DP', 'left_thumb_DP', 'left_pinky_DP']
    return M.FingertipContactReward(
        pool=[dict(handle_scale=[.2, .04, .04], head_scale=None)], num_envs=1,
        fingertip_names=names,
        robot_urdf=ROOT / 'assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf',
        table_urdf=ROOT / 'assets/urdf/table_narrow.urdf',
        config=dict(maxRewardPerSecond=30., supportThresholdN=.05, surfaceToleranceM=.015, tableClearanceM=.005),
        step_dt=1/60, device='cpu')


def test_reward_bounds_gates_weights_and_rotation():
    reward = make_reward()
    identity = torch.tensor([[0., 0, 0, 1.]])
    tip_q = identity[:, None].expand(-1, 5, -1)
    obj = torch.tensor([[0., 0, 1.]])
    tips = obj[:, None] + torch.tensor([[[0., 0, .02]]]).expand(-1, 5, -1)
    table = torch.zeros(1, 3)
    force = reward.normals[None].clone()
    value, quality = reward(force, tip_q, tips, obj, identity, table)
    torch.testing.assert_close(value, torch.tensor([.5]))
    torch.testing.assert_close(quality, torch.ones(1, 5))
    # Only thumb and one non-thumb: 0.30 + 0.25, independent of finger identity.
    for finger in [0, 1, 2, 4]:
        sparse = torch.zeros_like(force)
        sparse[:, [3, finger]] = force[:, [3, finger]]
        torch.testing.assert_close(reward(sparse, tip_q, tips, obj, identity, table)[0], torch.tensor([.275]))
    assert reward(-force, tip_q, tips, obj, identity, table)[0].item() == 0
    assert reward(force, tip_q, tips + .1, obj, identity, table)[0].item() == 0
    assert reward(force, tip_q, tips, obj, identity, torch.tensor([[0., 0., 1.]]))[0].item() == 0
    # A common rotation about gravity preserves proximity, inward alignment and reward.
    q = torch.tensor([[0., 0., 2**-.5, 2**-.5]])
    qtip = q[:, None].expand(-1, 5, -1)
    rotated_tips = M.quat_apply(qtip, tips - obj[:, None]) + obj[:, None]
    rotated_force = M.quat_apply(qtip, force)
    torch.testing.assert_close(reward(rotated_force, qtip, rotated_tips, obj, q, table)[0], value)
