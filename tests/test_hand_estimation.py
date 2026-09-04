import numpy as np

from dextoolbench.estimate_hands import (
    compute_palm_pose,
    rigid_alignment,
    transform_points,
)


def test_rigid_alignment_recovers_transform() -> None:
    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    angle = np.deg2rad(35.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = source @ rotation.T + np.array([0.2, -0.4, 1.3])
    transform = rigid_alignment(source, target)
    np.testing.assert_allclose(transform_points(transform, source), target, atol=1e-10)
    np.testing.assert_allclose(transform[:3, :3], rotation, atol=1e-10)


def test_palm_pose_is_right_handed_and_orthonormal() -> None:
    keypoints = {
        "wrist_back": np.array([0.0, 0.0, -0.01]),
        "wrist_front": np.array([0.0, 0.0, 0.01]),
        "middle_0_front": np.array([0.0, 0.1, 0.01]),
        "index_0_front": np.array([-0.04, 0.08, 0.01]),
        "ring_0_front": np.array([0.04, 0.08, 0.01]),
    }
    pose = compute_palm_pose(keypoints)
    np.testing.assert_allclose(pose[:3, 3], np.zeros(3))
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-12)
