from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RobotSpec:
    name: str
    urdf_rel_path: str
    joint_names: tuple[str, ...]
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    palm_body_name: str
    palm_offset: np.ndarray
    fingertip_body_names: tuple[str, ...]
    fingertip_offsets: np.ndarray
    fingertip_labels: tuple[str, ...]
    fingertip_friction_body_names: tuple[str, ...]

    @property
    def num_arm_dofs(self) -> int:
        return 7

    @property
    def num_hand_dofs(self) -> int:
        return len(self.joint_names) - self.num_arm_dofs

    @property
    def num_hand_arm_dofs(self) -> int:
        return len(self.joint_names)

    @property
    def num_fingertips(self) -> int:
        return len(self.fingertip_body_names)

    @property
    def restricted_lower_limits(self) -> np.ndarray:
        lower = self.lower_limits.copy()
        lower[: self.num_arm_dofs] += np.deg2rad(10.0)
        return lower

    @property
    def restricted_upper_limits(self) -> np.ndarray:
        upper = self.upper_limits.copy()
        upper[: self.num_arm_dofs] -= np.deg2rad(10.0)
        return upper


IIWA_JOINT_NAMES = (
    "iiwa14_joint_1",
    "iiwa14_joint_2",
    "iiwa14_joint_3",
    "iiwa14_joint_4",
    "iiwa14_joint_5",
    "iiwa14_joint_6",
    "iiwa14_joint_7",
)


SHARPA_JOINT_NAMES = IIWA_JOINT_NAMES + (
    "left_1_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_2_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_3_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_4_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_5_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
)

SHARPA_LOWER_LIMITS = np.array(
    [
        -2.9671,
        -2.0944,
        -2.9671,
        -2.0944,
        -2.9671,
        -2.0944,
        -3.0543,
        -0.1745,
        -0.3491,
        -0.5236,
        -0.3491,
        0.0000,
        -0.1745,
        -0.0349,
        0.0000,
        0.0000,
        -0.1745,
        -0.0349,
        0.0000,
        0.0000,
        -0.1745,
        -0.0349,
        0.0000,
        0.0000,
        0.0000,
        -0.1745,
        -0.0349,
        0.0000,
        0.0000,
    ],
    dtype=np.float32,
)
SHARPA_UPPER_LIMITS = np.array(
    [
        2.9671,
        2.0944,
        2.9671,
        2.0944,
        2.9671,
        2.0944,
        3.0543,
        1.9199,
        0.1309,
        1.3963,
        0.3491,
        1.7453,
        1.5708,
        0.0349,
        1.7453,
        1.3963,
        1.5708,
        0.0349,
        1.7453,
        1.3963,
        1.5708,
        0.0349,
        1.7453,
        1.3963,
        0.2618,
        1.5708,
        0.0349,
        1.7453,
        1.3963,
    ],
    dtype=np.float32,
)

WUJI_JOINT_NAMES = IIWA_JOINT_NAMES + tuple(
    f"left_finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)
)
WUJI_LOWER_LIMITS = np.array(
    [
        -2.9671,
        -2.0944,
        -2.9671,
        -2.0944,
        -2.9671,
        -2.0944,
        -3.0543,
        0.0475,
        -0.1387,
        -0.4642,
        -0.4699,
        -0.1585,
        -0.37,
        -0.4777,
        -0.4683,
        -0.1644,
        -0.37,
        -0.4739,
        -0.4684,
        -0.1554,
        -0.37,
        -0.4765,
        -0.4777,
        -0.1626,
        -0.37,
        -0.4768,
        -0.4683,
    ],
    dtype=np.float32,
)
WUJI_UPPER_LIMITS = np.array(
    [
        2.9671,
        2.0944,
        2.9671,
        2.0944,
        2.9671,
        2.0944,
        3.0543,
        1.6033,
        0.9324,
        1.5623,
        1.5568,
        1.5604,
        0.37,
        1.5485,
        1.5753,
        1.5516,
        0.37,
        1.5512,
        1.5745,
        1.5585,
        0.37,
        1.5487,
        1.5634,
        1.5585,
        0.37,
        1.5490,
        1.5735,
    ],
    dtype=np.float32,
)


SHARPA_SPEC = RobotSpec(
    name="sharpa",
    urdf_rel_path="urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf",
    joint_names=SHARPA_JOINT_NAMES,
    lower_limits=SHARPA_LOWER_LIMITS,
    upper_limits=SHARPA_UPPER_LIMITS,
    palm_body_name="iiwa14_link_7",
    palm_offset=np.array([-0.00, -0.02, 0.16], dtype=np.float32),
    fingertip_body_names=(
        "left_index_DP",
        "left_middle_DP",
        "left_ring_DP",
        "left_thumb_DP",
        "left_pinky_DP",
    ),
    fingertip_offsets=np.array(
        [
            [0.02, 0.002, 0.0],
            [0.02, 0.002, 0.0],
            [0.02, 0.002, 0.0],
            [0.02, 0.002, 0.0],
            [0.02, 0.002, 0.0],
        ],
        dtype=np.float32,
    ),
    fingertip_labels=("index", "middle", "ring", "thumb", "pinky"),
    fingertip_friction_body_names=(
        "left_index_DP",
        "left_middle_DP",
        "left_ring_DP",
        "left_thumb_DP",
        "left_pinky_DP",
    ),
)

WUJI_SPEC = RobotSpec(
    name="wuji",
    urdf_rel_path="urdf/kuka_wuji_description/iiwa14_left_wuji_adjusted_restricted.urdf",
    joint_names=WUJI_JOINT_NAMES,
    lower_limits=WUJI_LOWER_LIMITS,
    upper_limits=WUJI_UPPER_LIMITS,
    palm_body_name="left_palm_link",
    palm_offset=np.array([-0.0030, -0.0005, 0.0367], dtype=np.float32),
    fingertip_body_names=(
        "left_finger2_tip_link",
        "left_finger3_tip_link",
        "left_finger4_tip_link",
        "left_finger1_tip_link",
        "left_finger5_tip_link",
    ),
    fingertip_offsets=np.zeros((5, 3), dtype=np.float32),
    fingertip_labels=("index", "middle", "ring", "thumb", "pinky"),
    fingertip_friction_body_names=(
        "left_finger2_tip_link",
        "left_finger3_tip_link",
        "left_finger4_tip_link",
        "left_finger1_tip_link",
        "left_finger5_tip_link",
    ),
)

ROBOT_SPECS = {
    SHARPA_SPEC.name: SHARPA_SPEC,
    WUJI_SPEC.name: WUJI_SPEC,
}

OBJECT_KEYPOINT_OFFSETS = np.array(
    [[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]],
    dtype=np.float32,
)

T_W_R = np.eye(4, dtype=np.float32)
T_W_R[:3, 3] = np.array([0.0, 0.8, 0.0], dtype=np.float32)


def infer_robot_name(robot_asset_file: str) -> str:
    robot_asset_file = robot_asset_file.lower()
    if "sharpa" in robot_asset_file:
        return "sharpa"
    if "wuji" in robot_asset_file:
        return "wuji"
    raise ValueError(f"Unsupported robot asset: {robot_asset_file}")


def get_robot_spec(robot_asset_file: str) -> RobotSpec:
    return ROBOT_SPECS[infer_robot_name(robot_asset_file)]


def get_repo_asset_root() -> Path:
    return Path(__file__).resolve().parents[2] / "assets"


def get_robot_urdf_path(robot_asset_file: str) -> Path:
    spec = get_robot_spec(robot_asset_file)
    asset_root = get_repo_asset_root()
    urdf_path = asset_root / spec.urdf_rel_path
    if not urdf_path.exists():
        raise FileNotFoundError(f"Robot URDF not found: {urdf_path}")
    return urdf_path


def get_obs_name_to_names(robot_asset_file: str) -> dict[str, list[str]]:
    spec = get_robot_spec(robot_asset_file)
    joint_names = list(spec.joint_names)
    return {
        "joint_pos": [f"{name}_q" for name in joint_names],
        "joint_vel": [f"{name}_qd" for name in joint_names],
        "prev_action_targets": [f"{name}_prev_action_target" for name in joint_names],
        "palm_pos": [f"palm_center_pos_{axis}" for axis in "xyz"],
        "palm_rot": [f"palm_rot_{axis}" for axis in "xyzw"],
        "object_rot": [f"object_rot_{axis}" for axis in "xyzw"],
        "keypoints_rel_palm": [
            f"keypoints_rel_palm_{idx}_{axis}" for idx in range(4) for axis in "xyz"
        ],
        "keypoints_rel_goal": [
            f"keypoints_rel_goal_{idx}_{axis}" for idx in range(4) for axis in "xyz"
        ],
        "fingertip_pos_rel_palm": [
            f"fingertip_rel_pos_{finger}_{axis}"
            for finger in spec.fingertip_labels
            for axis in "xyz"
        ],
        "object_scales": [f"object_scales_{axis}" for axis in "xyz"],
    }


def get_obs_names(robot_asset_file: str) -> list[str]:
    obs_name_to_names = get_obs_name_to_names(robot_asset_file)
    return sum(obs_name_to_names.values(), [])


def get_num_observations(robot_asset_file: str, obs_list: list[str]) -> int:
    obs_name_to_names = get_obs_name_to_names(robot_asset_file)
    return sum(len(obs_name_to_names[name]) for name in obs_list)
