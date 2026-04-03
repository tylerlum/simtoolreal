#!/usr/bin/env python

import copy
import datetime
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import rospy
import torch
import tyro
from geometry_msgs.msg import Pose, PoseStamped
from rl_player import RlPlayer
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from termcolor import colored

from deployment.rl_player_utils import read_cfg
from dextoolbench.objects import (
    NAME_TO_OBJECT,
)
from isaacgymenvs.utils.observation_action_utils import (
    compute_joint_pos_targets,
    compute_observation,
    create_urdf_object,
)
from isaacgymenvs.utils.robot_info import get_num_observations, get_robot_spec

FORCE_FIXED_ORIENTATION = False


T_W_R = np.eye(4)
T_W_R[:3, 3] = np.array([0.0, 0.8, 0.0])


def xyzw_to_wxyz(xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = xyzw
    return np.array([w, x, y, z])


def wxyz_to_xyzw(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = wxyz
    return np.array([x, y, z, w])


def error(message: str):
    print(colored(message, "red"))


def warn(message: str):
    print(colored(message, "yellow"))


def warn_every(message: str, n_seconds: float, key=None):
    """
    Print a warning message at most once every n_seconds per unique key.
    Stores state inside the function itself (no globals).
    """
    if not hasattr(warn_every, "_last_times"):
        warn_every._last_times = {}  # create on first call

    key = key or message
    last_times = warn_every._last_times
    last_time = last_times.get(key, 0)

    if time.time() - last_time > n_seconds:
        warn(message)
        last_times[key] = time.time()


def info(message: str):
    print(colored(message, "green"))


def assert_equals(a, b):
    assert a == b, f"a: {a}, b: {b}"


def get_ros_loop_rate_str(
    start_time: rospy.Time,
    before_sleep_time: rospy.Time,
    after_sleep_time: rospy.Time,
    node_name: Optional[str] = None,
) -> str:
    max_rate_dt = (before_sleep_time - start_time).to_sec()
    max_rate_hz = 1 / max_rate_dt
    actual_rate_dt = (after_sleep_time - start_time).to_sec()
    actual_rate_hz = 1 / actual_rate_dt
    loop_rate_str = f"Max rate: {np.round(max_rate_hz, 1)} Hz ({np.round(max_rate_dt * 1000, 1)} ms), Actual rate: {np.round(actual_rate_hz, 1)} Hz"
    return f"{node_name} {loop_rate_str}" if node_name is not None else loop_rate_str


def var_to_is_none_str(var) -> str:
    if var is None:
        return "None"
    return "Not None"


def pose_msg_to_T(msg: Pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.array([msg.position.x, msg.position.y, msg.position.z])

    # Get the original rotation matrix
    R_original = R.from_quat(
        [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    ).as_matrix()

    if FORCE_FIXED_ORIENTATION:
        # 1. Keep the original local X axis (we trust this direction)
        x_local = R_original[:, 0]

        # 2. Define the desired "Up" direction (Global Z)
        z_global = np.array([0, 0, 1])

        # 3. Project Global Z onto the plane perpendicular to x_local
        # Formula: v_perp = v - (v dot u) * u
        # This makes z_new perpendicular to x_local while staying closest to z_global
        proj_factor = np.dot(z_global, x_local)
        z_new_raw = z_global - (proj_factor * x_local)

        # Normalize to get the unit vector for the new local Z
        # (Add small epsilon to avoid div by zero if x_local is pointing perfectly up/down)
        if np.linalg.norm(z_new_raw) < 0.001:
            print(f"z_new_raw: {z_new_raw}")
            breakpoint()
        z_new = z_new_raw / (np.linalg.norm(z_new_raw) + 1e-6)

        # 4. Compute the new local Y using the cross product (Right-Hand Rule)
        # z cross x = y
        y_new = np.cross(z_new, x_local)

        # 5. Construct the new rotation matrix
        # Columns are [x_local, y_new, z_new]
        T[:3, :3] = np.column_stack((x_local, y_new, z_new))

    else:
        T[:3, :3] = R_original

    return T


def pos_quat_xyzw_to_pose_msg(pos: np.ndarray, quat_xyzw: np.ndarray) -> Pose:
    pose_msg = Pose()
    pose_msg.position.x = pos[0]
    pose_msg.position.y = pos[1]
    pose_msg.position.z = pos[2]
    pose_msg.orientation.x = quat_xyzw[0]
    pose_msg.orientation.y = quat_xyzw[1]
    pose_msg.orientation.z = quat_xyzw[2]
    pose_msg.orientation.w = quat_xyzw[3]
    return pose_msg


def T_to_pos_quat_xyzw(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos = T[:3, 3]
    quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    return pos, quat_xyzw


def T_to_pose(T: np.ndarray) -> np.ndarray:
    pos, quat_xyzw = T_to_pos_quat_xyzw(T)
    pose = np.concatenate([pos, quat_xyzw])
    return pose


def pos_quat_xyzw_to_T(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    return T


def pose_to_T(pose: np.ndarray) -> np.ndarray:
    pos, quat_xyzw = pose[:3], pose[3:]
    return pos_quat_xyzw_to_T(pos, quat_xyzw)


def dist_Ts(T1s: np.ndarray, T2s: np.ndarray, rot_weight: float = 1.0) -> np.ndarray:
    N = T1s.shape[0]
    assert T1s.shape == (N, 4, 4), f"T1s.shape: {T1s.shape}, expected: (N, 4, 4)"
    assert T2s.shape == (N, 4, 4), f"T2s.shape: {T2s.shape}, expected: (N, 4, 4)"

    # 1. Translation Distance (L2 Norm)
    # Norm along the last dimension of the translation vector
    pos_err = np.linalg.norm(T1s[..., :3, 3] - T2s[..., :3, 3], axis=-1)

    # 2. Rotation Distance (Geodesic / Angle of Rotation)
    # We want trace(R1 @ R2.T).
    # Efficiently computed as element-wise dot product of the rotation blocks.
    # sum(A_ij * B_ij) is equivalent to trace(A @ B.T)
    # We sum over the last two dimensions (-1, -2) which are the 3x3 rows/cols.
    R_prod_trace = np.sum(T1s[..., :3, :3] * T2s[..., :3, :3], axis=(-1, -2))

    # Clip trace to valid domain [-1, 3] to prevent NaN in arccos due to float error
    R_prod_trace = np.clip(R_prod_trace, -1.0, 3.0)

    # theta = arccos((Trace - 1) / 2)
    rot_err = np.arccos((R_prod_trace - 1.0) / 2.0)

    return pos_err + (rot_weight * rot_err)


class RLPolicyNode:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        hand_moving_average: float,
        arm_moving_average: float,
        hand_dof_speed_scale: float,
        object_scales: np.ndarray,
        save_foldername: Optional[str] = None,
        overwrite_targets_filepath: Optional[Path] = None,
        use_relative_object_pose_once_lifted: bool = False,
        object_name: Optional[str] = None,
        automatically_detect_object_lifted: bool = False,
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.hand_moving_average = hand_moving_average
        self.arm_moving_average = arm_moving_average
        self.hand_dof_speed_scale = hand_dof_speed_scale
        self.object_scales = object_scales
        self.save_foldername = save_foldername
        self.overwrite_targets_filepath = overwrite_targets_filepath
        self.use_relative_object_pose_once_lifted = use_relative_object_pose_once_lifted
        self.object_name = object_name
        self.automatically_detect_object_lifted = automatically_detect_object_lifted

        assert_equals(object_scales.shape, (3,))

        # Initialize the ROS node
        rospy.init_node("rl_policy_node")

        if self.save_foldername is not None:
            # ##############################################################################
            # Signal handling to save on shutdown
            # When in progress saving to file, stop updating latest joint states and commands
            # ##############################################################################
            import signal

            # Signal handling to save on shutdown
            # When in progress saving to file, stop updating latest joint states and commands
            self._is_in_progress_saving_to_file = False
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            # Store history of joint states and commands
            self.time_history: list[float] = []
            self.q_history: list[np.ndarray] = []
            self.qd_history: list[np.ndarray] = []
            self.q_target_history: list[np.ndarray] = []
            self.object_pose_history: list[np.ndarray] = []
            self.goal_object_pose_history: list[np.ndarray] = []

        # Faster with CPU
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"

        assert self.config_path.exists(), (
            f"config_path: {self.config_path} does not exist"
        )
        assert self.checkpoint_path.exists(), (
            f"checkpoint_path: {self.checkpoint_path} does not exist"
        )

        self.cfg = read_cfg(str(self.config_path), device=self.device)
        self.robot_asset_file = self.cfg["task"]["env"]["asset"]["robot"]
        self.robot_spec = get_robot_spec(self.robot_asset_file)
        self.hand_topic = f"/{self.robot_spec.name}"
        self.obs_list = self.cfg["task"]["env"]["obsList"]
        self.num_observations = get_num_observations(
            self.robot_asset_file, self.obs_list
        )
        self.num_actions = self.robot_spec.num_hand_arm_dofs
        self.q_lower_limits = self.robot_spec.restricted_lower_limits
        self.q_upper_limits = self.robot_spec.restricted_upper_limits
        if self.robot_spec.name == "sharpa":
            self.hand_command_joint_names = [
                f"joint_{i}.0" for i in range(self.robot_spec.num_hand_dofs)
            ]
        else:
            self.hand_command_joint_names = list(self.robot_spec.joint_names[7:])

        # Publisher for iiwa and hand joint commands
        self.iiwa_joint_cmd_pub = rospy.Publisher(
            "/iiwa/joint_cmd", JointState, queue_size=1
        )
        self.hand_joint_cmd_pub = rospy.Publisher(
            f"{self.hand_topic}/joint_cmd", JointState, queue_size=1
        )

        # Variables to store the latest messages
        self.object_pose_msg = None
        self.goal_object_pose_msg = None
        self.iiwa_joint_state_msg = None
        self.hand_joint_state_msg = None

        # Subscribers
        self.object_pose_sub = rospy.Subscriber(
            "/robot_frame/current_object_pose",
            PoseStamped,
            self.object_pose_callback,
            queue_size=1,
        )
        self.goal_object_pose_sub = rospy.Subscriber(
            "/robot_frame/goal_object_pose",
            Pose,
            self.goal_object_pose_callback,
            queue_size=1,
        )
        self.iiwa_joint_state_sub = rospy.Subscriber(
            "/iiwa/joint_states",
            JointState,
            self.iiwa_joint_state_callback,
            queue_size=1,
        )
        self.hand_joint_state_sub = rospy.Subscriber(
            f"{self.hand_topic}/joint_states",
            JointState,
            self.hand_joint_state_callback,
            queue_size=1,
        )

        # Create the RL player
        self.player = RlPlayer(
            num_observations=self.num_observations,
            num_actions=self.num_actions,
            config_path=str(self.config_path),
            checkpoint_path=str(self.checkpoint_path),
            device=self.device,
        )

        # ROS rate
        self.control_dt = 1.0 / 60

        # Set up chain
        self.urdf_object = create_urdf_object(self.robot_asset_file)

        # State: prev_targets
        self.prev_targets = None
        self._warmup_completed = False

        if self.overwrite_targets_filepath is not None:
            info(f"Overwriting targets from file: {self.overwrite_targets_filepath}")
            from recorded_data import RecordedData

            data_path = self.overwrite_targets_filepath
            assert data_path.exists(), f"File {data_path} does not exist"
            data = RecordedData.from_file(data_path)
            self.q_targets_from_file = data.robot_joint_pos_targets_array
            T, D = self.q_targets_from_file.shape
            print(f"T: {T}, D: {D}")
            assert D == self.num_actions, (
                f"D: {D}, expected: {self.num_actions}"
            )
            self.current_step = 0

        if self.use_relative_object_pose_once_lifted:
            if self.robot_spec.name != "sharpa":
                raise NotImplementedError(
                    "use_relative_object_pose_once_lifted is currently only wired for the Sharpa hand path."
                )
            # ##############################################################################
            # Signal handling to save relative object pose once lifted
            # ##############################################################################
            import signal

            signal.signal(signal.SIGUSR1, self._save_relative_object_pose_once_lifted)
            self.object_lifted = False
            self.initialized_relative_object_pose_logic = False

    def _save_relative_object_pose_once_lifted(self, signum, frame):
        info("Received signal to save relative object pose once lifted")
        self.object_lifted = True
        info(f"Object lifted: {self.object_lifted}")

    def object_pose_callback(self, msg: PoseStamped):
        self.object_pose_msg = msg

    def goal_object_pose_callback(self, msg: Pose):
        self.goal_object_pose_msg = msg

    def iiwa_joint_state_callback(self, msg: JointState):
        self.iiwa_joint_state_msg = msg

    def hand_joint_state_callback(self, msg: JointState):
        self.hand_joint_state_msg = msg

    def create_observation(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], Optional[rospy.Time]]:
        # Ensure all messages are received before processing
        if (
            self.iiwa_joint_state_msg is None
            or self.hand_joint_state_msg is None
            or self.object_pose_msg is None
            or self.goal_object_pose_msg is None
        ):
            warn_every(
                f"Waiting for all messages to be received... iiwa_joint_state_msg: {var_to_is_none_str(self.iiwa_joint_state_msg)}, hand_joint_state_msg: {var_to_is_none_str(self.hand_joint_state_msg)}, object_pose_msg: {var_to_is_none_str(self.object_pose_msg)}, goal_object_pose_msg: {var_to_is_none_str(self.goal_object_pose_msg)}",
                n_seconds=1.0,
            )
            return None, None, None

        iiwa_joint_state_msg = copy.copy(self.iiwa_joint_state_msg)
        hand_joint_state_msg = copy.copy(self.hand_joint_state_msg)
        object_pose_msg = copy.copy(self.object_pose_msg)
        goal_object_pose_msg = copy.copy(self.goal_object_pose_msg)

        timestamp_object_pose = object_pose_msg.header.stamp
        timestamp_iiwa_joint_state = iiwa_joint_state_msg.header.stamp
        timestamp_hand_joint_state = hand_joint_state_msg.header.stamp
        min_timestamp = min(
            timestamp_object_pose,
            timestamp_iiwa_joint_state,
            timestamp_hand_joint_state,
        )

        # Concatenate the data from joint states and object pose
        iiwa_position = np.array(iiwa_joint_state_msg.position)
        iiwa_velocity = np.array(iiwa_joint_state_msg.velocity)

        hand_position = np.array(hand_joint_state_msg.position)
        hand_velocity = np.array(hand_joint_state_msg.velocity)

        T_R_O = pose_msg_to_T(object_pose_msg.pose)
        T_R_G = pose_msg_to_T(goal_object_pose_msg)

        T_W_O = T_W_R @ T_R_O
        T_W_G = T_W_R @ T_R_G

        object_position_W, object_quat_xyzw_W = T_to_pos_quat_xyzw(T_W_O)
        object_pose_W = np.concatenate([object_position_W, object_quat_xyzw_W])

        goal_object_pos_W, goal_object_quat_xyzw_W = T_to_pos_quat_xyzw(T_W_G)
        goal_object_pose_W = np.concatenate(
            [goal_object_pos_W, goal_object_quat_xyzw_W]
        )

        q = np.concatenate([iiwa_position, hand_position])
        qd = np.concatenate([iiwa_velocity, hand_velocity])

        prev_action_targets = self.prev_targets if self.prev_targets is not None else q
        with torch.no_grad():
            observation = compute_observation(
                q=q[None],
                qd=qd[None],
                prev_action_targets=prev_action_targets[None],
                object_pose=object_pose_W[None],
                goal_object_pose=goal_object_pose_W[None],
                object_scales=self.object_scales[None],
                urdf=self.urdf_object,
                obs_list=self.obs_list,
                robot_asset_file=self.robot_asset_file,
            )
            observation = torch.from_numpy(observation).float().to(self.device)
        assert_equals(
            observation.shape,
            (
                1,
                self.num_observations,
            ),
        )

        DEBUG = False
        if DEBUG:
            print(f"q: {q}")
            print(f"qd: {qd}")
            print(f"object_pose_W: {object_pose_W}")
            print(f"goal_object_pose_W: {goal_object_pose_W}")
            print(f"object_scales: {self.object_scales}")
            breakpoint()

        # ##############################################################################
        # Record time and joint states and commands and object pose and goal object pose
        # ##############################################################################
        if self.save_foldername is not None and self._warmup_completed:
            if not hasattr(self, "start_run_time"):
                self.start_run_time = time.time()
            current_time = time.time()
            dt = current_time - self.start_run_time

            self.time_history.append(dt)
            self.q_history.append(q)
            self.qd_history.append(qd)
            self.q_target_history.append(prev_action_targets)
            self.object_pose_history.append(object_pose_W)
            self.goal_object_pose_history.append(goal_object_pose_W)

        return observation, q, min_timestamp

    def publish_targets(self, joint_pos_targets: np.ndarray):
        assert_equals(joint_pos_targets.shape, (1, self.num_actions))
        joint_pos_targets = joint_pos_targets[0]

        # Clamp joint position to joint limits with buffer
        # Hardware has 5 deg buffer, so we add 7.5 deg buffer here to avoid getting overshoots that hits limit
        CLAMP_ARM = False
        if CLAMP_ARM:
            BUFFER = np.deg2rad(7.5)
            J_arm = 7
            arm_lower_limits = (
                np.array(
                    [
                        self.urdf_object.actuated_joints[i].limit.lower
                        for i in range(J_arm)
                    ]
                )
                + BUFFER
            )
            arm_upper_limits = (
                np.array(
                    [
                        self.urdf_object.actuated_joints[i].limit.upper
                        for i in range(J_arm)
                    ]
                )
                - BUFFER
            )
            joint_pos_targets[:J_arm] = np.clip(
                joint_pos_targets[:J_arm], arm_lower_limits, arm_upper_limits
            )

        iiwa_msg = JointState()
        iiwa_msg.header.stamp = rospy.Time.now()
        iiwa_msg.header.frame_id = ""
        iiwa_msg.name = [
            "iiwa_joint_1",
            "iiwa_joint_2",
            "iiwa_joint_3",
            "iiwa_joint_4",
            "iiwa_joint_5",
            "iiwa_joint_6",
            "iiwa_joint_7",
        ]
        iiwa_msg.position = joint_pos_targets[:7].tolist()
        self.iiwa_joint_cmd_pub.publish(iiwa_msg)
        hand_msg = JointState()
        hand_msg.header.stamp = rospy.Time.now()
        hand_msg.header.frame_id = ""
        hand_msg.name = self.hand_command_joint_names
        hand_msg.position = joint_pos_targets[7:].tolist()
        self.hand_joint_cmd_pub.publish(hand_msg)

    def _initialize_relative_object_pose_logic(
        self,
        q: np.ndarray,
        q_target: np.ndarray,
        T_R_O_lifted: np.ndarray,
    ):
        assert_equals(q.shape, (29,))
        assert_equals(q_target.shape, (29,))
        assert_equals(T_R_O_lifted.shape, (4, 4))

        info("=" * 100)
        info("Initializing relative object pose logic")
        info("=" * 100)

        # Load PK chain for fk and jacobian for ik
        import pytorch_kinematics as pk

        from isaacgymenvs.utils.utils import get_repo_root_dir

        KUKA_SHARPA_URDF_PATH = (
            get_repo_root_dir()
            / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
        )
        assert KUKA_SHARPA_URDF_PATH.exists(), (
            f"KUKA_SHARPA_URDF_PATH not found: {KUKA_SHARPA_URDF_PATH}"
        )
        with open(KUKA_SHARPA_URDF_PATH, "rb") as f:
            urdf_str = f.read()
        DEVICE = "cpu"
        self.arm_pk_chain = pk.build_serial_chain_from_urdf(
            urdf_str,
            end_link_name="left_hand_C_MC",
        ).to(device=DEVICE)

        # Store the hand target when the object was lifted
        # Will keep this hand target fixed moving forward
        q_hand_lifted_target = q_target[
            7:
        ].copy()  # Will keep this hand target fixed moving forward
        assert q_hand_lifted_target.shape == (22,), (
            f"q_hand_lifted_target.shape: {q_hand_lifted_target.shape}, expected: (22,)"
        )

        # We want to store T_O_P_lifted, which is the pose of the palm relative to the object at the time of lifting
        # We want this constant over time moving forward
        # Note that O here refers to the OBJECT not the GOAL OBJECT
        # This is because we are initializing based on the object being lifted, not necessarily the goal object pose being reached
        # Thus, we care about the palm relative to the object, not relative to the goal object
        from baselines.visualize_demo_with_hand import compute_current_T_R_P

        q_arm_lifted = q[:7].copy()
        assert q_arm_lifted.shape == (7,), (
            f"q_arm_lifted.shape: {q_arm_lifted.shape}, expected: (7,)"
        )
        T_R_P_lifted = compute_current_T_R_P(
            arm_pk_chain=self.arm_pk_chain, q_arm=q_arm_lifted
        )
        T_O_P_lifted = np.linalg.inv(T_R_O_lifted) @ T_R_P_lifted

        # Load goal object pose trajectory
        # Assumes this is in world frame
        # Stop listening to the published one
        import json

        OBJECT_CATEGORY = "hammer"
        OBJECT_NAME = "claw_hammer"
        TASK_NAME = "swing_down"
        assert self.object_name == OBJECT_NAME, (
            f"self.object_name: {self.object_name}, OBJECT_NAME: {OBJECT_NAME}"
        )
        object_pose_trajectory_filepath = (
            get_repo_root_dir()
            / "dextoolbench/trajectories"
            / OBJECT_CATEGORY
            / OBJECT_NAME
            / f"{TASK_NAME}.json"
        )
        assert object_pose_trajectory_filepath.exists(), (
            f"object_pose_trajectory_filepath not found: {object_pose_trajectory_filepath}"
        )
        with open(object_pose_trajectory_filepath, "r") as f:
            goal_pose_trajectory_full = np.array(json.load(f)["goals"])

        # HACK: Offset same as goal_pose_ros_node.py
        goal_pose_trajectory_full[:, 0] -= 0.05

        old_T = len(goal_pose_trajectory_full)

        # Upsample the goal object pose trajectory
        SLOWDOWN_FACTOR = (
            4  # This runs at 60Hz, data is at 30Hz, prob run 2x slower too
        )
        assert goal_pose_trajectory_full.shape == (old_T, 7), (
            f"goal_pose_trajectory_full.shape: {goal_pose_trajectory_full.shape}, expected: (old_T, 7)"
        )
        new_T = old_T * SLOWDOWN_FACTOR
        goal_pose_trajectory_full_repeated = (
            goal_pose_trajectory_full.reshape(old_T, 1, 7)
            .repeat(SLOWDOWN_FACTOR, axis=1)
            .reshape(new_T, 7)
        )

        # Remove the initial part of trajectory that is before the object was lifted
        # Find idx that minimizes dist(goal_object_pose_trajectory[idx], T_W_O_lifted)
        # TODO: Consider using T_W_G_lifted instead of T_W_O_lifted
        T_W_O_lifted = T_W_R @ T_R_O_lifted
        T_W_O_trajectory_full_repeated = np.array(
            [pose_to_T(goal_pose_trajectory_full_repeated[idx]) for idx in range(new_T)]
        )
        distances = dist_Ts(
            T1s=T_W_O_trajectory_full_repeated,
            T2s=T_W_O_lifted[None].repeat(new_T, axis=0),
        )
        self.goal_idx = np.argmin(distances)
        info(f"Distances: {distances.tolist()}")
        info(f"Goal idx starting at: {self.goal_idx}")

        # Store the trajectory after the object was lifted
        self.T_W_O_trajectory = np.array(
            [
                T_W_O_trajectory_full_repeated[i]
                for i in range(len(T_W_O_trajectory_full_repeated))
                if i >= self.goal_idx
            ]
        )

        # Compute the wrist pose trajectory for the trajectory after the object was lifted
        self.T_W_Ps_using_lifted_object_pose = np.array(
            [
                self.T_W_O_trajectory[i] @ T_O_P_lifted
                for i in range(len(self.T_W_O_trajectory))
            ]
        )

        # Filter the poses
        from baselines.visualize_demo_with_hand import filter_poses

        self.T_W_Ps_using_lifted_object_pose = filter_poses(
            self.T_W_Ps_using_lifted_object_pose
        )

        TRAJECTORY_LENGTH = self.T_W_Ps_using_lifted_object_pose.shape[0]
        assert self.T_W_O_trajectory.shape == (TRAJECTORY_LENGTH, 4, 4), (
            f"self.T_W_O_trajectory.shape: {self.T_W_O_trajectory.shape}, expected: (TRAJECTORY_LENGTH, 4, 4)"
        )
        assert self.T_W_Ps_using_lifted_object_pose.shape == (
            TRAJECTORY_LENGTH,
            4,
            4,
        ), (
            f"self.T_W_Ps_using_lifted_object_pose.shape: {self.T_W_Ps_using_lifted_object_pose.shape}, expected: (TRAJECTORY_LENGTH, 4, 4)"
        )

        # IK for the arm targets, starting from the current arm joint positions
        # This uses pseudoinverse IK which needs to be relative to some reference arm joint positions
        # We start from the current arm joint positions and iteratively compute the next arm joint positions
        MODE: Literal["IK", "TRAJOPT", "SAVE_AND_LOAD_FROM_FILE"] = (
            "SAVE_AND_LOAD_FROM_FILE"
        )
        if MODE == "IK":
            from baselines.visualize_demo_with_hand import compute_new_q_arm

            q_arm = q[:7].copy()
            q_arm_targets_using_lifted_object_pose = []
            from tqdm import tqdm

            for i in tqdm(
                range(TRAJECTORY_LENGTH),
                desc="Computing arm targets using lifted object pose",
            ):
                T_W_P_using_lifted_object_pose = self.T_W_Ps_using_lifted_object_pose[i]
                T_R_W = np.linalg.inv(T_W_R)
                T_R_P_using_lifted_object_pose = T_R_W @ T_W_P_using_lifted_object_pose
                q_arm = compute_new_q_arm(
                    arm_pk_chain=self.arm_pk_chain,
                    target_wrist_pose=T_R_P_using_lifted_object_pose,
                    q_arm=q_arm,
                )
                q_arm_targets_using_lifted_object_pose.append(q_arm.copy())
            q_arm_targets_using_lifted_object_pose = np.array(
                q_arm_targets_using_lifted_object_pose
            )
            assert q_arm_targets_using_lifted_object_pose.shape == (
                TRAJECTORY_LENGTH,
                7,
            ), (
                f"q_arm_targets_using_lifted_object_pose.shape: {q_arm_targets_using_lifted_object_pose.shape}, expected: (TRAJECTORY_LENGTH, 7)"
            )
        elif MODE == "TRAJOPT":
            from baselines.visualize_demo_with_hand_trajopt import (
                interpolate_traj,
                solve_trajopt,
            )

            T_R_W = np.linalg.inv(T_W_R)
            T_R_Ps_using_lifted_object_pose = np.array(
                [T_R_W @ T_W_P for T_W_P in self.T_W_Ps_using_lifted_object_pose]
            )
            DOWNSAMPLE_FACTOR = 10
            retargeted_qs = solve_trajopt(
                T_R_Ps=T_R_Ps_using_lifted_object_pose[::DOWNSAMPLE_FACTOR],
                q_start=q.copy(),
                dt=1 / 30,
            )
            print(
                f"TRAJECTORY_LENGTH: {TRAJECTORY_LENGTH}, retargeted_qs.shape: {retargeted_qs.shape}"
            )
            q_arm_targets_using_lifted_object_pose = interpolate_traj(
                retargeted_qs[:, :7], n_steps=DOWNSAMPLE_FACTOR
            )
            print(
                f"After interpolation, q_arm_targets_using_lifted_object_pose.shape: {q_arm_targets_using_lifted_object_pose.shape}"
            )
            breakpoint()
            if q_arm_targets_using_lifted_object_pose.shape[0] < TRAJECTORY_LENGTH:
                extra = (
                    TRAJECTORY_LENGTH - q_arm_targets_using_lifted_object_pose.shape[0]
                )
                q_arm_targets_using_lifted_object_pose = np.concatenate(
                    [
                        q_arm_targets_using_lifted_object_pose,
                        q_arm_targets_using_lifted_object_pose[-1][None].repeat(
                            extra, axis=0
                        ),
                    ],
                    axis=0,
                )
            elif q_arm_targets_using_lifted_object_pose.shape[0] > TRAJECTORY_LENGTH:
                extra = (
                    q_arm_targets_using_lifted_object_pose.shape[0] - TRAJECTORY_LENGTH
                )
                q_arm_targets_using_lifted_object_pose = (
                    q_arm_targets_using_lifted_object_pose[:-extra]
                )
            assert q_arm_targets_using_lifted_object_pose.shape == (
                TRAJECTORY_LENGTH,
                7,
            ), (
                f"q_arm_targets_using_lifted_object_pose.shape: {q_arm_targets_using_lifted_object_pose.shape}, expected: ({TRAJECTORY_LENGTH}, 7)"
            )
        elif MODE == "SAVE_AND_LOAD_FROM_FILE":
            T_R_W = np.linalg.inv(T_W_R)
            T_R_Ps_using_lifted_object_pose = np.array(
                [T_R_W @ T_W_P for T_W_P in self.T_W_Ps_using_lifted_object_pose]
            )
            # Output T_R_Ps_using_lifted_object_pose and q to json
            with open("trajopt_inputs.json", "w") as f:
                json.dump(
                    {
                        "T_R_Ps_using_lifted_object_pose": T_R_Ps_using_lifted_object_pose.tolist(),
                        "q": q.tolist(),
                    },
                    f,
                    indent=4,
                )
            print("Saved trajopt inputs to trajopt_inputs.json")
            print("python baselines/run_trajopt.py")
            print("Run trajopt and save the outputs to trajopt_outputs.json")
            breakpoint()
            with open("trajopt_outputs.json", "r") as f:
                retargeted_qs = np.array(json.load(f)["retargeted_qs"])
            q_arm_targets_using_lifted_object_pose = retargeted_qs[:, :7]
            print(
                f"TRAJECTORY_LENGTH: {TRAJECTORY_LENGTH}, retargeted_qs.shape: {retargeted_qs.shape}"
            )
            assert q_arm_targets_using_lifted_object_pose.shape == (
                TRAJECTORY_LENGTH,
                7,
            ), (
                f"q_arm_targets_using_lifted_object_pose.shape: {q_arm_targets_using_lifted_object_pose.shape}, expected: ({TRAJECTORY_LENGTH}, 7)"
            )
        else:
            raise ValueError(f"Invalid MODE: {MODE}")

        # Concatenate the arm targets and the hand target
        self.q_targets_using_lifted_object_pose = np.concatenate(
            [
                q_arm_targets_using_lifted_object_pose,
                q_hand_lifted_target[None].repeat(TRAJECTORY_LENGTH, axis=0),
            ],
            axis=1,
        )
        assert self.q_targets_using_lifted_object_pose.shape == (
            TRAJECTORY_LENGTH,
            29,
        ), (
            f"self.q_targets_using_lifted_object_pose.shape: {self.q_targets_using_lifted_object_pose.shape}, expected: (TRAJECTORY_LENGTH, 29)"
        )

    def _visualize_relative_object_pose_logic(
        self,
        q_targets_using_lifted_object_pose: np.ndarray,
        T_W_Ps_using_lifted_object_pose: np.ndarray,
        T_W_O_trajectory: np.ndarray,
    ) -> None:
        TRAJECTORY_LENGTH = q_targets_using_lifted_object_pose.shape[0]
        assert q_targets_using_lifted_object_pose.shape == (TRAJECTORY_LENGTH, 29), (
            f"q_targets_using_lifted_object_pose.shape: {q_targets_using_lifted_object_pose.shape}, expected: (TRAJECTORY_LENGTH, 29)"
        )
        assert T_W_Ps_using_lifted_object_pose.shape == (TRAJECTORY_LENGTH, 4, 4), (
            f"T_W_Ps_using_lifted_object_pose.shape: {T_W_Ps_using_lifted_object_pose.shape}, expected: (TRAJECTORY_LENGTH, 4, 4)"
        )
        assert T_W_O_trajectory.shape == (TRAJECTORY_LENGTH, 4, 4), (
            f"T_W_O_trajectory.shape: {T_W_O_trajectory.shape}, expected: (TRAJECTORY_LENGTH, 4, 4)"
        )

        import viser
        from viser.extras import ViserUrdf

        from isaacgymenvs.utils.utils import get_repo_root_dir

        SERVER = viser.ViserServer()

        @SERVER.on_client_connect
        def _(client):
            client.camera.position = (0.0, -1.0, 1.03)
            client.camera.look_at = (0.0, 0.0, 0.53)
            # client.camera.wxyz = (w, x, y, z)

        AXES_LENGTH = 0.0
        AXES_RADIUS = 0.0

        # Load table
        # TABLE_URDF_PATH = get_repo_root_dir() / "assets/urdf/table_narrow.urdf"
        TABLE_URDF_PATH = (
            get_repo_root_dir() / "assets/urdf/table_narrow_whiteboard.urdf"
        )
        assert TABLE_URDF_PATH.exists(), f"TABLE_URDF_PATH not found: {TABLE_URDF_PATH}"

        _table_frame = SERVER.scene.add_frame(
            "/table",
            show_axes=True,
            axes_length=AXES_LENGTH,
            axes_radius=AXES_RADIUS,
            position=(0, 0, 0.38),
            wxyz=(1, 0, 0, 0),
        )
        _table_viser = ViserUrdf(SERVER, TABLE_URDF_PATH, root_node_name="/table")

        # Load robot
        KUKA_SHARPA_URDF_PATH = (
            get_repo_root_dir()
            / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
        )
        assert KUKA_SHARPA_URDF_PATH.exists(), (
            f"KUKA_SHARPA_URDF_PATH not found: {KUKA_SHARPA_URDF_PATH}"
        )
        _kuka_sharpa_frame = SERVER.scene.add_frame(
            "/robot/state",
            show_axes=True,
            axes_length=AXES_LENGTH,
            axes_radius=AXES_RADIUS,
            position=(0, 0.8, 0),
            wxyz=(1, 0, 0, 0),
        )
        kuka_sharpa_viser = ViserUrdf(
            SERVER, KUKA_SHARPA_URDF_PATH, root_node_name="/robot/state"
        )
        HOME_JOINT_POS_IIWA = np.array(
            [
                -1.571,
                1.571 - np.deg2rad(10),
                -0.000,
                1.376 + np.deg2rad(10),
                -0.000,
                1.485,
                1.308,
            ]
        )
        HOME_JOINT_POS_SHARPA = np.zeros(22)
        HOME_JOINT_POS = np.concatenate([HOME_JOINT_POS_IIWA, HOME_JOINT_POS_SHARPA])
        kuka_sharpa_viser.update_cfg(HOME_JOINT_POS)

        # Load floating hand
        SHARPA_URDF_PATH = (
            get_repo_root_dir()
            / "assets/urdf/left_sharpa_ha4/left_sharpa_ha4_v2_1_adjusted_restricted.urdf"
        )
        assert SHARPA_URDF_PATH.exists(), (
            f"SHARPA_URDF_PATH not found: {SHARPA_URDF_PATH}"
        )
        sharpa_frame = SERVER.scene.add_frame(
            "/sharpa",
            show_axes=True,
            axes_length=AXES_LENGTH,
            axes_radius=AXES_RADIUS,
            position=(100, 0, 0),
            wxyz=(1, 0, 0, 0),
        )
        sharpa_viser = ViserUrdf(SERVER, SHARPA_URDF_PATH, root_node_name="/sharpa")
        sharpa_viser.update_cfg(HOME_JOINT_POS_SHARPA)

        # Plot the joint targets and limits
        joint_names = kuka_sharpa_viser._urdf.actuated_joint_names
        J_arm = 7
        joint_limit_mins = [
            kuka_sharpa_viser._urdf.actuated_joints[i].limit.lower for i in range(J_arm)
        ]
        joint_limit_maxs = [
            kuka_sharpa_viser._urdf.actuated_joints[i].limit.upper for i in range(J_arm)
        ]

        import matplotlib.pyplot as plt

        nrows = int(np.ceil(np.sqrt(J_arm)))
        ncols = int(np.ceil(J_arm / nrows))
        fig, axes = plt.subplots(nrows, ncols)
        axes = axes.flatten()
        for i in range(J_arm):
            axes[i].plot(q_targets_using_lifted_object_pose[:, i], label="Target")
            axes[i].plot([joint_limit_mins[i]] * TRAJECTORY_LENGTH, label="Limit min")
            axes[i].plot([joint_limit_maxs[i]] * TRAJECTORY_LENGTH, label="Limit max")
            axes[i].legend()
            axes[i].set_title(joint_names[i])
        plt.suptitle("Position")
        plt.tight_layout()
        plt.show()

        joint_limit_velocity_maxs = [
            kuka_sharpa_viser._urdf.actuated_joints[i].limit.velocity
            for i in range(J_arm)
        ]
        joint_limit_velocity_mins = [
            -joint_limit_velocity_maxs[i] for i in range(J_arm)
        ]

        fig, axes = plt.subplots(nrows, ncols)
        axes = axes.flatten()
        qd_targets_using_lifted_object_pose = (
            np.diff(q_targets_using_lifted_object_pose, axis=0) / self.control_dt
        )
        for i in range(J_arm):
            axes[i].plot(qd_targets_using_lifted_object_pose[:, i], label="Velocity")
            axes[i].plot(
                [joint_limit_velocity_mins[i]] * (TRAJECTORY_LENGTH - 1),
                label="Limit min",
            )
            axes[i].plot(
                [joint_limit_velocity_maxs[i]] * (TRAJECTORY_LENGTH - 1),
                label="Limit max",
            )
            axes[i].legend()
            axes[i].set_title(joint_names[i])
        plt.suptitle("Velocity")
        plt.tight_layout()
        plt.show()

        # Load object
        from dextoolbench.objects import NAME_TO_OBJECT

        OBJECT_URDF_PATH = NAME_TO_OBJECT[self.object_name].urdf_path
        object_frame_viser = SERVER.scene.add_frame(
            "/object",
            position=(100, 0, 0),
            wxyz=(1, 0, 0, 0),
            show_axes=True,
            axes_length=AXES_LENGTH,
            axes_radius=AXES_RADIUS,
        )
        _object_viser = ViserUrdf(
            SERVER, OBJECT_URDF_PATH, root_node_name=object_frame_viser.name
        )

        from tqdm import tqdm

        while True:
            # Ask user if they want to continue or visualize again
            while True:
                user_input = input(
                    colored(
                        "Press 'v' to visualize, or 'c' to continue, or 'b' to breakpoint: ",
                        "cyan",
                    )
                )
                if user_input.lower() == "v":
                    info("Visualizing trajectory")
                    break
                elif user_input.lower() == "c":
                    info("Continuing to use this trajectory")
                    break
                elif user_input.lower() == "b":
                    info("Breakpointing")
                    breakpoint()
                else:
                    warn("Invalid input. Please enter 'v' or 'c'.")

            if user_input.lower() == "c":
                break

            for i in tqdm(range(TRAJECTORY_LENGTH), desc="Visualizing trajectory"):
                start_time = time.time()

                # Update joints
                q_target = q_targets_using_lifted_object_pose[i]
                assert q_target.shape == (29,), (
                    f"q_target.shape: {q_target.shape}, expected: (29,)"
                )
                kuka_sharpa_viser.update_cfg(q_target)
                sharpa_viser.update_cfg(q_target[7:])

                # Update floating hand position
                T_W_P_using_lifted_object_pose = T_W_Ps_using_lifted_object_pose[i]
                sharpa_frame.position = T_W_P_using_lifted_object_pose[:3, 3]
                sharpa_frame.wxyz = xyzw_to_wxyz(
                    R.from_matrix(T_W_P_using_lifted_object_pose[:3, :3]).as_quat()
                )

                # Update object position
                object_frame_viser.position = T_W_O_trajectory[i][:3, 3]
                object_frame_viser.wxyz = xyzw_to_wxyz(
                    R.from_matrix(T_W_O_trajectory[i][:3, :3]).as_quat()
                )

                end_time = time.time()
                loop_dt = end_time - start_time
                sleep_dt = 1 / 60 - loop_dt
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
                else:
                    warn(
                        f"Loop too slow! Desired FPS = 60, Actual FPS = {1 / loop_dt:.1f}"
                    )

    def _wait_and_warmup(self):
        assert not self._warmup_completed, "Warmup already completed"

        # Wait
        while not rospy.is_shutdown():
            obs, q, _ = self.create_observation()
            if obs is not None and q is not None:
                break
            time.sleep(self.control_dt)

        # Done waiting
        info("=" * 100)
        info("First observations received, starting to publish sim state")
        info("=" * 100)

        self.prev_targets = q

        # Warm up the policy and publishing
        info("=" * 100)
        info("Warming up policy and publishing current targets")
        info("=" * 100)
        # THIS IS NOT THE REAL LOOP, DON'T CARE ABOUT THESE NUMBERs
        num_steps = 0
        NUM_WARMUP_STEPS = 100
        while not rospy.is_shutdown():
            num_steps += 1
            info(f"Warmup step {num_steps} of {NUM_WARMUP_STEPS}")
            if num_steps > NUM_WARMUP_STEPS:
                info(f"Reached {NUM_WARMUP_STEPS} steps, stopping warmup")
                break

            # Create observation from the latest messages
            obs, q, _ = self.create_observation()
            assert obs is not None and q is not None, f"obs: {obs}, q: {q}"
            assert_equals(obs.shape, (1, self.num_observations))

            # Get the normalized action from the RL player
            normalized_action = self.player.get_normalized_action(
                obs=obs,
                deterministic_actions=True,
            )
            # normalized_action = torch.zeros(1, self.num_actions, device=self.device)
            assert_equals(normalized_action.shape, (1, self.num_actions))

            DUMMY_HAND_MOVING_AVERAGE = 0.1
            DUMMY_ARM_MOVING_AVERAGE = 0.1
            DUMMY_HAND_DOF_SPEED_SCALE = 2.5
            DUMMY_DT = 1 / 60
            _ = compute_joint_pos_targets(
                actions=normalized_action.cpu().numpy(),
                prev_targets=self.prev_targets[None],
                hand_moving_average=DUMMY_HAND_MOVING_AVERAGE,
                arm_moving_average=DUMMY_ARM_MOVING_AVERAGE,
                hand_dof_speed_scale=DUMMY_HAND_DOF_SPEED_SCALE,
                dt=DUMMY_DT,
                robot_asset_file=self.robot_asset_file,
            )

            # We do not actually use the joint pos targets computed by the policy, we use the actual joint states so it doesn't move
            joint_pos_targets = np.clip(
                q[None],
                # self.prev_targets[None],
                self.q_lower_limits,
                self.q_upper_limits,
            )

            # Publish the targets
            self.publish_targets(joint_pos_targets)
            self.prev_targets = joint_pos_targets[0]
            time.sleep(self.control_dt)

        # Reset rnn state
        self.player.reset()

        # Done warming up
        self._warmup_completed = True
        info("=" * 100)
        info("Warmup complete")
        info("=" * 100)
        assert self._warmup_completed, "Warmup completed"

    def run(self):
        self._wait_and_warmup()

        loop_no_sleep_dts, loop_dts = [], []
        while not rospy.is_shutdown():
            start_loop_no_sleep_time = time.time()

            # Profiling:
            # t0 is the time at which the earliest ROS message was received that was used to create the observation
            # t1 is the time at which the policy loop started
            # t1_5 is the time at which the targets were computed (computed obs, policy forward pass, compute targets)
            # t2 is the time at which the targets were done being published
            # t3 is the time that the robot receives the targets (not captured here)

            # Create observation from the latest messages
            obs, q, _ = self.create_observation()
            assert obs is not None and q is not None, f"obs: {obs}, q: {q}"

            assert_equals(obs.shape, (1, self.num_observations))

            # Get the normalized action from the RL player
            normalized_action = self.player.get_normalized_action(
                obs=obs,
                deterministic_actions=True,
            )
            assert_equals(normalized_action.shape, (1, self.num_actions))

            DT = 1 / 60
            joint_pos_targets = compute_joint_pos_targets(
                actions=normalized_action.cpu().numpy(),
                prev_targets=self.prev_targets[None],
                hand_moving_average=self.hand_moving_average,
                arm_moving_average=self.arm_moving_average,
                hand_dof_speed_scale=self.hand_dof_speed_scale,
                dt=DT,
                robot_asset_file=self.robot_asset_file,
            )
            assert_equals(joint_pos_targets.shape, (1, self.num_actions))

            # Clamp
            joint_pos_targets = np.clip(
                joint_pos_targets,
                self.q_lower_limits,
                self.q_upper_limits,
            )

            if self.overwrite_targets_filepath is not None:
                if self.current_step >= self.q_targets_from_file.shape[0]:
                    self.current_step = self.q_targets_from_file.shape[0] - 1
                    info("Reached end of targets, holding last target")
                assert self.current_step < self.q_targets_from_file.shape[0], (
                    f"current_step: {self.current_step}, expected: < {self.q_targets_from_file.shape[0]}"
                )
                joint_pos_targets = self.q_targets_from_file[self.current_step][None]
                self.current_step += 1

            if self.use_relative_object_pose_once_lifted:
                # Lifted object threshold
                if self.automatically_detect_object_lifted:
                    LIFTED_Z = 0.6
                    object_z = self.object_pose_msg.pose.position.z
                    if object_z >= LIFTED_Z:
                        self.object_lifted = True

                # When just lifted, initialize the relative object pose logic and visualize it
                if (
                    self.object_lifted
                    and not self.initialized_relative_object_pose_logic
                ):
                    self.initialized_relative_object_pose_logic = True
                    T_R_O_lifted = pose_msg_to_T(
                        copy.deepcopy(self.object_pose_msg.pose)
                    )  # Object pose in robot frame
                    self._initialize_relative_object_pose_logic(
                        q=q,
                        q_target=joint_pos_targets[0],
                        T_R_O_lifted=T_R_O_lifted,
                    )
                    self._visualize_relative_object_pose_logic(
                        q_targets_using_lifted_object_pose=self.q_targets_using_lifted_object_pose,
                        T_W_Ps_using_lifted_object_pose=self.T_W_Ps_using_lifted_object_pose,
                        T_W_O_trajectory=self.T_W_O_trajectory,
                    )

                    self.goal_object_pose_pub = rospy.Publisher(
                        "/robot_frame/goal_object_pose", Pose, queue_size=1
                    )

                # When the object is lifted, use the relative object pose logic to compute the joint pos targets
                if self.object_lifted:
                    # Update the joint pos targets
                    if not hasattr(self, "trajectory_idx"):
                        self.trajectory_idx = 0
                    joint_pos_targets = self.q_targets_using_lifted_object_pose[
                        self.trajectory_idx
                    ][None]

                    # Publish the goal object pose
                    current_T_W_G = self.T_W_O_trajectory[self.trajectory_idx]
                    current_T_R_G = np.linalg.inv(T_W_R) @ current_T_W_G
                    current_goal_object_pos, current_goal_object_quat_xyzw = (
                        T_to_pos_quat_xyzw(current_T_R_G)
                    )
                    goal_object_pose_msg = pos_quat_xyzw_to_pose_msg(
                        pos=current_goal_object_pos,
                        quat_xyzw=current_goal_object_quat_xyzw,
                    )
                    self.goal_object_pose_pub.publish(goal_object_pose_msg)

                    # Update the trajectory index
                    self.trajectory_idx += 1
                    if (
                        self.trajectory_idx
                        >= self.q_targets_using_lifted_object_pose.shape[0]
                    ):
                        self.trajectory_idx = (
                            self.q_targets_using_lifted_object_pose.shape[0] - 1
                        )
                        info(
                            f"Reached end of q_targets_using_lifted_object_pose, setting to last target {self.trajectory_idx}"
                        )

            # Sanity check that the joint pos targets are not too far from the current joint positions
            q_arm_diff_deg = np.rad2deg(np.abs(joint_pos_targets[0, :7] - q[:7]))
            MAX_Q_ARM_DIFF_DEG = 10
            if q_arm_diff_deg.max() > MAX_Q_ARM_DIFF_DEG:
                error(
                    f"Joint pos targets are too far from current joint positions, q_arm_diff: {q_arm_diff_deg} (max: {MAX_Q_ARM_DIFF_DEG})"
                )
                breakpoint()

            # Publish the targets
            self.publish_targets(joint_pos_targets)
            self.prev_targets = joint_pos_targets[0]

            # End of loop timekeeping
            end_loop_no_sleep_time = time.time()
            loop_no_sleep_dt = end_loop_no_sleep_time - start_loop_no_sleep_time
            loop_no_sleep_dts.append(loop_no_sleep_dt)

            sleep_dt = self.control_dt - loop_no_sleep_dt
            if sleep_dt > 0:
                time.sleep(sleep_dt)
                loop_dt = loop_no_sleep_dt + sleep_dt
            else:
                loop_dt = loop_no_sleep_dt
                warn(
                    f"Simulation is running slower than real time, desired FPS = {1.0 / self.control_dt:.1f}, actual FPS = {1.0 / loop_dt:.1f}"
                )
            loop_dts.append(loop_dt)

            PRINT_FPS_EVERY_N_SECONDS = 5.0
            PRINT_FPS_EVERY_N_STEPS = int(PRINT_FPS_EVERY_N_SECONDS / self.control_dt)
            if len(loop_dts) == PRINT_FPS_EVERY_N_STEPS:
                loop_dt_array = np.array(loop_dts)
                loop_no_sleep_dt_array = np.array(loop_no_sleep_dts)
                fps_array = 1.0 / loop_dt_array
                fps_no_sleep_array = 1.0 / loop_no_sleep_dt_array
                print("FPS with sleep:")
                print(f"  Mean: {np.mean(fps_array):.1f}")
                print(f"  Median: {np.median(fps_array):.1f}")
                print(f"  Max: {np.max(fps_array):.1f}")
                print(f"  Min: {np.min(fps_array):.1f}")
                print(f"  Std: {np.std(fps_array):.1f}")
                print("FPS without sleep:")
                print(f"  Mean: {np.mean(fps_no_sleep_array):.1f}")
                print(f"  Median: {np.median(fps_no_sleep_array):.1f}")
                print(f"  Max: {np.max(fps_no_sleep_array):.1f}")
                print(f"  Min: {np.min(fps_no_sleep_array):.1f}")
                print(f"  Std: {np.std(fps_no_sleep_array):.1f}")
                print()
                loop_no_sleep_dts, loop_dts = [], []

    def _signal_handler(self, signum, frame):
        assert self.save_foldername is not None, (
            "save_foldername must be set to save to file"
        )

        if self._is_in_progress_saving_to_file:
            warn("Already in progress of saving to file, skipping")
            return

        self._is_in_progress_saving_to_file = True
        if len(self.time_history) == 0:
            warn("No data recorded, skipping")
        else:
            info(f"Received signal {signum}, saving to file")
            datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            filename = f"{datetime_str}_{self.checkpoint_path.stem}_arm{self.arm_moving_average}_{self.object_name}"
            if self.overwrite_targets_filepath is not None:
                filename = (
                    f"{datetime_str}_replay_{self.overwrite_targets_filepath.stem}"
                )
            output_path = (
                Path("recorded_robot_inputs") / self.save_foldername / f"{filename}.npz"
            )
            self.save_to_file(output_path)
            info(f"Saved to file: {output_path}")

        rospy.signal_shutdown("Shutting down")

    def save_to_file(self, file_path: Path):
        assert self.save_foldername is not None, (
            "save_foldername must be set to save to file"
        )

        file_path.parent.mkdir(parents=True, exist_ok=True)
        info(f"Saving to file: {file_path}")

        T = len(self.time_history)
        robot_root_states_array = np.zeros((T, 13))
        robot_root_states_array[:, 1] = 0.8
        robot_root_states_array[:, 6] = 1.0  # quaternion xyzw has w=1
        object_root_states_array = np.zeros((T, 13))
        object_root_states_array[:, :7] = np.array(self.object_pose_history)
        table_root_states_array = np.zeros((T, 13))
        table_root_states_array[:, :3] = np.array([0.0, 0.0, 0.38])[None]
        goal_root_states_array = np.zeros((T, 13))
        goal_root_states_array[:, :7] = np.array(self.goal_object_pose_history)

        robot_joint_positions = np.array(self.q_history)
        robot_joint_velocities = np.array(self.qd_history)

        robot_joint_pos_targets = np.array(self.q_target_history)
        time_array = np.array(self.time_history)

        assert robot_joint_positions.shape == (T, 29), (
            f"robot_joint_positions.shape: {robot_joint_positions.shape}, expected: (T, 29)"
        )
        assert robot_joint_velocities.shape == (T, 29), (
            f"robot_joint_velocities.shape: {robot_joint_velocities.shape}, expected: (T, 29)"
        )
        assert robot_joint_pos_targets.shape == (T, 29), (
            f"robot_joint_pos_targets.shape: {robot_joint_pos_targets.shape}, expected: (T, 29)"
        )
        assert object_root_states_array.shape == (T, 13), (
            f"object_root_states_array.shape: {object_root_states_array.shape}, expected: (T, 13)"
        )
        assert time_array.shape == (T,), (
            f"time_array.shape: {time_array.shape}, expected: (T,)"
        )

        JOINT_NAMES = [
            "iiwa14_joint_1",
            "iiwa14_joint_2",
            "iiwa14_joint_3",
            "iiwa14_joint_4",
            "iiwa14_joint_5",
            "iiwa14_joint_6",
            "iiwa14_joint_7",
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
        ]

        from recorded_data import RecordedData

        recorded_data = RecordedData(
            robot_root_states_array=robot_root_states_array,
            object_root_states_array=object_root_states_array,
            robot_joint_positions_array=robot_joint_positions,
            time_array=time_array,
            robot_joint_names=JOINT_NAMES,
            robot_joint_velocities_array=robot_joint_velocities,
            robot_joint_pos_targets_array=robot_joint_pos_targets,
            goal_root_states_array=goal_root_states_array,
            object_name=self.object_name,
        )
        recorded_data.to_file(file_path)


@dataclass
class RLPolicyNodeArgs:
    policy_path: Path = Path("pretrained_policy")
    """Path to the policy directory."""

    object_name: str = "claw_hammer"
    """The name of the object whose grasp bounding box will be used as input to the policy."""


def main():
    args: RLPolicyNodeArgs = tyro.cli(RLPolicyNodeArgs)

    config_path = args.policy_path / "config.yaml"
    checkpoint_path = args.policy_path / "model.pth"
    assert config_path.exists(), f"Config path not found: {config_path}"
    assert checkpoint_path.exists(), f"Checkpoint path not found: {checkpoint_path}"

    try:
        rl_policy_node = RLPolicyNode(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            hand_moving_average=0.1,
            arm_moving_average=0.1,
            hand_dof_speed_scale=1.5,
            object_scales=np.array(NAME_TO_OBJECT[args.object_name].scale),
            save_foldername=f"{datetime.datetime.now().strftime('%Y-%m-%d')}_testing",
            use_relative_object_pose_once_lifted=True,
            object_name=args.object_name,
            automatically_detect_object_lifted=False,
        )
        rl_policy_node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
