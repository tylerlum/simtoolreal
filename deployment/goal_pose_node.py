#!/usr/bin/env python
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rospy
import tyro
from geometry_msgs.msg import Pose, PoseStamped
from termcolor import colored

from isaacgymenvs.utils.observation_action_utils_sharpa import (
    _compute_keypoint_positions,
)
from isaacgymenvs.utils.utils import get_repo_root_dir

# HARDCODED_TRAJECTORY_PATH = Path.home() / "Downloads/brush_object_goal_poses_NEW.json"


def info(message: str):
    print(colored(message, "green"))


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


def keypoint_distance(
    pose1_xyzw: np.ndarray, pose2_xyzw: np.ndarray, object_scales: np.ndarray
) -> float:
    """Compute the distance between two keypoints."""
    object_keypoint_positions = _compute_keypoint_positions(
        pose=pose1_xyzw[None], scales=object_scales[None]
    )
    goal_keypoint_positions = _compute_keypoint_positions(
        pose=pose2_xyzw[None], scales=object_scales[None]
    )
    keypoints_rel_goal = object_keypoint_positions - goal_keypoint_positions
    N_KEYPOINTS = 4
    N = 1
    assert keypoints_rel_goal.shape == (N, N_KEYPOINTS, 3), (
        f"keypoints_rel_goal.shape: {keypoints_rel_goal.shape}, expected: (N, N_KEYPOINTS, 3)"
    )
    keypoint_distances_l2 = np.linalg.norm(keypoints_rel_goal, axis=-1).max(axis=-1)
    return keypoint_distances_l2


class GoalPoseNode:
    def __init__(
        self,
        goal_poses_robot_frame: np.ndarray,  # Assumes xyzw quat convention and robot frame
        object_scales: np.ndarray,
        success_threshold: float,
        success_steps: int,
        force_open_loop: bool = False,
        force_fixed_orientation: bool = False,
    ):
        # ROS setup
        rospy.init_node("goal_pose_node")

        KEYPOINT_SCALE = 1.5
        self.object_scales = object_scales
        self.success_threshold = success_threshold
        self.keypoint_success_threshold = success_threshold * KEYPOINT_SCALE
        self.success_steps = success_steps
        self.force_fixed_orientation = force_fixed_orientation
        self.current_success_steps = 0

        # Goal object pose
        self.goal_object_poses = goal_poses_robot_frame
        N = len(self.goal_object_poses)
        assert self.goal_object_poses.shape == (N, 7), (
            f"goal_object_poses.shape: {self.goal_object_poses.shape}, expected: (N, 7)"
        )

        # State
        self.current_goal_object_pose_index = 0

        # ROS msgs
        self.latest_current_object_pose = None

        # Force open loop mode, i.e., do not use the current object pose to update the goal pose, but update it at a fixed rate.
        if force_open_loop:
            self.latest_current_object_pose = Pose()
            self.success_threshold = 10.0
            self.keypoint_success_threshold = self.success_threshold * KEYPOINT_SCALE
            self.success_steps = 30

        # Publisher and subscriber
        self.goal_object_pose_pub = rospy.Publisher(
            "/robot_frame/goal_object_pose", Pose, queue_size=1
        )
        self.current_object_pose_sub = rospy.Subscriber(
            "/robot_frame/current_object_pose",
            PoseStamped,
            self.current_object_pose_callback,
            queue_size=1,
        )

        # Set control rate to 60Hz
        self.rate_hz = 60
        self.dt = 1 / self.rate_hz
        self.rate = rospy.Rate(self.rate_hz)

    def current_object_pose_callback(self, msg: PoseStamped):
        """Callback to update the current object pose."""
        self.latest_current_object_pose = msg.pose

    def update_goal_object_pose(self):
        """Update the goal object pose."""
        num_goals = self.goal_object_poses.shape[0]
        if self.current_goal_object_pose_index >= num_goals:
            print(colored("Reached end of goal object poses", "blue"))
            print(
                colored(
                    f"self.current_goal_object_pose_index/num_goals: {self.current_goal_object_pose_index}/{num_goals} = {self.current_goal_object_pose_index / num_goals:.2%}",
                    "blue",
                )
            )
            return

        latest_current_object_pose = deepcopy(self.latest_current_object_pose)
        p = latest_current_object_pose

        current_object_pose_xyzw = np.array(
            [
                p.position.x,
                p.position.y,
                p.position.z,
                p.orientation.x,
                p.orientation.y,
                p.orientation.z,
                p.orientation.w,
            ]
        )
        current_goal_object_pose_xyzw = self.goal_object_poses[
            self.current_goal_object_pose_index
        ]

        if self.force_fixed_orientation:
            # Overwrite with fixed orientation
            current_object_pose_xyzw = np.copy(current_object_pose_xyzw)
            current_goal_object_pose_xyzw = np.copy(current_goal_object_pose_xyzw)
            current_object_pose_xyzw[3:7] = np.array([0, 0, 0, 1])
            current_goal_object_pose_xyzw[3:7] = np.array([0, 0, 0, 1])

        distance = keypoint_distance(
            pose1_xyzw=current_object_pose_xyzw,
            pose2_xyzw=current_goal_object_pose_xyzw,
            object_scales=self.object_scales,
        )
        num_goals = self.goal_object_poses.shape[0]
        print(
            f"Distance: {distance}, self.current_goal_object_pose_index/num_goals: {self.current_goal_object_pose_index}/{num_goals} = {self.current_goal_object_pose_index / num_goals:.2%}"
        )

        # HACK: Different threshold per idx
        threshold = self.keypoint_success_threshold
        # if self.current_goal_object_pose_index > 1:
        #     # threshold = self.keypoint_success_threshold * 2
        #     threshold = self.keypoint_success_threshold * 2.5
        #     print(f"Using LOOSER threshold because self.current_goal_object_pose_index = {self.current_goal_object_pose_index}")
        # else:
        #     print(f"Using TIGHTER threshold because self.current_goal_object_pose_index = {self.current_goal_object_pose_index}")

        if distance < threshold:
            self.current_success_steps += 1
            if self.current_success_steps >= self.success_steps:
                info(
                    f"Success threshold reached, updating goal object pose index to {self.current_goal_object_pose_index + 1}"
                )
                self.current_success_steps = 0
                self.current_goal_object_pose_index += 1
                # if self.current_goal_object_pose_index >= self.goal_object_poses.shape[0]:
                #     self.current_goal_object_pose_index = self.goal_object_poses.shape[0] - 1
            else:
                info(
                    f"Success threshold reached, at {self.current_success_steps} of {self.success_steps} steps"
                )

    def publish_goal_object_pose(self):
        """Publish the goal object pose."""
        idx = self.current_goal_object_pose_index
        if idx >= self.goal_object_poses.shape[0]:
            idx = self.goal_object_poses.shape[0] - 1
        elif idx < 0:
            idx = 0

        current_goal_object_pose_xyzw = self.goal_object_poses[idx]
        goal_object_pose_msg = Pose()
        goal_object_pose_msg.position.x = current_goal_object_pose_xyzw[0]
        goal_object_pose_msg.position.y = current_goal_object_pose_xyzw[1]
        goal_object_pose_msg.position.z = current_goal_object_pose_xyzw[2]
        goal_object_pose_msg.orientation.x = current_goal_object_pose_xyzw[3]
        goal_object_pose_msg.orientation.y = current_goal_object_pose_xyzw[4]
        goal_object_pose_msg.orientation.z = current_goal_object_pose_xyzw[5]
        goal_object_pose_msg.orientation.w = current_goal_object_pose_xyzw[6]

        self.goal_object_pose_pub.publish(goal_object_pose_msg)

    def run(self):
        """Main loop to run the node, update simulation, and publish joint states."""

        # Wait for the current object pose to be received
        while not rospy.is_shutdown():
            if self.latest_current_object_pose is None:
                warn_every("Waiting for current object pose", n_seconds=1.0)
                time.sleep(0.1)
            else:
                info("Current object pose received, starting goal pose node")
                break  # All messages received, exit loop

        loop_no_sleep_dts, loop_dts = [], []
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Update the goal object pose
            self.update_goal_object_pose()

            # Publish the goal object pose
            self.publish_goal_object_pose()

            # Sleep to maintain the loop rate
            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()

            loop_no_sleep_dt = (before_sleep_time - start_time).to_sec()
            loop_no_sleep_dts.append(loop_no_sleep_dt)
            loop_dt = (after_sleep_time - start_time).to_sec()
            loop_dts.append(loop_dt)

            PRINT_FPS_EVERY_N_SECONDS = 5.0
            PRINT_FPS_EVERY_N_STEPS = int(PRINT_FPS_EVERY_N_SECONDS / self.dt)
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


@dataclass
class GoalPoseNodeArgs:
    object_category: str = "hammer"
    object_name: str = "claw_hammer"
    task_name: str = "swing_down"

    success_threshold: float = 0.02
    """Success threshold in meters."""

    success_steps: int = 1
    """Number of steps to consider a success."""

    force_open_loop: bool = False
    """Force open loop mode, i.e., do not use the current object pose to update the goal pose, but update it at a fixed rate."""

    force_fixed_orientation: bool = False
    """Force fixed orientation mode, i.e., overwrite the orientation with a fixed one."""


def main():
    args: GoalPoseNodeArgs = tyro.cli(GoalPoseNodeArgs)

    # Load trajectory
    # trajectory_path = HARDCODED_TRAJECTORY_PATH
    # trajectory_path = (
    #     get_repo_root_dir()
    #     / "dextoolbench/trajectories"
    #     / args.object_category
    #     / args.object_name
    #     / f"{args.task_name}.json"
    # )
    # assert trajectory_path.exists(), f"Trajectory file not found: {trajectory_path}"
    # with open(trajectory_path) as f:
    #     traj_data = json.load(f)

    # # Account for robot to world frame
    # goal_poses_world_frame = traj_data["goals"]
    # goal_poses_robot_frame = [
    #     [x, y - 0.8, z, qx, qy, qz, qw]
    #     for x, y, z, qx, qy, qz, qw in goal_poses_world_frame
    # ]

    OVERWRITE = True
    if OVERWRITE:
        # 2026-05-25 First Try part 2 (vertical)
        #   position:
        #     x: -0.08844813556818426
        #     y: -0.7765577369600304
        #     z: 0.649823525142813
        #   orientation:
        #     x: 0.002249703855947019
        #     y: -0.7059066758501827
        #     z: 0.021154703502566832
        #     w: 0.7079852981117781
        # insert_pose = np.array([
        #     -0.08844,
        #     -0.7765,
        #     0.649,
        #     0.0,
        #     -0.707,
        #     0.0,
        #     0.707,
        # ])

        # 2026-06-13 Part 2 (vertical)
        #   position: 
        #     x: -0.09571262756182047
        #     y: -0.7639454461950859
        #     z: 0.6324156484909513
        #   orientation: 
        #     x: 0.021573868183511386
        #     y: -0.7199091668954553
        #     z: 0.011502597041645
        #     w: 0.6936375493676722
        # insert_pose = np.array([
        #     -0.09571,
        #     -0.76394,
        #     # 0.63241,
        #     0.649,
        #     0.0,
        #     -0.707,
        #     0.0,
        #     0.707,
        # ])

        # 2026-06-13 v2 Part 2 (horizontal)
        #   position: 
        #     x: -0.09244088208872892
        #     y: -0.7603925938400313
        #     z: 0.6876884632801282
        # insert_pose = np.array([
        #     # -0.09571,
        #     # -0.76394,
        #     # 0.63241,
        #     -0.08452915171043296,
        #     -0.7722208424055638,
        #     0.649,
        #     0.0,
        #     -0.707,
        #     0.0,
        #     0.707,
        # ])

        #     x: -0.0892717019451631
        #     y: -0.7612253518112828
        #     z: 0.682896966697796
        # insert_pose = np.array([
        #     -0.08927,
        #     -0.76122,
        #     0.68289 - 0.0375,
        #     0.0,
        #     -0.707,
        #     0.0,
        #     0.707,
        # ])

        #     x: -0.08624174066673934
        #     y: -0.7693611909191991
        #     z: 0.6852802427130255
        # insert_pose = np.array([
        #     -0.08624,
        #     -0.76936,
        #     0.68528 - 0.0375,
        #     0.0,
        #     -0.707,
        #     0.0,
        #     0.707,
        # ])

        #     x: -0.08322641169514289
        #     y: -0.7688177412617602
        #     z: 0.6825694397159632
        # insert_pose = np.array([
        #     -0.08322,
        #     -0.7688,
        #     0.68528 - 0.0375,
        #     0.0,
        #     -0.707,
        #     0.0,
        #     0.707,
        # ])

        # 2026-08-06 v1
        #   position:
        #     x: -0.09097027262045054
        #     y: -0.7628996776483543
        #     z: 0.6460198927747884
        #   orientation:
        #     x: 0.032962811296989
        #     y: -0.6922682008323167
        #     z: 0.08748273418201578
        #     w: 0.7155591956000741
        insert_pose = np.array([
            -0.09097027262045054,
            -0.7628996776483543,
            0.6460198927747884,
            0.032962811296989,
            -0.6922682008323167,
            0.08748273418201578,
            0.7155591956000741,
        ])

        # 2026-08-06 v2: Pre-insert pose
        #   position: 
        #     x: -0.09111804405729668
        #     y: -0.7637948569732023
        #     z: 0.6819707579804075
        #   orientation: 
        #     x: 0.012066834117826468
        #     y: -0.7010927913258167
        #     z: 0.04477382147611119
        #     w: 0.7115606751189759

        # 2026-08-06 v3: pre-insert pose
        #   position:
        #     x: -0.09624589721783183
        #     y: -0.7726821816176856
        #     z: 0.6820946898028508
        #   orientation:
        #     x: 0.042055281047737966
        #     y: -0.7081854618551252
        #     z: 0.05189763344871776
        #     w: 0.7028594031493508
        # insert_pose = np.array([
        #     -0.09624589721783183,
        #     -0.7726821816176856,
        #     0.6820946898028508 - 0.0375,
        #     0.0,
        #     -np.sqrt(0.5),
        #     0.0,
        #     np.sqrt(0.5),
        # ])


        # goal_mode = "screw"
        goal_mode = "preinsert"
        if goal_mode == "preinsert":
            preinsert_pose = insert_pose.copy()
            print("OVERWRITING GOAL POSES WITH INSERT POSE")
            DZ = 0.0375
            # DZ = 0.05
            preinsert_pose[2] += DZ
            goal_poses_robot_frame = [preinsert_pose.tolist(), insert_pose.tolist()]
        elif goal_mode == "screw":
            import sys

            from pathlib import Path
            root_dir = Path(__file__).parent.parent
            print(f"Adding {root_dir} to path")
            sys.path.insert(0, str(root_dir))
            from peg_in_hole_dynamic.furniture_bench.problems import _one_leg_super_dense_insert_waypoints

            DY = 0.005
            insert_pose[1] += DY

            DZ = 0.005
            # DZ = 0.0025
            # insert_pose[2] -= DZ
            waypoints = np.array(_one_leg_super_dense_insert_waypoints(insert_pose.tolist()))
            # waypoints[0, 2] += 
            # waypoints[1:, 2] -= DZ
            # print(f"waypoints = {waypoints}")
            # print(f"waypoints[0] = {waypoints[0]}")
            # print(f"waypoints[1] = {waypoints[1]}")
            # print(f"waypoints[2] = {waypoints[2]}")
            # breakpoint()
            goal_poses_robot_frame = waypoints.tolist()
        else:
            raise ValueError("Bad")

    try:
        # Create and run the GoalPoseNode
        node = GoalPoseNode(
            goal_poses_robot_frame=np.array(goal_poses_robot_frame),
            object_scales=np.array([0.141, 0.03025, 0.0271]) * 25,  # fixed size
            success_threshold=args.success_threshold,
            success_steps=args.success_steps,
            force_open_loop=args.force_open_loop,
            force_fixed_orientation=args.force_fixed_orientation,
        )
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
