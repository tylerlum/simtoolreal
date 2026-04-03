# NOTE: torch must be imported AFTER isaacgym imports
# isort: off
from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
import torch
# isort: on

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import tyro
import yourdfpy
from termcolor import colored

from deployment.isaac.isaac_env import create_env
from deployment.rl_player import RlPlayer
from isaacgymenvs.utils.observation_action_utils import (
    compute_joint_pos_targets,
    compute_observation,
    create_urdf_object,
)
from isaacgymenvs.utils.utils import get_repo_root_dir

HAND_MOVING_AVERAGE = 0.1
ARM_MOVING_AVERAGE = 0.05
HAND_DOF_SPEED_SCALE = 2.5


def warn(message: str):
    print(colored(message, "yellow"))


def info(message: str):
    print(colored(message, "green"))


class IsaacEnvNoRos:
    def __init__(
        self,
        env: SimToolReal,
        control_dt: float,
        device: str,
        urdf: yourdfpy.URDF,
        robot_asset_file: str,
    ):
        self.env = env
        self.control_dt = control_dt
        self.device = device
        self.urdf = urdf
        self.robot_asset_file = robot_asset_file
        self.num_actions = self.env.cfg["env"]["numActions"]

    def reset(self) -> torch.Tensor:
        obs, _, _, _ = self.env.step(
            torch.zeros((self.env.num_envs, self.num_actions), device=self.device)
        )
        return obs["obs"]

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        joint_pos_targets = compute_joint_pos_targets(
            actions=action.cpu().numpy(),
            prev_targets=self.env.prev_targets.cpu().numpy(),
            hand_moving_average=HAND_MOVING_AVERAGE,
            arm_moving_average=ARM_MOVING_AVERAGE,
            hand_dof_speed_scale=HAND_DOF_SPEED_SCALE,
            dt=self.control_dt,
            robot_asset_file=self.robot_asset_file,
        )
        joint_pos_targets = torch.from_numpy(joint_pos_targets).float().to(self.device)

        obs, reward, done, info = self.env.step(
            action, joint_pos_targets=joint_pos_targets
        )
        q = self.env.arm_hand_dof_pos
        qd = self.env.arm_hand_dof_vel
        object_pose = self.env.object_pose
        goal_object_pose = self.env.goal_pose
        object_scales = self.env.object_scales

        DEBUG = False
        if DEBUG:
            print(f"q = {q}")
            print(f"qd = {qd}")
            print(f"object_pose = {object_pose}")
            print(f"goal_object_pose = {goal_object_pose}")
            print(f"object_scales = {object_scales}")
            breakpoint()

        new_obs = compute_observation(
            q=q.cpu().numpy(),
            qd=qd.cpu().numpy(),
            prev_action_targets=self.env.prev_targets.cpu().numpy(),
            object_pose=object_pose.cpu().numpy(),
            goal_object_pose=goal_object_pose.cpu().numpy(),
            object_scales=object_scales.cpu().numpy(),
            urdf=self.urdf,
            obs_list=self.env.obs_list,
            robot_asset_file=self.robot_asset_file,
        )
        new_obs = torch.from_numpy(new_obs).float().to(self.device)

        DEBUG = False
        if DEBUG:
            diff = (obs["obs"] - new_obs).abs()[0]
            print(f"diff = {diff}")
            print(f"diff.max() = {diff.max()}")
            print(f"diff.argsort() = {diff.argsort()}")

            from isaacgymenvs.utils.robot_info import get_obs_names

            obs_names = get_obs_names(self.robot_asset_file)
            idxs = diff.argsort()
            for idx in idxs:
                print(f"OBS_NAMES[{idx}] = {obs_names[idx]}")
                print(f"obs['obs'][{idx}] = {obs['obs'][0, idx]}")
                print(f"new_obs[{idx}] = {new_obs[0, idx]}")
                print(f"diff[{idx}] = {diff[idx]}")
                print("--------------------------------")

            breakpoint()
        return new_obs, reward, done, info


@dataclass
class IsaacEnvNoRosArgs:
    config_path: Path = Path("pretrained_policy/config.yaml")
    """Path to the policy config YAML."""

    checkpoint_path: Path = Path("pretrained_policy/model.pth")
    """Path to the policy checkpoint."""

    object_category: str = "hammer"
    """Object category (e.g. hammer, marker, spatula)."""

    object_name: str = "claw_hammer"
    """Object name within the category."""

    task_name: str = "swing_down"
    """Task / trajectory name."""

    control_hz: float = 60.0
    """Control loop frequency in Hz."""

    headless: bool = False
    """Run IsaacGym without rendering."""


def main():
    args: IsaacEnvNoRosArgs = tyro.cli(IsaacEnvNoRosArgs)

    assert args.config_path.exists(), f"Config not found: {args.config_path}"
    assert args.checkpoint_path.exists(), (
        f"Checkpoint not found: {args.checkpoint_path}"
    )

    control_dt = 1.0 / args.control_hz

    # Load trajectory
    trajectory_path = (
        get_repo_root_dir()
        / "dextoolbench/trajectories"
        / args.object_category
        / args.object_name
        / f"{args.task_name}.json"
    )
    assert trajectory_path.exists(), f"Trajectory file not found: {trajectory_path}"
    with open(trajectory_path) as f:
        traj_data = json.load(f)

    # NOTE: cpu has different physics than training
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    env = create_env(
        config_path=str(args.config_path),
        headless=args.headless,
        device=DEVICE,
        overrides={
            # Turn off randomization noise
            "task.env.resetPositionNoiseX": 0.0,
            "task.env.resetPositionNoiseY": 0.0,
            "task.env.resetPositionNoiseZ": 0.0,
            "task.env.randomizeObjectRotation": False,
            "task.env.resetDofPosRandomIntervalFingers": 0.0,
            "task.env.resetDofPosRandomIntervalArm": 0.0,
            "task.env.resetDofVelRandomInterval": 0.0,
            "task.env.tableResetZRange": 0.0,
            # Object
            "task.env.objectName": args.object_name,
            # Set up environment parameters
            "task.env.numEnvs": 1,
            "task.env.envSpacing": 0.4,
            "task.env.capture_video": False,
            # Goal settings
            "task.env.useFixedGoalStates": True,
            "task.env.fixedGoalStates": traj_data["goals"],
            # Delays and noise
            "task.env.useActionDelay": False,
            "task.env.useObsDelay": False,
            "task.env.useObjectStateDelayNoise": False,
            "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],
            # Reset
            "task.env.resetWhenDropped": False,
            # Moving average
            "task.env.armMovingAverage": 0.1,
            # Success criteria
            "task.env.evalSuccessTolerance": 0.01,
            "task.env.successSteps": 1,
            "task.env.fixedSizeKeypointReward": True,
            # Initialization
            "task.env.useFixedInitObjectPose": True,
            "task.env.objectStartPose": traj_data["start_pose"],
            "task.env.startArmHigher": True,
            # Forces/torques (all zero for eval)
            "task.env.forceScale": 0.0,
            "task.env.torqueScale": 0.0,
            "task.env.linVelImpulseScale": 0.0,
            "task.env.angVelImpulseScale": 0.0,
            "task.env.forceOnlyWhenLifted": True,
            "task.env.torqueOnlyWhenLifted": True,
            "task.env.linVelImpulseOnlyWhenLifted": True,
            "task.env.angVelImpulseOnlyWhenLifted": True,
            "task.env.forceProbRange": [0.0001, 0.0001],
            "task.env.torqueProbRange": [0.0001, 0.0001],
            "task.env.linVelImpulseProbRange": [0.0001, 0.0001],
            "task.env.angVelImpulseProbRange": [0.0001, 0.0001],
        },
    )

    # Set env state from checkpoint to match things like success_tolerance
    checkpoint = torch.load(args.checkpoint_path)
    env_state = checkpoint[0]["env_state"]
    env.set_env_state(env_state)

    robot_asset_file = env.cfg["env"]["asset"]["robot"]
    num_observations = env.cfg["env"]["numObservations"]
    num_actions = env.cfg["env"]["numActions"]
    policy = RlPlayer(
        num_observations=num_observations,
        num_actions=num_actions,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=DEVICE,
        num_envs=env.num_envs,
    )

    urdf = create_urdf_object(robot_asset_file)

    isaac_env = IsaacEnvNoRos(
        env=env,
        control_dt=control_dt,
        device=DEVICE,
        urdf=urdf,
        robot_asset_file=robot_asset_file,
    )
    observation = isaac_env.reset()

    while True:
        start_time = time.time()
        action = policy.get_normalized_action(observation, deterministic_actions=True)
        observation, _, _, _ = isaac_env.step(action)
        end_time = time.time()
        sleep_time = control_dt - (end_time - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            warn(
                f"Control loop too slow! Desired FPS: {args.control_hz:.1f}, Actual FPS: {1.0 / (end_time - start_time):.1f}"
            )


if __name__ == "__main__":
    main()
