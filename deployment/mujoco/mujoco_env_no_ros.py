import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
import yourdfpy
from termcolor import colored

from deployment.mujoco.mujoco_sim import (
    MujocoSim,
    MujocoSimConfig,
)
from deployment.rl_player import RlPlayer
from deployment.rl_player_utils import read_cfg
from dextoolbench.objects import NAME_TO_OBJECT
from isaacgymenvs.utils.observation_action_utils import (
    compute_joint_pos_targets,
    compute_observation,
    create_urdf_object,
)
from isaacgymenvs.utils.robot_info import get_num_observations, get_robot_spec


def warn(message: str):
    print(colored(message, "yellow"))


def info(message: str):
    print(colored(message, "green"))


class MujocoEnvNoRos:
    def __init__(
        self,
        sim: MujocoSim,
        object_scales: np.ndarray,
        urdf: yourdfpy.URDF,
        hand_moving_average: float,
        arm_moving_average: float,
        hand_dof_speed_scale: float,
        control_dt: float,
        device: str,
        obs_list: list[str],
        robot_asset_file: str,
    ):
        self.sim = sim
        self.object_scales = object_scales
        self.urdf = urdf
        self.hand_moving_average = hand_moving_average
        self.arm_moving_average = arm_moving_average
        self.hand_dof_speed_scale = hand_dof_speed_scale
        self.control_dt = control_dt
        self.device = device
        self.obs_list = obs_list
        self.robot_asset_file = robot_asset_file

    def compute_observation(self) -> torch.Tensor:
        sim_state = self.sim.get_sim_state()

        object_pos = sim_state["object_pos"]
        object_quat_wxyz = sim_state["object_quat_wxyz"]
        object_quat_xyzw = object_quat_wxyz[[1, 2, 3, 0]]
        object_pose_W = np.concatenate([object_pos, object_quat_xyzw])

        goal_object_pos = sim_state["goal_object_pos"]
        goal_object_quat_wxyz = sim_state["goal_object_quat_wxyz"]
        goal_object_quat_xyzw = goal_object_quat_wxyz[[1, 2, 3, 0]]
        goal_object_pose_W = np.concatenate([goal_object_pos, goal_object_quat_xyzw])

        q = sim_state["joint_positions"]
        qd = sim_state["joint_velocities"]

        observation = compute_observation(
            q=q[None],
            qd=qd[None],
            prev_action_targets=self.sim.robot_joint_pos_targets[None],
            object_pose=object_pose_W[None],
            goal_object_pose=goal_object_pose_W[None],
            object_scales=self.object_scales[None],
            urdf=self.urdf,
            obs_list=self.obs_list,
            robot_asset_file=self.robot_asset_file,
        )
        observation = torch.from_numpy(observation).float().to(self.device)
        return observation

    def step(self, action: torch.Tensor) -> None:
        joint_pos_targets = compute_joint_pos_targets(
            actions=action.cpu().numpy(),
            prev_targets=self.sim.robot_joint_pos_targets[None],
            hand_moving_average=self.hand_moving_average,
            arm_moving_average=self.arm_moving_average,
            hand_dof_speed_scale=self.hand_dof_speed_scale,
            dt=self.control_dt,
            robot_asset_file=self.robot_asset_file,
        )
        self.sim.set_robot_joint_pos_targets(joint_pos_targets[0])

        for _ in range(self.sim_steps_per_control_step):
            self.sim.sim_step()

            if self.sim.config.enable_viewer:
                self.sim.viewer.sync()

        return

    @property
    def sim_steps_per_control_step(self) -> int:
        return int(self.control_dt / self.sim.config.sim_dt)


@dataclass
class MujocoEnvNoRosArgs:
    config_path: Path = Path("pretrained_policy/config.yaml")
    """Path to the policy config YAML."""

    checkpoint_path: Path = Path("pretrained_policy/model.pth")
    """Path to the policy checkpoint."""

    object_name: str = "claw_hammer"
    """Object name."""


def main():
    args: MujocoEnvNoRosArgs = tyro.cli(MujocoEnvNoRosArgs)

    # Parameters
    SIM_DT = 1.0 / 600.0  # Mujoco sim step (needs to be small to get stable physics)
    CONTROL_DT = 1.0 / 60.0  # Control loop frequency (policy loop rate)
    HAND_MOVING_AVERAGE = 0.1
    ARM_MOVING_AVERAGE = 0.1
    HAND_DOF_SPEED_SCALE = 1.5

    # Cuboid
    OBJECT_SCALES = np.array(NAME_TO_OBJECT[args.object_name].scale)

    assert args.config_path.exists(), f"Config not found: {args.config_path}"
    assert args.checkpoint_path.exists(), (
        f"Checkpoint not found: {args.checkpoint_path}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = MujocoSim(
        MujocoSimConfig(
            enable_viewer=True,
            sim_dt=SIM_DT,
            object_name="cuboidal_mallet",
            object_start_pos=np.array([0.0, 0.0, 0.58]),
            object_start_quat_wxyz=np.array([0.0, 0.0, 0.0, 1.0]),
            goal_object_start_pos=np.array([0.0, 0.0, 0.78]),
            goal_object_start_quat_wxyz=np.array([0.0, 0.0, 0.0, 1.0]),
        )
    )
    cfg = read_cfg(str(args.config_path), device=device)
    robot_asset_file = cfg["task"]["env"]["asset"]["robot"]
    robot_spec = get_robot_spec(robot_asset_file)
    if robot_spec.name == "wuji":
        raise NotImplementedError(
            "WUJI support has been added to the Isaac Gym and ROS paths, but deployment/mujoco/mujoco_sim.py is still Sharpa-specific."
        )
    obs_list = cfg["task"]["env"]["obsList"]
    num_observations = get_num_observations(robot_asset_file, obs_list)
    num_actions = robot_spec.num_hand_arm_dofs
    policy = RlPlayer(
        num_observations=num_observations,
        num_actions=num_actions,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=device,
    )

    urdf = create_urdf_object(robot_asset_file)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"obs_list: {obs_list}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    mujoco_env = MujocoEnvNoRos(
        sim=sim,
        object_scales=OBJECT_SCALES,
        urdf=urdf,
        hand_moving_average=HAND_MOVING_AVERAGE,
        arm_moving_average=ARM_MOVING_AVERAGE,
        hand_dof_speed_scale=HAND_DOF_SPEED_SCALE,
        control_dt=CONTROL_DT,
        device=device,
        obs_list=obs_list,
        robot_asset_file=robot_asset_file,
    )

    while True:
        start_time = time.time()
        # Get observation, get action, step simulation
        observation = mujoco_env.compute_observation()
        action = policy.get_normalized_action(observation, deterministic_actions=True)
        mujoco_env.step(action)
        end_time = time.time()

        # Sleep to maintain control loop frequency
        sleep_time = CONTROL_DT - (end_time - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(
                f"Control loop too slow! Desired FPS: {1.0 / CONTROL_DT:.1f}, Actual FPS: {1.0 / (end_time - start_time):.1f}"
            )


if __name__ == "__main__":
    main()
