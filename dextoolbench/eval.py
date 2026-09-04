"""Evaluation script for dexterous manipulation with viser visualization."""

# NOTE: torch must be imported AFTER isaacgym imports
# isort: off
from isaacgym import gymapi, gymtorch, gymutil
import torch
# isort: on

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import imageio
import numpy as np
import tyro
import viser
from termcolor import colored
from viser.extras import ViserUrdf

from deployment.isaac.isaac_env import create_env
from deployment.rl_player import RlPlayer
from dextoolbench.objects import NAME_TO_OBJECT
from isaacgymenvs.utils.utils import get_repo_root_dir

TABLE_Z = 0.38


def quat_xyzw_to_wxyz(q):
    """Convert quaternion from xyzw to wxyz format."""
    return (q[3], q[0], q[1], q[2])


def log_info(text):
    print(colored(text, "cyan"))


def log_success(text):
    print(colored(text, "green"))


def log_warn(text):
    print(colored(text, "yellow"))


class ViserServer:
    """Viser-based visualization server for robot manipulation."""

    def __init__(
        self,
        object_name: str,
        task_name: str,
        num_keypoints: int,
        table_urdf: str,
        policy_name: str,
        port: int = 8080,
    ):
        self.port = port
        self.num_keypoints = num_keypoints
        self.is_paused = False
        self.show_keypoints = True
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.table_urdf = table_urdf
        self._setup_scene(
            object_name=object_name,
            task_name=task_name,
            policy_name=policy_name,
        )

    def _setup_scene(self, object_name: str, task_name: str, policy_name: str):
        """Initialize the 3D scene with robot, table, object, and GUI elements."""

        @self.server.on_client_connect
        def _(client):
            client.camera.position = (0.0, -1.0, 1.0)
            client.camera.look_at = (0.0, 0.0, 0.5)

        # Ground grid
        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        # Robot
        robot_urdf = (
            get_repo_root_dir()
            / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
        )
        self.server.scene.add_frame(
            "/robot", position=(0, 0.8, 0), wxyz=(1, 0, 0, 0), show_axes=False
        )
        self.robot = ViserUrdf(self.server, robot_urdf, root_node_name="/robot")
        self.robot.update_cfg(np.zeros(29))

        # Table
        table_urdf = get_repo_root_dir() / "assets" / self.table_urdf
        self.server.scene.add_frame(
            "/table", position=(0, 0, TABLE_Z), wxyz=(1, 0, 0, 0), show_axes=False
        )
        # ViserUrdf(self.server, table_urdf, root_node_name="/table", mesh_color_override=(0, 0, 0, 0.5))
        ViserUrdf(
            self.server,
            table_urdf,
            root_node_name="/table",
            mesh_color_override=(0, 0, 0, 1.0),
        )

        # Object and goal
        object_urdf = NAME_TO_OBJECT[object_name].urdf_path
        self.object_frame = self.server.scene.add_frame(
            "/object", show_axes=True, axes_length=0.1, axes_radius=0.001
        )
        ViserUrdf(self.server, object_urdf, root_node_name="/object")
        self.goal_frame = self.server.scene.add_frame(
            "/goal", show_axes=True, axes_length=0.1, axes_radius=0.001
        )
        ViserUrdf(
            self.server,
            object_urdf,
            root_node_name="/goal",
            mesh_color_override=(0, 255, 0, 0.5),
        )

        # Keypoint spheres (red for object, green for goal)
        self.obj_keypoint_spheres = []
        self.goal_keypoint_spheres = []
        self.obj_keypoint_spheres_fixed_size = []
        self.goal_keypoint_spheres_fixed_size = []
        for i in range(self.num_keypoints):
            self.obj_keypoint_spheres.append(
                self.server.scene.add_icosphere(
                    f"/obj_keypoint_{i}", radius=0.01, color=(255, 0, 0)
                )
            )
            self.goal_keypoint_spheres.append(
                self.server.scene.add_icosphere(
                    f"/goal_keypoint_{i}", radius=0.01, color=(0, 255, 0)
                )
            )
            self.obj_keypoint_spheres_fixed_size.append(
                self.server.scene.add_icosphere(
                    f"/obj_keypoint_{i}_fixed_size",
                    radius=0.01,
                    color=(255, 0, 0),
                    opacity=0.6,
                )
            )
            self.goal_keypoint_spheres_fixed_size.append(
                self.server.scene.add_icosphere(
                    f"/goal_keypoint_{i}_fixed_size",
                    radius=0.01,
                    color=(0, 255, 0),
                    opacity=0.6,
                )
            )

        # GUI elements
        self.server.gui.add_markdown(f"**Policy:** {policy_name}")
        self.server.gui.add_markdown(f"**Task:** {task_name}")
        self.server.gui.add_markdown(f"**Object:** {object_name}")
        self.server.gui.add_markdown("---")
        self.progress_text = self.server.gui.add_markdown("**Progress:** --")
        self.stats_text = self.server.gui.add_markdown(
            "**Stats:** No episodes completed"
        )
        self.object_state_text = self.server.gui.add_markdown("**Object State:** --")
        self.server.gui.add_markdown("---")

        # Controls
        self.keypoint_toggle = self.server.gui.add_checkbox(
            "Show Keypoints", initial_value=True
        )
        self.keypoint_toggle.on_update(lambda _: self._toggle_keypoints())
        self.keypoint_toggle_fixed_size = self.server.gui.add_checkbox(
            "Show Keypoints Fixed Size", initial_value=True
        )
        self.keypoint_toggle_fixed_size.on_update(
            lambda _: self._toggle_keypoints_fixed_size()
        )
        self.keypoint_toggle_fixed_size.value = (
            False  # start as True, then set to False to hide them
        )

    def _toggle_keypoints(self):
        """Toggle visibility of keypoint spheres."""
        self.show_keypoints = self.keypoint_toggle.value
        for sphere in self.obj_keypoint_spheres + self.goal_keypoint_spheres:
            sphere.visible = self.show_keypoints

    def _toggle_keypoints_fixed_size(self):
        """Toggle visibility of keypoint spheres fixed size."""
        self.show_keypoints_fixed_size = self.keypoint_toggle_fixed_size.value
        for sphere in (
            self.obj_keypoint_spheres_fixed_size + self.goal_keypoint_spheres_fixed_size
        ):
            sphere.visible = self.show_keypoints_fixed_size

    def _toggle_pause(self):
        """Toggle pause state."""
        self.is_paused = not self.is_paused
        self.pause_button.name = "Resume" if self.is_paused else "Pause"
        log_info(f"Paused: {self.is_paused}")

    def add_controls(self, run_callback):
        """Add run and pause buttons."""
        self.server.gui.add_button("Run Episode").on_click(lambda _: run_callback())
        self.pause_button = self.server.gui.add_button("Pause")
        self.pause_button.on_click(lambda _: self._toggle_pause())

    def update(
        self,
        joint_pos,
        object_pose,
        goal_pose,
        obj_keypoints=None,
        goal_keypoints=None,
        obj_keypoints_fixed_size=None,
        goal_keypoints_fixed_size=None,
    ):
        """Update visualization with current state."""
        self.robot.update_cfg(joint_pos)
        self.object_frame.position = object_pose[:3]
        self.object_frame.wxyz = quat_xyzw_to_wxyz(object_pose[3:7])
        self.goal_frame.position = goal_pose[:3]
        self.goal_frame.wxyz = quat_xyzw_to_wxyz(goal_pose[3:7])

        if obj_keypoints is not None:
            for i, sphere in enumerate(self.obj_keypoint_spheres):
                sphere.position = tuple(obj_keypoints[i])
        if goal_keypoints is not None:
            for i, sphere in enumerate(self.goal_keypoint_spheres):
                sphere.position = tuple(goal_keypoints[i])
        if obj_keypoints_fixed_size is not None:
            for i, sphere in enumerate(self.obj_keypoint_spheres_fixed_size):
                sphere.position = tuple(obj_keypoints_fixed_size[i])
        if goal_keypoints_fixed_size is not None:
            for i, sphere in enumerate(self.goal_keypoint_spheres_fixed_size):
                sphere.position = tuple(goal_keypoints_fixed_size[i])

    def update_progress(
        self, current: int, total: int, timestep: int, control_hz: float = 60.0
    ):
        """Update progress display."""
        pct = 100 * current / total if total > 0 else 0
        self.progress_text.content = f"**Time:** {timestep / control_hz:.1f}s | **Goal:** {current}/{total} ({pct:.0f}%)"

    def update_object_state(self, object_state: np.ndarray):
        object_pos = object_state[:3]
        self.object_state_text.content = f"**Object State:** {object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f}"

    def update_stats(self, num_episodes: int, avg_goal_pct: float, avg_time_sec: float):
        """Update statistics display."""
        self.stats_text.content = f"**Episodes:** {num_episodes} | **Avg Goal:** {avg_goal_pct:.1f}% | **Avg Time:** {avg_time_sec:.1f}s"

    def get_frame(self) -> np.ndarray:
        """Capture current view as image."""
        clients = list(self.server.get_clients().values())
        if clients:
            return clients[0].camera.get_render(height=480, width=640)
        return np.zeros((480, 640, 3), dtype=np.uint8)


class EvalRunner:
    """Runs policy evaluation with viser visualization."""

    def __init__(
        self,
        env,
        config_path: Path,
        checkpoint_path: Path,
        object_name: str,
        task_name: str,
        table_urdf: str,
        output_dir: Optional[Path] = None,
        record_video: bool = False,
        policy_name: str = None,
    ):
        self.env = env
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_act = 29
        self.control_hz = 60.0
        self.control_dt = 1.0 / self.control_hz
        self.record_fps = 10
        self.record_interval = int(self.control_hz / self.record_fps)

        # Joint limits for denormalization
        self.joint_lower = env.arm_hand_dof_lower_limits[: self.n_act].cpu().numpy()
        self.joint_upper = env.arm_hand_dof_upper_limits[: self.n_act].cpu().numpy()

        # Full training checkpoints store resume-only state under integer key 0,
        # while inference-only checkpoints may contain just {"model": ...}.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_root = checkpoint.get(0, checkpoint)
        env_state = checkpoint_root.get("env_state")
        if env_state is not None:
            # SimToolReal.set_env_state removes entries as it consumes them.
            self.env.set_env_state(env_state.copy())
        else:
            log_info(
                "Checkpoint has no env_state; using fresh deterministic "
                "evaluation environment state."
            )
        self.policy = RlPlayer(
            140, self.n_act, config_path, checkpoint_path, self.device, env.num_envs
        )

        self.output_dir = output_dir

        # Recording setup
        self.record_video = record_video
        self.episode_count = 0
        self.episode_goal_pcts = []
        self.episode_lengths = []
        if self.record_video:
            assert self.output_dir is not None, (
                "Output directory must be provided if recording video"
            )
            self.session_dir = self.output_dir / datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S"
            )
            self.session_dir.mkdir(parents=True, exist_ok=True)
            log_info(f"Recording to: {self.session_dir}")

        # Visualization
        if policy_name is None:
            policy_name = checkpoint_path.name

        self.viser = ViserServer(
            object_name=object_name,
            task_name=task_name,
            num_keypoints=env.num_keypoints,
            table_urdf=table_urdf,
            policy_name=policy_name,
        )
        self.obs = self._reset()

    def _reset(self):
        """Reset environment and return initial observation."""
        obs, _, _, _ = self.env.step(
            torch.zeros((self.env.num_envs, self.n_act), device=self.device)
        )
        return obs["obs"]

    def _step(self, action) -> Tuple[torch.Tensor, bool]:
        """Step environment with action."""
        obs, _, done, _ = self.env.step(action)
        return obs["obs"], done[0].item()

    def _get_state(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract current state for visualization."""
        obs_np = self.obs[0].cpu().numpy()
        joint_pos = (
            0.5 * (obs_np[:29] + 1.0) * (self.joint_upper - self.joint_lower)
            + self.joint_lower
        )
        return (
            joint_pos,
            self.env.object_state[0, :7].cpu().numpy(),
            self.env.goal_pose[0].cpu().numpy(),
            self.env.obj_keypoint_pos[0].cpu().numpy(),
            self.env.goal_keypoint_pos[0].cpu().numpy(),
            self.env.obj_keypoint_pos_fixed_size[0].cpu().numpy(),
            self.env.goal_keypoint_pos_fixed_size[0].cpu().numpy(),
        )

    def _sim_step(self, timestep: int) -> bool:
        """Execute one simulation step with timing control."""
        t0 = time.time()
        self.viser.update(*self._get_state())
        self.obs, done = self._step(
            self.policy.get_normalized_action(self.obs, deterministic_actions=True)
        )
        self.viser.update_progress(
            int(self.env.successes[0].item()),
            self.env.max_consecutive_successes,
            timestep,
            self.control_hz,
        )
        self.viser.update_object_state(self.env.object_state[0].cpu().numpy())

        elapsed = time.time() - t0
        if (sleep_time := self.control_dt - elapsed) > 0:
            time.sleep(sleep_time)
        return done

    def _render_video(self, states: list, path: Path):
        """Render recorded states to video file."""
        log_info(f"Rendering {len(states)} frames...")
        frames = []
        for i, state in enumerate(states):
            self.viser.update(*state)
            time.sleep(0.05)
            frames.append(self.viser.get_frame())
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(states)}")
        imageio.mimsave(str(path), frames, fps=self.record_fps)
        log_success(f"Saved: {path}")

    def _run_episode(self):
        """Run a single evaluation episode."""

        # Don't let this function be called twice at the same time
        if not hasattr(self, "_run_episode_in_progress"):
            self._run_episode_in_progress = False
        if self._run_episode_in_progress:
            log_warn("Episode already in progress. Skipping...")
            return
        self._run_episode_in_progress = True

        self.policy.reset()
        log_info("Reset...")
        self.obs = self._reset()
        self.viser.update(*self._get_state())

        log_success(f"Running{' (+ recording)' if self.record_video else ''}...")
        states, step, done = [], 0, False

        while not done:
            # Handle pause
            while self.viser.is_paused:
                time.sleep(0.1)

            if self.record_video and step % self.record_interval == 0:
                states.append(tuple(x.copy() for x in self._get_state()))
            done = self._sim_step(step)
            step += 1

        # Update stats
        goal_pct = (
            100 * int(self.env.successes[0].item()) / self.env.max_consecutive_successes
        )
        self.episode_goal_pcts.append(goal_pct)
        self.episode_lengths.append(step)
        self.episode_count += 1
        avg_goal_pct = sum(self.episode_goal_pcts) / len(self.episode_goal_pcts)
        avg_time_sec = (
            sum(self.episode_lengths) / len(self.episode_lengths) / self.control_hz
        )
        self.viser.update_stats(self.episode_count, avg_goal_pct, avg_time_sec)

        if states and self.record_video:
            self._render_video(states, self.session_dir / f"{self.episode_count}.mp4")

        log_success(f"Done: {step / self.control_hz:.1f}s, {goal_pct:.0f}% goals")
        self._run_episode_in_progress = False

    def run_interactive_eval(self):
        """Start the interactive evaluation loop (click 'Run Episode' in GUI)."""
        self.viser.add_controls(self._run_episode)
        log_info(f"Open http://localhost:{self.viser.port}")
        log_info("Click 'Run Episode' to start.")
        while True:
            time.sleep(1.0)

    def run_eval(self, num_episodes: int):
        assert self.output_dir is not None, "Output directory must be provided"
        output_dir = self.output_dir
        output_json_file = self.output_dir / "eval.json"

        for i in range(num_episodes):
            self._run_episode()
        log_success(f"Done: {num_episodes} episodes")

        output_json_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_file, "w") as f:
            json.dump(
                {
                    "avg_goal_pct": np.mean(self.episode_goal_pcts),
                    "avg_time_sec": np.mean(self.episode_lengths) / self.control_hz,
                    "episode_goal_pcts": self.episode_goal_pcts,
                    "episode_lengths": self.episode_lengths,
                },
                f,
                indent=4,
            )

        # Also need to save the policy config
        # And save the env cfg because of overrides
        from omegaconf import OmegaConf

        with open(output_dir / "policy_config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(self.policy.cfg))
        with open(output_dir / "env_cfg.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(self.env.cfg))
        log_success(f"Saved: {output_json_file}")


@dataclass
class EvalArgs:
    object_category: str
    """Object category (e.g. hammer, marker, spatula)."""

    object_name: str
    """Object name within the category."""

    task_name: str
    """Task / trajectory name."""

    config_path: Path
    """Path to the policy config YAML."""

    checkpoint_path: Path
    """Path to the policy checkpoint."""

    output_dir: Optional[Path] = None
    """Directory to save evaluation results."""

    num_episodes: int = 1
    """Number of evaluation episodes to run."""

    downsample_factor: int = 1
    """Downsample factor for trajectory goals."""

    policy_name: Optional[str] = None
    """Name of the policy (for display)."""

    interactive: bool = False
    """If True, run interactive eval (GUI button to trigger episodes). Otherwise run all episodes automatically."""

    force_table_urdf: bool = True
    """If True, always use the default table URDF regardless of object category."""

    z_offset: float = 0.03
    """Z offset added to start pose to avoid the table."""


TABLE_URDF = "urdf/table_narrow.urdf"
TABLE_WHITEBOARD_URDF = "urdf/table_narrow_whiteboard.urdf"
TABLE_NAIL_URDF = "urdf/table_narrow_nail.urdf"
TABLE_BOWL_PLATE_URDF = "urdf/table_narrow_bowl_plate.urdf"

OBJECT_CATEGORY_TO_TABLE_URDF = {
    "hammer": TABLE_NAIL_URDF,
    "spatula": TABLE_BOWL_PLATE_URDF,
    "eraser": TABLE_WHITEBOARD_URDF,
    "screwdriver": TABLE_URDF,
    "marker": TABLE_WHITEBOARD_URDF,
    "brush": TABLE_URDF,
}


def main():
    args: EvalArgs = tyro.cli(EvalArgs)

    if args.force_table_urdf:
        selected_table_urdf = TABLE_URDF
    else:
        selected_table_urdf = OBJECT_CATEGORY_TO_TABLE_URDF[args.object_category]

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

    # Raise the start pose by z_offset to avoid the table
    traj_data["start_pose"][2] += args.z_offset

    # Downsample goals
    traj_data["goals"] = traj_data["goals"][:: args.downsample_factor]

    # Create environment
    env = create_env(
        config_path=str(args.config_path),
        headless=True,  # Set to False to see the isaacgym viewer
        device="cuda" if torch.cuda.is_available() else "cpu",
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
            # Table
            "task.env.asset.table": str(selected_table_urdf),
            "task.env.tableResetZ": TABLE_Z,
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

    runner = EvalRunner(
        env=env,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        object_name=args.object_name,
        task_name=args.task_name,
        table_urdf=selected_table_urdf,
        output_dir=args.output_dir,
        policy_name=args.policy_name,
    )

    if args.interactive:
        runner.run_interactive_eval()
    else:
        runner.run_eval(num_episodes=args.num_episodes)


if __name__ == "__main__":
    main()
