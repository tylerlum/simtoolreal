"""Interactive SimToolReal playground backed by Isaac Gym and Viser.

This mirrors the project website's MuJoCo demo: one deterministic pretrained
policy runs continuously while the user moves/randomizes the goal, teleports
the object on the table, or resets the scene.

Browser controls live in the Viser sidebar.  When stdin is a TTY, the original
desktop hotkeys are also available in the terminal that launched this script:

  R             randomize goal pose
  arrows        move goal in X/Z
  [ / ]         move goal in Y
  W/A/S/D       teleport object on the table
  Backspace     reset scene
  Space         pause/resume

Isaac Gym is isolated in a subprocess so the Viser process stays lightweight.
"""

import argparse
import multiprocessing
import os
import select
import sys
import tempfile
import termios
import time
import traceback
import tty
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

from dextoolbench.eval_interactive import (
    DEFAULT_DOF_POS,
    REPO_ROOT,
    TABLE_Z,
    quat_xyzw_to_wxyz,
)


N_ACT = 29
CONTROL_HZ = 60.0
DEFAULT_OBJECT_POSE = [0.0, 0.0, 0.545, 0.0, 0.0, 0.0, 1.0]
DEFAULT_GOAL_POSE = [0.0, 0.1, 0.83, 0.7071068, 0.0, 0.0, 0.7071068]
GOAL_STEP = 0.025
OBJECT_STEP = 0.035


def _state(env, obs, joint_lower, joint_upper, reward=0.0, step=0, paused=False):
    obs_np = obs[0].detach().cpu().numpy()
    joints = 0.5 * (obs_np[:N_ACT] + 1.0) * (joint_upper - joint_lower) + joint_lower
    return {
        "joints": joints,
        "object": env.object_state[0, :7].detach().cpu().numpy(),
        "goal": env.goal_pose[0].detach().cpu().numpy(),
        "reward": float(reward),
        "successes": int(env.successes[0].item()),
        "step": int(step),
        "paused": bool(paused),
    }


def _flush_state_writes(env):
    env.set_actor_root_state_tensor_indexed()
    env.set_dof_state_tensor_indexed()


def _set_goal(env, pose):
    """Set both the visible goal actor and the fixed goal used after success."""
    import torch

    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    value = torch.as_tensor(pose, dtype=env.goal_states.dtype, device=env.device)
    value = value / torch.cat((torch.ones(3, device=env.device), value[3:7].norm().repeat(4)))
    env.goal_states[ids, :7] = value
    env.trajectory_states = value[None, :].clone()
    goal_indices = env.goal_object_indices[ids]
    env.root_state_tensor[goal_indices, :7] = value
    env.root_state_tensor[goal_indices, 7:13] = 0.0
    env.reset_goal_buf[ids] = 0
    env.near_goal_steps[ids] = 0
    env.successes[ids] = 0
    env.deferred_set_actor_root_state_tensor_indexed([goal_indices])
    env.set_actor_root_state_tensor_indexed()


def _nudge_goal(env, delta):
    pose = env.goal_states[0, :7].detach().clone()
    pose[:3] += pose.new_tensor(delta)
    mins = env.target_volume_origin + env.target_volume_extent[:, 0]
    maxs = env.target_volume_origin + env.target_volume_extent[:, 1]
    pose[:3] = pose[:3].clamp(mins, maxs)
    _set_goal(env, pose)


def _randomize_goal(env):
    import torch

    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    mins = env.target_volume_origin + env.target_volume_extent[:, 0]
    maxs = env.target_volume_origin + env.target_volume_extent[:, 1]
    pose = env.goal_states[0, :7].detach().clone()
    pose[:3] = mins + torch.rand(3, device=env.device) * (maxs - mins)
    pose[3:7] = env.get_random_quat(ids)[0]
    _set_goal(env, pose)


def _teleport_object(env, delta):
    import torch

    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    obj_indices = env.object_indices[ids]
    env.root_state_tensor[obj_indices, :3] += torch.as_tensor(
        delta, dtype=env.root_state_tensor.dtype, device=env.device
    )
    env.root_state_tensor[obj_indices, 0].clamp_(-0.35, 0.35)
    env.root_state_tensor[obj_indices, 1].clamp_(-0.12, 0.22)
    env.root_state_tensor[obj_indices, 7:13] = 0.0
    env.closest_fingertip_dist[ids] = -1
    env.furthest_hand_dist[ids] = -1
    env.lifted_object[ids] = False
    env.deferred_set_actor_root_state_tensor_indexed([obj_indices])
    env.set_actor_root_state_tensor_indexed()


def _reset_scene(env, policy):
    import torch

    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    _set_goal(env, DEFAULT_GOAL_POSE)
    env.reset_idx(ids)
    _flush_state_writes(env)
    policy.reset()


def sim_worker(conn, config_path, checkpoint_path):
    """Isaac Gym + policy loop. Heavy GPU imports intentionally stay here."""
    try:
        from isaacgym import gymapi  # noqa: F401
    except ImportError:
        conn.send(("error", "Isaac Gym Preview 4 is not installed in this Python environment."))
        return

    import torch
    from deployment.isaac.isaac_env import create_env
    from deployment.rl_player import RlPlayer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        env = create_env(
            config_path=str(config_path),
            headless=True,
            device=device,
            overrides={
                "task.env.numEnvs": 1,
                "task.env.envSpacing": 0.4,
                "task.env.capture_video": False,
                "task.env.objectName": "handle_head_primitives",
                "task.env.handleHeadTypes": ["hammer"],
                "task.env.numAssetsPerType": 1,
                "task.env.randomizeAssetOrder": False,
                "task.env.useFixedInitObjectPose": True,
                "task.env.objectStartPose": DEFAULT_OBJECT_POSE,
                "task.env.startArmHigher": False,
                "task.env.tableResetZ": TABLE_Z,
                "task.env.tableResetZRange": 0.0,
                "task.env.useFixedGoalStates": True,
                "task.env.fixedGoalStates": [DEFAULT_GOAL_POSE],
                "task.env.fixedGoalStatesJsonPath": None,
                "task.env.forceNoReset": True,
                "task.env.resetWhenDropped": False,
                "task.env.resetPositionNoiseX": 0.0,
                "task.env.resetPositionNoiseY": 0.0,
                "task.env.resetPositionNoiseZ": 0.0,
                "task.env.randomizeObjectRotation": False,
                "task.env.resetDofPosRandomIntervalFingers": 0.0,
                "task.env.resetDofPosRandomIntervalArm": 0.0,
                "task.env.resetDofVelRandomInterval": 0.0,
                "task.env.useActionDelay": False,
                "task.env.useObsDelay": False,
                "task.env.useObjectStateDelayNoise": False,
                "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],
                "task.env.forceScale": 0.0,
                "task.env.torqueScale": 0.0,
                "task.env.linVelImpulseScale": 0.0,
                "task.env.angVelImpulseScale": 0.0,
            },
        )
        joint_lower = env.arm_hand_dof_lower_limits[:N_ACT].cpu().numpy()
        joint_upper = env.arm_hand_dof_upper_limits[:N_ACT].cpu().numpy()

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_root = checkpoint.get(0, checkpoint)
        env_state = checkpoint_root.get("env_state")
        if env_state is not None:
            env.set_env_state(env_state.copy())

        policy = RlPlayer(140, N_ACT, config_path, checkpoint_path, device, 1)
        policy.reset()
        obs_dict, reward, _, _ = env.step(torch.zeros((1, N_ACT), device=device))
        obs = obs_dict["obs"]
        _set_goal(env, DEFAULT_GOAL_POSE)

        conn.send(("ready", env.viewer_object_urdf_text, _state(
            env, obs, joint_lower, joint_upper
        )))
        paused = False
        step = 0
        control_dt = 1.0 / CONTROL_HZ

        while True:
            while conn.poll(0):
                cmd = conn.recv()
                name = cmd[0]
                if name == "quit":
                    conn.close()
                    return
                if name == "pause":
                    paused = not paused
                elif name == "random_goal":
                    _randomize_goal(env)
                elif name == "nudge_goal":
                    _nudge_goal(env, cmd[1])
                elif name == "teleport_object":
                    _teleport_object(env, cmd[1])
                elif name == "reset":
                    _reset_scene(env, policy)
                    step = 0

            tick = time.time()
            if not paused:
                action = policy.get_normalized_action(obs, deterministic_actions=True)
                obs_dict, reward, _, _ = env.step(action)
                obs = obs_dict["obs"]
                step += 1
            conn.send(("state", _state(
                env, obs, joint_lower, joint_upper, reward[0].item(), step, paused
            )))
            delay = control_dt - (time.time() - tick)
            if delay > 0:
                time.sleep(delay)
    except Exception as exc:
        conn.send(("error", "%s\n%s" % (exc, traceback.format_exc())))
    finally:
        conn.close()


class Playground:
    def __init__(self, config_path, checkpoint_path, port):
        self.port = port
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self._tmp = tempfile.TemporaryDirectory(prefix="simtoolreal_viser_")
        self.object_frame = None
        self.goal_frame = None
        self.object_vis = None
        self.goal_vis = None
        self._paused = False
        self._stdin_state = None

        self._build_scene()
        self._build_gui()

        ctx = multiprocessing.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(
            target=sim_worker,
            args=(child, str(config_path), str(checkpoint_path)),
            daemon=True,
        )
        self.proc.start()
        child.close()

    def _build_scene(self):
        @self.server.on_client_connect
        def _(client):
            client.camera.position = (0.0, -1.0, 0.95)
            client.camera.look_at = (0.0, 0.0, 0.62)

        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        self.server.scene.add_box(
            "/table", position=(0, 0, TABLE_Z), dimensions=(0.475, 0.4, 0.3),
            color=(180, 130, 70), opacity=0.9,
        )
        self.server.scene.add_frame(
            "/robot", position=(0, 0.8, 0), wxyz=(1, 0, 0, 0), show_axes=False,
        )
        robot_path = REPO_ROOT / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
        self.robot = ViserUrdf(self.server, robot_path, root_node_name="/robot")
        self.robot.update_cfg(DEFAULT_DOF_POS)

    def _build_gui(self):
        self.server.gui.add_markdown(
            "# SimToolReal Playground\n"
            "Isaac Gym · pretrained policy · **deterministic=True**"
        )
        self.status = self.server.gui.add_markdown("**Status:** Loading Isaac Gym…")
        with self.server.gui.add_folder("Goal", expand_by_default=True):
            self.server.gui.add_button("🎲 Randomize goal (R)").on_click(
                lambda _: self._send(("random_goal",))
            )
            self.server.gui.add_button("← Goal X−").on_click(
                lambda _: self._send(("nudge_goal", [-GOAL_STEP, 0, 0]))
            )
            self.server.gui.add_button("Goal X+ →").on_click(
                lambda _: self._send(("nudge_goal", [GOAL_STEP, 0, 0]))
            )
            self.server.gui.add_button("↓ Goal Z−").on_click(
                lambda _: self._send(("nudge_goal", [0, 0, -GOAL_STEP]))
            )
            self.server.gui.add_button("Goal Z+ ↑").on_click(
                lambda _: self._send(("nudge_goal", [0, 0, GOAL_STEP]))
            )
            self.server.gui.add_button("[ Goal Y−").on_click(
                lambda _: self._send(("nudge_goal", [0, -GOAL_STEP, 0]))
            )
            self.server.gui.add_button("] Goal Y+").on_click(
                lambda _: self._send(("nudge_goal", [0, GOAL_STEP, 0]))
            )
        with self.server.gui.add_folder("Object teleport", expand_by_default=True):
            self.server.gui.add_button("W: Object +Y").on_click(
                lambda _: self._send(("teleport_object", [0, OBJECT_STEP, 0]))
            )
            self.server.gui.add_button("S: Object −Y").on_click(
                lambda _: self._send(("teleport_object", [0, -OBJECT_STEP, 0]))
            )
            self.server.gui.add_button("A: Object −X").on_click(
                lambda _: self._send(("teleport_object", [-OBJECT_STEP, 0, 0]))
            )
            self.server.gui.add_button("D: Object +X").on_click(
                lambda _: self._send(("teleport_object", [OBJECT_STEP, 0, 0]))
            )
        with self.server.gui.add_folder("Simulation", expand_by_default=True):
            self.pause_button = self.server.gui.add_button("Pause (Space)")
            self.pause_button.on_click(lambda _: self._toggle_pause())
            self.server.gui.add_button("Reset scene (Backspace)").on_click(
                lambda _: self._send(("reset",))
            )
        self.metrics = self.server.gui.add_markdown("**Reward:** --")

    def _send(self, msg):
        try:
            self.conn.send(msg)
        except (BrokenPipeError, OSError):
            pass

    def _toggle_pause(self):
        self._paused = not self._paused
        self.pause_button.name = "Resume (Space)" if self._paused else "Pause (Space)"
        self._send(("pause",))

    def _install_object(self, urdf_text):
        path = Path(self._tmp.name) / "playground_hammer.urdf"
        path.write_text(urdf_text)
        self.object_frame = self.server.scene.add_frame("/object", show_axes=False)
        self.goal_frame = self.server.scene.add_frame("/goal", show_axes=True, axes_length=0.08)
        self.object_vis = ViserUrdf(self.server, path, root_node_name="/object")
        self.goal_vis = ViserUrdf(
            self.server, path, root_node_name="/goal", mesh_color_override=(0, 255, 0, 0.45)
        )

    def _update(self, state):
        self.robot.update_cfg(state["joints"])
        if self.object_frame is not None:
            self.object_frame.position = tuple(state["object"][:3])
            self.object_frame.wxyz = quat_xyzw_to_wxyz(state["object"][3:7])
            self.goal_frame.position = tuple(state["goal"][:3])
            self.goal_frame.wxyz = quat_xyzw_to_wxyz(state["goal"][3:7])
        self._paused = state["paused"]
        self.pause_button.name = "Resume (Space)" if self._paused else "Pause (Space)"
        self.status.content = "**Status:** %s" % ("Paused" if self._paused else "Running")
        obj = state["object"][:3]
        goal = state["goal"][:3]
        self.metrics.content = (
            "**Reward:** %.3f  \n**Time:** %.1fs  \n**Successes:** %d"
            "  \n**Object:** (%.3f, %.3f, %.3f)"
            "  \n**Goal:** (%.3f, %.3f, %.3f)"
            % (state["reward"], state["step"] / CONTROL_HZ, state["successes"],
               obj[0], obj[1], obj[2], goal[0], goal[1], goal[2])
        )

    def _enable_terminal_keys(self):
        if not sys.stdin.isatty():
            return
        self._stdin_state = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())

    def _poll_terminal(self):
        if self._stdin_state is None or not select.select([sys.stdin], [], [], 0)[0]:
            return
        key = os.read(sys.stdin.fileno(), 3)
        mapping = {
            b"r": ("random_goal",), b"R": ("random_goal",),
            b"[": ("nudge_goal", [0, -GOAL_STEP, 0]),
            b"]": ("nudge_goal", [0, GOAL_STEP, 0]),
            b"\x1b[D": ("nudge_goal", [-GOAL_STEP, 0, 0]),
            b"\x1b[C": ("nudge_goal", [GOAL_STEP, 0, 0]),
            b"\x1b[B": ("nudge_goal", [0, 0, -GOAL_STEP]),
            b"\x1b[A": ("nudge_goal", [0, 0, GOAL_STEP]),
            b"w": ("teleport_object", [0, OBJECT_STEP, 0]),
            b"s": ("teleport_object", [0, -OBJECT_STEP, 0]),
            b"a": ("teleport_object", [-OBJECT_STEP, 0, 0]),
            b"d": ("teleport_object", [OBJECT_STEP, 0, 0]),
            b"\x7f": ("reset",),
        }
        if key == b" ":
            self._toggle_pause()
        elif key in mapping:
            self._send(mapping[key])

    def run(self):
        print("Isaac + Viser playground: http://localhost:%d" % self.port)
        print("Browser buttons are always active; terminal hotkeys require terminal focus.")
        self._enable_terminal_keys()
        try:
            while True:
                self._poll_terminal()
                while self.conn.poll(0):
                    msg = self.conn.recv()
                    if msg[0] == "ready":
                        self._install_object(msg[1])
                        self._update(msg[2])
                    elif msg[0] == "state":
                        self._update(msg[1])
                    elif msg[0] == "error":
                        self.status.content = "**Status:** Isaac error (see terminal)"
                        print(msg[1])
                if not self.proc.is_alive() and not self.conn.poll(0):
                    raise RuntimeError("Isaac worker exited")
                time.sleep(1.0 / 120.0)
        except KeyboardInterrupt:
            pass
        finally:
            if self._stdin_state is not None:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._stdin_state)
            try:
                self._send(("quit",))
                self.proc.join(timeout=5)
                if self.proc.is_alive():
                    self.proc.kill()
            finally:
                self._tmp.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--config-path", type=Path, default=Path("pretrained_policy/config.yaml"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("pretrained_policy/model.pth"))
    args = parser.parse_args()
    if not args.config_path.exists():
        parser.error("config not found: %s" % args.config_path)
    if not args.checkpoint_path.exists():
        parser.error("checkpoint not found: %s" % args.checkpoint_path)
    Playground(args.config_path, args.checkpoint_path, args.port).run()


if __name__ == "__main__":
    main()
