"""Evaluate one SimToolReal checkpoint on deterministic seeded hammer trials.

Run this script in a fresh process for each checkpoint because Isaac Gym does
not reliably release and recreate a GPU simulation in the same process.
"""

# Isaac Gym must be imported before torch.
from isaacgym import gymapi  # noqa: F401
import torch

import argparse
import json
from pathlib import Path

import numpy as np

from deployment.isaac.isaac_env import create_env
from deployment.rl_player import RlPlayer


N_ACT = 29
N_OBS = 140
OBJECT_POSE = [0.0, 0.0, 0.545, 0.0, 0.0, 0.0, 1.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=Path("pretrained_policy/config.yaml"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = create_env(
        config_path=str(args.config_path),
        headless=True,
        device=device,
        overrides={
            "seed": args.seed,
            "task.env.numEnvs": args.num_envs,
            "task.env.envSpacing": 0.4,
            "task.env.episodeLength": args.num_steps,
            "task.env.capture_video": False,
            "task.env.capture_viewer": False,
            "task.env.viserViz": False,
            "task.env.enableCameraSensors": False,
            "task.env.objectName": "handle_head_primitives",
            "task.env.handleHeadTypes": ["hammer"],
            "task.env.numAssetsPerType": 1,
            "task.env.randomizeAssetOrder": False,
            "task.env.useFixedInitObjectPose": True,
            "task.env.objectStartPose": OBJECT_POSE,
            "task.env.startArmHigher": False,
            "task.env.tableResetZ": 0.38,
            "task.env.tableResetZRange": 0.0,
            "task.env.useFixedGoalStates": False,
            "task.env.fixedGoalStates": None,
            "task.env.fixedGoalStatesJsonPath": None,
            "task.env.maxConsecutiveSuccesses": 1,
            "task.env.successTolerance": 0.01,
            "task.env.targetSuccessTolerance": 0.01,
            "task.env.evalSuccessTolerance": 0.01,
            "task.env.forceNoReset": False,
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

    # Reset and sample goals before constructing the policy. Model construction
    # consumes a checkpoint-dependent amount of RNG, so reversing this order
    # silently gives different goal poses to differently packaged checkpoints.
    obs_dict, _, _, _ = env.step(torch.zeros((args.num_envs, N_ACT), device=device))
    obs = obs_dict["obs"]
    initial_goals = env.goal_pose[:, :7].detach().cpu().numpy().copy()

    policy = RlPlayer(
        N_OBS, N_ACT, args.config_path, args.checkpoint_path, device, args.num_envs
    )
    policy.reset()

    active = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    hit = torch.zeros_like(active)
    elapsed_steps = torch.full(
        (args.num_envs,), args.num_steps, dtype=torch.long, device=device
    )
    returns = torch.zeros(args.num_envs, device=device)

    for step in range(1, args.num_steps + 1):
        action = policy.get_normalized_action(obs, deterministic_actions=True)
        obs_dict, reward, done, _ = env.step(action)
        obs = obs_dict["obs"]
        returns[active] += reward[active]

        successful_now = active & (env.successes >= 1)
        hit[successful_now] = True
        elapsed_steps[successful_now] = step
        active &= ~done.bool()
        if not active.any():
            break

    hit_np = hit.cpu().numpy()
    times = elapsed_steps.cpu().numpy() / 60.0
    returns_np = returns.cpu().numpy()
    result = {
        "label": args.label,
        "checkpoint": str(args.checkpoint_path),
        "seed": args.seed,
        "num_trials": args.num_envs,
        "max_steps": args.num_steps,
        "deterministic_actions": True,
        "success_tolerance": 0.01,
        "hits": int(hit_np.sum()),
        "hit_rate": float(hit_np.mean()),
        "mean_time_to_hit_seconds": float(times[hit_np].mean()) if hit_np.any() else None,
        "median_time_to_hit_seconds": float(np.median(times[hit_np])) if hit_np.any() else None,
        "mean_return": float(returns_np.mean()),
        "median_return": float(np.median(returns_np)),
        "per_trial_hit": hit_np.tolist(),
        "per_trial_time_seconds": times.tolist(),
        "per_trial_return": returns_np.tolist(),
        "initial_goal_poses_xyzw": initial_goals.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT_JSON=" + json.dumps({
        key: result[key] for key in (
            "label", "hits", "num_trials", "hit_rate",
            "mean_time_to_hit_seconds", "mean_return", "median_return"
        )
    }))


if __name__ == "__main__":
    main()
