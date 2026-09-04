"""Compare five checkpoints side-by-side on one DexToolBench task."""

# Isaac Gym must be imported before torch.
from isaacgym import gymapi  # noqa: F401
import torch

import argparse
import json
from pathlib import Path

from deployment.isaac.isaac_env import create_env
from deployment.rl_player import RlPlayer
from isaacgymenvs.utils.utils import get_repo_root_dir, set_seed


N_OBS = 140
N_ACT = 29
CONTROL_HZ = 60.0
REPO_ROOT = Path(get_repo_root_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--config-path", type=Path, default=Path("pretrained_policy/config.yaml"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("simtoolreal_eval_checkpoints_20260904"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--episode-length", type=int, default=600)
    args = parser.parse_args()

    policies = [
        ("Pretrained", Path("pretrained_policy/model.pth")),
        ("Eigenoise 01", args.checkpoint_dir / "eigenoise_01.pth"),
        ("Eigenoise 02", args.checkpoint_dir / "eigenoise_02.pth"),
        ("Jabs 01", args.checkpoint_dir / "jabs_01.pth"),
        ("Jabs 02", args.checkpoint_dir / "jabs_02.pth"),
    ]
    for _, checkpoint in policies:
        if not checkpoint.exists():
            parser.error("checkpoint not found: %s" % checkpoint)

    trajectory_path = (
        REPO_ROOT / "dextoolbench" / "trajectories" / args.category
        / args.object_name / (args.task_name + ".json")
    )
    with trajectory_path.open() as f:
        trajectory = json.load(f)
    start_pose = list(trajectory["start_pose"])
    start_pose[2] += 0.03
    goals = trajectory["goals"]
    table_urdf = (
        "urdf/dextoolbench/environments/%s/%s/%s.urdf"
        % (args.category, args.object_name, args.task_name)
    )

    set_seed(args.seed, torch_deterministic=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = create_env(
        config_path=str(args.config_path),
        headless=True,
        device=device,
        overrides={
            "seed": args.seed,
            "task.env.numEnvs": len(policies),
            "task.env.envSpacing": 1.2,
            "task.env.episodeLength": args.episode_length,
            "task.env.capture_video": False,
            "task.env.capture_viewer": False,
            "task.env.viserViz": False,
            "task.env.enableCameraSensors": False,
            "task.env.objectName": args.object_name,
            "task.env.asset.table": table_urdf,
            "task.env.tableResetZ": 0.38,
            "task.env.tableResetZRange": 0.0,
            "task.env.useFixedGoalStates": True,
            "task.env.fixedGoalStates": goals,
            "task.env.fixedGoalStatesJsonPath": None,
            "task.env.useFixedInitObjectPose": True,
            "task.env.objectStartPose": start_pose,
            "task.env.startArmHigher": True,
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
            "task.env.resetWhenDropped": False,
            "task.env.armMovingAverage": 0.1,
            "task.env.evalSuccessTolerance": 0.01,
            "task.env.successSteps": 1,
            "task.env.fixedSizeKeypointReward": True,
            "task.env.forceNoReset": False,
            "task.env.forceScale": 0.0,
            "task.env.torqueScale": 0.0,
            "task.env.linVelImpulseScale": 0.0,
            "task.env.angVelImpulseScale": 0.0,
        },
    )

    # Establish the identical task reset before model construction consumes RNG.
    obs_dict, _, _, _ = env.step(torch.zeros((len(policies), N_ACT), device=device))
    obs = obs_dict["obs"]
    players = []
    for _, checkpoint in policies:
        player = RlPlayer(N_OBS, N_ACT, args.config_path, checkpoint, device, 1)
        player.reset()
        players.append(player)

    active = torch.ones(len(policies), dtype=torch.bool, device=device)
    max_goals = torch.zeros(len(policies), dtype=torch.long, device=device)
    finish_step = torch.full(
        (len(policies),), args.episode_length, dtype=torch.long, device=device
    )
    returns = torch.zeros(len(policies), device=device)

    # A success resets the task progress counter, so total runtime can exceed
    # episode_length while a policy advances through a multi-waypoint trajectory.
    absolute_step_limit = args.episode_length * (len(goals) + 1)
    for step in range(1, absolute_step_limit + 1):
        actions = torch.cat([
            player.get_normalized_action(obs[i:i + 1], deterministic_actions=True)
            for i, player in enumerate(players)
        ])
        obs_dict, reward, done, _ = env.step(actions)
        obs = obs_dict["obs"]
        returns[active] += reward[active]
        max_goals = torch.maximum(max_goals, env.successes.long())

        newly_done = active & done.bool()
        finish_step[newly_done] = step
        active &= ~newly_done
        if not active.any():
            break

    results = []
    for i, (label, checkpoint) in enumerate(policies):
        goals_reached = int(max_goals[i].item())
        results.append({
            "policy": label,
            "checkpoint": str(checkpoint),
            "goals_reached": goals_reached,
            "total_goals": len(goals),
            "goal_completion": goals_reached / len(goals),
            "all_goals_hit": goals_reached >= len(goals),
            "steps": int(finish_step[i].item()),
            "seconds_at_60hz": float(finish_step[i].item() / CONTROL_HZ),
            "return": float(returns[i].item()),
        })

    output = {
        "category": args.category,
        "object_name": args.object_name,
        "task_name": args.task_name,
        "seed": args.seed,
        "deterministic_actions": True,
        "success_tolerance": 0.01,
        "num_goals": len(goals),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("RESULT_JSON=" + json.dumps({
        "task": "%s/%s/%s" % (args.category, args.object_name, args.task_name),
        "results": [
            [r["policy"], r["goals_reached"], r["total_goals"], r["all_goals_hit"]]
            for r in results
        ],
    }))


if __name__ == "__main__":
    main()
