# SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation

## Exploration defaults

Use balanced EigenDExplore for new runs and SAPG
`expl_reward_coef_scale: 0.002` for both JABS and eigennoise. Balanced uses
hand IID sigma and decoded eigen RMS `sqrt(0.5)`, spectrum power `1.0`.
Keep the exact covariance KL and independent per-SAPG-group eigen scales.
Hot is an explicit ablation; existing checkpoint noise is retained on resume.
The Action-Bench benchmark arm is `str_joint_abs_eignoise_espp_bal`.

The native Gym default pair is documented in [the launcher guide](experiments/isaacgym_four/README.md).


[Project Page](https://simtoolreal.github.io/)

https://github.com/user-attachments/assets/e2d0db98-2e31-46aa-9480-c4c6f4a48f7d

# Overview

This repository is the official implementation of _SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation_. It provides:

* Simulation environments: Isaac Sim (recommended) and Isaac Gym (legacy) environments for training and evaluating dexterous tool manipulation policies.

* DexToolBench: A benchmark for dexterous tool manipulation.

* RL training: Reinforcement learning algorithms for training these policies.

* Deployment: Policy deployment in simulation and the real world.

# Project Structure

```
simtoolreal
  ├── assets
  │   └── // Robot URDFs, object models, and other assets
  ├── baselines
  │   └── // Kinematic retargeting and fixed-grasp baselines
  ├── deployment
  │   └── // Sim-to-real and sim-to-sim deployment of the policy
  ├── dextoolbench
  │   ├── data
  │   │   └── // DexToolBench data (needs to be downloaded)
  │   ├── // Scripts for evaluating policies on DexToolBench
  │   └── // Scripts for visualizing DexToolBench objects and trajectories
  ├── docs
  │   └── // Documentation
  ├── isaacsimenvs
  │   └── // Simulation environment (Isaac Sim / Isaac Lab — recommended)
  ├── isaacgymenvs
  │   └── // Simulation environment (Isaac Gym — legacy, used for the paper's results)
  ├── pretrained_policy
  │   └── // Checkpoint of the pretrained policy (needs to be downloaded)
  ├── recorded_data
  │   └── // Interface and tools for saving, loading, and visualizing recorded data
  └── rl_games
      └── // RL algorithms, including PPO and SAPG
```

**External repos:**
[FoundationPose](https://github.com/kushal2000/FoundationPose) — Perception system (SAM + FoundationPose pose tracking)

# Installation

The recommended setup runs SimToolReal in **Isaac Sim** (via Isaac Lab, Python 3.11, pip-installable): see the [IsaacSim Installation](docs/isaacsim_installation.md) documentation.

The legacy **Isaac Gym** environment (Python 3.8, manual binary download) is covered in the [IsaacGym Installation](docs/isaacgym_installation.md) documentation. The two environments live in separate venvs (`.venv_isaacsim` and `.venv`) and can coexist.

# Quick Start

Run all commands from the repository root. Most commands accept `--help` to list available options.

## 1. Download the Pretrained Policy

```
python download_pretrained_policy.py
```

This downloads `config.yaml` and `model.pth` into `pretrained_policy/`.

## 2. Interactive Evaluation on DexToolBench

Launch the web-based interactive demo (default `http://localhost:8080`): pick a tool category, object instance, and task from the dropdown menus, then load the environment and run episodes.

```
.venv_isaacsim/bin/python dextoolbench/eval_interactive_isaacsim.py \
--config-path pretrained_policy/config.yaml \
--checkpoint-path pretrained_policy/model.pth
```

On the Isaac Gym setup, run `dextoolbench/eval_interactive_isaacgym.py` (same arguments) from the `.venv` environment instead.

https://github.com/user-attachments/assets/58eb188b-662c-4190-8148-29710c9eb20f

## 3. Train a Policy

We recommend training in Isaac Sim. Training logs are tracked with [Weights & Biases](https://wandb.ai/); run `wandb login` before launching:

```
.venv_isaacsim/bin/python isaacsimenvs/train.py \
--task Isaacsimenvs-SimToolReal-Direct-v0 \
--agent rl_games_sapg_cfg_entry_point \
--headless \
--capture_viewer \
--wandb_activate \
--wandb_project <project> \
--wandb_entity <entity> \
env.scene.num_envs=24576 \
agent.params.config.expl_coef_block_size=4096
```

`--capture_viewer` periodically uploads an interactive 3D rollout viewer to W&B (pose-only, no cameras) so you can visually check the policy during training.

To finetune from a checkpoint, add `--checkpoint pretrained_policy/model.pth --checkpoint_load_mode weights` (`resume` also restores the optimizer state). If you run out of GPU memory, reduce `env.scene.num_envs` and `expl_coef_block_size` together (`num_envs / expl_coef_block_size` is the SAPG block count — keep it at 6).

### Isaac Gym (legacy)

The paper's results were trained with the Isaac Gym pipeline. It logs to [Weights & Biases](https://wandb.ai/); run `wandb login` and set `wandb_entity` in `isaacgymenvs/launch_training.py` before launching:

```
python isaacgymenvs/launch_training.py \
--custom_experiment_name my_experiment
```

To finetune from a checkpoint, add `--checkpoint <checkpoint_path>` (e.g. `pretrained_policy/model.pth`). If you run out of GPU memory, reduce `--num_envs` (must be divisible by `num_blocks`, default 6).

## 4. Evaluate a Policy on DexToolBench

Evaluate a policy numerically on all 24 DexToolBench combinations:

```
.venv_isaacsim/bin/python dextoolbench/run_all_evals_isaacsim.py
```

(Isaac Gym variant: `python dextoolbench/run_all_evals_isaacgym.py`.)

# DexToolBench

DexToolBench covers 6 tool categories (hammer, marker, eraser, brush, spatula, screwdriver) × 2 objects × 2 tasks each. See the [DexToolBench Reference](docs/dextoolbench.md) for dataset download, visualization tooling, creating new tasks and objects, real-world data collection, and acquiring the physical objects.

# Deployment

Sim2Real deployment runs four nodes: an RL policy node and a goal pose node (this repo), a perception node (SAM + FoundationPose, in the [FoundationPose fork](https://github.com/kushal2000/FoundationPose)), and a robot node. For safe testing, the Sim2Sim setup replaces the robot and perception nodes with a simulation node. See the [Deployment](docs/deployment.md) documentation for node diagrams and step-by-step run instructions.

# Formatting

Python files:

```
./format_pys.sh
```

URDF files:

```
./format_urdfs.sh
```

# Acknowledgements

This implementation builds on the following codebases:

1. [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs)
2. [rl_games](https://github.com/Denys88/rl_games)
3. [SAPG](https://github.com/jayeshs999/sapg)

# Citation

```
@misc{kedia2026simtoolrealobjectcentricpolicyzeroshot,
      title={SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation},
      author={Kushal Kedia and Tyler Ga Wei Lum and Jeannette Bohg and C. Karen Liu},
      year={2026},
      eprint={2602.16863},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.16863}
}
```

# Contact

If you have any questions, issues, or feedback, please contact [Tyler Lum](https://tylerlum.github.io/) or [Kushal Kedia](https://kushal2000.github.io/).

## Action-Bench IsaacGym fine-tuning recipe

The four-run JABS/EigenDExplore fine-tuning and scratch setup, including frozen
configs, persistent launch and resume commands, is documented in
[experiments/isaacgym_four/README.md](experiments/isaacgym_four/README.md).
