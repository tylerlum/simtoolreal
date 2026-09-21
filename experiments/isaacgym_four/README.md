# Native IsaacGym JABS and balanced EigenDExplore

The default launch is a fresh JABS / balanced EigenDExplore pair, both seed 43
and SAPG `expl_reward_coef_scale: 0.002`. Online W&B logging is required.
The existing exact covariance KL and independent six-group eigen scales are
retained. Use the companion `codex/isaacgym-finetune` SimToolReal host commit
`codex/isaacgym-finetune`.

Both runs use 24,576 environments, six SAPG blocks, horizon 16,
minibatch 98,304, two mini-epochs, actor LR 1e-4, expanded Sharpa joint limits,
and initial goal tolerance 0.075 annealed to 0.01. There is no stopping cap.
Balanced uses egosuite++ k=9, hand IID sigma sqrt(0.5), decoded eigen RMS
sqrt(0.5), and spectrum power 1.0. The total initial hand budget matches JABS.

Existing runs and checkpoints retain their resolved settings. The old hot
scratch recipe is an explicit ablation via `--runs eigendexplore_scratch --basis ...`; it also uses SAPG 0.002. Fine-tuning is explicit and preserves
checkpoint noise: it never converts hot weights into balanced weights.
The original shared-scale ~61B eigen checkpoint is incompatible with the
corrected six-group shape and fails explicitly; use historical Action-Bench
commit 4a9fa3d with host 5abb5b33 to reproduce that old setup.

## Environment and required inputs

The validated environment is Linux, Python 3.8.20, Isaac Gym Preview 4,
PyTorch 2.4.1+cu121 and NVIDIA L40S. A C++ compiler, Ninja and tmux are needed.
The current training runs use this environment; other GPU architectures have
not been validated for this recipe.

Install the separately downloaded NVIDIA Isaac Gym Preview 4 SDK into a
Python 3.8 environment (see [the Gym installation guide](https://github.com/hgupt3/simtoolreal/blob/codex/isaacgym-finetune/docs/isaacgym_installation.md)).
Use the following commands from the repository root in that environment.
`ISAACGYM_ROOT` points to the extracted SDK directory containing `python/`.
The worker uses this checkout and its vendored `rl_games`. Obtain the pinned host:

```bash
export STR_HOST_ROOT=/your/source/storage/simtoolreal
git clone --branch codex/isaacgym-finetune https://github.com/hgupt3/simtoolreal.git "$STR_HOST_ROOT"
git -C "$STR_HOST_ROOT" checkout codex/isaacgym-finetune
```

```bash
python -m pip install 'setuptools==75.3.4' 'wheel==0.45.1'
python -m pip install 'torch==2.4.1' 'torchvision==0.19.1' \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r experiments/isaacgym_four/requirements.txt
python -m pip install --no-deps -e "$ISAACGYM_ROOT/python"
PYTORCH3D_NO_EXTENSION=1 python -m pip install --no-build-isolation --no-deps \
  'git+https://github.com/facebookresearch/pytorch3d.git@89653419d0973396f3eff1a381ba09a07fffc2ed'
python -m wandb login
```

Only PyTorch3D's pure-PyTorch transforms are used by this training path.
The requirements file records versions from the working environment; a complete
installation in a new environment was not repeated for this packaging change.
Do not install the root project's broad dependency list over these pins.

## Prepare and launch the default pair

Provide `sharpa_espp_eignoise_bal_k9_simtoolreal.npz`, writable external run
and cache directories, and an authenticated W&B account. No checkpoint or hot
basis is needed for the default pair. From the Python 3.8 environment:

```bash
export STR_PYTHON="$(command -v python)"
export STR_RUN_ROOT=/your/run/storage/isaacgym-balanced
export STR_PREPARED=/your/run/storage/isaacgym-balanced-configs
export STR_BALANCED_BASIS=/your/artifacts/sharpa_espp_eignoise_bal_k9_simtoolreal.npz
export STR_WANDB_ENTITY=your-wandb-entity
export STR_CACHE_ROOT=/your/cache/storage/isaacgym

"$STR_PYTHON" experiments/isaacgym_four/prepare.py \
  --output-dir "$STR_PREPARED" --run-root "$STR_RUN_ROOT" \
  --balanced-basis "$STR_BALANCED_BASIS" \
  --gpus 0 1 --wandb-entity "$STR_WANDB_ENTITY"

"$STR_PYTHON" experiments/isaacgym_four/launch.py \
  "$STR_PREPARED/manifest.json" --python "$STR_PYTHON" --dry-run
```

Omit `--dry-run` to start. The launcher refuses occupied GPUs and existing
run directories. Override paired seeds with `--seeds 42 42` when preparing.
Use `--runs jabs_scratch` for JABS alone; that needs no basis artifact.

Both runs log rewards, environment steps and learning rate to W&B; eigen
runs additionally log group RMS under `info/noise_eigadd_*/block_N`.
Check local `progress.json`, `startup_contract.json`, `worker.log`, and
`last/model.pth` below each run directory. SIGTERM saves the final checkpoint
and finishes W&B before exit. Capture records 30 seconds of states from four
environments at launch, every 500 epochs through 3000, then every 1000; MP4
rendering and W&B video upload are separate from this launcher.

## Monitor, stop and resume

```bash
python experiments/isaacgym_four/check_status.py "$STR_PREPARED/manifest.json"
```

The status command reports live workers, progress and first-five-minute
samples measured from actual training start. W&B's `train/episode_reward`
is the native rolling mean task return for the final SAPG block, which has
zero intrinsic-reward coefficient. `train/env_steps` counts this Gym run;
`train/parent_plus_env_steps` also includes the source training. Native
TensorBoard metrics are also synchronized to W&B.

Checkpoints include the complete native training state. The initial checkpoint
is saved after epoch 1, rolling checkpoints every 15 minutes, archives every
four hours, plus native best and graceful-shutdown checkpoints. Each run also
records local 30-second state windows for four worlds at launch, every 500
epochs through 3000, then every 1000. These are state files, not uploaded videos.

To stop one run, send SIGTERM to the PID in that run's `worker.pid`, then check
its exit status and shutdown checkpoint. For a restart, copy its resolved YAML
to a new config, set `campaign.run_dir` and `train.params.config.train_dir` to
a new segment directory, and keep its W&B ID. On the same authorized GPU, use
this command inside a named tmux session:

```bash
CUDA_VISIBLE_DEVICES=GPU-UUID \
  experiments/isaacgym_four/run_worker.sh /your/new-segment-config.yaml \
  --resume /your/previous-run/0_gym_RUN_sSEED/last/model.pth
```

`--resume` uses W&B `resume=must` and restores learner state, counters,
adaptive learning rate and curriculum while starting fresh Gym episodes.
Transient simulator tensors are deliberately not restored: restoring them
caused PhysX contact-capacity errors and non-finite observations in a restart
smoke. The corrected path passed actual PPO updates. Native behavior still
counts the first resumed rollout but skips its PPO update.
