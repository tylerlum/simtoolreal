# Checkpoint comparison — 2026-09-04

Controlled Isaac Gym evaluation of the pretrained SimToolReal policy and four
checkpoints supplied by Harsh.

| Policy | Strict hits | Hit rate | Mean time to hit | Mean return | Median return |
|---|---:|---:|---:|---:|---:|
| Pretrained | 31 / 32 | 96.9% | 4.14 s | 1395.08 | 1438.91 |
| Eigenoise 01 | 0 / 32 | 0.0% | — | 416.07 | 410.34 |
| Eigenoise 02 | 0 / 32 | 0.0% | — | 385.27 | 403.36 |
| Jabs 01 | 0 / 32 | 0.0% | — | 394.92 | 399.65 |
| Jabs 02 | 0 / 32 | 0.0% | — | 403.03 | 400.79 |

## Protocol

- 32 parallel trials per checkpoint, 600 policy steps maximum.
- The same seeded procedural cuboid-handle hammer and the same 32 goal poses.
- Goal-array SHA-256 prefix for every run: `30d40b3a758165b3`.
- Fixed initial object pose and no observation/action delays, state noise,
  reset noise, external forces, or torque disturbances.
- Deterministic policy actions (`deterministic_actions=True`).
- Strict 0.01 goal tolerance and one goal required for episode success.
- Undiscounted task reward is accumulated until episode termination.

## Checkpoint sanity check

All five checkpoints load successfully into the same policy architecture. Each
contains 26 model tensors and compatible observation normalization state with
140-element running mean and variance tensors. The four new files are
inference-only (`{"model": ...}`), while the pretrained file additionally has
training/environment state under key `0`; the benchmark deliberately does not
restore environment state from any checkpoint.

The roughly 1000-point return gap is consistent with the task's +1000
goal-success bonus: the new policies still collect several hundred points of
shaped pickup/lifting/manipulation reward, but never satisfy the strict final
pose criterion in these trials.
