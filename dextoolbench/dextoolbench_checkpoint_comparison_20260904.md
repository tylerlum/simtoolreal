# Full DexToolBench checkpoint comparison — 2026-09-04

Each cell reports `waypoints reached / total waypoints` in one deterministic
episode. A task is complete only when all waypoints are reached.

| Policy | Complete tasks | Tasks with any progress | Waypoints |
|---|---:|---:|---:|
| Pretrained | 20/24 | 24/24 | 890/954 (93.3%) |
| Eigenoise 01 | 0/24 | 0/24 | 0/954 (0.0%) |
| Eigenoise 02 | 0/24 | 2/24 | 3/954 (0.31%) |
| Jabs 01 | 0/24 | 0/24 | 0/954 (0.0%) |
| Jabs 02 | 0/24 | 0/24 | 0/954 (0.0%) |

| Category | Object | Task | Pretrained | Eigen 01 | Eigen 02 | Jabs 01 | Jabs 02 |
|---|---|---|---:|---:|---:|---:|---:|
| brush | blue_brush | sweep_forward | 34/34 | 0/34 | 0/34 | 0/34 | 0/34 |
| brush | blue_brush | sweep_right | 21/21 | 0/21 | 0/21 | 0/21 | 0/21 |
| brush | red_brush | sweep_forward | 32/38 | 0/38 | 0/38 | 0/38 | 0/38 |
| brush | red_brush | sweep_right | 38/38 | 0/38 | 0/38 | 0/38 | 0/38 |
| eraser | flat_eraser | wipe_c | 6/35 | 0/35 | 0/35 | 0/35 | 0/35 |
| eraser | flat_eraser | wipe_smile | 33/33 | 0/33 | 0/33 | 0/33 | 0/33 |
| eraser | handle_eraser | wipe_c | 31/31 | 0/31 | 0/31 | 0/31 | 0/31 |
| eraser | handle_eraser | wipe_smile | 29/29 | 0/29 | 0/29 | 0/29 | 0/29 |
| hammer | claw_hammer | swing_down | 37/37 | 0/37 | 0/37 | 0/37 | 0/37 |
| hammer | claw_hammer | swing_side | 40/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| hammer | mallet_hammer | swing_down | 36/36 | 0/36 | 0/36 | 0/36 | 0/36 |
| hammer | mallet_hammer | swing_side | 32/32 | 0/32 | 2/32 | 0/32 | 0/32 |
| marker | sharpie_marker | draw_smile | 36/36 | 0/36 | 0/36 | 0/36 | 0/36 |
| marker | sharpie_marker | write_c | 25/25 | 0/25 | 0/25 | 0/25 | 0/25 |
| marker | staples_marker | draw_smile | 49/49 | 0/49 | 0/49 | 0/49 | 0/49 |
| marker | staples_marker | write_c | 29/29 | 0/29 | 0/29 | 0/29 | 0/29 |
| screwdriver | long_screwdriver | spin_horizontal | 31/31 | 0/31 | 0/31 | 0/31 | 0/31 |
| screwdriver | long_screwdriver | spin_vertical | 84/100 | 0/100 | 0/100 | 0/100 | 0/100 |
| screwdriver | short_screwdriver | spin_horizontal | 25/38 | 0/38 | 0/38 | 0/38 | 0/38 |
| screwdriver | short_screwdriver | spin_vertical | 76/76 | 0/76 | 1/76 | 0/76 | 0/76 |
| spatula | flat_spatula | flip_over | 45/45 | 0/45 | 0/45 | 0/45 | 0/45 |
| spatula | flat_spatula | serve_plate | 33/33 | 0/33 | 0/33 | 0/33 | 0/33 |
| spatula | spoon_spatula | flip_over | 40/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| spatula | spoon_spatula | serve_plate | 48/48 | 0/48 | 0/48 | 0/48 | 0/48 |

## Protocol

- Five side-by-side Isaac Gym environments per task, one per policy.
- Exact dataset object, start pose, task environment, and full waypoint list.
- Deterministic actions for every policy.
- Observation/action delays, reset noise, state noise, forces, and impulses off.
- Fixed-size keypoint reward, success tolerance 0.01, and one near-goal step.
- No checkpoint environment-state restoration, so all policies use identical
  evaluation configuration.

The four new checkpoints load successfully with compatible model and
normalization tensor shapes. Their near-total failure across every object class
is therefore not localized to hammer orientation or one DexToolBench task.
