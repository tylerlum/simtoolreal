# Hand estimation and visualization

The hand pipeline has two inputs that must not be confused:

- `masks/` contains the tracked tool/object mask.
- `hand_masks/` contains the tracked human-hand mask used by `hamer_depth` for
  its depth alignment.

## 1. Create hand masks

Use the sibling SAM2 checkout. You can click the first-frame hand interactively:

```bash
cd /home/tylerlum/github_repos/segment-anything-2-real-time
source .venv/bin/activate
python video_sam2.py \
  --input_dir /home/tylerlum/github_repos/simtoolreal/dextoolbench/data/hammer/claw_hammer/swing_down/rgb \
  --output_dir /home/tylerlum/github_repos/simtoolreal/dextoolbench/data/hammer/claw_hammer/swing_down/hand_masks \
  --use_second_prompt
```

For this particular `swing_down` recording, the following non-interactive
first-frame prompts land on the palm/fingers:

```bash
python video_sam2.py \
  --input_dir /home/tylerlum/github_repos/simtoolreal/dextoolbench/data/hammer/claw_hammer/swing_down/rgb \
  --output_dir /home/tylerlum/github_repos/simtoolreal/dextoolbench/data/hammer/claw_hammer/swing_down/hand_masks \
  --use_second_prompt \
  --prompt_x 660 --prompt_y 65 \
  --second_prompt_x 700 --second_prompt_y 62
```

Inspect several masks across the recording before estimating hands. A mask that
drifts onto the tool or sleeve will make the depth refinement inaccurate.

## 2. Estimate hands

Run the frontend with the tested `hamer_depth` environment:

```bash
cd /home/tylerlum/github_repos/simtoolreal
/home/tylerlum/github_repos/hamer_depth/.venv/bin/python \
  -m dextoolbench.estimate_hands \
  --demo-dir dextoolbench/data/hammer/claw_hammer/swing_down \
  --ignore-exceptions
```

Use `--only-idx 0` for a one-frame smoke test and `--overwrite` to recompute
existing outputs. Results go to `hand_pose_trajectory/` by default:

- `frame_NNNN.obj`: depth-aligned MANO hand mesh.
- `frame_NNNN.png`: before/after depth-refinement overlay.
- `frame_NNNN.json`: 21 MANO joints, five fingertips, the existing 13 surface
  landmarks, depth alignment transform, MANO pose matrices, and `palm_pose_cam`.
- `failures.json`: skipped frames when `--ignore-exceptions` is enabled.

All 3D hand quantities are in the camera frame and use meters. The palm pose
uses the same landmark convention as the existing retargeting baseline:
position at the midpoint of the two wrist surface points, Z toward the middle
knuckle, Y from index toward ring, and X along the palm normal.

## 3. Visualize

```bash
cd /home/tylerlum/github_repos/simtoolreal
.venv/bin/python -m dextoolbench.visualize_hand_estimation \
  --demo-dir dextoolbench/data/hammer/claw_hammer/swing_down
```

Open the URL printed by Viser. The `Visibility` panel independently toggles the
RGB-D cloud, object, MANO mesh, 21-joint skeleton, surface landmarks,
fingertips, wrist points, and palm pose.
