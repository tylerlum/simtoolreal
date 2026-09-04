"""Estimate depth-aligned MANO hands for a DexToolBench RGB-D sequence.

This is a thin, richer-output frontend for the sibling ``hamer_depth`` repo.
It deliberately expects hand masks rather than the object masks in ``masks/``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tyro


SELECTED_KEYPOINT_NAMES = (
    "wrist_back",
    "wrist_front",
    "index_0_back",
    "index_0_front",
    "middle_0_back",
    "middle_0_front",
    "ring_0_back",
    "ring_0_front",
    "index_3",
    "middle_3",
    "ring_3",
    "thumb_3",
    "pinky_3",
)

# HaMeR's MANO wrapper emits OpenPose hand ordering.
MANO_JOINT_NAMES = (
    "wrist",
    "thumb_1",
    "thumb_2",
    "thumb_3",
    "thumb_tip",
    "index_1",
    "index_2",
    "index_3",
    "index_tip",
    "middle_1",
    "middle_2",
    "middle_3",
    "middle_tip",
    "ring_1",
    "ring_2",
    "ring_3",
    "ring_tip",
    "pinky_1",
    "pinky_2",
    "pinky_3",
    "pinky_tip",
)


@dataclass
class Args:
    """Run HaMeR-depth and save meshes, keypoints, MANO parameters, and palm poses."""

    demo_dir: Path = Path("dextoolbench/data/hammer/claw_hammer/swing_down")
    """Sequence containing rgb/, depth/, hand_masks/, and cam_K.txt."""

    hamer_depth_repo: Path = Path("/home/tylerlum/github_repos/hamer_depth")
    """Path to the hamer_depth checkout."""

    hand_masks_dir: Optional[Path] = None
    """Hand mask directory. Defaults to DEMO_DIR/hand_masks (not DEMO_DIR/masks)."""

    output_dir: Optional[Path] = None
    """Output directory. Defaults to DEMO_DIR/hand_pose_trajectory."""

    hand_type: str = "LEFT"
    """RIGHT or LEFT."""

    only_idx: Optional[int] = None
    """Process just this zero-based frame index."""

    ignore_exceptions: bool = False
    """Log failed frames and continue."""

    overwrite: bool = False
    """Recompute frames whose JSON and OBJ outputs already exist."""

    debug: bool = False
    """Show hamer_depth debug visualizations."""


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Cannot normalize a near-zero vector")
    return vector / norm


def compute_palm_pose(keypoints: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute the palm frame used by the existing hand-retargeting baseline.

    Position is the midpoint of the front/back wrist landmarks. Frame columns
    are X=palm normal, Y=index-to-ring direction, Z=wrist-to-middle direction.
    """

    position = np.mean([keypoints["wrist_back"], keypoints["wrist_front"]], axis=0)
    z_axis = normalize(keypoints["middle_0_front"] - keypoints["wrist_front"])
    y_hint = normalize(keypoints["ring_0_front"] - keypoints["index_0_front"])
    y_axis = normalize(y_hint - np.dot(y_hint, z_axis) * z_axis)
    x_axis = normalize(np.cross(y_axis, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))

    transform = np.eye(4)
    transform[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    transform[:3, 3] = position
    return transform


def rigid_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the rigid transform mapping paired source points to target points."""

    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"Expected matching (N,3) arrays, got {source.shape}, {target.shape}"
        )
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def paths_by_stem(directory: Path) -> Dict[str, Path]:
    paths = sorted(directory.glob("*.png"))
    return {path.stem: path for path in paths}


def validate_inputs(args: Args) -> Tuple[Path, Path, Path, Path, Path]:
    rgb_dir = args.demo_dir / "rgb"
    depth_dir = args.demo_dir / "depth"
    mask_dir = args.hand_masks_dir or (args.demo_dir / "hand_masks")
    intrinsics_path = args.demo_dir / "cam_K.txt"
    output_dir = args.output_dir or (args.demo_dir / "hand_pose_trajectory")
    for label, path in (
        ("RGB directory", rgb_dir),
        ("depth directory", depth_dir),
        ("hand-mask directory", mask_dir),
        ("camera intrinsics", intrinsics_path),
        ("hamer_depth checkout", args.hamer_depth_repo),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if mask_dir.resolve() == (args.demo_dir / "masks").resolve():
        print(
            "WARNING: DEMO_DIR/masks is normally the object mask; hand_masks is expected."
        )
    return rgb_dir, depth_dir, mask_dir, intrinsics_path, output_dir


def run(args: Args) -> None:
    rgb_dir, depth_dir, mask_dir, intrinsics_path, output_dir = validate_inputs(args)
    rgb_by_stem = paths_by_stem(rgb_dir)
    depth_by_stem = paths_by_stem(depth_dir)
    masks_by_stem = paths_by_stem(mask_dir)
    common_stems = sorted(set(rgb_by_stem) & set(depth_by_stem) & set(masks_by_stem))
    if not common_stems:
        raise ValueError("No matching PNG stems across rgb, depth, and hand masks")
    missing = {
        "depth": sorted(set(rgb_by_stem) - set(depth_by_stem)),
        "hand masks": sorted(set(rgb_by_stem) - set(masks_by_stem)),
    }
    for label, stems in missing.items():
        if stems:
            raise ValueError(
                f"Missing {label} for {len(stems)} RGB frames; first: {stems[:3]}"
            )

    hamer_type = args.hand_type.upper()
    if hamer_type not in ("LEFT", "RIGHT"):
        raise ValueError(f"hand_type must be LEFT or RIGHT, got {args.hand_type!r}")

    # Import from the requested checkout while running in its tested environment.
    sys.path.insert(0, str(args.hamer_depth_repo))
    import cv2
    from PIL import Image
    from tqdm import tqdm

    from hamer_depth.detectors.detector_hamer import DetectorHamer
    from hamer_depth.utils.cam_intrinsics_utils import (
        convert_intrinsics_matrix_to_dict,
        get_camera_matrix_from_file,
    )
    from hamer_depth.utils.hand_type import HandType
    from hamer_depth.utils.run_utils import process_image_with_hamer

    hand_enum = HandType.RIGHT if hamer_type == "RIGHT" else HandType.LEFT
    detector = DetectorHamer()
    camera_matrix = get_camera_matrix_from_file(intrinsics_path)
    camera_intrinsics = convert_intrinsics_matrix_to_dict(camera_matrix)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_stems = common_stems
    if args.only_idx is not None:
        if not 0 <= args.only_idx < len(common_stems):
            raise IndexError(
                f"only_idx={args.only_idx} outside [0, {len(common_stems) - 1}]"
            )
        selected_stems = [common_stems[args.only_idx]]

    failures = []
    for stem in tqdm(selected_stems, desc="HaMeR-depth", dynamic_ncols=True):
        json_path = output_dir / f"{stem}.json"
        mesh_path = output_dir / f"{stem}.obj"
        if not args.overwrite and json_path.exists() and mesh_path.exists():
            continue
        try:
            rgb = np.asarray(Image.open(rgb_by_stem[stem]).convert("RGB"))
            depth = np.asarray(Image.open(depth_by_stem[stem])).astype(np.float32)
            if np.nanmax(depth) > 100.0:
                depth /= 1000.0
            mask = np.asarray(Image.open(masks_by_stem[stem]))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask > 0
            if not np.any(mask):
                raise ValueError("hand mask is empty")

            (
                hamer_out,
                selected_keypoints_raw,
                refined_mesh,
                annotated_before,
                annotated_after,
            ) = process_image_with_hamer(
                img_rgb=rgb,
                img_depth=depth,
                mask=mask,
                cam_intrinsics=camera_intrinsics,
                detector_hamer=detector,
                hand_type=hand_enum,
                debug=args.debug,
            )

            selected_keypoints = {
                name: np.asarray(selected_keypoints_raw[name], dtype=np.float64)
                for name in SELECTED_KEYPOINT_NAMES
            }
            source_vertices = np.asarray(hamer_out["verts"], dtype=np.float64)
            refined_vertices = np.asarray(refined_mesh.vertices, dtype=np.float64)
            depth_alignment = rigid_alignment(source_vertices, refined_vertices)
            joints_3d = transform_points(
                depth_alignment, np.asarray(hamer_out["kpts_3d"], dtype=np.float64)
            )
            if joints_3d.shape != (len(MANO_JOINT_NAMES), 3):
                raise ValueError(f"Unexpected HaMeR joints shape: {joints_3d.shape}")
            palm_pose = compute_palm_pose(selected_keypoints)

            refined_mesh.export(mesh_path)
            annotated = np.concatenate([annotated_before, annotated_after], axis=1)
            cv2.imwrite(
                str(output_dir / f"{stem}.png"),
                cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
            )

            # Keep the original flat keys for compatibility with existing baselines.
            frame_data = {
                name: selected_keypoints[name].tolist()
                for name in SELECTED_KEYPOINT_NAMES
            }
            frame_data.update(
                {
                    "global_orient": np.asarray(hamer_out["global_orient"]).tolist(),
                    "schema_version": 2,
                    "coordinate_frame": "camera",
                    "hand_type": hamer_type,
                    "keypoints": {
                        name: selected_keypoints[name].tolist()
                        for name in SELECTED_KEYPOINT_NAMES
                    },
                    "joints_3d": {
                        name: joints_3d[i].tolist()
                        for i, name in enumerate(MANO_JOINT_NAMES)
                    },
                    "fingertips": {
                        name: joints_3d[MANO_JOINT_NAMES.index(name)].tolist()
                        for name in (
                            "thumb_tip",
                            "index_tip",
                            "middle_tip",
                            "ring_tip",
                            "pinky_tip",
                        )
                    },
                    "palm_pose_cam": palm_pose.tolist(),
                    "depth_alignment_transform": depth_alignment.tolist(),
                    "mano": {
                        "mesh_file": mesh_path.name,
                        "global_orient": np.asarray(
                            hamer_out["global_orient"]
                        ).tolist(),
                        "hand_pose": np.asarray(hamer_out["hand_pose"]).tolist(),
                    },
                }
            )
            json_path.write_text(json.dumps(frame_data, indent=2) + "\n")
        except Exception as exc:
            if not args.ignore_exceptions:
                raise
            failures.append((stem, str(exc)))
            print(f"WARNING: skipping {stem}: {exc}")

    print(f"Outputs: {output_dir}")
    print(
        f"Processed/available frames: {len(selected_stems) - len(failures)}/{len(selected_stems)}"
    )
    if failures:
        failure_path = output_dir / "failures.json"
        failure_path.write_text(json.dumps(dict(failures), indent=2) + "\n")
        print(f"Failures: {failure_path}")


def main() -> None:
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()
