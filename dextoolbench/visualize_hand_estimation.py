"""Visualize DexToolBench RGB-D, object poses, and HaMeR-depth hands in Viser."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import trimesh
import tyro
import viser
from PIL import Image
from scipy.spatial.transform import Rotation as R

from dextoolbench.estimate_hands import (
    MANO_JOINT_NAMES,
    SELECTED_KEYPOINT_NAMES,
    compute_palm_pose,
)
from dextoolbench.objects import NAME_TO_OBJECT


T_W_R = np.eye(4)
T_W_R[:3, 3] = np.array([0.0, 0.8, 0.0])
T_R_C = np.array(
    [
        [
            0.9552763064728893,
            -0.17920451516639435,
            0.2352295050275207,
            -0.5002050422666431,
        ],
        [
            -0.2889023075483251,
            -0.3958074425064433,
            0.8717063296487887,
            -1.4385715691360608,
        ],
        [
            -0.06310812138518884,
            -0.9006787497218348,
            -0.42987806970668574,
            1.0201893282998005,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
T_W_C = T_W_R @ T_R_C

FINGERTIP_NAMES = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
WRIST_NAMES = ("wrist_back", "wrist_front")
MANO_BONES = tuple(
    (0, finger_start + offset)
    if offset == 0
    else (finger_start + offset - 1, finger_start + offset)
    for finger_start in (1, 5, 9, 13, 17)
    for offset in range(4)
)


@dataclass
class Args:
    """Play a hand/object/RGB-D trajectory with independent visibility controls."""

    demo_dir: Path = Path("dextoolbench/data/hammer/claw_hammer/swing_down")
    """Directory containing rgb/, depth/, cam_K.txt, poses.json, and hand outputs."""

    object_name: Optional[str] = "claw_hammer"
    """DexToolBench object name, or None to omit the object mesh."""

    object_poses_path: Optional[Path] = None
    """Object poses; defaults to DEMO_DIR/poses.json. Poses are in robot frame."""

    hand_dir: Optional[Path] = None
    """HaMeR-depth results; defaults to DEMO_DIR/hand_pose_trajectory."""

    port: int = 8081
    dt: float = 1.0 / 30.0
    start_idx: int = 0
    point_cloud_stride: int = 2
    point_cloud_subsample: int = 2
    max_depth_m: float = 5.0
    point_size: float = 0.002
    hand_opacity: float = 0.72
    hand_point_size: float = 0.008
    axes_length: float = 0.08
    cache_size: int = 8

    show_point_cloud: bool = True
    show_object: bool = True
    show_hand_mesh: bool = True
    show_hand_skeleton: bool = True
    show_selected_keypoints: bool = False
    show_fingertips: bool = True
    show_wrist_keypoints: bool = True
    show_palm_pose: bool = True


@dataclass
class HandFrame:
    mesh: trimesh.Trimesh
    selected: Dict[str, np.ndarray]
    joints: Dict[str, np.ndarray]
    palm_pose: np.ndarray


def xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    return quaternion[[3, 0, 1, 2]]


def pose_to_transform(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"Expected [x,y,z,qx,qy,qz,qw], got {pose.shape}")
    transform = np.eye(4)
    transform[:3, :3] = R.from_quat(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def transform_pose(left_transform: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return left_transform @ pose


def pose_components(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return transform[:3, 3], xyzw_to_wxyz(R.from_matrix(transform[:3, :3]).as_quat())


def load_hand_frame(hand_dir: Path, stem: str) -> Optional[HandFrame]:
    json_path = hand_dir / f"{stem}.json"
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text())
    selected_data = data.get("keypoints", data)
    selected = {
        name: np.asarray(selected_data[name], dtype=np.float64)
        for name in SELECTED_KEYPOINT_NAMES
        if name in selected_data
    }
    if len(selected) != len(SELECTED_KEYPOINT_NAMES):
        missing = sorted(set(SELECTED_KEYPOINT_NAMES) - set(selected))
        raise KeyError(f"{json_path} is missing selected keypoints: {missing}")
    joints = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in data.get("joints_3d", {}).items()
    }
    mesh_filename = data.get("mano", {}).get("mesh_file", f"{stem}.obj")
    mesh_path = hand_dir / mesh_filename
    if not mesh_path.exists():
        raise FileNotFoundError(
            f"Hand mesh referenced by {json_path} is missing: {mesh_path}"
        )
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    palm_pose = np.asarray(
        data.get("palm_pose_cam", compute_palm_pose(selected)), dtype=np.float64
    )
    return HandFrame(mesh=mesh, selected=selected, joints=joints, palm_pose=palm_pose)


def depth_to_points(
    depth_path: Path,
    rgb_path: Path,
    intrinsics: np.ndarray,
    args: Args,
) -> Tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth = np.asarray(Image.open(depth_path)).astype(np.float32) / 1000.0
    stride = args.point_cloud_stride
    vv, uu = np.indices(depth.shape)
    z = depth[::stride, ::stride].reshape(-1)
    u = uu[::stride, ::stride].reshape(-1)
    v = vv[::stride, ::stride].reshape(-1)
    colors = rgb[::stride, ::stride].reshape(-1, 3)
    valid = (z > 0.0) & (z < args.max_depth_m)
    z, u, v, colors = z[valid], u[valid], v[valid], colors[valid]
    subsample = args.point_cloud_subsample
    z, u, v, colors = (
        z[::subsample],
        u[::subsample],
        v[::subsample],
        colors[::subsample],
    )
    fx, fy, cx, cy = (
        intrinsics[0, 0],
        intrinsics[1, 1],
        intrinsics[0, 2],
        intrinsics[1, 2],
    )
    points = np.stack(((u - cx) / fx * z, (v - cy) / fy * z, z), axis=1)
    return transform_points(T_W_C, points).astype(np.float32), colors.astype(np.uint8)


def main() -> None:
    args = tyro.cli(Args)
    rgb_dir = args.demo_dir / "rgb"
    depth_dir = args.demo_dir / "depth"
    hand_dir = args.hand_dir or (args.demo_dir / "hand_pose_trajectory")
    object_poses_path = args.object_poses_path or (args.demo_dir / "poses.json")
    intrinsics_path = args.demo_dir / "cam_K.txt"
    for path in (rgb_dir, depth_dir, hand_dir, object_poses_path, intrinsics_path):
        if not path.exists():
            raise FileNotFoundError(path)

    rgb_paths = sorted(rgb_dir.glob("*.png"))
    depth_by_stem = {path.stem: path for path in depth_dir.glob("*.png")}
    if not rgb_paths:
        raise ValueError(f"No PNG frames in {rgb_dir}")
    for rgb_path in rgb_paths:
        if rgb_path.stem not in depth_by_stem:
            raise FileNotFoundError(f"No depth frame matching {rgb_path.name}")
    intrinsics = np.loadtxt(intrinsics_path).reshape(3, 3)

    pose_data = json.loads(object_poses_path.read_text())
    if isinstance(pose_data, dict):
        pose_data = pose_data.get("poses_robot", pose_data.get("goals"))
    if pose_data is None:
        raise ValueError(f"No robot-frame pose list found in {object_poses_path}")
    object_poses = [T_W_R @ pose_to_transform(np.asarray(pose)) for pose in pose_data]
    n_frames = min(len(rgb_paths), len(object_poses))
    rgb_paths = rgb_paths[:n_frames]
    object_poses = object_poses[:n_frames]
    frame_idx = int(np.clip(args.start_idx, 0, n_frames - 1))

    object_mesh = None
    if args.object_name is not None:
        if args.object_name not in NAME_TO_OBJECT:
            raise KeyError(
                f"Unknown object {args.object_name!r}; choices: {sorted(NAME_TO_OBJECT)}"
            )
        object_mesh = NAME_TO_OBJECT[args.object_name].get_object_mesh()

    server = viser.ViserServer(port=args.port)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = T_W_C[:3, 3]
        client.camera.wxyz = xyzw_to_wxyz(R.from_matrix(T_W_C[:3, :3]).as_quat())

    object_handle = None
    if object_mesh is not None:
        object_position, object_wxyz = pose_components(object_poses[frame_idx])
        object_handle = server.scene.add_mesh_simple(
            "/object",
            vertices=np.asarray(object_mesh.vertices),
            faces=np.asarray(object_mesh.faces),
            color=(235, 180, 80),
            position=object_position,
            wxyz=object_wxyz,
        )

    hand_cache: Dict[int, Optional[HandFrame]] = {}

    def get_hand(index: int) -> Optional[HandFrame]:
        if index not in hand_cache:
            hand_cache[index] = load_hand_frame(hand_dir, rgb_paths[index].stem)
            while len(hand_cache) > args.cache_size:
                hand_cache.pop(next(iter(hand_cache)))
        return hand_cache[index]

    initial_hand = get_hand(frame_idx)
    if initial_hand is None:
        first_available = next(
            (
                i
                for i in range(n_frames)
                if (hand_dir / f"{rgb_paths[i].stem}.json").exists()
            ),
            None,
        )
        if first_available is None:
            raise ValueError(f"No hand JSON frames found in {hand_dir}")
        initial_hand = get_hand(first_available)
    assert initial_hand is not None
    initial_vertices = transform_points(T_W_C, np.asarray(initial_hand.mesh.vertices))
    hand_mesh_handle = server.scene.add_mesh_simple(
        "/hand/mano_mesh",
        vertices=initial_vertices,
        faces=np.asarray(initial_hand.mesh.faces),
        color=(90, 180, 255),
        opacity=args.hand_opacity,
        side="double",
    )
    placeholder = np.zeros((1, 3), dtype=np.float32)
    skeleton_handle = server.scene.add_line_segments(
        "/hand/skeleton",
        points=np.zeros((1, 2, 3)),
        colors=(50, 220, 255),
        line_width=3.0,
    )
    joint_handle = server.scene.add_point_cloud(
        "/hand/joints",
        points=placeholder,
        colors=(50, 220, 255),
        point_size=args.hand_point_size,
    )
    selected_handle = server.scene.add_point_cloud(
        "/hand/selected_keypoints",
        points=placeholder,
        colors=(255, 190, 60),
        point_size=args.hand_point_size,
    )
    fingertip_handle = server.scene.add_point_cloud(
        "/hand/fingertips",
        points=placeholder,
        colors=(255, 40, 190),
        point_size=args.hand_point_size * 1.35,
    )
    wrist_handle = server.scene.add_point_cloud(
        "/hand/wrist_keypoints",
        points=placeholder,
        colors=(255, 70, 70),
        point_size=args.hand_point_size * 1.25,
    )
    palm_position, palm_wxyz = pose_components(
        transform_pose(T_W_C, initial_hand.palm_pose)
    )
    palm_handle = server.scene.add_frame(
        "/hand/palm_pose",
        position=palm_position,
        wxyz=palm_wxyz,
        axes_length=args.axes_length,
        axes_radius=args.axes_length * 0.04,
    )

    point_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def get_cloud(index: int) -> Tuple[np.ndarray, np.ndarray]:
        if index not in point_cache:
            rgb_path = rgb_paths[index]
            point_cache[index] = depth_to_points(
                depth_by_stem[rgb_path.stem], rgb_path, intrinsics, args
            )
            while len(point_cache) > args.cache_size:
                point_cache.pop(next(iter(point_cache)))
        return point_cache[index]

    initial_points, initial_colors = get_cloud(frame_idx)
    pcd_handle = server.scene.add_point_cloud(
        "/camera/rgbd_points",
        points=initial_points,
        colors=initial_colors,
        point_size=args.point_size,
        precision="float32",
    )

    fps = 1.0 / args.dt

    def frame_label(index: int) -> str:
        return f"{index * args.dt:.3f}s ({index:04d}/{n_frames - 1:04d}, {fps:.0f} fps)"

    with server.gui.add_folder("Frame Controls"):
        slider = server.gui.add_slider(
            "Frame", min=0, max=n_frames - 1, step=1, initial_value=frame_idx
        )
        label = server.gui.add_markdown(frame_label(frame_idx))
        pause_button = server.gui.add_button("Pause")
        decrement_button = server.gui.add_button("Previous")
        increment_button = server.gui.add_button("Next")
        reset_button = server.gui.add_button("Reset")
    with server.gui.add_folder("Visibility"):
        object_checkbox = server.gui.add_checkbox(
            "Object", initial_value=args.show_object
        )
        pcd_checkbox = server.gui.add_checkbox(
            "RGB-D point cloud", initial_value=args.show_point_cloud
        )
        mesh_checkbox = server.gui.add_checkbox(
            "MANO hand mesh", initial_value=args.show_hand_mesh
        )
        skeleton_checkbox = server.gui.add_checkbox(
            "21-joint hand skeleton", initial_value=args.show_hand_skeleton
        )
        selected_checkbox = server.gui.add_checkbox(
            "Surface keypoints", initial_value=args.show_selected_keypoints
        )
        fingertips_checkbox = server.gui.add_checkbox(
            "Fingertips", initial_value=args.show_fingertips
        )
        wrist_checkbox = server.gui.add_checkbox(
            "Wrist keypoints", initial_value=args.show_wrist_keypoints
        )
        palm_checkbox = server.gui.add_checkbox(
            "Palm pose", initial_value=args.show_palm_pose
        )

    paused = False

    def apply_visibility(has_hand: bool) -> None:
        if object_handle is not None:
            object_handle.visible = bool(object_checkbox.value)
        pcd_handle.visible = bool(pcd_checkbox.value)
        hand_mesh_handle.visible = has_hand and bool(mesh_checkbox.value)
        skeleton_handle.visible = has_hand and bool(skeleton_checkbox.value)
        joint_handle.visible = has_hand and bool(skeleton_checkbox.value)
        selected_handle.visible = has_hand and bool(selected_checkbox.value)
        fingertip_handle.visible = has_hand and bool(fingertips_checkbox.value)
        wrist_handle.visible = has_hand and bool(wrist_checkbox.value)
        palm_handle.visible = has_hand and bool(palm_checkbox.value)

    def set_frame(index: int) -> None:
        nonlocal frame_idx
        frame_idx = int(np.clip(index, 0, n_frames - 1))
        label.content = frame_label(frame_idx)
        if object_handle is not None:
            object_handle.position, object_handle.wxyz = pose_components(
                object_poses[frame_idx]
            )
        points, colors = get_cloud(frame_idx)
        pcd_handle.points, pcd_handle.colors = points, colors

        hand = get_hand(frame_idx)
        if hand is not None:
            hand_mesh_handle.vertices = transform_points(
                T_W_C, np.asarray(hand.mesh.vertices)
            )
            selected_points = transform_points(
                T_W_C, np.stack(list(hand.selected.values()))
            )
            selected_handle.points = selected_points.astype(np.float32)
            wrist_handle.points = transform_points(
                T_W_C, np.stack([hand.selected[name] for name in WRIST_NAMES])
            ).astype(np.float32)
            if all(name in hand.joints for name in MANO_JOINT_NAMES):
                joints = transform_points(
                    T_W_C, np.stack([hand.joints[name] for name in MANO_JOINT_NAMES])
                ).astype(np.float32)
                joint_handle.points = joints
                skeleton_handle.points = np.stack(
                    [[joints[start], joints[end]] for start, end in MANO_BONES]
                )
                fingertip_handle.points = np.stack(
                    [joints[MANO_JOINT_NAMES.index(name)] for name in FINGERTIP_NAMES]
                )
            else:
                fingertip_handle.points = transform_points(
                    T_W_C,
                    np.stack(
                        [
                            hand.selected[f"{finger}_3"]
                            for finger in ("thumb", "index", "middle", "ring", "pinky")
                        ]
                    ),
                ).astype(np.float32)
            palm_handle.position, palm_handle.wxyz = pose_components(
                T_W_C @ hand.palm_pose
            )
        apply_visibility(hand is not None)

    @slider.on_update
    def _(_) -> None:
        set_frame(int(slider.value))

    @pause_button.on_click
    def _(_) -> None:
        nonlocal paused
        paused = not paused
        pause_button.label = "Play" if paused else "Pause"

    @decrement_button.on_click
    def _(_) -> None:
        nonlocal paused
        paused = True
        pause_button.label = "Play"
        slider.value = max(0, frame_idx - 1)

    @increment_button.on_click
    def _(_) -> None:
        nonlocal paused
        paused = True
        pause_button.label = "Play"
        slider.value = min(n_frames - 1, frame_idx + 1)

    @reset_button.on_click
    def _(_) -> None:
        slider.value = 0

    for checkbox in (
        object_checkbox,
        pcd_checkbox,
        mesh_checkbox,
        skeleton_checkbox,
        selected_checkbox,
        fingertips_checkbox,
        wrist_checkbox,
        palm_checkbox,
    ):
        checkbox.on_update(lambda _: apply_visibility(get_hand(frame_idx) is not None))

    set_frame(frame_idx)
    print(f"Serving {n_frames} frames from {args.demo_dir}")
    print(f"Hand results: {hand_dir}")
    try:
        while True:
            loop_start = time.time()
            if not paused:
                slider.value = 0 if frame_idx == n_frames - 1 else frame_idx + 1
            time.sleep(max(0.0, args.dt - (time.time() - loop_start)))
    except KeyboardInterrupt:
        print("Stopping Viser server")


if __name__ == "__main__":
    main()
