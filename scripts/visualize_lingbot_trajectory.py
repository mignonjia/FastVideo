import argparse
import time
from pathlib import Path

import numpy as np
import torch


def load_npy(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return np.load(str(p), allow_pickle=True)


def normalize_to_first_pose(c2ws: np.ndarray) -> np.ndarray:
    first_inv = np.linalg.inv(c2ws[0])
    return np.einsum("ij,tjk->tik", first_inv, c2ws)


def load_action_from_npy(action_path: str) -> tuple[np.ndarray, np.ndarray]:
    data = load_npy(action_path)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()
    if not isinstance(data, dict):
        raise ValueError(
            "Action file must be a dict-like npy with keys 'keyboard' and 'mouse'."
        )
    if "keyboard" not in data or "mouse" not in data:
        raise ValueError("Action dict must contain both 'keyboard' and 'mouse'.")
    keyboard = np.asarray(data["keyboard"], dtype=np.float32)
    mouse = np.asarray(data["mouse"], dtype=np.float32)
    return keyboard, mouse


def _reformat_keyboard_and_mouse_cond(
    num_frames: int,
    keyboard_cond: torch.Tensor,
    mouse_cond: torch.Tensor,
    compression_ratio: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert frame-level controls to latent-level controls."""

    # keyboard / mouse shape: [T, C]
    assert keyboard_cond.shape[0] == num_frames and mouse_cond.shape[0] == num_frames, (
        "keyboard_cond and mouse_cond must have the same number of frames as num_frames, "
        f"num_frames: {num_frames} "
        f"got keyboard_cond shape: {keyboard_cond.shape}, mouse_cond shape: {mouse_cond.shape}"
    )
    assert (num_frames - 1) % compression_ratio == 0, (
        f"num_frames must satisfy (num_frames - 1) % {compression_ratio} == 0, "
        f"got {num_frames}"
    )
    keyboard_cond = keyboard_cond[1:, :]
    mouse_cond = mouse_cond[1:, :]

    groups = keyboard_cond.view(-1, compression_ratio, keyboard_cond.shape[1])
    assert (groups == groups[:, 0:1]).all(dim=1).all(), (
        "keyboard_tensor must be constant within each compression group"
    )
    groups = mouse_cond.view(-1, compression_ratio, mouse_cond.shape[1])
    assert (groups == groups[:, 0:1]).all(dim=1).all(), (
        "mouse_tensor must be constant within each compression group"
    )
    return keyboard_cond[::compression_ratio], mouse_cond[::compression_ratio]


def _motions_to_c2ws(
    keyboard_cond: torch.Tensor,
    mouse_cond: torch.Tensor,
    forward_speed: float = 3.0,
) -> torch.Tensor:
    n_steps = keyboard_cond.shape[0]
    poses = []
    pose = torch.eye(4, dtype=torch.float32)
    poses.append(pose.clone())

    for t in range(n_steps):
        forward = 0.0
        right = 0.0
        if keyboard_cond[t, 0] > 0.9:  # W
            forward += forward_speed
        if keyboard_cond[t, 1] > 0.9:  # S
            forward -= forward_speed
        if keyboard_cond[t, 2] > 0.9:  # A
            right -= forward_speed
        if keyboard_cond[t, 3] > 0.9:  # D
            right += forward_speed

        pitch = float(mouse_cond[t, 0].item())
        yaw = float(mouse_cond[t, 1].item())

        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        r_pitch = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=torch.float32
        )
        r_yaw = torch.tensor(
            [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=torch.float32
        )
        r_delta = r_yaw @ r_pitch

        delta = torch.eye(4, dtype=torch.float32)
        delta[:3, :3] = r_delta
        delta[:3, 3] = torch.tensor([right, 0.0, forward], dtype=torch.float32)
        pose = pose @ delta
        poses.append(pose.clone())

    return torch.stack(poses, dim=0)


def action_to_c2ws(
    keyboard: np.ndarray,
    mouse: np.ndarray,
    num_frames: int,
    forward_speed: float = 3.0,
) -> np.ndarray:
    if keyboard.ndim != 2 or mouse.ndim != 2:
        raise ValueError("keyboard/mouse must be [T, C].")
    if keyboard.shape[1] < 4:
        raise ValueError("keyboard must have at least 4 dims for W/S/A/D.")
    if mouse.shape[1] != 2:
        raise ValueError("mouse must be [T,2] as [pitch,yaw].")
    
    keyboard_t = torch.from_numpy(keyboard)
    mouse_t = torch.from_numpy(mouse)
    
    keyboard_lat, mouse_lat = _reformat_keyboard_and_mouse_cond(
        num_frames=num_frames,
        keyboard_cond=keyboard_t,
        mouse_cond=mouse_t,
    )
    c2ws_t = _motions_to_c2ws(
        keyboard_cond=keyboard_lat,
        mouse_cond=mouse_lat,
        forward_speed=forward_speed,
    )
    return c2ws_t.numpy()


def camera_centers(c2ws: np.ndarray) -> np.ndarray:
    return c2ws[:, :3, 3]


def compute_center_metrics(
    gt_c2ws: np.ndarray, pred_c2ws: np.ndarray
) -> tuple[float, float]:
    gt = camera_centers(gt_c2ws)
    pred = camera_centers(pred_c2ws)
    n = min(len(gt), len(pred))
    dist = np.linalg.norm(gt[:n] - pred[:n], axis=-1)
    return float(dist.mean()), float(dist.max())


def make_path_lines(points: np.ndarray, color: list[float]):
    import open3d as o3d

    if len(points) < 2:
        return o3d.geometry.LineSet()
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def make_camera_axes(c2w: np.ndarray, scale: float = 0.08):
    import open3d as o3d

    origin = c2w[:3, 3]
    r = c2w[:3, :3]
    points = np.stack(
        [
            origin,
            origin + r[:, 0] * scale,
            origin,
            origin + r[:, 1] * scale,
            origin,
            origin + r[:, 2] * scale,
        ]
    )
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    axis.lines = o3d.utility.Vector2iVector([[0, 1], [2, 3], [4, 5]])
    axis.colors = o3d.utility.Vector3dVector(
        [[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.3, 1.0]]
    )
    return axis


def make_camera_frustum(
    c2w: np.ndarray,
    intrinsic: np.ndarray,
    image_width: int,
    image_height: int,
    depth: float = 0.35,
    color: list[float] | None = None,
):
    import open3d as o3d

    fx, fy, cx, cy = [float(v) for v in intrinsic]
    px = np.array(
        [[0, 0], [image_width - 1, 0], [image_width - 1, image_height - 1], [0, image_height - 1]],
        dtype=np.float32,
    )
    xs = (px[:, 0] - cx) / fx * depth
    ys = (px[:, 1] - cy) / fy * depth
    zs = np.full_like(xs, depth)
    corners_cam = np.stack([xs, ys, zs], axis=1)
    corners_h = np.concatenate([corners_cam, np.ones((4, 1), dtype=np.float32)], axis=1)
    corners_world = (c2w @ corners_h.T).T[:, :3]
    origin = c2w[:3, 3]

    points = np.vstack([origin[None], corners_world]).astype(np.float64)
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    c = color or [1.0, 0.6, 0.0]
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([c] * len(lines))
    return frustum


def make_ground_grid(grid_size: int = 100, step: float = 1.0, y: float = 0.0):
    import open3d as o3d

    points = []
    lines = []
    colors = []
    half = grid_size * step
    idx = 0
    # Create an enhanced grid with brighter colors at the center
    for i in range(-grid_size, grid_size + 1):
        z = i * step
        points.append([-half, y, z])
        points.append([half, y, z])
        lines.append([idx, idx + 1])
        c = 0.5 if i % 4 == 0 else 0.2
        colors.append([c, c, c])
        idx += 2

        x = i * step
        points.append([x, y, -half])
        points.append([x, y, half])
        lines.append([idx, idx + 1])
        c = 0.5 if i % 4 == 0 else 0.2
        colors.append([c, c, c])
        idx += 2

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector(colors)
    return grid


def make_landmarks(y_offset: float = 0.0, landmark_count: int = 80):
    import open3d as o3d
    import random
    
    random.seed(42)
    meshes = []
    
    # Generate a procedural "city" or obstacle course
    for _ in range(landmark_count):
        w = random.uniform(1.0, 5.0)
        d = random.uniform(1.0, 5.0)
        h = random.uniform(1.0, 15.0)
        
        x = random.uniform(-60, 60)
        z = random.uniform(-20, 100)
        
        # Keep clear of the starting area
        if abs(x) < 2.0 and abs(z) < 2.5:
            continue
            
        color = [
            random.uniform(0.3, 0.8),
            random.uniform(0.3, 0.8),
            random.uniform(0.3, 0.9)
        ]
        
        m = o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=d)
        m.compute_vertex_normals()
        m.paint_uniform_color(color)
        
        # NOTE: In this camera coordinate system, +Y is DOWN. 
        # So "UP" is negative Y. 
        # create_box creates the box in [0, h] in Y. 
        # To make it stand on the floor (y=y_offset) and go UP, we translate Y by y_offset - h.
        m.translate([x - w / 2.0, y_offset - h, z - d / 2.0])
        meshes.append(m)
        
    return meshes


def build_scene(
    gt_c2ws: np.ndarray | None,
    pred_c2ws: np.ndarray | None,
    intrinsics: np.ndarray,
    axis_stride: int,
    image_width: int,
    image_height: int,
    show_frustums: bool,
    landmark_count: int,
):
    import open3d as o3d

    # Move the floor and objects down by 20.0 units (+Y is down)
    geoms = [make_ground_grid(y=20.0), o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)]
    geoms.extend(make_landmarks(y_offset=20.0, landmark_count=landmark_count))

    if gt_c2ws is not None:
        gt_pts = camera_centers(gt_c2ws)
        geoms.append(make_path_lines(gt_pts, [1.0, 0.2, 0.2]))
        n_gt = min(len(gt_c2ws), len(intrinsics))
        for i in range(0, n_gt, max(1, axis_stride)):
            geoms.append(make_camera_axes(gt_c2ws[i], 0.08))
            if show_frustums:
                geoms.append(
                    make_camera_frustum(
                        gt_c2ws[i], intrinsics[i], image_width, image_height, color=[1.0, 0.6, 0.0]
                    )
                )

    if pred_c2ws is not None:
        pred_pts = camera_centers(pred_c2ws)
        
        # Add a tall thin marker at action mode camera initial position
        marker = o3d.geometry.TriangleMesh.create_box(width=0.05, height=100.0, depth=0.05)
        marker.compute_vertex_normals()
        marker.paint_uniform_color([0.9, 0.1, 0.1]) # Red marker
        marker.translate([pred_pts[0][0] - 0.025, pred_pts[0][1] - 20.0, pred_pts[0][2] - 0.025])
        geoms.append(marker)
        
        geoms.append(make_path_lines(pred_pts, [0.0, 0.9, 0.9]))
        n = min(len(pred_c2ws), len(intrinsics))
        for i in range(0, n, max(1, axis_stride)):
            geoms.append(make_camera_axes(pred_c2ws[i], 0.06))
            if show_frustums:
                geoms.append(
                    make_camera_frustum(
                        pred_c2ws[i], intrinsics[i], image_width, image_height, color=[0.1, 0.9, 0.9]
                    )
                )
    return geoms


def visualize_scene(geoms):
    import open3d as o3d

    o3d.visualization.draw_geometries(geoms)


def play_first_person(
    c2ws: np.ndarray,
    geoms,
    fps: float = 24.0,
):
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LingBot FPV", width=1280, height=720)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    
    for g in geoms:
        vis.add_geometry(g)

    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    dt = 1.0 / max(1.0, fps)
    next_tick = time.perf_counter()

    for i in range(len(c2ws)):
        pose = c2ws[i]
        r_wc = pose[:3, :3]
        t_wc = pose[:3, 3]
        r_cw = r_wc.T
        t_cw = -r_cw @ t_wc

        # Open3D camera extrinsic expects world->camera.
        ext = np.eye(4, dtype=np.float64)
        ext[:3, :3] = r_cw.astype(np.float64)
        ext[:3, 3] = t_cw.astype(np.float64)

        cam_params.extrinsic = ext
        ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        
        vis.poll_events()
        vis.update_renderer()
        next_tick += dt
        sleep_s = next_tick - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)

    while vis.poll_events():
        vis.update_renderer()
        time.sleep(0.01)
    vis.destroy_window()


def expand_latent_poses_to_video_frames(
    latent_c2ws: np.ndarray,
    num_video_frames: int,
    compression_ratio: int = 4,
) -> np.ndarray:
    """Expand latent poses [L,4,4] to video-frame poses [T,4,4] with interpolation."""
    if latent_c2ws.ndim != 3 or latent_c2ws.shape[1:] != (4, 4):
        raise ValueError(f"latent_c2ws must be [L,4,4], got {latent_c2ws.shape}")
    if num_video_frames <= 0:
        raise ValueError("num_video_frames must be positive.")
    if (num_video_frames - 1) % compression_ratio != 0:
        raise ValueError(
            f"num_video_frames must satisfy (T-1)%{compression_ratio}==0, got {num_video_frames}"
        )

    expected_latent = (num_video_frames - 1) // compression_ratio + 1
    if len(latent_c2ws) != expected_latent:
        raise ValueError(
            f"latent length mismatch: got {len(latent_c2ws)}, expected {expected_latent}"
        )

    def rot_to_quat(r: np.ndarray) -> np.ndarray:
        # Returns quaternion as [w, x, y, z]
        tr = float(np.trace(r))
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (r[2, 1] - r[1, 2]) / s
            y = (r[0, 2] - r[2, 0]) / s
            z = (r[1, 0] - r[0, 1]) / s
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
        q = np.array([w, x, y, z], dtype=np.float64)
        q /= np.linalg.norm(q) + 1e-12
        return q

    def quat_to_rot(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=np.float64,
        )

    def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        if dot > 0.9995:
            q = q0 + t * (q1 - q0)
            q /= np.linalg.norm(q) + 1e-12
            return q
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        s0 = np.sin(theta_0 - theta) / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return s0 * q0 + s1 * q1

    out = np.zeros((num_video_frames, 4, 4), dtype=np.float64)
    out[:, 3, 3] = 1.0
    for f in range(num_video_frames):
        if f == num_video_frames - 1:
            out[f] = latent_c2ws[-1]
            continue
        i = f // compression_ratio
        alpha = (f % compression_ratio) / float(compression_ratio)
        p0 = latent_c2ws[i]
        p1 = latent_c2ws[i + 1]

        t0 = p0[:3, 3]
        t1 = p1[:3, 3]
        t_interp = (1.0 - alpha) * t0 + alpha * t1

        q0 = rot_to_quat(p0[:3, :3])
        q1 = rot_to_quat(p1[:3, :3])
        q_interp = quat_slerp(q0, q1, alpha)
        r_interp = quat_to_rot(q_interp)

        out[f, :3, :3] = r_interp
        out[f, :3, 3] = t_interp
    return out.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize LingBot trajectory with paired poses+intrinsics, and optional "
            "overlay of action-reconstructed trajectory."
        )
    )
    parser.add_argument("--poses", type=str, default=None, help="Optional path to poses.npy")
    parser.add_argument(
        "--intrinsics", type=str, default=None, help="Path to intrinsics.npy (paired with poses)"
    )
    parser.add_argument("--action", type=str, default=None, help="Optional action npy dict: keyboard/mouse")
    parser.add_argument("--forward-speed", type=float, default=3.0)
    parser.add_argument("--axis-stride", type=int, default=20)
    parser.add_argument("--image-width", type=int, default=832)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Use only the first N frames from poses/intrinsics (and action if provided).",
    )
    parser.add_argument("--no-normalize-first", action="store_true")
    parser.add_argument("--fpv", type=str, default="none", choices=["none", "gt", "pred"])
    parser.add_argument("--fpv-fps", type=float, default=24.0)
    parser.add_argument("--show-frustums", action="store_true")
    parser.add_argument("--landmark-count", type=int, default=80)
    parser.add_argument(
        "--compression-ratio",
        type=int,
        default=4,
        help="Temporal compression ratio between video frames and latent action poses.",
    )
    args = parser.parse_args()

    gt_c2ws = None
    if args.poses is not None:
        gt_c2ws = np.asarray(load_npy(args.poses), dtype=np.float32)
        if gt_c2ws.ndim != 3 or gt_c2ws.shape[1:] != (4, 4):
            raise ValueError(f"poses must be [T,4,4], got {gt_c2ws.shape}")

    keyboard = None
    mouse = None
    pred_c2ws = None
    if args.action:
        keyboard, mouse = load_action_from_npy(args.action)
        print(f"[info] action keyboard={keyboard.shape}, mouse={mouse.shape}")

    if gt_c2ws is None and keyboard is None:
        raise ValueError("Provide at least one of --poses or --action.")

    lengths = []
    if gt_c2ws is not None:
        lengths.append(len(gt_c2ws))
    if keyboard is not None:
        lengths.append(len(keyboard))
    target_len = min(lengths)
    if args.num_frames is not None:
        if args.num_frames <= 0:
            raise ValueError("--num-frames must be positive.")
        target_len = min(target_len, args.num_frames)

    if gt_c2ws is not None:
        gt_c2ws = gt_c2ws[:target_len]
        print(f"[info] poses={gt_c2ws.shape}")
    if keyboard is not None and mouse is not None:
        keyboard = keyboard[:target_len]
        mouse = mouse[:target_len]
        pred_c2ws = action_to_c2ws(
            keyboard=keyboard,
            mouse=mouse,
            num_frames=target_len,
            forward_speed=args.forward_speed,
        )
        print(f"[info] reconstructed poses={pred_c2ws.shape}")

    intrinsics_path = args.intrinsics
    if intrinsics_path is None and args.poses is not None:
        cand = Path(args.poses).with_name("intrinsics.npy")
        if cand.exists():
            intrinsics_path = str(cand)
    if intrinsics_path is not None:
        intrinsics = np.asarray(load_npy(intrinsics_path), dtype=np.float32)
        if intrinsics.ndim != 2 or intrinsics.shape[1] != 4:
            raise ValueError(f"intrinsics must be [T,4], got {intrinsics.shape}")
        if len(intrinsics) < target_len:
            raise ValueError(
                f"intrinsics length {len(intrinsics)} is shorter than required {target_len}"
            )
        intrinsics = intrinsics[:target_len]
        print(f"[info] intrinsics={intrinsics.shape}")
    else:
        fx = float(args.image_width)
        fy = float(args.image_height)
        cx = float(args.image_width) / 2.0
        cy = float(args.image_height) / 2.0
        intrinsics = np.tile(
            np.array([[fx, fy, cx, cy]], dtype=np.float32), (target_len, 1)
        )
        print("[info] intrinsics not provided, using fallback pinhole intrinsics")

    if not args.no_normalize_first:
        if gt_c2ws is not None:
            gt_c2ws = normalize_to_first_pose(gt_c2ws)
        if pred_c2ws is not None:
            pred_c2ws = normalize_to_first_pose(pred_c2ws)

    if pred_c2ws is not None and gt_c2ws is not None:
        mean_err, max_err = compute_center_metrics(gt_c2ws, pred_c2ws)
        print(f"[metric] center L2 mean={mean_err:.6f}, max={max_err:.6f}")

    try:
        geoms = build_scene(
            gt_c2ws=gt_c2ws,
            pred_c2ws=pred_c2ws,
            intrinsics=intrinsics,
            axis_stride=max(1, args.axis_stride),
            image_width=args.image_width,
            image_height=args.image_height,
            show_frustums=args.show_frustums,
            landmark_count=max(0, args.landmark_count),
        )
        if args.fpv == "none":
            visualize_scene(geoms)
        elif args.fpv == "gt":
            if gt_c2ws is None:
                raise ValueError("--fpv gt requires --poses.")
            play_first_person(gt_c2ws, geoms, fps=args.fpv_fps)
        else:
            if pred_c2ws is None:
                raise ValueError("--fpv pred requires --action.")
            pred_fpv = pred_c2ws
            if len(pred_c2ws) != target_len:
                pred_fpv = expand_latent_poses_to_video_frames(
                    pred_c2ws,
                    num_video_frames=target_len,
                    compression_ratio=args.compression_ratio,
                )
            play_first_person(pred_fpv, geoms, fps=args.fpv_fps)
    except ImportError as exc:
        raise ImportError(
            "open3d is required. Install with: pip install open3d"
        ) from exc


if __name__ == "__main__":
    main()
