import numpy as np
import torch
from fastvideo.models.dits.lingbotworld.cam_utils import (
    SE3_inverse,
    compute_relative_poses,
    get_Ks_transformed,
    get_plucker_embeddings,
    interpolate_camera_poses,
)


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
    forward_speed: float = 0.08,  # from HYW
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


def process_custom_actions(
    num_frames: int,
    keyboard_cond: torch.Tensor,
    mouse_cond: torch.Tensor,
    *,
    latent_height: int,
    latent_width: int,
    height_org: int = 480,
    width_org: int = 832,
    spatial_scale: int = 8,
    intrinsics: torch.Tensor | None = None,
    forward_speed: float = 0.08,  # from HYW
    normalize_trans: bool = True,
) -> torch.Tensor:
    """
    Convert frame-level keyboard/mouse controls to camera condition.

    Returns:
        c2ws_plucker_emb with shape [B=1, C=6*spatial_scale^2, F_lat, H_lat, W_lat]
    """
    if keyboard_cond.ndim == 3:
        keyboard_cond = keyboard_cond.squeeze(0)
    if mouse_cond.ndim == 3:
        mouse_cond = mouse_cond.squeeze(0)
    assert keyboard_cond.ndim == 2 and mouse_cond.ndim == 2, (
        "keyboard_cond and mouse_cond must be [T, C]"
    )

    keyboard_cond, mouse_cond = _reformat_keyboard_and_mouse_cond(
        num_frames, keyboard_cond.float(), mouse_cond.float()
    )
    c2ws = _motions_to_c2ws(
        keyboard_cond=keyboard_cond,
        mouse_cond=mouse_cond,
        forward_speed=forward_speed,
    )
    c2ws = compute_relative_poses(c2ws, framewise=True, normalize_trans=normalize_trans)

    lat_f = c2ws.shape[0]
    h = latent_height * spatial_scale
    w = latent_width * spatial_scale

    if intrinsics is None:
        # Fallback pinhole intrinsics centered at image center.
        fx = float(width_org)
        fy = float(height_org)
        cx = float(width_org) / 2.0
        cy = float(height_org) / 2.0
        intrinsics = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)

    intrinsics = intrinsics.float()
    if intrinsics.ndim == 1:
        intrinsics = intrinsics.unsqueeze(0).repeat(lat_f, 1)
    assert intrinsics.shape == (lat_f, 4), (
        f"intrinsics must be [F_lat, 4], got {tuple(intrinsics.shape)}"
    )
    Ks = get_Ks_transformed(
        intrinsics,
        height_org=height_org,
        width_org=width_org,
        height_resize=h,
        width_resize=w,
        height_final=h,
        width_final=w,
    )
    plucker = get_plucker_embeddings(c2ws, Ks, h, w)  # [F, H, W, 6]

    c1 = h // latent_height
    c2 = w // latent_width
    assert c1 == spatial_scale and c2 == spatial_scale
    plucker = plucker.view(lat_f, latent_height, c1, latent_width, c2, 6)
    plucker = plucker.permute(0, 1, 3, 5, 2, 4).contiguous()
    plucker = plucker.view(lat_f, latent_height, latent_width, 6 * c1 * c2)
    c2ws_plucker_emb = plucker.permute(3, 0, 1, 2).contiguous().unsqueeze(0)
    return c2ws_plucker_emb