from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _imports():
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    return cv2, np, pd


def _load_actions(parquet_path: Path):
    _, np, pd = _imports()
    df = pd.read_parquet(parquet_path)
    j_left = np.array(df["j_left"].tolist(), dtype=np.float32)
    j_right = np.array(df["j_right"].tolist(), dtype=np.float32)
    btn_cols = [c for c in df.columns if c not in ("j_left", "j_right")]
    btn = (
        df[btn_cols].to_numpy(dtype=np.float32)
        if btn_cols
        else np.zeros((len(df), 0), dtype=np.float32)
    )
    return j_left, j_right, btn_cols, btn, int(len(df))


def _dir_to_xy(name: str) -> tuple[float, float]:
    mapping = {
        "neutral": (0.0, 0.0),
        "up": (0.0, 1.0),
        "down": (0.0, -1.0),
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
        "up_left": (-0.70710678, 0.70710678),
        "up_right": (0.70710678, 0.70710678),
        "down_left": (-0.70710678, -0.70710678),
        "down_right": (0.70710678, -0.70710678),
    }
    return mapping.get(str(name or "neutral"), (0.0, 0.0))


def _load_processed_json(action_json_path: Path):
    _, np, _ = _imports()
    data = json.loads(action_json_path.read_text())
    if not isinstance(data, dict):
        raise RuntimeError(f"expected dict action json: {action_json_path}")

    max_idx = max(int(k) for k in data.keys()) if data else -1
    n = max_idx + 1
    j_left = np.zeros((n, 2), dtype=np.float32)
    j_right = np.zeros((n, 2), dtype=np.float32)

    button_names = sorted(
        {
            str(btn)
            for rec in data.values()
            if isinstance(rec, dict)
            for btn in rec.get("buttons", [])
        }
    )
    btn = np.zeros((n, len(button_names)), dtype=np.float32)
    btn_index = {name: i for i, name in enumerate(button_names)}

    for key, rec in data.items():
        if not isinstance(rec, dict):
            continue
        idx = int(key)
        j_left[idx] = _dir_to_xy(rec.get("left_joystick"))
        j_right[idx] = _dir_to_xy(rec.get("right_joystick"))
        for name in rec.get("buttons", []):
            col = btn_index.get(str(name))
            if col is not None:
                btn[idx, col] = 1.0

    return j_left, j_right, button_names, btn, n


def _load_segment_npy(segment_path: Path):
    _, np, _ = _imports()
    arr = np.load(segment_path, allow_pickle=True)
    obj = arr.item() if getattr(arr, "dtype", None) == object else arr
    if not isinstance(obj, dict):
        raise RuntimeError(f"expected dict segment npy: {segment_path}")
    keyboard = np.asarray(obj["keyboard"], dtype=np.float32)
    mouse = np.asarray(obj["mouse"], dtype=np.float32)
    if keyboard.ndim != 2 or keyboard.shape[1] < 4:
        raise RuntimeError(f"invalid keyboard shape in {segment_path}: {keyboard.shape}")
    if mouse.ndim != 2 or mouse.shape[1] != 2:
        raise RuntimeError(f"invalid mouse shape in {segment_path}: {mouse.shape}")
    movement = np.stack(
        [keyboard[:, 3] - keyboard[:, 2], keyboard[:, 0] - keyboard[:, 1]],
        axis=1,
    ).astype(np.float32)
    camera = np.stack([mouse[:, 1], -mouse[:, 0]], axis=1).astype(np.float32)
    return keyboard, mouse, movement, camera, int(len(keyboard))


def _stick_scale(arr, start: int, end: int) -> float:
    hi = 0.0
    for i in range(int(start), int(end) + 1):
        x = float(arr[i][0])
        y = float(arr[i][1])
        hi = max(hi, math.hypot(x, y))
    target_extent = 0.95
    return hi / target_extent if hi > 1e-6 else 1.0


def _short_label(name: str) -> str:
    mapping = {
        "back": "BACK",
        "start": "START",
        "guide": "HOME",
        "south": "S",
        "east": "E",
        "west": "W",
        "north": "N",
        "left_shoulder": "L1",
        "right_shoulder": "R1",
        "left_trigger": "L2",
        "right_trigger": "R2",
        "left_thumb": "L3",
        "right_thumb": "R3",
        "dpad_up": "DU",
        "dpad_down": "DD",
        "dpad_left": "DL",
        "dpad_right": "DR",
    }
    return mapping.get(name, name.upper())


def _draw_circle_pad(img, *, center, radius, xy, title, value_text):
    cv2, _, _ = _imports()
    cx, cy = center
    fg = (240, 240, 240)
    grid = (120, 130, 138)
    accent = (80, 210, 255)

    cv2.circle(img, (cx, cy), radius, grid, 2, cv2.LINE_AA)
    cv2.line(img, (cx - radius, cy), (cx + radius, cy), grid, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - radius), (cx, cy + radius), grid, 1, cv2.LINE_AA)

    x = max(-1.0, min(1.0, float(xy[0])))
    y = max(-1.0, min(1.0, float(xy[1])))
    travel = 0.70 * radius
    px = int(round(cx + x * travel))
    py = int(round(cy - y * travel))
    cv2.circle(img, (px, py), 12, accent, -1, cv2.LINE_AA)
    cv2.circle(img, (px, py), 12, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(
        img,
        title,
        (cx - radius, cy - radius - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        fg,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        value_text,
        (cx - radius, cy + radius + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        fg,
        1,
        cv2.LINE_AA,
    )


def _draw_buttons(img, *, origin, size, btn_cols, btn_row):
    cv2, _, _ = _imports()
    x0, y0 = origin
    w, h = size
    fg = (240, 240, 240)
    off = (44, 52, 58)
    on = (80, 210, 255)
    border = (118, 126, 132)

    cv2.putText(
        img,
        "Buttons",
        (x0, y0 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        fg,
        2,
        cv2.LINE_AA,
    )
    if not btn_cols:
        return

    cols = 4
    rows = (len(btn_cols) + cols - 1) // cols
    gap_x = 10
    gap_y = 10
    cell_w = max(42, (w - gap_x * (cols - 1)) // cols)
    cell_h = max(28, (h - gap_y * (rows - 1)) // rows)

    for idx, name in enumerate(btn_cols):
        row = idx // cols
        col = idx % cols
        x = x0 + col * (cell_w + gap_x)
        y = y0 + row * (cell_h + gap_y)
        active = False
        try:
            active = float(btn_row[idx]) > 0.0
        except Exception:
            active = False
        fill = on if active else off
        cv2.rectangle(img, (x, y), (x + cell_w, y + cell_h), fill, -1)
        cv2.rectangle(img, (x, y), (x + cell_w, y + cell_h), border, 1)
        cv2.putText(
            img,
            _short_label(str(name)),
            (x + 8, y + int(cell_h * 0.65)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            fg,
            1,
            cv2.LINE_AA,
        )


def render(
    *,
    video_path: Path,
    action_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
):
    cv2, np, _ = _imports()
    mode = "sticks"
    if action_path.suffix.lower() == ".parquet":
        j_left, j_right, btn_cols, btn, n_actions = _load_actions(action_path)
        action_fps = None
        action_label = "Raw actions"
    elif action_path.suffix.lower() == ".json":
        j_left, j_right, btn_cols, btn, n_actions = _load_processed_json(action_path)
        action_fps = 30.0
        action_label = "Processed actions"
    elif action_path.suffix.lower() == ".npy":
        keyboard, mouse, j_left, j_right, n_actions = _load_segment_npy(action_path)
        btn_cols = []
        btn = np.zeros((n_actions, 0), dtype=np.float32)
        action_fps = 30.0
        action_label = "Processed segment"
        mode = "wangame"
    else:
        raise RuntimeError(f"unsupported action format: {action_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if action_fps is None:
        n_total = min(n_actions, n_video) if n_video > 0 else n_actions
    else:
        n_total = n_video if n_video > 0 else int(round(n_actions * fps / action_fps))
    if n_total <= 0:
        raise RuntimeError("no frames to render")

    start = max(0, int(start_frame))
    end = n_total - 1 if int(end_frame) < 0 else min(int(end_frame), n_total - 1)
    if end < start:
        raise ValueError(f"invalid range: {start}-{end}")

    panel_w = 520
    out_w = width + panel_w
    out_h = height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {output_path}")

    if action_fps is None:
        left_scale = _stick_scale(j_left, start, end)
        right_scale = _stick_scale(j_right, start, end)
    else:
        left_scale = _stick_scale(j_left, 0, n_actions - 1)
        right_scale = _stick_scale(j_right, 0, n_actions - 1)

    panel_bg = np.zeros((out_h, panel_w, 3), dtype=np.uint8)
    panel_bg[:] = (18, 24, 28)
    left_center = (panel_w // 2, 170)
    right_center = (panel_w // 2, 395)
    stick_radius = 88

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
    frame_idx = start
    while frame_idx <= end:
        ok, frame = cap.read()
        if not ok:
            break

        if action_fps is None:
            ai = min(frame_idx, n_actions - 1)
        else:
            ai = min(int(round(frame_idx * action_fps / fps)), n_actions - 1)
        jl = j_left[ai]
        jr = j_right[ai]
        btn_row = btn[ai] if ai < len(btn) else []

        panel = panel_bg.copy()
        cv2.putText(
            panel,
            f"{action_label}  frame={frame_idx}  t={frame_idx / fps:0.3f}s",
            (22, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"scale left/{left_scale:.3f}  right/{right_scale:.3f}",
            (22, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (200, 210, 214),
            1,
            cv2.LINE_AA,
        )

        _draw_circle_pad(
            panel,
            center=left_center,
            radius=stick_radius,
            xy=(float(jl[0]) / left_scale, -float(jl[1]) / left_scale),
            title="Left Stick" if mode == "sticks" else "Movement",
            value_text=(
                f"x={float(jl[0]):+.3f}  y={float(jl[1]):+.3f}  "
                f"m={math.hypot(float(jl[0]), float(jl[1])):.3f}"
                if mode == "sticks"
                else (
                    f"W={float(keyboard[ai][0]):.0f} S={float(keyboard[ai][1]):.0f} "
                    f"A={float(keyboard[ai][2]):.0f} D={float(keyboard[ai][3]):.0f}"
                )
            ),
        )
        _draw_circle_pad(
            panel,
            center=right_center,
            radius=stick_radius,
            xy=(float(jr[0]) / right_scale, float(jr[1]) / right_scale),
            title="Right Stick" if mode == "sticks" else "Camera",
            value_text=(
                f"x={float(jr[0]):+.3f}  y={float(jr[1]):+.3f}  "
                f"m={math.hypot(float(jr[0]), float(jr[1])):.3f}"
                if mode == "sticks"
                else (
                    f"pitch={float(mouse[ai][0]):+.3f} "
                    f"yaw={float(mouse[ai][1]):+.3f}"
                )
            ),
        )
        if btn_cols:
            _draw_buttons(
                panel,
                origin=(34, 520),
                size=(panel_w - 68, out_h - 560),
                btn_cols=btn_cols,
                btn_row=btn_row,
            )

        writer.write(np.hstack([frame, panel]))
        frame_idx += 1

    cap.release()
    writer.release()


def main():
    ap = argparse.ArgumentParser(
        description="Render original video with action panel on the right."
    )
    ap.add_argument("--video", required=True)
    ap.add_argument("--action", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=-1)
    args = ap.parse_args()

    render(
        video_path=Path(args.video),
        action_path=Path(args.action),
        output_path=Path(args.output),
        start_frame=int(args.start_frame),
        end_frame=int(args.end_frame),
    )


if __name__ == "__main__":
    main()
