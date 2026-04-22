from __future__ import annotations

import argparse
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
    btn = df[btn_cols].to_numpy(dtype=np.float32) if btn_cols else np.zeros((len(df), 0), dtype=np.float32)
    return j_left, j_right, btn_cols, btn, int(len(df))


def _stick_scale(arr, start: int, end: int) -> float:
    hi = 0.0
    for i in range(int(start), int(end) + 1):
        x = float(arr[i][0])
        y = float(arr[i][1])
        hi = max(hi, math.hypot(x, y))
    return hi if hi > 1e-6 else 1.0


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


def _draw_alpha_box(img, *, xy, wh, color, alpha: float, radius: int = 18):
    cv2, np, _ = _imports()
    x, y = xy
    w, h = wh
    x = max(0, int(x))
    y = max(0, int(y))
    w = max(1, int(w))
    h = max(1, int(h))
    x2 = min(img.shape[1], x + w)
    y2 = min(img.shape[0], y + h)
    if x2 <= x or y2 <= y:
        return

    overlay = img.copy()
    rr = max(0, min(int(radius), (x2 - x) // 2, (y2 - y) // 2))
    cv2.rectangle(overlay, (x + rr, y), (x2 - rr, y2), color, -1)
    cv2.rectangle(overlay, (x, y + rr), (x2, y2 - rr), color, -1)
    if rr > 0:
        cv2.circle(overlay, (x + rr, y + rr), rr, color, -1)
        cv2.circle(overlay, (x2 - rr, y + rr), rr, color, -1)
        cv2.circle(overlay, (x + rr, y2 - rr), rr, color, -1)
        cv2.circle(overlay, (x2 - rr, y2 - rr), rr, color, -1)

    cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0.0, dst=img)


def _draw_stick_overlay(img, *, center, radius, xy, title, value_text):
    cv2, _, _ = _imports()
    cx, cy = center
    fg = (240, 240, 240)
    grid = (145, 145, 145)
    accent = (80, 210, 255)
    dot_outline = (255, 255, 255)

    cv2.circle(img, (cx, cy), radius, grid, 2, cv2.LINE_AA)
    cv2.line(img, (cx - radius, cy), (cx + radius, cy), grid, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - radius), (cx, cy + radius), grid, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 3, fg, -1, cv2.LINE_AA)

    x = max(-1.0, min(1.0, float(xy[0])))
    y = max(-1.0, min(1.0, float(xy[1])))
    travel = 0.56 * radius
    px = int(round(cx + x * travel))
    py = int(round(cy - y * travel))
    dot_r = max(10, radius // 6)
    cv2.circle(img, (px, py), dot_r, accent, -1, cv2.LINE_AA)
    cv2.circle(img, (px, py), dot_r, dot_outline, 2, cv2.LINE_AA)

    cv2.putText(img, title, (cx - radius, cy - radius - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.56, fg, 2, cv2.LINE_AA)
    cv2.putText(img, value_text, (cx - radius, cy + radius + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, fg, 1, cv2.LINE_AA)


def _draw_buttons_overlay(img, *, origin, size, btn_cols, btn_row):
    cv2, _, _ = _imports()
    x0, y0 = origin
    w, h = size
    fg = (240, 240, 240)
    off = (55, 55, 55)
    on = (80, 210, 255)
    border = (110, 110, 110)

    cv2.putText(img, "Buttons", (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.62, fg, 2, cv2.LINE_AA)
    if not btn_cols:
        return

    cols = 4
    rows = (len(btn_cols) + cols - 1) // cols
    gap_x = 10
    gap_y = 8
    cell_w = max(36, (w - gap_x * (cols - 1)) // cols)
    cell_h = max(24, (h - gap_y * (rows - 1)) // rows)

    for idx, name in enumerate(btn_cols):
        row = idx // cols
        col = idx % cols
        x = x0 + col * (cell_w + gap_x)
        y = y0 + row * (cell_h + gap_y)
        try:
            active = float(btn_row[idx]) > 0.0
        except Exception:
            active = False
        fill = on if active else off
        cv2.rectangle(img, (x, y), (x + cell_w, y + cell_h), fill, -1)
        cv2.rectangle(img, (x, y), (x + cell_w, y + cell_h), border, 1)
        label = _short_label(str(name))
        label_scale = 0.42 if len(label) <= 4 else 0.34
        cv2.putText(
            img,
            label,
            (x + 7, y + int(cell_h * 0.63)),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            fg,
            1,
            cv2.LINE_AA,
        )


def render(
    *,
    video_path: Path,
    parquet_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
):
    cv2, _, _ = _imports()
    j_left, j_right, btn_cols, btn, n_actions = _load_actions(parquet_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    n_total = min(n_actions, n_video) if n_video > 0 else n_actions
    if n_total <= 0:
        raise RuntimeError("no frames to render")

    start = max(0, int(start_frame))
    end = n_total - 1 if int(end_frame) < 0 else min(int(end_frame), n_total - 1)
    if end < start:
        raise ValueError(f"invalid frame range: start={start}, end={end}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {output_path}")

    left_scale = _stick_scale(j_left, start, end)
    right_scale = _stick_scale(j_right, start, end)

    radius = max(48, min(width, height) // 15)
    sticks_box_w = radius * 4 + 90
    sticks_box_h = radius * 2 + 92
    sticks_box_x = 20
    sticks_box_y = height - sticks_box_h - 92

    left_center = (sticks_box_x + radius + 26, sticks_box_y + radius + 28)
    right_center = (sticks_box_x + radius * 3 + 64, sticks_box_y + radius + 28)

    buttons_box_w = min(560, max(420, width // 3))
    buttons_box_h = min(270, max(190, height // 4))
    buttons_box_x = width - buttons_box_w - 20
    buttons_box_y = height - buttons_box_h - 20

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
    frame_idx = start
    while frame_idx <= end:
        ok, frame = cap.read()
        if not ok:
            break

        _draw_alpha_box(frame, xy=(sticks_box_x, sticks_box_y), wh=(sticks_box_w, sticks_box_h), color=(18, 18, 18), alpha=0.68)
        _draw_alpha_box(frame, xy=(buttons_box_x, buttons_box_y), wh=(buttons_box_w, buttons_box_h), color=(18, 18, 18), alpha=0.68)
        _draw_alpha_box(frame, xy=(20, 18), wh=(460, 66), color=(18, 18, 18), alpha=0.60, radius=14)

        jl = j_left[frame_idx]
        jr = j_right[frame_idx]
        t = frame_idx / fps

        header = f"Processed controls  frame={frame_idx}  t={t:0.3f}s"
        scale_line = f"display norm  left/{left_scale:.3f} (Y flipped)  right/{right_scale:.3f}"
        cv2.putText(frame, header, (34, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(frame, scale_line, (34, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

        _draw_stick_overlay(
            frame,
            center=left_center,
            radius=radius,
            xy=(float(jl[0]) / left_scale, -float(jl[1]) / left_scale),
            title="Left Stick (Y flipped)",
            value_text=f"x={float(jl[0]):+.3f}  y={float(jl[1]):+.3f}  m={math.hypot(float(jl[0]), float(jl[1])):.3f}",
        )
        _draw_stick_overlay(
            frame,
            center=right_center,
            radius=radius,
            xy=(float(jr[0]) / right_scale, float(jr[1]) / right_scale),
            title="Right Stick",
            value_text=f"x={float(jr[0]):+.3f}  y={float(jr[1]):+.3f}  m={math.hypot(float(jr[0]), float(jr[1])):.3f}",
        )

        btn_row = btn[frame_idx] if frame_idx < len(btn) else []
        _draw_buttons_overlay(
            frame,
            origin=(buttons_box_x + 16, buttons_box_y + 30),
            size=(buttons_box_w - 32, buttons_box_h - 46),
            btn_cols=btn_cols,
            btn_row=btn_row,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def main():
    ap = argparse.ArgumentParser(description="Render a clip with compact processed controller overlays.")
    ap.add_argument("--video", required=True, help="Path to source clip.mp4")
    ap.add_argument("--parquet", required=True, help="Path to actions parquet")
    ap.add_argument("--output", required=True, help="Path to output .mp4")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=-1, help="Inclusive; -1 means last available frame")
    args = ap.parse_args()

    render(
        video_path=Path(args.video),
        parquet_path=Path(args.parquet),
        output_path=Path(args.output),
        start_frame=int(args.start_frame),
        end_frame=int(args.end_frame),
    )


if __name__ == "__main__":
    main()
