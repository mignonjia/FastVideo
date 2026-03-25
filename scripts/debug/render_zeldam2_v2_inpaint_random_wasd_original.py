#!/usr/bin/env python3
"""Render original zeldam2 clips with action overlays for target idxs.

This script reads zeldam2-v2 manifest files, finds samples whose original
action npy contains non-zero W/S/A/D activity, then selects random chunk/seg
examples for each idx with broad chunk-range coverage. It renders the original
clip segment from `video_source_path` with the original action overlay.

Outputs:
    /mnt/weka/home/hao.zhang/mhuo/FastVideo/scripts/decoded_latent_samples/
    zeldam2-v2/idx{idx}/
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import re
import subprocess
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_WM_LAB_ROOT = Path("/mnt/weka/home/hao.zhang/alex/wm-lab")
DEFAULT_MANIFEST_ROOT = (
    DEFAULT_WM_LAB_ROOT / "datas/datasets/zeldam2-v2/action_latent"
)
DEFAULT_IDX_FILE = (
    DEFAULT_WM_LAB_ROOT / "datas/spec/zeldam2/inpaint.txt"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/scripts/decoded_latent_samples/zeldam2-v2"
)

ID_RE = re.compile(r"^idx(?P<idx>\d+)_chunk(?P<chunk>.+)_seg(?P<seg>\d+)$")
CHUNK_NUM_RE = re.compile(r"_chunk_(\d+)$")
KEY_NAMES = ["W", "S", "A", "D", "left", "right"]
KEY_ICONS = {"W": "W", "A": "A", "S": "S", "D": "D", "left": "L", "right": "R"}


@dataclass(frozen=True)
class Candidate:
    idx: int
    sample_id: str
    chunk_id: str
    chunk_number: int
    segment_idx: int
    start_frame: int
    end_frame: int
    video_source_path: str
    action_segment_path: str
    manifest_path: str
    wasd_nonzero: int
    keyboard_nonzero: int
    mouse_nonzero: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render original zeldam2 clips with random WASD overlay samples."
    )
    parser.add_argument(
        "--wm-lab-root",
        type=Path,
        default=DEFAULT_WM_LAB_ROOT,
        help="Base root for wm-lab paths referenced by the manifests.",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=DEFAULT_MANIFEST_ROOT,
        help="Root directory containing zeldam2-v2 manifests.",
    )
    parser.add_argument(
        "--idx-file",
        type=Path,
        default=DEFAULT_IDX_FILE,
        help="Input txt file listing idxs to inspect.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root. Results go to output_root/idx{idx}/.",
    )
    parser.add_argument(
        "--samples-per-idx",
        type=int,
        default=5,
        help="How many samples to render per idx.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel render workers. GPU is not required here.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260322,
        help="Random seed for candidate selection.",
    )
    parser.add_argument(
        "--only-idxs",
        type=str,
        default=None,
        help="Optional comma-separated subset of idxs to process.",
    )
    parser.add_argument(
        "--idx-limit",
        type=int,
        default=None,
        help="Optional limit after filtering.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rendering if the expected output file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build selection metadata; do not render videos.",
    )
    return parser.parse_args()


def parse_idx_file(idx_file: Path) -> list[int]:
    values: list[int] = []
    for raw_line in idx_file.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.split(","):
            token = token.strip()
            if token:
                values.append(int(token))
    return values


def parse_only_idxs(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(token.strip()) for token in value.split(",") if token.strip()}


def parse_sample_id(sample_id: str) -> tuple[int, str, int]:
    match = ID_RE.match(sample_id)
    if not match:
        raise ValueError(f"Unexpected sample id format: {sample_id}")
    return (
        int(match.group("idx")),
        match.group("chunk"),
        int(match.group("seg")),
    )


def parse_chunk_number(chunk_id: str) -> int:
    match = CHUNK_NUM_RE.search(chunk_id)
    if not match:
        raise ValueError(f"Could not parse chunk number from {chunk_id}")
    return int(match.group(1))


def resolve_path(root: Path, manifest_path: str) -> Path:
    path = Path(manifest_path)
    if path.is_absolute():
        return path
    return root / path


def expected_output_path(output_dir: Path, candidate: Candidate) -> Path:
    return output_dir / f"{candidate.sample_id}_original_overlay.mp4"


def scan_candidates(
    manifest_root: Path,
    wm_lab_root: Path,
    target_idxs: set[int],
) -> dict[int, list[Candidate]]:
    import numpy as np

    candidates: dict[int, list[Candidate]] = defaultdict(list)
    manifest_paths = sorted(manifest_root.rglob("manifest.rank*.jsonl"))
    total = len(manifest_paths)
    for manifest_index, manifest_path in enumerate(manifest_paths, start=1):
        with manifest_path.open() as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                idx = int(record["idx"])
                if idx not in target_idxs:
                    continue

                action_path = resolve_path(wm_lab_root, record["action_segment_path"])
                if not action_path.is_file():
                    continue

                action = np.load(action_path, allow_pickle=True).item()
                keyboard = np.asarray(action["keyboard"], dtype=np.float32)
                mouse = np.asarray(action.get("mouse"), dtype=np.float32)

                wasd_width = min(4, keyboard.shape[1])
                wasd_nonzero = int((keyboard[:, :wasd_width] != 0).sum())
                if wasd_nonzero <= 0:
                    continue

                sample_id = record["id"]
                _, chunk_id, segment_idx = parse_sample_id(sample_id)
                candidates[idx].append(
                    Candidate(
                        idx=idx,
                        sample_id=sample_id,
                        chunk_id=chunk_id,
                        chunk_number=parse_chunk_number(chunk_id),
                        segment_idx=int(record["segment_idx"]),
                        start_frame=int(record["start_frame"]),
                        end_frame=int(record["end_frame"]),
                        video_source_path=record["video_source_path"],
                        action_segment_path=record["action_segment_path"],
                        manifest_path=str(manifest_path),
                        wasd_nonzero=wasd_nonzero,
                        keyboard_nonzero=int((keyboard != 0).sum()),
                        mouse_nonzero=int((mouse != 0).sum()) if mouse is not None else 0,
                    )
                )

        if manifest_index % 50 == 0 or manifest_index == total:
            print(
                f"[scan] processed {manifest_index}/{total} manifest files",
                flush=True,
            )
    return candidates


def pick_diverse_random_samples(
    idx_order: list[int],
    candidates_by_idx: dict[int, list[Candidate]],
    samples_per_idx: int,
    seed: int,
) -> dict[int, list[Candidate]]:
    selected: dict[int, list[Candidate]] = {}
    for idx in idx_order:
        candidates = sorted(
            candidates_by_idx.get(idx, []),
            key=lambda item: (
                item.chunk_number,
                item.segment_idx,
                item.sample_id,
            ),
        )
        rng = random.Random(seed + idx)
        by_chunk: dict[int, list[Candidate]] = defaultdict(list)
        for item in candidates:
            by_chunk[item.chunk_number].append(item)

        chunk_numbers = sorted(by_chunk.keys())
        picks: list[Candidate] = []
        used_ids: set[str] = set()

        if chunk_numbers:
            n_chunks = len(chunk_numbers)
            n_bins = min(samples_per_idx, n_chunks)
            for bin_index in range(n_bins):
                start = round(bin_index * n_chunks / n_bins)
                end = round((bin_index + 1) * n_chunks / n_bins)
                bucket = chunk_numbers[start:end]
                if not bucket:
                    continue
                chosen_chunk = rng.choice(bucket)
                chosen_candidate = rng.choice(by_chunk[chosen_chunk])
                picks.append(chosen_candidate)
                used_ids.add(chosen_candidate.sample_id)

        if len(picks) < samples_per_idx:
            remaining = [c for c in candidates if c.sample_id not in used_ids]
            rng.shuffle(remaining)
            picks.extend(remaining[: samples_per_idx - len(picks)])

        selected[idx] = picks
    return selected


def write_selection_summary(
    output_root: Path,
    idx_order: list[int],
    all_candidates: dict[int, list[Candidate]],
    selected: dict[int, list[Candidate]],
) -> None:
    payload = {
        "generated_at_unix_s": time.time(),
        "idxs": [],
    }
    for idx in idx_order:
        payload["idxs"].append(
            {
                "idx": idx,
                "num_candidates": len(all_candidates.get(idx, [])),
                "selected": [asdict(item) for item in selected.get(idx, [])],
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "original_overlay_selection_summary.json").write_text(
        json.dumps(payload, indent=2)
    )


def write_render_summary(output_root: Path, results: list[dict[str, object]]) -> None:
    payload = {
        "generated_at_unix_s": time.time(),
        "num_results": len(results),
        "results": results,
    }
    (output_root / "original_overlay_render_results.json").write_text(
        json.dumps(payload, indent=2)
    )


def build_jobs(
    selected: dict[int, list[Candidate]],
    output_root: Path,
    skip_existing: bool,
) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for idx, candidates in selected.items():
        output_dir = output_root / f"idx{idx}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for candidate in candidates:
            output_path = expected_output_path(output_dir, candidate)
            if skip_existing and output_path.is_file():
                continue
            jobs.append(
                {
                    "idx": idx,
                    "candidate": asdict(candidate),
                    "output_path": str(output_path),
                }
            )
    return jobs


def render_worker(task: dict[str, object], wm_lab_root: str) -> dict[str, object]:
    import cv2
    import numpy as np

    started_at = time.time()
    candidate = task["candidate"]
    output_path = Path(task["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix(".log")

    def draw_rounded_rectangle(image, top_left, bottom_right, color, radius=10, alpha=0.5):
        overlay = image.copy()
        x1, y1 = top_left
        x2, y2 = bottom_right
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.ellipse(overlay, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1)
        cv2.ellipse(overlay, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1)
        cv2.ellipse(overlay, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
        cv2.ellipse(overlay, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def draw_keys_on_frame(frame, keys_pressed, key_size=(30, 30), top_margin=15):
        left_margin = 15
        gap = 3
        key_positions = {
            "W": (left_margin + key_size[0] + gap, top_margin),
            "A": (left_margin, top_margin + key_size[1] + gap),
            "S": (left_margin + key_size[0] + gap, top_margin + key_size[1] + gap),
            "D": (left_margin + (key_size[0] + gap) * 2, top_margin + key_size[1] + gap),
            "left": (left_margin + (key_size[0] + gap) * 3 + 10, top_margin + key_size[1] + gap),
            "right": (left_margin + (key_size[0] + gap) * 4 + 15, top_margin + key_size[1] + gap),
        }
        for key, (x, y) in key_positions.items():
            is_pressed = keys_pressed.get(key, False)
            color = (0, 255, 0) if is_pressed else (200, 200, 200)
            alpha = 0.8 if is_pressed else 0.5
            draw_rounded_rectangle(
                frame,
                (x, y),
                (x + key_size[0], y + key_size[1]),
                color,
                radius=5,
                alpha=alpha,
            )
            icon = KEY_ICONS[key]
            text_size = cv2.getTextSize(icon, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x + (key_size[0] - text_size[0]) // 2
            text_y = y + (key_size[1] + text_size[1]) // 2
            cv2.putText(
                frame,
                icon,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

    def draw_mouse_on_frame(frame, pitch, yaw, top_margin=15):
        height, width = frame.shape[:2]
        right_margin = 15
        radius = 25
        cx = width - right_margin - radius
        cy = top_margin + radius
        dx = int(yaw * radius * 8)
        dy = int(-pitch * radius * 8)
        max_arrow = radius - 5
        dx = max(-max_arrow, min(max_arrow, dx))
        dy = max(-max_arrow, min(max_arrow, dy))
        cv2.circle(frame, (cx, cy), radius, (50, 50, 50), -1)
        cv2.circle(frame, (cx, cy), radius, (200, 200, 200), 1)
        cv2.line(frame, (cx - radius + 5, cy), (cx + radius - 5, cy), (100, 100, 100), 1)
        cv2.line(frame, (cx, cy - radius + 5), (cx, cy + radius - 5), (100, 100, 100), 1)
        if abs(dx) > 1 or abs(dy) > 1:
            cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy), (0, 255, 0), 2, tipLength=0.3)

    try:
        root = Path(wm_lab_root)
        video_path = resolve_path(root, candidate["video_source_path"])
        action_path = resolve_path(root, candidate["action_segment_path"])
        temp_output_path = output_path.with_name(
            output_path.stem + "_tmp_mp4v.mp4"
        )

        action = np.load(action_path, allow_pickle=True).item()
        keyboard = np.asarray(action["keyboard"], dtype=np.float32)
        mouse = np.asarray(action.get("mouse"), dtype=np.float32)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = int(candidate["start_frame"])
        end_frame = int(candidate["end_frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open writer: {temp_output_path}")

        action_len = min(len(keyboard), len(mouse)) if mouse is not None else len(keyboard)
        expected_frames = max(0, end_frame - start_frame + 1)
        frame_budget = min(expected_frames, action_len)

        frame_idx = 0
        while frame_idx < frame_budget:
            ok, frame = cap.read()
            if not ok:
                break
            frame = frame.copy()
            keyboard_vec = keyboard[frame_idx]
            keys_pressed = {
                name: bool(keyboard_vec[i] > 0.5)
                for i, name in enumerate(KEY_NAMES[: len(keyboard_vec)])
            }
            draw_keys_on_frame(frame, keys_pressed)
            if mouse is not None and len(mouse[frame_idx]) >= 2:
                draw_mouse_on_frame(
                    frame,
                    float(mouse[frame_idx][0]),
                    float(mouse[frame_idx][1]),
                )
            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_output_path),
            "-c:v",
            "libopenh264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        ffmpeg_result = subprocess.run(
            ffmpeg_cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if ffmpeg_result.returncode != 0:
            raise RuntimeError(
                "ffmpeg transcode failed:\n"
                f"stdout:\n{ffmpeg_result.stdout}\n"
                f"stderr:\n{ffmpeg_result.stderr}"
            )
        temp_output_path.unlink(missing_ok=True)
        result = {
            "ok": True,
            "idx": candidate["idx"],
            "sample_id": candidate["sample_id"],
            "chunk_number": candidate["chunk_number"],
            "segment_idx": candidate["segment_idx"],
            "start_frame": candidate["start_frame"],
            "end_frame": candidate["end_frame"],
            "frames_written": frame_idx,
            "output_path": str(output_path),
            "elapsed_s": round(time.time() - started_at, 3),
        }
        log_path.write_text(json.dumps(result, indent=2))
        return result
    except Exception:
        error_text = traceback.format_exc()
        log_path.write_text(error_text)
        try:
            temp_output_path.unlink(missing_ok=True)
        except Exception:
            pass
        return {
            "ok": False,
            "idx": candidate["idx"],
            "sample_id": candidate["sample_id"],
            "chunk_number": candidate["chunk_number"],
            "segment_idx": candidate["segment_idx"],
            "output_path": str(output_path),
            "elapsed_s": round(time.time() - started_at, 3),
            "error": error_text,
        }


def main() -> int:
    args = parse_args()
    idx_order = parse_idx_file(args.idx_file)
    only_idxs = parse_only_idxs(args.only_idxs)
    if only_idxs is not None:
        idx_order = [idx for idx in idx_order if idx in only_idxs]
    if args.idx_limit is not None:
        idx_order = idx_order[: args.idx_limit]
    if not idx_order:
        raise ValueError("No idxs selected")

    target_idxs = set(idx_order)
    print(
        f"[setup] idxs={len(idx_order)} samples_per_idx={args.samples_per_idx} "
        f"workers={args.num_workers}",
        flush=True,
    )
    print(f"[setup] scanning manifests under {args.manifest_root}", flush=True)

    all_candidates = scan_candidates(
        args.manifest_root,
        args.wm_lab_root,
        target_idxs,
    )
    selected = pick_diverse_random_samples(
        idx_order,
        all_candidates,
        args.samples_per_idx,
        args.seed,
    )
    write_selection_summary(args.output_root, idx_order, all_candidates, selected)

    for idx in idx_order:
        picked = selected.get(idx, [])
        chunk_numbers = [item.chunk_number for item in picked]
        print(
            f"[select] idx={idx} candidates={len(all_candidates.get(idx, []))} "
            f"picked_chunks={chunk_numbers}",
            flush=True,
        )

    jobs = build_jobs(selected, args.output_root, args.skip_existing)
    print(f"[jobs] queued {len(jobs)} render jobs", flush=True)
    if args.dry_run or not jobs:
        return 0

    worker_count = max(1, min(args.num_workers, len(jobs)))
    results: list[dict[str, object]] = []
    with mp.get_context("spawn").Pool(processes=worker_count) as pool:
        async_results = [
            pool.apply_async(render_worker, (job, str(args.wm_lab_root)))
            for job in jobs
        ]
        for job_index, async_result in enumerate(async_results, start=1):
            result = async_result.get()
            results.append(result)
            status = "ok" if result["ok"] else "fail"
            print(
                f"[render] {job_index}/{len(jobs)} {status} "
                f"idx={result['idx']} sample={result['sample_id']} "
                f"chunk={result['chunk_number']} seg={result['segment_idx']}",
                flush=True,
            )

    write_render_summary(args.output_root, results)
    failures = sum(1 for item in results if not item["ok"])
    print(f"[done] finished {len(results)} jobs, failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
