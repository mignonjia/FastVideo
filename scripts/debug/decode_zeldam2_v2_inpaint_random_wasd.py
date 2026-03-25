#!/usr/bin/env python3
"""Batch decode random zeldam2-v2 WASD-active samples for target idxs.

For each idx listed in the input txt file, this script:
1. Scans zeldam2-v2 action latents for samples whose keyboard action contains
   at least one non-zero W/S/A/D entry.
2. Randomly selects N samples across the full idx candidate pool, preferring
   distinct chunks first so coverage is spread across the video.
3. Decodes the raw VAE latents with one persistent Wan VAE worker per GPU.
4. Writes overlaid mp4s to:
   scripts/decoded_latent_samples/zeldam2-v2/{idx}/
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import re
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path

import pyarrow.parquet as pq


DEFAULT_DATASET_ROOT = Path(
    "/mnt/weka/home/hao.zhang/alex/wm-lab/datas/datasets/zeldam2-v2/action_latent"
)
DEFAULT_IDX_FILE = Path(
    "/mnt/weka/home/hao.zhang/alex/wm-lab/datas/spec/zeldam2/inpaint.txt"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/scripts/decoded_latent_samples/zeldam2-v2"
)
DEFAULT_CACHE_SNAPSHOT = Path(
    "/mnt/weka/home/hao.zhang/.cache/huggingface/hub/"
    "models--weizhou03--Wan2.1-Game-Fun-1.3B-InP-Diffusers/snapshots/"
    "646c2c907816063473b6238f3dad5a971d353be3"
)

ID_RE = re.compile(r"^idx(?P<idx>\d+)_chunk(?P<chunk>.+)_seg(?P<seg>\d+)$")


@dataclass(frozen=True)
class SampleMeta:
    idx: int
    sample_id: str
    chunk_id: str
    segment_idx: int
    parquet_path: str
    row_index: int
    parquet_stem: str
    wasd_nonzero: int
    keyboard_nonzero: int
    mouse_nonzero: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode random WASD-active zeldam2-v2 samples per idx."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing zeldam2-v2 action_latent parquet files.",
    )
    parser.add_argument(
        "--idx-file",
        type=Path,
        default=DEFAULT_IDX_FILE,
        help="Text file listing target idxs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root. Each idx gets its own subdirectory.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_CACHE_SNAPSHOT),
        help="Wan model root path or repo id containing a vae/ subfolder.",
    )
    parser.add_argument(
        "--samples-per-idx",
        type=int,
        default=5,
        help="Number of random WASD-active samples to decode per idx.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated GPU ids to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260322,
        help="Random seed for sample selection.",
    )
    parser.add_argument(
        "--master-port-base",
        type=int,
        default=29530,
        help="Base port used if a worker needs distributed-style env vars.",
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
        help="Optional limit after filtering, useful for testing.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip decoding if the expected overlay mp4 already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only scan/select samples and write metadata, without decoding.",
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


def expected_output_path(output_dir: Path, sample: SampleMeta) -> Path:
    return output_dir / (
        f"wangame_{sample.sample_id}_{sample.parquet_stem}_row"
        f"{sample.row_index}_overlay.mp4"
    )


def resolve_model_root(model_path: str) -> Path:
    candidate = Path(model_path)
    if candidate.is_dir():
        if candidate.name == "vae":
            candidate = candidate.parent
        vae_dir = candidate / "vae"
        if not vae_dir.is_dir():
            raise FileNotFoundError(f"Could not find vae/ under {candidate}")
        return candidate
    raise FileNotFoundError(
        f"Model path must be a local directory in this environment: {model_path}"
    )


def scan_candidates(
    dataset_root: Path,
    target_idxs: set[int],
) -> dict[int, list[SampleMeta]]:
    candidates: dict[int, list[SampleMeta]] = defaultdict(list)
    parquet_files = sorted(dataset_root.rglob("*.parquet"))
    total_files = len(parquet_files)
    for file_idx, parquet_path in enumerate(parquet_files, start=1):
        table = pq.read_table(
            parquet_path,
            columns=[
                "id",
                "keyboard_cond_bytes",
                "keyboard_cond_shape",
                "keyboard_cond_dtype",
                "mouse_cond_bytes",
                "mouse_cond_shape",
                "mouse_cond_dtype",
            ],
        )
        ids = table["id"]
        keyboard_bytes_col = table["keyboard_cond_bytes"]
        keyboard_shape_col = table["keyboard_cond_shape"]
        keyboard_dtype_col = table["keyboard_cond_dtype"]
        mouse_bytes_col = table["mouse_cond_bytes"]
        mouse_shape_col = table["mouse_cond_shape"]
        mouse_dtype_col = table["mouse_cond_dtype"]

        for row_index in range(table.num_rows):
            sample_id = ids[row_index].as_py()
            try:
                idx, chunk_id, segment_idx = parse_sample_id(sample_id)
            except ValueError:
                continue
            if idx not in target_idxs:
                continue

            keyboard_bytes = keyboard_bytes_col[row_index].as_py()
            if keyboard_bytes is None:
                continue

            import numpy as np

            keyboard_shape = keyboard_shape_col[row_index].as_py()
            keyboard_dtype = keyboard_dtype_col[row_index].as_py()
            keyboard = np.frombuffer(
                keyboard_bytes, dtype=np.dtype(keyboard_dtype)
            ).reshape(keyboard_shape)
            wasd_width = min(4, keyboard.shape[1])
            wasd_nonzero = int((keyboard[:, :wasd_width] != 0).sum())
            if wasd_nonzero <= 0:
                continue

            mouse_nonzero = 0
            mouse_bytes = mouse_bytes_col[row_index].as_py()
            if mouse_bytes is not None:
                mouse_shape = mouse_shape_col[row_index].as_py()
                mouse_dtype = mouse_dtype_col[row_index].as_py()
                mouse = np.frombuffer(
                    mouse_bytes, dtype=np.dtype(mouse_dtype)
                ).reshape(mouse_shape)
                mouse_nonzero = int((mouse != 0).sum())

            candidates[idx].append(
                SampleMeta(
                    idx=idx,
                    sample_id=sample_id,
                    chunk_id=chunk_id,
                    segment_idx=segment_idx,
                    parquet_path=str(parquet_path),
                    row_index=row_index,
                    parquet_stem=parquet_path.stem,
                    wasd_nonzero=wasd_nonzero,
                    keyboard_nonzero=int((keyboard != 0).sum()),
                    mouse_nonzero=mouse_nonzero,
                )
            )

        if file_idx % 500 == 0 or file_idx == total_files:
            print(
                f"[scan] processed {file_idx}/{total_files} parquet files",
                flush=True,
            )
    return candidates


def select_random_samples(
    all_candidates: dict[int, list[SampleMeta]],
    idx_order: list[int],
    samples_per_idx: int,
    seed: int,
) -> dict[int, list[SampleMeta]]:
    selected: dict[int, list[SampleMeta]] = {}
    for idx in idx_order:
        candidates = sorted(
            all_candidates.get(idx, []),
            key=lambda sample: (
                sample.chunk_id,
                sample.segment_idx,
                sample.sample_id,
                sample.row_index,
            ),
        )
        rng = random.Random(seed + idx)

        by_chunk: dict[str, list[SampleMeta]] = defaultdict(list)
        for sample in candidates:
            by_chunk[sample.chunk_id].append(sample)

        chunk_ids = list(by_chunk.keys())
        rng.shuffle(chunk_ids)

        picked: list[SampleMeta] = []
        used_ids: set[str] = set()

        for chunk_id in chunk_ids:
            if len(picked) >= samples_per_idx:
                break
            chunk_samples = by_chunk[chunk_id][:]
            rng.shuffle(chunk_samples)
            choice = chunk_samples[0]
            picked.append(choice)
            used_ids.add(choice.sample_id)

        if len(picked) < samples_per_idx:
            remaining = [s for s in candidates if s.sample_id not in used_ids]
            rng.shuffle(remaining)
            picked.extend(remaining[: samples_per_idx - len(picked)])

        selected[idx] = picked
    return selected


def build_jobs(
    selected: dict[int, list[SampleMeta]],
    output_root: Path,
    skip_existing: bool,
) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for idx, samples in selected.items():
        output_dir = output_root / str(idx)
        output_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            output_path = expected_output_path(output_dir, sample)
            if skip_existing and output_path.is_file():
                continue
            jobs.append(
                {
                    "idx": idx,
                    "sample": asdict(sample),
                    "output_path": str(output_path),
                }
            )
    return jobs


def write_selection_summary(
    output_root: Path,
    idx_order: list[int],
    all_candidates: dict[int, list[SampleMeta]],
    selected: dict[int, list[SampleMeta]],
) -> None:
    summary = {
        "generated_at_unix_s": time.time(),
        "idxs": [],
    }
    for idx in idx_order:
        summary["idxs"].append(
            {
                "idx": idx,
                "num_candidates": len(all_candidates.get(idx, [])),
                "selected": [asdict(sample) for sample in selected.get(idx, [])],
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "selection_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


def write_decode_summary(output_root: Path, results: list[dict[str, object]]) -> None:
    payload = {
        "generated_at_unix_s": time.time(),
        "num_results": len(results),
        "results": results,
    }
    (output_root / "decode_results.json").write_text(json.dumps(payload, indent=2))


def decode_worker(
    worker_index: int,
    gpu_id: int,
    model_root: str,
    task_queue: mp.queues.Queue,
    result_queue: mp.queues.Queue,
    master_port: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["PYTHONUNBUFFERED"] = "1"

    import cv2
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import torchvision
    from diffusers import AutoencoderKLWan
    from einops import rearrange

    def draw_rounded_rectangle(
        image,
        top_left,
        bottom_right,
        color,
        radius=10,
        alpha=0.5,
    ):
        overlay = image.copy()
        x1, y1 = top_left
        x2, y2 = bottom_right
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.ellipse(
            overlay,
            (x1 + radius, y1 + radius),
            (radius, radius),
            180,
            0,
            90,
            color,
            -1,
        )
        cv2.ellipse(
            overlay,
            (x2 - radius, y1 + radius),
            (radius, radius),
            270,
            0,
            90,
            color,
            -1,
        )
        cv2.ellipse(
            overlay,
            (x1 + radius, y2 - radius),
            (radius, radius),
            90,
            0,
            90,
            color,
            -1,
        )
        cv2.ellipse(
            overlay,
            (x2 - radius, y2 - radius),
            (radius, radius),
            0,
            0,
            90,
            color,
            -1,
        )
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def draw_keys_on_frame(frame, keys, key_size=(30, 30), top_margin=15):
        left_margin = 15
        gap = 3
        key_positions = {
            "W": (left_margin + key_size[0] + gap, top_margin),
            "A": (left_margin, top_margin + key_size[1] + gap),
            "S": (
                left_margin + key_size[0] + gap,
                top_margin + key_size[1] + gap,
            ),
            "D": (
                left_margin + (key_size[0] + gap) * 2,
                top_margin + key_size[1] + gap,
            ),
            "left": (
                left_margin + (key_size[0] + gap) * 3 + 10,
                top_margin + key_size[1] + gap,
            ),
            "right": (
                left_margin + (key_size[0] + gap) * 4 + 15,
                top_margin + key_size[1] + gap,
            ),
        }
        key_icon = {
            "W": "W",
            "A": "A",
            "S": "S",
            "D": "D",
            "left": "L",
            "right": "R",
        }
        for key, (x, y) in key_positions.items():
            is_pressed = keys.get(key, False)
            top_left = (x, y)
            bottom_right = (x + key_size[0], y + key_size[1])
            color = (0, 255, 0) if is_pressed else (200, 200, 200)
            alpha = 0.8 if is_pressed else 0.5
            draw_rounded_rectangle(
                frame, top_left, bottom_right, color, radius=5, alpha=alpha
            )
            text_size = cv2.getTextSize(
                key_icon[key], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )[0]
            text_x = x + (key_size[0] - text_size[0]) // 2
            text_y = y + (key_size[1] + text_size[1]) // 2
            cv2.putText(
                frame,
                key_icon[key],
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

    def draw_mouse_on_frame(frame, pitch, yaw, top_margin=15):
        height, width, _ = frame.shape
        right_margin = 15
        crosshair_radius = 25
        crosshair_x = width - right_margin - crosshair_radius
        crosshair_y = top_margin + crosshair_radius
        dx = int(yaw * crosshair_radius * 8)
        dy = int(-pitch * crosshair_radius * 8)
        max_arrow = crosshair_radius - 5
        dx = max(-max_arrow, min(max_arrow, dx))
        dy = max(-max_arrow, min(max_arrow, dy))
        cv2.circle(frame, (crosshair_x, crosshair_y), crosshair_radius, (50, 50, 50), -1)
        cv2.circle(
            frame, (crosshair_x, crosshair_y), crosshair_radius, (200, 200, 200), 1
        )
        cv2.line(
            frame,
            (crosshair_x - crosshair_radius + 5, crosshair_y),
            (crosshair_x + crosshair_radius - 5, crosshair_y),
            (100, 100, 100),
            1,
        )
        cv2.line(
            frame,
            (crosshair_x, crosshair_y - crosshair_radius + 5),
            (crosshair_x, crosshair_y + crosshair_radius - 5),
            (100, 100, 100),
            1,
        )
        if abs(dx) > 1 or abs(dy) > 1:
            cv2.arrowedLine(
                frame,
                (crosshair_x, crosshair_y),
                (crosshair_x + dx, crosshair_y + dy),
                (0, 255, 0),
                2,
                tipLength=0.3,
            )

    def load_latent_row(parquet_path: str, row_index: int):
        table = pq.read_table(
            parquet_path,
            columns=[
                "id",
                "file_name",
                "vae_latent_bytes",
                "vae_latent_shape",
                "vae_latent_dtype",
                "fps",
            ],
        )
        latent = np.frombuffer(
            table["vae_latent_bytes"][row_index].as_py(),
            dtype=np.dtype(table["vae_latent_dtype"][row_index].as_py()),
        ).reshape(table["vae_latent_shape"][row_index].as_py())
        return {
            "id": table["id"][row_index].as_py(),
            "file_name": table["file_name"][row_index].as_py(),
            "fps": table["fps"][row_index].as_py(),
            "latent": latent.copy(),
        }

    def load_action_row(parquet_path: str, row_index: int):
        table = pq.read_table(
            parquet_path,
            columns=[
                "keyboard_cond_bytes",
                "keyboard_cond_shape",
                "keyboard_cond_dtype",
                "mouse_cond_bytes",
                "mouse_cond_shape",
                "mouse_cond_dtype",
            ],
        )
        keyboard = None
        mouse = None
        keyboard_bytes = table["keyboard_cond_bytes"][row_index].as_py()
        if keyboard_bytes is not None:
            keyboard = np.frombuffer(
                keyboard_bytes,
                dtype=np.dtype(table["keyboard_cond_dtype"][row_index].as_py()),
            ).reshape(table["keyboard_cond_shape"][row_index].as_py()).copy()
        mouse_bytes = table["mouse_cond_bytes"][row_index].as_py()
        if mouse_bytes is not None:
            mouse = np.frombuffer(
                mouse_bytes,
                dtype=np.dtype(table["mouse_cond_dtype"][row_index].as_py()),
            ).reshape(table["mouse_cond_shape"][row_index].as_py()).copy()
        return keyboard, mouse

    def decode_to_frames(vae, latent):
        latents = torch.from_numpy(latent).unsqueeze(0).to(
            device="cuda", dtype=torch.float32
        )
        with torch.inference_mode():
            decoded = vae.decode(latents)
            if hasattr(decoded, "sample"):
                samples = decoded.sample
            elif isinstance(decoded, (tuple, list)):
                samples = decoded[0]
            else:
                samples = decoded
            samples = (samples / 2 + 0.5).clamp(0, 1)
            videos = rearrange(samples, "b c t h w -> t b c h w")
            frames = []
            for frame in videos:
                frame = torchvision.utils.make_grid(frame, nrow=6)
                frame = frame.permute(1, 2, 0).squeeze(-1)
                frame = (frame * 255).to(torch.uint8).cpu().numpy()
                frames.append(frame)
        return frames

    model_root_path = resolve_model_root(model_root)
    vae = AutoencoderKLWan.from_pretrained(
        model_root_path, subfolder="vae", torch_dtype=torch.float32
    )
    vae = vae.to("cuda").eval()
    vae.requires_grad_(False)
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()

    while True:
        try:
            task = task_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if task is None:
            break
        sample = task["sample"]
        output_path = Path(task["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = output_path.with_suffix(".log")
        started_at = time.time()
        try:
            row = load_latent_row(sample["parquet_path"], sample["row_index"])
            keyboard_cond, mouse_cond = load_action_row(
                sample["parquet_path"], sample["row_index"]
            )
            frames = decode_to_frames(vae, row["latent"])
            key_names = ["W", "S", "A", "D", "left", "right"]
            overlaid_frames = []
            for frame_idx, frame in enumerate(frames):
                frame = np.ascontiguousarray(frame.copy())
                if keyboard_cond is not None and frame_idx < len(keyboard_cond):
                    keys = {
                        key_names[i]: bool(keyboard_cond[frame_idx, i])
                        for i in range(
                            min(len(key_names), keyboard_cond.shape[1])
                        )
                    }
                    draw_keys_on_frame(frame, keys)
                if mouse_cond is not None and frame_idx < len(mouse_cond):
                    draw_mouse_on_frame(
                        frame,
                        float(mouse_cond[frame_idx, 0]),
                        float(mouse_cond[frame_idx, 1]),
                    )
                overlaid_frames.append(frame)
            fps = row["fps"] or 25.0
            imageio.mimsave(output_path, overlaid_frames, fps=fps, format="mp4")
            log_path.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "worker_index": worker_index,
                        "gpu_id": gpu_id,
                        "sample_id": sample["sample_id"],
                        "idx": sample["idx"],
                        "parquet_path": sample["parquet_path"],
                        "row_index": sample["row_index"],
                        "fps": fps,
                        "num_frames": len(overlaid_frames),
                        "output_path": str(output_path),
                        "elapsed_s": time.time() - started_at,
                    },
                    indent=2,
                )
            )
            result_queue.put(
                {
                    "ok": True,
                    "idx": sample["idx"],
                    "sample_id": sample["sample_id"],
                    "gpu_id": gpu_id,
                    "worker_index": worker_index,
                    "output_path": str(output_path),
                    "elapsed_s": round(time.time() - started_at, 3),
                }
            )
        except Exception:
            error_text = traceback.format_exc()
            log_path.write_text(error_text)
            result_queue.put(
                {
                    "ok": False,
                    "idx": sample["idx"],
                    "sample_id": sample["sample_id"],
                    "gpu_id": gpu_id,
                    "worker_index": worker_index,
                    "output_path": str(output_path),
                    "elapsed_s": round(time.time() - started_at, 3),
                    "error": error_text,
                }
            )


def main() -> int:
    args = parse_args()
    model_root = resolve_model_root(args.model_path)
    gpu_ids = [int(token.strip()) for token in args.gpu_ids.split(",") if token.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")

    idx_order = parse_idx_file(args.idx_file)
    only_idxs = parse_only_idxs(args.only_idxs)
    if only_idxs is not None:
        idx_order = [idx for idx in idx_order if idx in only_idxs]
    if args.idx_limit is not None:
        idx_order = idx_order[: args.idx_limit]
    target_idxs = set(idx_order)
    if not idx_order:
        raise ValueError("No idxs selected")

    print(
        f"[setup] selected {len(idx_order)} idxs, "
        f"{args.samples_per_idx} samples each, GPUs={gpu_ids}",
        flush=True,
    )
    print(f"[setup] scanning {args.dataset_root}", flush=True)
    all_candidates = scan_candidates(args.dataset_root, target_idxs)
    selected = select_random_samples(
        all_candidates, idx_order, args.samples_per_idx, args.seed
    )
    write_selection_summary(args.output_root, idx_order, all_candidates, selected)

    for idx in idx_order:
        num_candidates = len(all_candidates.get(idx, []))
        num_selected = len(selected.get(idx, []))
        print(
            f"[select] idx={idx} candidates={num_candidates} selected={num_selected}",
            flush=True,
        )

    jobs = build_jobs(selected, args.output_root, args.skip_existing)
    print(f"[jobs] queued {len(jobs)} decode jobs", flush=True)
    if args.dry_run or not jobs:
        return 0

    ctx = mp.get_context("spawn")
    task_queue: mp.queues.Queue = ctx.Queue()
    result_queue: mp.queues.Queue = ctx.Queue()

    for job in jobs:
        task_queue.put(job)
    for _ in gpu_ids:
        task_queue.put(None)

    workers: list[mp.Process] = []
    port_counter = count(args.master_port_base)
    for worker_index, gpu_id in enumerate(gpu_ids):
        process = ctx.Process(
            target=decode_worker,
            args=(
                worker_index,
                gpu_id,
                str(model_root),
                task_queue,
                result_queue,
                next(port_counter),
            ),
            daemon=False,
        )
        process.start()
        workers.append(process)

    results: list[dict[str, object]] = []
    completed = 0
    failed = 0
    total = len(jobs)
    while completed < total:
        result = result_queue.get()
        results.append(result)
        completed += 1
        status = "ok" if result["ok"] else "fail"
        if not result["ok"]:
            failed += 1
        print(
            f"[decode] {completed}/{total} {status} "
            f"idx={result['idx']} sample={result['sample_id']} "
            f"gpu={result['gpu_id']} elapsed={result['elapsed_s']}s",
            flush=True,
        )

    for process in workers:
        process.join()

    write_decode_summary(args.output_root, results)
    print(f"[done] finished {total} jobs, failures={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
