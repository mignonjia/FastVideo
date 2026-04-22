#!/usr/bin/env python3
"""Compare one W-only synthetic flow against camera-only group blocks.

This script evaluates a fixed W-only synthetic action against GT PTLFlow from
the dataset:
    /mnt/weka/home/hao.zhang/mhuo/traindata_0206_1200/data/wasdonly_alpha1/videos

By default it assumes the four contiguous 1k blocks correspond to:
- 000000-000999: left
- 001000-001999: right
- 002000-002999: up
- 003000-003999: down

It samples videos from each block, optionally excludes filtered bad samples,
computes the 9 FastVideo metrics, and plots one box per group for each metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _ptlflow_root import PTLFLOW_ROOT, ensure_ptlflow_root_on_path

ROOT = PTLFLOW_ROOT
ensure_ptlflow_root_on_path()

import ptlflow
from eval_flow_divergence import (
    compute_flow_sequence,
    compute_frame_metrics,
    compute_temporal_metrics,
    extract_frames,
)
from ptlflow.utils.io_adapter import IOAdapter
from synthetic_flow import SyntheticFlowGenerator, load_calibration


DEFAULT_PARENT_DIR = Path(
    "/mnt/weka/home/hao.zhang/mhuo/traindata_0206_1200/data/wasdonly_alpha1"
)
DEFAULT_VIDEO_DIR = DEFAULT_PARENT_DIR / "videos"
DEFAULT_FILTER_DIR = DEFAULT_PARENT_DIR / "filter"
DEFAULT_OUTPUT_DIR = ROOT / "analyze" / "outputs" / "w_synth_vs_camera_groups"
DEFAULT_CALIBRATION = ROOT / "calibration.json"
DEFAULT_CKPT = ROOT / "dpflow-things-2012b5d6.ckpt"
DEFAULT_SYNTH_ACTION = Path(
    "/mnt/weka/home/hao.zhang/mhuo/traindata_0205_1330/data/1_wasd_only/videos/004090_action.npy"
)

GROUP_DEFS = [
    ("left", 0, 1000),
    ("right", 1000, 2000),
    ("up", 2000, 3000),
    ("down", 3000, 4000),
]
GROUP_ORDER = [name for name, _, _ in GROUP_DEFS]
PLOT_METRICS = [
    "mf_epe_mean",
    "mf_angle_err_mean",
    "mf_cosine_mean",
    "mf_mag_ratio_mean",
    "pixel_epe_mean_mean",
    "px_angle_rmse_mean",
    "fl_all_mean",
    "foe_dist_mean",
    "flow_kl_2d_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one W-only synthetic flow against left/right/up/down GT groups."
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--synthetic-action-path", type=Path, default=DEFAULT_SYNTH_ACTION)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--model", type=str, default="dpflow")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--n-per-group", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--gpu-ids", type=str, default="")
    parser.add_argument(
        "--filter-jsons",
        nargs="*",
        default=[
            str(DEFAULT_FILTER_DIR / "bot_died.json"),
            str(DEFAULT_FILTER_DIR / "blue_water.json"),
        ],
    )
    parser.add_argument("--no-depth", action="store_true")
    return parser.parse_args()


def parse_gpu_ids(gpu_ids_arg: str) -> List[int]:
    if gpu_ids_arg.strip():
        return [int(x) for x in gpu_ids_arg.split(",") if x.strip()]
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {output_path}")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_excluded_ids(filter_jsons: Sequence[str]) -> set[str]:
    excluded: set[str] = set()
    for path_str in filter_jsons:
        filter_path = Path(path_str)
        if not filter_path.exists():
            print(f"Warning: filter json not found, skipping: {filter_path}")
            continue
        with open(filter_path) as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Expected list in {filter_path}")
        for item in payload:
            if isinstance(item, int):
                excluded.add(f"{item:06d}")
            elif isinstance(item, str) and item.isdigit():
                excluded.add(f"{int(item):06d}")
            else:
                raise ValueError(f"Unsupported filter entry {item!r} in {filter_path}")
    return excluded


def load_actions(action_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(action_path, allow_pickle=True).item()
    return {
        "keyboard": np.asarray(data["keyboard"], dtype=np.float32),
        "mouse": np.asarray(data["mouse"], dtype=np.float32),
    }


def build_io_adapter(model: torch.nn.Module, frame_shape: tuple[int, int]) -> IOAdapter:
    return IOAdapter(
        output_stride=model.output_stride,
        input_size=frame_shape,
        cuda=torch.cuda.is_available(),
    )


def compute_depth_maps(
    generator: SyntheticFlowGenerator,
    frames: Sequence[np.ndarray],
) -> List[np.ndarray]:
    return [generator.estimate_depth(frame) for frame in frames[:-1]]


def generate_synthetic_flow_sequence(
    generator: SyntheticFlowGenerator,
    actions: Dict[str, np.ndarray],
    frames: Sequence[np.ndarray],
    depth_maps: Sequence[np.ndarray] | None,
) -> List[np.ndarray]:
    keyboard = actions["keyboard"]
    mouse = actions["mouse"]
    n_flows = min(len(frames) - 1, len(keyboard) - 1)
    flows = []
    for idx in range(n_flows):
        flow = generator.generate_flow(
            keyboard[idx],
            mouse[idx],
            frame_bgr=frames[idx] if depth_maps is None else None,
            depth_map=None if depth_maps is None else depth_maps[idx],
        )
        flows.append(flow)
    return flows


def compute_reference_relative_metrics(
    flow_ref: np.ndarray,
    flow_cmp: np.ndarray,
    min_mag: float = 0.5,
    max_mag_pct: float = 80.0,
) -> Dict[str, float]:
    ref_mag_map = np.linalg.norm(flow_ref, axis=2)
    cmp_mag_map = np.linalg.norm(flow_cmp, axis=2)
    max_mag_map = np.maximum(ref_mag_map, cmp_mag_map)
    mag_hi = np.percentile(max_mag_map, max_mag_pct)
    mag_mask = (max_mag_map >= min_mag) & (max_mag_map <= mag_hi)

    if mag_mask.sum() > 0:
        ref_mag = float(ref_mag_map[mag_mask].mean())
        cmp_mag = float(cmp_mag_map[mag_mask].mean())
        epe_map = np.linalg.norm(flow_ref - flow_cmp, axis=2)
        pixel_epe = float(epe_map[mag_mask].mean())
    else:
        ref_mag = float(ref_mag_map.mean())
        cmp_mag = float(cmp_mag_map.mean())
        epe_map = np.linalg.norm(flow_ref - flow_cmp, axis=2)
        pixel_epe = float(epe_map.mean())

    if ref_mag > 1e-6:
        pixel_mag_ratio = cmp_mag / ref_mag
        epe_over_ref = pixel_epe / ref_mag
    else:
        pixel_mag_ratio = 1.0
        epe_over_ref = 0.0

    return {
        "ref_mean_mag": ref_mag,
        "cmp_mean_mag": cmp_mag,
        "pixel_mag_ratio": float(pixel_mag_ratio),
        "pixel_epe_over_ref_mean_mag": float(epe_over_ref),
    }


def build_group_candidates(video_dir: Path, excluded_ids: set[str]) -> Dict[str, List[str]]:
    existing_ids = {p.stem.replace("_action", "") for p in video_dir.glob("*_action.npy")}
    groups: Dict[str, List[str]] = {}
    for name, start, end in GROUP_DEFS:
        ids = [
            f"{idx:06d}"
            for idx in range(start, end)
            if f"{idx:06d}" in existing_ids and f"{idx:06d}" not in excluded_ids
        ]
        groups[name] = ids
    return groups


def sample_video_ids(
    group_candidates: Dict[str, List[str]],
    n_per_group: int,
    random_seed: int,
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(random_seed)
    rows: List[Dict[str, object]] = []
    for group_name in GROUP_ORDER:
        candidates = list(group_candidates[group_name])
        if len(candidates) < n_per_group:
            raise ValueError(
                f"Need at least {n_per_group} valid ids for {group_name}, found {len(candidates)}"
            )
        chosen = sorted(rng.choice(candidates, size=n_per_group, replace=False).tolist())
        for video_id in chosen:
            rows.append({"video_id": video_id, "group_name": group_name})
    return rows


def evaluate_single_video(
    item: Dict[str, object],
    video_dir: Path,
    synthetic_action_name: str,
    synthetic_action_data: Dict[str, np.ndarray],
    model: torch.nn.Module,
    calibration: dict,
    use_depth: bool,
    grid_size: int,
    generator_cache: Dict[tuple[int, int], SyntheticFlowGenerator],
) -> Dict[str, object]:
    video_id = str(item["video_id"])
    group_name = str(item["group_name"])
    video_path = video_dir / f"{video_id}.mp4"

    frames = extract_frames(str(video_path))
    if len(frames) < 2:
        raise ValueError(f"Need at least 2 frames in {video_path}")

    io_adapter = build_io_adapter(model, frames[0].shape[:2])
    with torch.no_grad():
        flows_gt = compute_flow_sequence(model, frames, io_adapter)

    frame_shape = frames[0].shape[:2]
    generator = generator_cache.get(frame_shape)
    if generator is None:
        generator = SyntheticFlowGenerator(
            calibration=calibration,
            frame_shape=frame_shape,
            use_depth=use_depth,
        )
        if use_depth:
            generator.load_depth_model()
        generator_cache[frame_shape] = generator

    depth_maps = compute_depth_maps(generator, frames) if use_depth else None
    flows_synth = generate_synthetic_flow_sequence(
        generator=generator,
        actions=synthetic_action_data,
        frames=frames,
        depth_maps=depth_maps,
    )

    n_flows = min(len(flows_gt), len(flows_synth))
    flows_gt = flows_gt[:n_flows]
    flows_synth = flows_synth[:n_flows]

    frame_metrics = []
    for idx in range(n_flows):
        metrics = compute_frame_metrics(flows_gt[idx], flows_synth[idx], grid_size=grid_size)
        metrics.update(compute_reference_relative_metrics(flows_gt[idx], flows_synth[idx]))
        metrics["frame_idx"] = idx
        frame_metrics.append(metrics)

    summary = compute_temporal_metrics(frame_metrics)
    summary["video_id"] = video_id
    summary["group_name"] = group_name
    summary["synthetic_action_name"] = synthetic_action_name
    summary["n_flows"] = n_flows
    return summary


def extract_chunk(
    manifest_rows: List[Dict[str, object]],
    video_dir: Path,
    synthetic_action_name: str,
    synthetic_action_data: Dict[str, np.ndarray],
    model_name: str,
    ckpt: str,
    calibration_path: Path,
    use_depth: bool,
    grid_size: int,
    gpu_id: int | None,
) -> List[Dict[str, object]]:
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)

    calibration = load_calibration(calibration_path)
    model = ptlflow.get_model(model_name, ckpt)
    model.eval()
    if gpu_id is not None and torch.cuda.is_available():
        model = model.cuda(gpu_id)

    generator_cache: Dict[tuple[int, int], SyntheticFlowGenerator] = {}
    rows = []
    for item in manifest_rows:
        video_id = str(item["video_id"])
        print(f"[gpu {gpu_id if gpu_id is not None else 'cpu'}] [{video_id}] evaluating")
        rows.append(
            evaluate_single_video(
                item=item,
                video_dir=video_dir,
                synthetic_action_name=synthetic_action_name,
                synthetic_action_data=synthetic_action_data,
                model=model,
                calibration=calibration,
                use_depth=use_depth,
                grid_size=grid_size,
                generator_cache=generator_cache,
            )
        )
    return rows


def worker_main(
    manifest_rows: List[Dict[str, object]],
    video_dir: str,
    synthetic_action_name: str,
    synthetic_action_path: str,
    model_name: str,
    ckpt: str,
    calibration_path: str,
    use_depth: bool,
    grid_size: int,
    gpu_id: int,
    out_path: str,
) -> None:
    try:
        rows = extract_chunk(
            manifest_rows=manifest_rows,
            video_dir=Path(video_dir),
            synthetic_action_name=synthetic_action_name,
            synthetic_action_data=load_actions(Path(synthetic_action_path)),
            model_name=model_name,
            ckpt=ckpt,
            calibration_path=Path(calibration_path),
            use_depth=use_depth,
            grid_size=grid_size,
            gpu_id=gpu_id,
        )
        with open(out_path, "w") as handle:
            json.dump({"ok": True, "rows": rows}, handle)
    except Exception as exc:
        with open(out_path, "w") as handle:
            json.dump(
                {
                    "ok": False,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
                handle,
            )


def extract_parallel(
    manifest_rows: List[Dict[str, object]],
    video_dir: Path,
    synthetic_action_name: str,
    synthetic_action_path: Path,
    model_name: str,
    ckpt: str,
    calibration_path: Path,
    use_depth: bool,
    grid_size: int,
    gpu_ids: List[int],
    output_dir: Path,
) -> List[Dict[str, object]]:
    if len(gpu_ids) <= 1:
        gpu_id = gpu_ids[0] if gpu_ids else None
        return extract_chunk(
            manifest_rows=manifest_rows,
            video_dir=video_dir,
            synthetic_action_name=synthetic_action_name,
            synthetic_action_data=load_actions(synthetic_action_path),
            model_name=model_name,
            ckpt=ckpt,
            calibration_path=calibration_path,
            use_depth=use_depth,
            grid_size=grid_size,
            gpu_id=gpu_id,
        )

    shards = [[] for _ in gpu_ids]
    for idx, row in enumerate(manifest_rows):
        shards[idx % len(gpu_ids)].append(row)

    ctx = get_context("spawn")
    processes = []
    shard_paths = []
    for shard_idx, (gpu_id, shard_rows) in enumerate(zip(gpu_ids, shards)):
        if not shard_rows:
            continue
        shard_path = output_dir / f"tmp_camera_group_worker_{shard_idx}_gpu_{gpu_id}.json"
        shard_paths.append(shard_path)
        proc = ctx.Process(
            target=worker_main,
            args=(
                shard_rows,
                str(video_dir),
                synthetic_action_name,
                str(synthetic_action_path),
                model_name,
                ckpt,
                str(calibration_path),
                use_depth,
                grid_size,
                gpu_id,
                str(shard_path),
            ),
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"A worker exited with code {proc.exitcode}")

    all_rows: List[Dict[str, object]] = []
    for shard_path in shard_paths:
        with open(shard_path) as handle:
            payload = json.load(handle)
        if not payload.get("ok", False):
            raise RuntimeError(
                f"Worker failed for {shard_path.name}: {payload.get('error')}\n"
                f"{payload.get('traceback', '')}"
            )
        all_rows.extend(payload["rows"])
        shard_path.unlink(missing_ok=True)

    by_id = {str(row["video_id"]): row for row in all_rows}
    return [by_id[str(row["video_id"])] for row in manifest_rows]


def compute_group_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group_name = str(row["group_name"])
        for metric in PLOT_METRICS:
            value = row.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                grouped[group_name][metric].append(float(value))

    out = []
    for group_name in GROUP_ORDER:
        for metric in PLOT_METRICS:
            values = grouped[group_name].get(metric, [])
            if not values:
                continue
            out.append(
                {
                    "group_name": group_name,
                    "metric": metric,
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return out


def plot_boxplots(per_video_rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    axes = axes.flatten()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax, metric in zip(axes, PLOT_METRICS):
        series = []
        for group_name in GROUP_ORDER:
            values = [
                float(row[metric])
                for row in per_video_rows
                if row.get("group_name") == group_name
                and isinstance(row.get(metric), (int, float))
                and math.isfinite(float(row[metric]))
            ]
            series.append(values)
        box = ax.boxplot(
            series,
            patch_artist=True,
            tick_labels=GROUP_ORDER,
            showfliers=True,
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)
        ax.set_xticklabels(GROUP_ORDER)
        ax.set_title(metric)
        ax.grid(True, axis="y", alpha=0.25)
        if metric == "mf_cosine_mean":
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
        if metric == "mf_mag_ratio_mean":
            ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.35)

    fig.suptitle("W-only synthetic flow vs left/right/up/down GT groups")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    excluded_ids = load_excluded_ids(args.filter_jsons)
    group_candidates = build_group_candidates(args.video_dir, excluded_ids)
    manifest_rows = sample_video_ids(
        group_candidates=group_candidates,
        n_per_group=args.n_per_group,
        random_seed=args.random_seed,
    )
    for row in manifest_rows:
        video_id = str(row["video_id"])
        row["video_path"] = str(args.video_dir / f"{video_id}.mp4")
        row["action_path"] = str(args.video_dir / f"{video_id}_action.npy")
    write_csv(manifest_rows, args.output_dir / "sample_manifest.csv")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    synthetic_action_name = args.synthetic_action_path.stem.replace("_action", "")
    print("Using GPUs:", gpu_ids if gpu_ids else "CPU")
    print("Synthetic W action:", args.synthetic_action_path)
    print("Excluded ids loaded:", len(excluded_ids))

    summary_rows = extract_parallel(
        manifest_rows=manifest_rows,
        video_dir=args.video_dir,
        synthetic_action_name=synthetic_action_name,
        synthetic_action_path=args.synthetic_action_path,
        model_name=args.model,
        ckpt=args.ckpt,
        calibration_path=args.calibration,
        use_depth=not args.no_depth,
        grid_size=args.grid_size,
        gpu_ids=gpu_ids,
        output_dir=args.output_dir,
    )
    write_csv(summary_rows, args.output_dir / "per_video_summary.csv")

    group_summary_rows = compute_group_summary(summary_rows)
    write_csv(group_summary_rows, args.output_dir / "group_metric_summary.csv")
    with open(args.output_dir / "group_metric_summary.json", "w") as handle:
        json.dump(
            {
                "config": {
                    "video_dir": str(args.video_dir),
                    "synthetic_action_path": str(args.synthetic_action_path),
                    "group_definitions": GROUP_DEFS,
                    "n_per_group": args.n_per_group,
                    "random_seed": args.random_seed,
                    "filter_jsons": list(args.filter_jsons),
                    "excluded_count": len(excluded_ids),
                    "model": args.model,
                    "ckpt": args.ckpt,
                    "grid_size": args.grid_size,
                    "gpu_ids": gpu_ids,
                    "use_depth": not args.no_depth,
                },
                "available_counts_after_filter": {
                    group_name: len(ids) for group_name, ids in group_candidates.items()
                },
            },
            handle,
            indent=2,
        )

    plot_boxplots(summary_rows, args.output_dir / "metric_boxplots.png")

    print("\nMetric medians by group")
    for metric in PLOT_METRICS:
        medians = {
            row["group_name"]: row["median"]
            for row in group_summary_rows
            if row["metric"] == metric
        }
        print(f"  {metric}: {medians}")

    print("\nWrote outputs to:", args.output_dir)


if __name__ == "__main__":
    main()
