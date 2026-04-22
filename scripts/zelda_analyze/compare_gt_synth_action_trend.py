#!/usr/bin/env python3
"""Compare GT-video PTLFlow against action-conditioned synthetic flow.

This script answers a narrow question for WanGame-style action data:
does the synthetic flow generated from the correct action match GT-video
PTLFlow better than a mismatched action?

For each GT video:
1. Compute PTLFlow on consecutive GT frames.
2. Generate synthetic flow from a supplied action file.
3. Compare GT PTLFlow vs synthetic flow with the same metrics used by
   ``eval_flow_divergence.py``.

Outputs:
- per-video/action frame metrics CSVs
- per-video/action summary CSV + JSON
- paired correct-vs-wrong comparison CSV
- aggregate JSON with win counts
- summary plot PNG
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
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


DEFAULT_VIDEO_DIR = Path(
    "/mnt/weka/home/hao.zhang/mhuo/traindata_0205_1330/data/1_a_only_plus_d_only/videos"
)
DEFAULT_OUTPUT_DIR = ROOT / "analyze" / "outputs" / "gt_synth_action_trend"
DEFAULT_CALIBRATION = ROOT / "calibration.json"
DEFAULT_CKPT = ROOT / "dpflow-things-2012b5d6.ckpt"

LOWER_BETTER_METRICS = {
    "mf_epe_mean",
    "mf_angle_err_mean",
    "pixel_epe_mean_mean",
    "px_angle_rmse_mean",
    "fl_all_mean",
    "foe_dist_mean",
    "flow_kl_2d_mean",
    "pixel_epe_over_ref_mean_mag_mean",
    "pixel_mag_abs_rel_mean",
}
HIGHER_BETTER_METRICS = {
    "mf_cosine_mean",
}
CLOSER_TO_ONE_METRICS = {
    "mf_mag_ratio_mean",
    "pixel_mag_ratio_mean",
}
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
        description="Compare GT PTLFlow to synthetic flow for correct vs wrong actions."
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help="Directory containing GT .mp4 files and *_action.npy files.",
    )
    parser.add_argument(
        "--video-ids",
        nargs="+",
        default=["000000", "000001", "000002"],
        help="GT video ids to evaluate.",
    )
    parser.add_argument(
        "--correct-action",
        type=Path,
        default=DEFAULT_VIDEO_DIR / "000000_action.npy",
        help="Action file considered the correct action for the GT videos.",
    )
    parser.add_argument(
        "--wrong-action",
        type=Path,
        default=DEFAULT_VIDEO_DIR / "001591_action.npy",
        help="Action file considered the wrong action baseline.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="Path to calibration.json.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dpflow",
        help="PTLFlow model name.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(DEFAULT_CKPT),
        help="PTLFlow checkpoint path or alias.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=8,
        help="Grid size for flow metrics.",
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Disable depth-aware synthetic flow and use constant depth.",
    )
    return parser.parse_args()


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
    depth_maps = []
    for frame in frames[:-1]:
        depth_maps.append(generator.estimate_depth(frame))
    return depth_maps


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
        pixel_mag_abs_rel = abs(cmp_mag - ref_mag) / ref_mag
    else:
        pixel_mag_ratio = 1.0
        epe_over_ref = 0.0
        pixel_mag_abs_rel = 0.0

    return {
        "ref_mean_mag": ref_mag,
        "cmp_mean_mag": cmp_mag,
        "pixel_mag_ratio": float(pixel_mag_ratio),
        "pixel_epe_over_ref_mean_mag": float(epe_over_ref),
        "pixel_mag_abs_rel": float(pixel_mag_abs_rel),
    }


def evaluate_action_against_gt(
    video_id: str,
    flows_gt: Sequence[np.ndarray],
    frames: Sequence[np.ndarray],
    action_name: str,
    action_path: Path,
    action_data: Dict[str, np.ndarray],
    generator: SyntheticFlowGenerator,
    depth_maps: Sequence[np.ndarray] | None,
    grid_size: int,
    per_case_dir: Path,
) -> Dict[str, object]:
    flows_synth = generate_synthetic_flow_sequence(generator, action_data, frames, depth_maps)
    n_flows = min(len(flows_gt), len(flows_synth))
    flows_synth = flows_synth[:n_flows]

    frame_metrics = []
    for idx in range(n_flows):
        metrics = compute_frame_metrics(flows_gt[idx], flows_synth[idx], grid_size=grid_size)
        metrics.update(compute_reference_relative_metrics(flows_gt[idx], flows_synth[idx]))
        metrics["frame_idx"] = idx
        frame_metrics.append(metrics)

    summary = compute_temporal_metrics(frame_metrics)
    summary["video_id"] = video_id
    summary["action_name"] = action_name
    summary["action_path"] = str(action_path)
    summary["n_flows"] = n_flows
    summary["pixel_mag_ratio_pct"] = 100.0 * summary["pixel_mag_ratio_mean"]
    summary["pixel_epe_over_ref_mean_mag_pct"] = (
        100.0 * summary["pixel_epe_over_ref_mean_mag_mean"]
    )

    per_case_dir.mkdir(parents=True, exist_ok=True)
    frame_csv_path = per_case_dir / "frame_metrics.csv"
    with open(frame_csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(frame_metrics)

    summary_json_path = per_case_dir / "summary.json"
    with open(summary_json_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    return {
        "video_id": video_id,
        "action_name": action_name,
        "action_path": str(action_path),
        "frame_metrics": frame_metrics,
        "summary": summary,
        "frame_csv_path": str(frame_csv_path),
        "summary_json_path": str(summary_json_path),
    }


def choose_winner(metric_name: str, correct_value: float, wrong_value: float) -> str:
    if not (math.isfinite(correct_value) and math.isfinite(wrong_value)):
        return "invalid"

    if metric_name in LOWER_BETTER_METRICS:
        if correct_value < wrong_value:
            return "correct"
        if wrong_value < correct_value:
            return "wrong"
        return "tie"

    if metric_name in HIGHER_BETTER_METRICS:
        if correct_value > wrong_value:
            return "correct"
        if wrong_value > correct_value:
            return "wrong"
        return "tie"

    if metric_name in CLOSER_TO_ONE_METRICS:
        correct_dist = abs(correct_value - 1.0)
        wrong_dist = abs(wrong_value - 1.0)
        if correct_dist < wrong_dist:
            return "correct"
        if wrong_dist < correct_dist:
            return "wrong"
        return "tie"

    return "unknown"


def aggregate_by_action(results: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        action_name = str(result["action_name"])
        summary = result["summary"]
        assert isinstance(summary, dict)
        for key, value in summary.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                grouped[action_name][key].append(float(value))

    aggregated: Dict[str, Dict[str, float]] = {}
    for action_name, metrics in grouped.items():
        aggregated[action_name] = {}
        for key, values in metrics.items():
            aggregated[action_name][key] = float(np.mean(values))
    return aggregated


def write_summary_csv(results: Sequence[Dict[str, object]], output_path: Path) -> None:
    rows = []
    for result in results:
        summary = dict(result["summary"])
        summary["video_id"] = result["video_id"]
        summary["action_name"] = result["action_name"]
        summary["action_path"] = result["action_path"]
        rows.append(summary)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pairwise_csv(results: Sequence[Dict[str, object]], output_path: Path) -> Dict[str, Dict[str, int]]:
    by_video: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for result in results:
        summary = result["summary"]
        assert isinstance(summary, dict)
        by_video[str(result["video_id"])][str(result["action_name"])] = summary

    metrics_to_compare = [
        "pixel_epe_mean_mean",
        "flow_kl_2d_mean",
        "mf_cosine_mean",
        "mf_mag_ratio_mean",
        "pixel_mag_ratio_mean",
        "pixel_epe_over_ref_mean_mag_mean",
        "fl_all_mean",
    ]

    rows = []
    win_counts: Dict[str, Dict[str, int]] = {
        metric: {"correct": 0, "wrong": 0, "tie": 0, "invalid": 0, "unknown": 0}
        for metric in metrics_to_compare
    }
    for video_id, summaries in sorted(by_video.items()):
        if "correct" not in summaries or "wrong" not in summaries:
            continue
        row: Dict[str, object] = {"video_id": video_id}
        correct_summary = summaries["correct"]
        wrong_summary = summaries["wrong"]
        for metric in metrics_to_compare:
            correct_value = float(correct_summary.get(metric, float("nan")))
            wrong_value = float(wrong_summary.get(metric, float("nan")))
            winner = choose_winner(metric, correct_value, wrong_value)
            row[f"correct_{metric}"] = correct_value
            row[f"wrong_{metric}"] = wrong_value
            row[f"winner_{metric}"] = winner
            win_counts[metric][winner] = win_counts[metric].get(winner, 0) + 1
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return win_counts


def plot_results(
    results: Sequence[Dict[str, object]],
    aggregate: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    by_video: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for result in results:
        summary = result["summary"]
        assert isinstance(summary, dict)
        by_video[str(result["video_id"])][str(result["action_name"])] = summary

    n_metrics = len(PLOT_METRICS)
    n_cols = 3
    n_rows = int(math.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    video_ids = sorted(by_video.keys())
    x = np.arange(len(video_ids))

    for ax, metric in zip(axes, PLOT_METRICS):
        correct_vals = [
            float(by_video[video_id]["correct"].get(metric, float("nan"))) for video_id in video_ids
        ]
        wrong_vals = [
            float(by_video[video_id]["wrong"].get(metric, float("nan"))) for video_id in video_ids
        ]

        ax.plot(x, correct_vals, marker="o", linewidth=2, label="correct")
        ax.plot(x, wrong_vals, marker="o", linewidth=2, label="wrong")
        ax.set_xticks(x)
        ax.set_xticklabels(video_ids, rotation=0)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)

        agg_correct = aggregate.get("correct", {}).get(metric)
        agg_wrong = aggregate.get("wrong", {}).get(metric)
        if agg_correct is not None:
            ax.axhline(float(agg_correct), color="C0", linestyle="--", alpha=0.35)
        if agg_wrong is not None:
            ax.axhline(float(agg_wrong), color="C1", linestyle="--", alpha=0.35)

    for ax in axes[n_metrics:]:
        ax.axis("off")

    axes[0].legend(loc="best")
    fig.suptitle("GT PTLFlow vs Synthetic Flow: correct vs wrong action")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    calibration = load_calibration(args.calibration)
    model = ptlflow.get_model(args.model, args.ckpt)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    correct_action = load_actions(args.correct_action)
    wrong_action = load_actions(args.wrong_action)

    results = []
    config = {
        "video_dir": str(args.video_dir),
        "video_ids": list(args.video_ids),
        "correct_action": str(args.correct_action),
        "wrong_action": str(args.wrong_action),
        "calibration": str(args.calibration),
        "model": args.model,
        "ckpt": args.ckpt,
        "use_depth": not args.no_depth,
    }

    generator_cache: Dict[tuple[int, int], SyntheticFlowGenerator] = {}

    for video_id in args.video_ids:
        video_path = args.video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Missing GT video: {video_path}")

        print(f"[video {video_id}] Loading frames")
        frames = extract_frames(str(video_path))
        if len(frames) < 2:
            raise ValueError(f"Need at least 2 frames in {video_path}")

        print(f"[video {video_id}] Computing GT PTLFlow")
        io_adapter = build_io_adapter(model, frames[0].shape[:2])
        with torch.no_grad():
            flows_gt = compute_flow_sequence(model, frames, io_adapter)

        frame_shape = frames[0].shape[:2]
        generator = generator_cache.get(frame_shape)
        if generator is None:
            generator = SyntheticFlowGenerator(
                calibration=calibration,
                frame_shape=frame_shape,
                use_depth=not args.no_depth,
            )
            if not args.no_depth:
                print(f"[video {video_id}] Loading depth model")
                generator.load_depth_model()
            generator_cache[frame_shape] = generator

        depth_maps = None
        if not args.no_depth:
            print(f"[video {video_id}] Estimating depth maps")
            depth_maps = compute_depth_maps(generator, frames)

        for action_name, action_path, action_data in [
            ("correct", args.correct_action, correct_action),
            ("wrong", args.wrong_action, wrong_action),
        ]:
            print(f"[video {video_id}] Comparing action={action_name}")
            case_dir = args.output_dir / f"{video_id}_{action_name}"
            result = evaluate_action_against_gt(
                video_id=video_id,
                flows_gt=flows_gt,
                frames=frames,
                action_name=action_name,
                action_path=action_path,
                action_data=action_data,
                generator=generator,
                depth_maps=depth_maps,
                grid_size=args.grid_size,
                per_case_dir=case_dir,
            )
            results.append(result)

    summary_csv_path = args.output_dir / "summary_rows.csv"
    write_summary_csv(results, summary_csv_path)

    pairwise_csv_path = args.output_dir / "pairwise_comparison.csv"
    win_counts = write_pairwise_csv(results, pairwise_csv_path)

    aggregate = aggregate_by_action(results)
    aggregate_json_path = args.output_dir / "aggregate_summary.json"
    aggregate_payload = {
        "config": config,
        "aggregate_by_action": aggregate,
        "pairwise_win_counts": win_counts,
    }
    with open(aggregate_json_path, "w") as handle:
        json.dump(aggregate_payload, handle, indent=2)

    plot_path = args.output_dir / "metric_overview.png"
    plot_results(results, aggregate, plot_path)

    print("\nAggregate means")
    for action_name in ("correct", "wrong"):
        action_metrics = aggregate.get(action_name, {})
        if not action_metrics:
            continue
        print(
            f"  {action_name}: "
            f"pixel_epe_mean_mean={action_metrics.get('pixel_epe_mean_mean', float('nan')):.3f}, "
            f"flow_kl_2d_mean={action_metrics.get('flow_kl_2d_mean', float('nan')):.3f}, "
            f"mf_cosine_mean={action_metrics.get('mf_cosine_mean', float('nan')):.3f}, "
            f"pixel_mag_ratio_mean={action_metrics.get('pixel_mag_ratio_mean', float('nan')):.3f}, "
            f"pixel_epe_over_ref_mean_mag_mean="
            f"{action_metrics.get('pixel_epe_over_ref_mean_mag_mean', float('nan')):.3f}"
        )

    print("\nPairwise wins")
    for metric, counts in win_counts.items():
        print(
            f"  {metric}: correct={counts.get('correct', 0)}, "
            f"wrong={counts.get('wrong', 0)}, tie={counts.get('tie', 0)}"
        )

    print("\nWrote outputs to:", args.output_dir)


if __name__ == "__main__":
    main()
