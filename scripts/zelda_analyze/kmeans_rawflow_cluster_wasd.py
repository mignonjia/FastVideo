#!/usr/bin/env python3
"""Cluster WanGame videos using sampled PTLFlow volumes.

This version avoids handcrafted per-frame summary statistics. Each video is
represented by a compact sampled flow tensor:

- sample 8 flow timesteps from the full PTLFlow sequence
- sample each flow map on a deterministic spatial grid
- keep both flow channels (u, v)
- flatten the resulting 8 x 16 x 16 x 2 tensor

Multiple feature modes can then be derived from the same sampled flow tensor:

- raw: flatten raw u, v values
- video_norm: divide raw u, v by a per-video motion scale
- unit: use unit direction vectors plus an active-motion mask
- angle_masked: use flow angle plus an active-motion mask
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from _ptlflow_root import PTLFLOW_ROOT, ensure_ptlflow_root_on_path

ROOT = PTLFLOW_ROOT
ensure_ptlflow_root_on_path()

import ptlflow
from eval_flow_divergence import compute_flow_sequence, extract_frames
from ptlflow.utils.io_adapter import IOAdapter


DEFAULT_VIDEO_DIR = Path(
    "/mnt/weka/home/hao.zhang/mhuo/traindata_0205_1330/data/1_wasd_only/videos"
)
DEFAULT_OUTPUT_DIR = ROOT / "analyze" / "outputs" / "kmeans_rawflow_cluster_wasd"
DEFAULT_CKPT = ROOT / "dpflow-things-2012b5d6.ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster sampled WanGame videos using sampled PTLFlow volumes."
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default="dpflow")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--gpu-ids", type=str, default="")
    parser.add_argument("--time-steps", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--grid-mode", type=str, default="uniform")
    parser.add_argument("--pca-dim", type=int, default=20)
    parser.add_argument("--feature-modes", type=str, default="raw")
    parser.add_argument("--motion-eps", type=float, default=0.5)
    return parser.parse_args()


def sample_video_ids() -> Dict[str, List[str]]:
    return {
        "A": [f"{idx:06d}" for idx in range(500, 510)],
        "D": [f"{idx:06d}" for idx in range(1500, 1510)],
        "S": [f"{idx:06d}" for idx in range(2500, 2510)],
        "still": [f"{idx:06d}" for idx in range(3500, 3510)],
        "W": [f"{idx:06d}" for idx in range(4500, 4510)],
    }


def infer_true_label(action_path: Path) -> str:
    actions = np.load(action_path, allow_pickle=True).item()["keyboard"]
    key_sums = actions.sum(axis=0)
    mapping = {0: "W", 1: "S", 2: "A", 3: "D"}
    if float(key_sums[:4].max()) < 0.5:
        return "still"
    return mapping[int(np.argmax(key_sums[:4]))]


def build_io_adapter(model: torch.nn.Module, frame_shape: tuple[int, int]) -> IOAdapter:
    return IOAdapter(
        output_stride=model.output_stride,
        input_size=frame_shape,
        cuda=torch.cuda.is_available(),
    )


def parse_gpu_ids(gpu_ids_arg: str) -> List[int]:
    if gpu_ids_arg.strip():
        return [int(x) for x in gpu_ids_arg.split(",") if x.strip()]
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def select_time_indices(n_flows: int, time_steps: int) -> np.ndarray:
    if n_flows <= 0:
        raise ValueError("Expected at least one flow field.")
    if n_flows == 1:
        return np.zeros((time_steps,), dtype=np.int64)
    return np.linspace(0, n_flows - 1, num=time_steps, dtype=np.int64)


def parse_feature_modes(feature_modes_arg: str) -> List[str]:
    supported_modes = {"raw", "video_norm", "unit", "angle_masked"}
    modes = [mode.strip() for mode in feature_modes_arg.split(",") if mode.strip()]
    if not modes:
        raise ValueError("Expected at least one feature mode.")
    invalid_modes = [mode for mode in modes if mode not in supported_modes]
    if invalid_modes:
        raise ValueError(
            f"Unsupported feature mode(s): {invalid_modes}. "
            f"Supported modes: {sorted(supported_modes)}"
        )
    return modes


def parse_grid_mode(grid_mode_arg: str) -> str:
    supported_modes = {"uniform", "center_bottom_weighted"}
    grid_mode = grid_mode_arg.strip()
    if grid_mode not in supported_modes:
        raise ValueError(
            f"Unsupported grid mode: {grid_mode}. Supported modes: {sorted(supported_modes)}"
        )
    return grid_mode


def sample_axis_positions(
    length: int,
    grid_size: int,
    grid_mode: str,
    axis: str,
) -> np.ndarray:
    if grid_mode == "uniform":
        return np.linspace(0.0, length - 1, num=grid_size, dtype=np.float32)

    if grid_mode == "center_bottom_weighted":
        global_count = max(4, grid_size // 2)
        focus_count = grid_size - global_count
        global_positions = np.linspace(0.0, length - 1, num=global_count, dtype=np.float32)
        if axis == "x":
            focus_positions = np.linspace(
                0.2 * (length - 1),
                0.8 * (length - 1),
                num=focus_count,
                dtype=np.float32,
            )
        elif axis == "y":
            focus_positions = np.linspace(
                0.35 * (length - 1),
                0.95 * (length - 1),
                num=focus_count,
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unsupported axis: {axis}")
        positions = np.sort(np.concatenate([global_positions, focus_positions], axis=0))
        return positions.astype(np.float32)

    raise ValueError(f"Unsupported grid mode: {grid_mode}")


def sample_spatial_grid(flow: np.ndarray, grid_size: int, grid_mode: str) -> np.ndarray:
    """Sample a deterministic spatial grid over the flow field."""
    height, width = flow.shape[:2]
    ys = sample_axis_positions(height, grid_size, grid_mode=grid_mode, axis="y")
    xs = sample_axis_positions(width, grid_size, grid_mode=grid_mode, axis="x")
    grid_x, grid_y = np.meshgrid(xs, ys)
    sampled = cv.remap(
        flow,
        grid_x,
        grid_y,
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_REPLICATE,
    )
    return sampled.astype(np.float32)


def sample_flow_volume(
    flows: List[np.ndarray],
    time_steps: int,
    grid_size: int,
    grid_mode: str,
) -> np.ndarray:
    indices = select_time_indices(len(flows), time_steps)
    chunks = []
    for idx in indices:
        flow = flows[int(idx)].astype(np.float32)
        chunks.append(sample_spatial_grid(flow, grid_size, grid_mode=grid_mode))
    return np.stack(chunks, axis=0).astype(np.float32)


def video_motion_scale(magnitudes: np.ndarray, active_mask: np.ndarray, motion_eps: float) -> float:
    active_values = magnitudes[active_mask]
    if active_values.size == 0:
        return max(motion_eps, 1.0)
    return float(max(np.median(active_values), motion_eps))


def build_feature_vector(
    sampled_volume: np.ndarray,
    feature_mode: str,
    motion_eps: float,
) -> np.ndarray:
    magnitudes = np.linalg.norm(sampled_volume, axis=-1, keepdims=True)
    active_mask = magnitudes >= motion_eps
    safe_magnitudes = np.maximum(magnitudes, motion_eps)

    if feature_mode == "raw":
        feature = sampled_volume
    elif feature_mode == "video_norm":
        scale = video_motion_scale(magnitudes[..., 0], active_mask[..., 0], motion_eps)
        feature = sampled_volume / scale
    elif feature_mode == "unit":
        unit_flow = np.where(active_mask, sampled_volume / safe_magnitudes, 0.0)
        feature = np.concatenate([unit_flow, active_mask.astype(np.float32)], axis=-1)
    elif feature_mode == "angle_masked":
        angles = np.arctan2(sampled_volume[..., 1], sampled_volume[..., 0])[..., None] / np.pi
        angles = np.where(active_mask, angles, 0.0)
        feature = np.concatenate([angles.astype(np.float32), active_mask.astype(np.float32)], axis=-1)
    else:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    return feature.astype(np.float32).reshape(-1)


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_chunk(
    manifest_rows: List[Dict[str, object]],
    model_name: str,
    ckpt: str,
    gpu_id: int | None,
    time_steps: int,
    grid_size: int,
    grid_mode: str,
) -> List[Dict[str, object]]:
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)

    model = ptlflow.get_model(model_name, ckpt)
    model.eval()
    if gpu_id is not None and torch.cuda.is_available():
        model = model.cuda(gpu_id)

    rows: List[Dict[str, object]] = []
    io_adapter: IOAdapter | None = None
    adapter_shape: tuple[int, int] | None = None

    for item in manifest_rows:
        video_id = str(item["video_id"])
        video_path = Path(str(item["video_path"]))
        print(f"[gpu {gpu_id if gpu_id is not None else 'cpu'}] [{video_id}] loading frames")
        frames = extract_frames(str(video_path))
        if len(frames) < 2:
            raise ValueError(f"Need at least 2 frames in {video_path}")

        frame_shape = tuple(frames[0].shape[:2])
        if io_adapter is None or adapter_shape != frame_shape:
            io_adapter = build_io_adapter(model, frame_shape)
            adapter_shape = frame_shape

        print(f"[gpu {gpu_id if gpu_id is not None else 'cpu'}] [{video_id}] computing flow")
        with torch.no_grad():
            flows = compute_flow_sequence(model, frames, io_adapter)

        sampled_volume = sample_flow_volume(
            flows,
            time_steps=time_steps,
            grid_size=grid_size,
            grid_mode=grid_mode,
        )
        row: Dict[str, object] = dict(item)
        row["sampled_volume"] = sampled_volume.tolist()
        row["n_flows"] = len(flows)
        rows.append(row)

    return rows


def worker_main(
    manifest_rows: List[Dict[str, object]],
    model_name: str,
    ckpt: str,
    gpu_id: int,
    time_steps: int,
    grid_size: int,
    grid_mode: str,
    out_path: str,
) -> None:
    try:
        rows = extract_chunk(
            manifest_rows=manifest_rows,
            model_name=model_name,
            ckpt=ckpt,
            gpu_id=gpu_id,
            time_steps=time_steps,
            grid_size=grid_size,
            grid_mode=grid_mode,
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
    model_name: str,
    ckpt: str,
    gpu_ids: List[int],
    time_steps: int,
    grid_size: int,
    grid_mode: str,
    output_dir: Path,
) -> List[Dict[str, object]]:
    if len(gpu_ids) <= 1:
        gpu_id = gpu_ids[0] if gpu_ids else None
        return extract_chunk(
            manifest_rows,
            model_name,
            ckpt,
            gpu_id,
            time_steps,
            grid_size,
            grid_mode,
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
        shard_path = output_dir / f"tmp_raw_worker_{shard_idx}_gpu_{gpu_id}.json"
        shard_paths.append(shard_path)
        proc = ctx.Process(
            target=worker_main,
            args=(
                shard_rows,
                model_name,
                ckpt,
                gpu_id,
                time_steps,
                grid_size,
                grid_mode,
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


def cluster_majority_summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    by_cluster: Dict[int, List[str]] = defaultdict(list)
    for row in rows:
        by_cluster[int(row["cluster_id"])].append(str(row["true_label"]))

    cluster_to_majority = {}
    correct = 0
    total = 0
    for cluster_id, labels in sorted(by_cluster.items()):
        counts = Counter(labels)
        majority_label, majority_count = counts.most_common(1)[0]
        cluster_to_majority[str(cluster_id)] = {
            "majority_label": majority_label,
            "counts": dict(sorted(counts.items())),
        }
        correct += majority_count
        total += len(labels)

    return {
        "cluster_to_majority": cluster_to_majority,
        "purity": float(correct / max(total, 1)),
    }


def plot_pca(
    rows: List[Dict[str, object]],
    output_path: Path,
    feature_mode: str,
    grid_mode: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    label_markers = {"A": "o", "D": "s", "S": "^", "W": "D", "still": "X"}
    cluster_colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    for row in rows:
        ax.scatter(
            float(row["pca_x"]),
            float(row["pca_y"]),
            c=cluster_colors[int(row["cluster_id"]) % len(cluster_colors)],
            marker=label_markers[str(row["true_label"])],
            s=80,
            alpha=0.85,
        )
        ax.text(
            float(row["pca_x"]) + 0.03,
            float(row["pca_y"]) + 0.03,
            str(row["video_id"]),
            fontsize=7,
            alpha=0.75,
        )

    ax.set_title(f"K-means clusters of sampled PTLFlow volumes ({feature_mode}, {grid_mode})")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_single_mode(
    extracted_rows: List[Dict[str, object]],
    feature_mode: str,
    motion_eps: float,
    output_dir: Path,
    k: int,
    random_state: int,
    pca_dim_limit: int,
    grid_mode: str,
) -> Dict[str, object]:
    mode_dir = output_dir / feature_mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    x = np.array(
        [
            build_feature_vector(
                np.array(row["sampled_volume"], dtype=np.float32),
                feature_mode=feature_mode,
                motion_eps=motion_eps,
            )
            for row in extracted_rows
        ],
        dtype=np.float32,
    )
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    pca_dim = min(pca_dim_limit, x_scaled.shape[0], x_scaled.shape[1])
    pca_model = PCA(n_components=pca_dim, random_state=random_state)
    x_pca_for_kmeans = pca_model.fit_transform(x_scaled)

    kmeans = KMeans(n_clusters=k, n_init=50, random_state=random_state)
    cluster_ids = kmeans.fit_predict(x_pca_for_kmeans)

    pca_plot = PCA(n_components=2, random_state=random_state)
    x_plot = pca_plot.fit_transform(x_scaled)

    assignment_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    for idx, row in enumerate(extracted_rows):
        feature_rows.append(
            {
                "video_id": row["video_id"],
                "true_label": row["true_label"],
                "requested_label": row["requested_label"],
                "n_flows": row["n_flows"],
                "feature_mode": feature_mode,
                "grid_mode": grid_mode,
                "feature_dim": x.shape[1],
            }
        )
        assignment_rows.append(
            {
                "video_id": row["video_id"],
                "requested_label": row["requested_label"],
                "true_label": row["true_label"],
                "feature_mode": feature_mode,
                "grid_mode": grid_mode,
                "cluster_id": int(cluster_ids[idx]),
                "pca_x": float(x_plot[idx, 0]),
                "pca_y": float(x_plot[idx, 1]),
            }
        )

    write_csv(feature_rows, mode_dir / "feature_manifest.csv")
    write_csv(assignment_rows, mode_dir / "cluster_assignments.csv")

    crosstab_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in assignment_rows:
        crosstab_counts[str(row["true_label"])][str(row["cluster_id"])] += 1

    cluster_summary_rows = []
    for true_label in ["still", "W", "A", "S", "D"]:
        row: Dict[str, object] = {
            "true_label": true_label,
            "feature_mode": feature_mode,
            "grid_mode": grid_mode,
        }
        for cluster_id in range(k):
            row[f"cluster_{cluster_id}"] = crosstab_counts[true_label].get(str(cluster_id), 0)
        cluster_summary_rows.append(row)
    write_csv(cluster_summary_rows, mode_dir / "cluster_summary.csv")

    majority_summary = cluster_majority_summary(assignment_rows)
    with open(mode_dir / "cluster_report.json", "w") as handle:
        json.dump(
            {
                "config": {
                    "output_dir": str(mode_dir),
                    "feature_mode": feature_mode,
                    "grid_mode": grid_mode,
                    "motion_eps": motion_eps,
                    "k": k,
                    "feature_dim": int(x.shape[1]),
                    "pca_dim": pca_dim,
                },
                "pca_explained_variance_ratio_for_kmeans": pca_model.explained_variance_ratio_.tolist(),
                "pca_explained_variance_ratio_for_plot": pca_plot.explained_variance_ratio_.tolist(),
                "kmeans_inertia": float(kmeans.inertia_),
                "majority_summary": majority_summary,
            },
            handle,
            indent=2,
        )

    plot_pca(
        assignment_rows,
        mode_dir / "pca_clusters.png",
        feature_mode=feature_mode,
        grid_mode=grid_mode,
    )

    return {
        "feature_mode": feature_mode,
        "grid_mode": grid_mode,
        "purity": float(majority_summary["purity"]),
        "feature_dim": int(x.shape[1]),
        "cluster_to_majority": majority_summary["cluster_to_majority"],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_modes = parse_feature_modes(args.feature_modes)
    grid_mode = parse_grid_mode(args.grid_mode)

    sample_groups = sample_video_ids()
    manifest_rows = []
    for requested_label, video_ids in sample_groups.items():
        for video_id in video_ids:
            action_path = args.video_dir / f"{video_id}_action.npy"
            inferred_label = infer_true_label(action_path)
            manifest_rows.append(
                {
                    "video_id": video_id,
                    "requested_label": requested_label,
                    "true_label": inferred_label,
                    "video_path": str(args.video_dir / f"{video_id}.mp4"),
                    "action_path": str(action_path),
                }
            )
    write_csv(manifest_rows, args.output_dir / "sample_manifest.csv")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    print("Using GPUs:", gpu_ids if gpu_ids else "CPU")
    extracted_rows = extract_parallel(
        manifest_rows=manifest_rows,
        model_name=args.model,
        ckpt=args.ckpt,
        gpu_ids=gpu_ids,
        time_steps=args.time_steps,
        grid_size=args.grid_size,
        grid_mode=grid_mode,
        output_dir=args.output_dir,
    )
    mode_results = []
    for feature_mode in feature_modes:
        result = run_single_mode(
            extracted_rows=extracted_rows,
            feature_mode=feature_mode,
            motion_eps=args.motion_eps,
            output_dir=args.output_dir,
            k=args.k,
            random_state=args.random_state,
            pca_dim_limit=args.pca_dim,
            grid_mode=grid_mode,
        )
        mode_results.append(result)

    write_csv(mode_results, args.output_dir / "mode_comparison.csv")
    with open(args.output_dir / "mode_comparison.json", "w") as handle:
        json.dump(
            {
                "config": {
                    "video_dir": str(args.video_dir),
                    "output_dir": str(args.output_dir),
                    "model": args.model,
                    "ckpt": args.ckpt,
                    "gpu_ids": gpu_ids,
                    "k": args.k,
                    "time_steps": args.time_steps,
                    "grid_size": args.grid_size,
                    "grid_mode": grid_mode,
                    "feature_modes": feature_modes,
                    "motion_eps": args.motion_eps,
                    "pca_dim_limit": args.pca_dim,
                },
                "mode_results": mode_results,
            },
            handle,
            indent=2,
        )

    print("\nMode comparison:")
    for result in mode_results:
        print(f"  {result['feature_mode']}: purity={result['purity']:.3f}, feature_dim={result['feature_dim']}")
        for cluster_id, info in result["cluster_to_majority"].items():
            print(f"    cluster {cluster_id}: {info['majority_label']} {info['counts']}")
    print("\nWrote outputs to:", args.output_dir)


if __name__ == "__main__":
    main()
