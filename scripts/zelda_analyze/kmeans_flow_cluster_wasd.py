#!/usr/bin/env python3
"""Cluster WanGame videos by PTLFlow-derived motion descriptors.

This script samples 10 videos from each of five known buckets in
`1_wasd_only/videos`:
- 000500-000509: A
- 001500-001509: D
- 002500-002509: S
- 003500-003509: still
- 004500-004509: W

For each video, it computes PTLFlow on consecutive frame pairs and reduces the
dense flow into a compact descriptor intended for clustering:
- global signed flow means
- mean absolute flow components
- motion magnitude/activity statistics
- signed radial/tangential components
- magnitude-weighted angle histogram

The script then standardizes features, runs K-means with k=5, saves cluster
assignments and summaries, and produces a PCA plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, Iterable, List

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
DEFAULT_OUTPUT_DIR = ROOT / "analyze" / "outputs" / "kmeans_flow_cluster_wasd"
DEFAULT_CKPT = ROOT / "dpflow-things-2012b5d6.ckpt"
ANGLE_BINS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample WanGame videos and cluster them by PTLFlow descriptors."
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default="dpflow")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--min-mag", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="",
        help="Comma-separated CUDA device ids to use. Empty means all visible GPUs.",
    )
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


def flow_descriptor(
    flows: Iterable[np.ndarray],
    min_mag: float,
) -> Dict[str, float]:
    flow_list = list(flows)
    if not flow_list:
        raise ValueError("Expected at least one flow field.")

    h, w = flow_list[0].shape[:2]
    xs = np.arange(w, dtype=np.float32) - (w / 2.0)
    ys = np.arange(h, dtype=np.float32) - (h / 2.0)
    x_grid, y_grid = np.meshgrid(xs, ys)
    radius = np.sqrt(x_grid ** 2 + y_grid ** 2) + 1e-6

    scalar_lists: Dict[str, List[float]] = defaultdict(list)
    hist_accum = np.zeros(ANGLE_BINS, dtype=np.float64)
    hist_weight = 0.0

    for flow in flow_list:
        u = flow[:, :, 0]
        v = flow[:, :, 1]
        mag = np.sqrt(u ** 2 + v ** 2)
        valid = mag >= min_mag

        scalar_lists["active_frac"].append(float(valid.mean()))
        if valid.sum() == 0:
            for key in (
                "mean_u",
                "mean_v",
                "mean_abs_u",
                "mean_abs_v",
                "mean_mag",
                "std_mag",
                "radial_mean",
                "tangential_mean",
            ):
                scalar_lists[key].append(0.0)
            continue

        u_valid = u[valid]
        v_valid = v[valid]
        mag_valid = mag[valid]
        x_valid = x_grid[valid]
        y_valid = y_grid[valid]
        r_valid = radius[valid]

        scalar_lists["mean_u"].append(float(u_valid.mean()))
        scalar_lists["mean_v"].append(float(v_valid.mean()))
        scalar_lists["mean_abs_u"].append(float(np.abs(u_valid).mean()))
        scalar_lists["mean_abs_v"].append(float(np.abs(v_valid).mean()))
        scalar_lists["mean_mag"].append(float(mag_valid.mean()))
        scalar_lists["std_mag"].append(float(mag_valid.std()))

        radial = (u_valid * x_valid + v_valid * y_valid) / r_valid
        tangential = (-u_valid * y_valid + v_valid * x_valid) / r_valid
        scalar_lists["radial_mean"].append(float(radial.mean()))
        scalar_lists["tangential_mean"].append(float(tangential.mean()))

        angles = (np.arctan2(v_valid, u_valid) + 2 * np.pi) % (2 * np.pi)
        hist, _ = np.histogram(
            angles,
            bins=ANGLE_BINS,
            range=(0.0, 2 * np.pi),
            weights=mag_valid,
        )
        hist_accum += hist
        hist_weight += float(mag_valid.sum())

    features: Dict[str, float] = {}
    for key, values in scalar_lists.items():
        features[key] = float(np.mean(values))

    if hist_weight > 1e-6:
        hist_norm = hist_accum / hist_weight
    else:
        hist_norm = hist_accum
    for idx, value in enumerate(hist_norm):
        features[f"angle_hist_{idx}"] = float(value)

    return features


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_gpu_ids(gpu_ids_arg: str) -> List[int]:
    if gpu_ids_arg.strip():
        return [int(x) for x in gpu_ids_arg.split(",") if x.strip()]
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def extract_features_chunk(
    manifest_rows: List[Dict[str, object]],
    model_name: str,
    ckpt: str,
    min_mag: float,
    gpu_id: int | None,
) -> List[Dict[str, object]]:
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)

    model = ptlflow.get_model(model_name, ckpt)
    model.eval()
    if gpu_id is not None and torch.cuda.is_available():
        model = model.cuda(gpu_id)

    feature_rows: List[Dict[str, object]] = []
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

        descriptor = flow_descriptor(flows, min_mag=min_mag)
        row: Dict[str, object] = dict(item)
        row.update(descriptor)
        feature_rows.append(row)

    return feature_rows


def worker_main(
    manifest_rows: List[Dict[str, object]],
    model_name: str,
    ckpt: str,
    min_mag: float,
    gpu_id: int,
    out_path: str,
) -> None:
    try:
        rows = extract_features_chunk(
            manifest_rows=manifest_rows,
            model_name=model_name,
            ckpt=ckpt,
            min_mag=min_mag,
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


def extract_features_parallel(
    manifest_rows: List[Dict[str, object]],
    model_name: str,
    ckpt: str,
    min_mag: float,
    gpu_ids: List[int],
    output_dir: Path,
) -> List[Dict[str, object]]:
    if len(gpu_ids) <= 1:
        gpu_id = gpu_ids[0] if gpu_ids else None
        return extract_features_chunk(manifest_rows, model_name, ckpt, min_mag, gpu_id)

    shards = [[] for _ in gpu_ids]
    for idx, row in enumerate(manifest_rows):
        shards[idx % len(gpu_ids)].append(row)

    ctx = get_context("spawn")
    processes = []
    shard_paths = []
    for shard_idx, (gpu_id, shard_rows) in enumerate(zip(gpu_ids, shards)):
        if not shard_rows:
            continue
        shard_path = output_dir / f"tmp_worker_{shard_idx}_gpu_{gpu_id}.json"
        shard_paths.append(shard_path)
        proc = ctx.Process(
            target=worker_main,
            args=(shard_rows, model_name, ckpt, min_mag, gpu_id, str(shard_path)),
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


def plot_pca(rows: List[Dict[str, object]], output_path: Path) -> None:
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

    ax.set_title("K-means clusters of PTLFlow video descriptors")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    feature_rows = extract_features_parallel(
        manifest_rows=manifest_rows,
        model_name=args.model,
        ckpt=args.ckpt,
        min_mag=args.min_mag,
        gpu_ids=gpu_ids,
        output_dir=args.output_dir,
    )

    feature_path = args.output_dir / "video_features.csv"
    write_csv(feature_rows, feature_path)

    feature_keys = [
        "mean_u",
        "mean_v",
        "mean_abs_u",
        "mean_abs_v",
        "mean_mag",
        "std_mag",
        "active_frac",
        "radial_mean",
        "tangential_mean",
    ] + [f"angle_hist_{idx}" for idx in range(ANGLE_BINS)]

    x = np.array([[float(row[key]) for key in feature_keys] for row in feature_rows], dtype=np.float64)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    kmeans = KMeans(n_clusters=args.k, n_init=50, random_state=args.random_state)
    cluster_ids = kmeans.fit_predict(x_scaled)

    pca = PCA(n_components=2, random_state=args.random_state)
    x_pca = pca.fit_transform(x_scaled)

    assignment_rows: List[Dict[str, object]] = []
    for idx, row in enumerate(feature_rows):
        assignment = dict(row)
        assignment["cluster_id"] = int(cluster_ids[idx])
        assignment["pca_x"] = float(x_pca[idx, 0])
        assignment["pca_y"] = float(x_pca[idx, 1])
        assignment_rows.append(assignment)

    assignment_path = args.output_dir / "cluster_assignments.csv"
    write_csv(assignment_rows, assignment_path)

    # Cross-tab summary
    crosstab_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in assignment_rows:
        crosstab_counts[str(row["true_label"])][str(row["cluster_id"])] += 1

    cluster_summary_rows = []
    for true_label in ["still", "W", "A", "S", "D"]:
        row: Dict[str, object] = {"true_label": true_label}
        for cluster_id in range(args.k):
            row[f"cluster_{cluster_id}"] = crosstab_counts[true_label].get(str(cluster_id), 0)
        cluster_summary_rows.append(row)
    write_csv(cluster_summary_rows, args.output_dir / "cluster_summary.csv")

    majority_summary = cluster_majority_summary(assignment_rows)

    payload = {
        "config": {
            "video_dir": str(args.video_dir),
            "output_dir": str(args.output_dir),
            "model": args.model,
            "ckpt": args.ckpt,
            "gpu_ids": gpu_ids,
            "k": args.k,
            "min_mag": args.min_mag,
            "feature_keys": feature_keys,
        },
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "kmeans_inertia": float(kmeans.inertia_),
        "cluster_centers_scaled": kmeans.cluster_centers_.tolist(),
        "majority_summary": majority_summary,
    }
    with open(args.output_dir / "cluster_report.json", "w") as handle:
        json.dump(payload, handle, indent=2)

    plot_pca(assignment_rows, args.output_dir / "pca_clusters.png")

    print("\nCluster purity:", f"{majority_summary['purity']:.3f}")
    print("Cluster majority labels:")
    for cluster_id, info in majority_summary["cluster_to_majority"].items():
        print(f"  cluster {cluster_id}: {info['majority_label']} {info['counts']}")
    print("\nWrote outputs to:", args.output_dir)


if __name__ == "__main__":
    main()
