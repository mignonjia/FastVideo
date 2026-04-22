#!/usr/bin/env python3
"""Generate dummy MatrixGame parquet data for train-speed profiling.

Creates one dataset under <output_dir>/latent_<T>/ for each T in LATENT_SIZES.
Spatial resolution matches the default training config: 480x832 → latent 60x104.
"""
import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from fastvideo.dataset.dataloader.schema import pyarrow_schema_matrixgame

LATENT_SIZES = [20, 40, 80, 120]
NUM_SAMPLES = 64  # enough for many batches
LATENT_C = 16
LATENT_H = 60   # 480 / 8
LATENT_W = 104  # 832 / 8
TEMPORAL_RATIO = 4


def _rand_bytes(shape: tuple) -> bytes:
    return np.random.randn(*shape).astype(np.float32).tobytes()


def make_row(sample_id: int, T: int) -> dict:
    num_video_frames = (T - 1) * TEMPORAL_RATIO + 1
    shape_CTHW = [LATENT_C, T, LATENT_H, LATENT_W]
    shape_action = [num_video_frames]

    return {
        "id": f"sample_{sample_id:04d}_T{T}",
        "vae_latent_bytes": _rand_bytes((LATENT_C, T, LATENT_H, LATENT_W)),
        "vae_latent_shape": shape_CTHW,
        "vae_latent_dtype": "float32",
        "clip_feature_bytes": _rand_bytes((257, 1280)),
        "clip_feature_shape": [257, 1280],
        "clip_feature_dtype": "float32",
        "first_frame_latent_bytes": _rand_bytes(
            (LATENT_C, T, LATENT_H, LATENT_W)),
        "first_frame_latent_shape": shape_CTHW,
        "first_frame_latent_dtype": "float32",
        "mouse_cond_bytes": _rand_bytes((num_video_frames, 2)),
        "mouse_cond_shape": [num_video_frames, 2],
        "mouse_cond_dtype": "float32",
        "keyboard_cond_bytes": _rand_bytes((num_video_frames, 4)),
        "keyboard_cond_shape": [num_video_frames, 4],
        "keyboard_cond_dtype": "float32",
        "pil_image_bytes": b"",
        "pil_image_shape": [],
        "pil_image_dtype": "",
        "file_name": f"dummy_{sample_id:04d}",
        "caption": "",
        "media_type": "video",
        "width": 832,
        "height": 480,
        "num_frames": num_video_frames,
        "duration_sec": num_video_frames / 16.0,
        "fps": 16.0,
    }


def generate(output_dir: str, num_samples: int = NUM_SAMPLES) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for T in LATENT_SIZES:
        parquet_dir = os.path.join(output_dir, f"latent_{T}", "worker_0")
        os.makedirs(parquet_dir, exist_ok=True)
        rows = [make_row(i, T) for i in range(num_samples)]
        table = pa.Table.from_pylist(rows, schema=pyarrow_schema_matrixgame)
        out_path = os.path.join(parquet_dir, "data_chunk_0.parquet")
        pq.write_table(table, out_path)
        print(f"[T={T:3d}] wrote {num_samples} rows → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",
                        default="/tmp/mg_dummy_profiling_data",
                        help="Root dir for dummy datasets")
    parser.add_argument("--num_samples",
                        type=int,
                        default=NUM_SAMPLES,
                        help="Samples per latent-size dataset")
    args = parser.parse_args()
    generate(args.output_dir, args.num_samples)
