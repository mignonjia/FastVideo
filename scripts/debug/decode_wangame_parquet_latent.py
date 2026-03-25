#!/usr/bin/env python3
"""Decode one Wangame parquet latent row into an mp4."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import imageio
import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision
from einops import rearrange

from fastvideo.configs.pipelines import WanGameI2V480PConfig
from fastvideo.pipelines.basic.wan.wangame_i2v_pipeline import (
    WanGameActionImageToVideoPipeline,
)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.utils import maybe_download_model


DEFAULT_MODEL_ID = "weizhou03/Wan2.1-Game-Fun-1.3B-InP-Diffusers"
DEFAULT_CACHE_SNAPSHOT = (
    "/mnt/weka/home/hao.zhang/.cache/huggingface/hub/"
    "models--weizhou03--Wan2.1-Game-Fun-1.3B-InP-Diffusers/snapshots/"
    "646c2c907816063473b6238f3dad5a971d353be3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode one Wangame latent row from parquet into a video."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/mnt/weka/home/hao.zhang/alex/wm-lab/datas/datasets/zeldam2-v2"
        ),
        help="Root directory to search recursively for parquet shards.",
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        help="Optional explicit parquet shard path. Overrides --dataset-root.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        help="Optional row index inside the chosen parquet shard.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used when choosing parquet shard and row.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=(
            "Wangame model root or HF repo id. If a root path is provided, "
            "the script will use its vae/ subdirectory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("decoded_latent_samples"),
        help="Directory for generated mp4 files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Optional explicit output mp4 path.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional fps override. Defaults to parquet fps or 25.",
    )
    parser.add_argument(
        "--latent-format",
        choices=["parquet_raw", "pipeline_normalized"],
        default="parquet_raw",
        help=(
            "Latent space of the input tensor. "
            "'parquet_raw' matches stored preprocessing latents "
            "(vae.encode(...).mean). "
            "'pipeline_normalized' matches model/pipeline latent space "
            "used before DecodingStage denormalization."
        ),
    )
    parser.add_argument(
        "--overlay-actions",
        action="store_true",
        help=(
            "Overlay keyboard and mouse action indicators using the same "
            "drawing functions as Wangame validation."
        ),
    )
    return parser.parse_args()


def resolve_model_root(model_path: str) -> Path:
    if model_path == DEFAULT_MODEL_ID and Path(DEFAULT_CACHE_SNAPSHOT).is_dir():
        resolved = Path(DEFAULT_CACHE_SNAPSHOT)
    else:
        resolved = Path(maybe_download_model(model_path))

    if resolved.name == "vae":
        resolved = resolved.parent

    vae_path = resolved / "vae"
    if not vae_path.is_dir():
        raise FileNotFoundError(f"Could not find vae directory under {resolved}")
    return resolved


def choose_parquet(args: argparse.Namespace, rng: random.Random) -> Path:
    if args.parquet_path is not None:
        if not args.parquet_path.is_file():
            raise FileNotFoundError(args.parquet_path)
        return args.parquet_path

    parquet_files = sorted(args.dataset_root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found under {args.dataset_root}"
        )
    return rng.choice(parquet_files)


def choose_row_index(
    parquet_path: Path, row_index: int | None, rng: random.Random
) -> int:
    parquet_file = pq.ParquetFile(parquet_path)
    num_rows = parquet_file.metadata.num_rows
    if num_rows <= 0:
        raise ValueError(f"No rows in parquet file: {parquet_path}")
    if row_index is None:
        return rng.randrange(num_rows)
    if row_index < 0 or row_index >= num_rows:
        raise IndexError(
            f"row_index={row_index} is out of range for {parquet_path} "
            f"with {num_rows} rows"
        )
    return row_index


def load_latent_row(parquet_path: Path, row_index: int) -> dict[str, object]:
    table = pq.read_table(
        parquet_path,
        columns=[
            "id",
            "file_name",
            "vae_latent_bytes",
            "vae_latent_shape",
            "vae_latent_dtype",
            "fps",
            "width",
            "height",
            "num_frames",
        ],
    )
    row = {
        "id": table["id"][row_index].as_py(),
        "file_name": table["file_name"][row_index].as_py(),
        "vae_latent_shape": table["vae_latent_shape"][row_index].as_py(),
        "vae_latent_dtype": table["vae_latent_dtype"][row_index].as_py(),
        "fps": table["fps"][row_index].as_py(),
        "width": table["width"][row_index].as_py(),
        "height": table["height"][row_index].as_py(),
        "num_frames": table["num_frames"][row_index].as_py(),
    }
    latent = np.frombuffer(
        table["vae_latent_bytes"][row_index].as_py(),
        dtype=np.dtype(row["vae_latent_dtype"]),
    ).reshape(row["vae_latent_shape"]).copy()
    row["latent"] = latent
    return row


def load_action_row(
    parquet_path: Path, row_index: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
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
        keyboard_shape = table["keyboard_cond_shape"][row_index].as_py()
        keyboard_dtype = table["keyboard_cond_dtype"][row_index].as_py()
        keyboard = np.frombuffer(
            keyboard_bytes, dtype=np.dtype(keyboard_dtype)
        ).reshape(keyboard_shape).copy()

    mouse_bytes = table["mouse_cond_bytes"][row_index].as_py()
    if mouse_bytes is not None:
        mouse_shape = table["mouse_cond_shape"][row_index].as_py()
        mouse_dtype = table["mouse_cond_dtype"][row_index].as_py()
        mouse = np.frombuffer(
            mouse_bytes, dtype=np.dtype(mouse_dtype)
        ).reshape(mouse_shape).copy()

    return keyboard, mouse


def decode_latent(
    latent: np.ndarray, model_root: Path, latent_format: str
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Wangame VAE decode script.")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")

    fastvideo_args = FastVideoArgs(
        model_path=str(model_root),
        pipeline_config=WanGameI2V480PConfig(),
        inference_mode=True,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        image_encoder_cpu_offload=True,
        pin_cpu_memory=False,
    )
    pipeline = WanGameActionImageToVideoPipeline(
        str(model_root), fastvideo_args
    )
    pipeline.post_init()
    assert isinstance(pipeline.fastvideo_args, FastVideoArgs)
    args = pipeline.fastvideo_args
    vae = pipeline.get_module("vae")
    device = next(vae.parameters()).device
    latents = torch.from_numpy(latent).unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    with torch.no_grad():
        if latent_format == "parquet_raw":
            # Preprocessed parquet stores raw encoder mean latents.
            # Decode them directly through the VAE without pipeline denorm.
            samples = vae.decode(latents)
            samples = (samples / 2 + 0.5).clamp(0, 1)
        elif latent_format == "pipeline_normalized":
            batch = ForwardBatch(data_type="video", latents=latents)
            output_batch = pipeline.decoding_stage.forward(batch, args)
            assert output_batch.output is not None
            samples = output_batch.output
        else:
            raise ValueError(f"Unsupported latent_format: {latent_format}")

    return samples.to(torch.float32).cpu()


def samples_to_video_generator_frames(samples: torch.Tensor) -> list[np.ndarray]:
    """Mirror VideoGenerator's frame packing for a single decoded sample."""
    videos = rearrange(samples, "b c t h w -> t b c h w")
    frames: list[np.ndarray] = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=6)
        x = x.permute(1, 2, 0).squeeze(-1)
        x = (x * 255).to(torch.uint8)
        frames.append(x.cpu().numpy())
    return frames


def apply_action_overlay(
    frames: list[np.ndarray],
    keyboard_cond: np.ndarray | None,
    mouse_cond: np.ndarray | None,
) -> list[np.ndarray]:
    from fastvideo.models.dits.matrixgame.utils import (
        draw_keys_on_frame,
        draw_mouse_on_frame,
    )

    key_names = ["W", "S", "A", "D", "left", "right"]
    overlaid_frames: list[np.ndarray] = []
    for frame_idx, frame in enumerate(frames):
        frame = np.ascontiguousarray(frame.copy())

        if keyboard_cond is not None and frame_idx < len(keyboard_cond):
            keys = {
                key_names[i]: bool(keyboard_cond[frame_idx, i])
                for i in range(min(len(key_names), keyboard_cond.shape[1]))
            }
            draw_keys_on_frame(frame, keys, mode="universal")

        if mouse_cond is not None and frame_idx < len(mouse_cond):
            pitch = float(mouse_cond[frame_idx, 0])
            yaw = float(mouse_cond[frame_idx, 1])
            draw_mouse_on_frame(frame, pitch, yaw)

        overlaid_frames.append(frame)

    return overlaid_frames


def build_output_path(
    args: argparse.Namespace, sample_id: str, parquet_path: Path, row_index: int
) -> Path:
    if args.output_path is not None:
        return args.output_path
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_overlay" if args.overlay_actions else ""
    return args.output_dir / (
        f"wangame_{sample_id}_{parquet_path.stem}_row{row_index}{suffix}.mp4"
    )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    parquet_path = choose_parquet(args, rng)
    row_index = choose_row_index(parquet_path, args.row_index, rng)
    row = load_latent_row(parquet_path, row_index)
    model_root = resolve_model_root(args.model_path)
    output_path = build_output_path(
        args, str(row["id"]), parquet_path, row_index
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fps = args.fps or row["fps"] or 25.0
    samples = decode_latent(row["latent"], model_root, args.latent_format)
    frames = samples_to_video_generator_frames(samples)
    if args.overlay_actions:
        keyboard_cond, mouse_cond = load_action_row(parquet_path, row_index)
        frames = apply_action_overlay(frames, keyboard_cond, mouse_cond)
    imageio.mimsave(output_path, frames, fps=fps, format="mp4")

    print(f"Saved decoded video to: {output_path}")
    print(f"Parquet shard: {parquet_path}")
    print(f"Row index: {row_index}")
    print(f"Sample id: {row['id']}")
    print(f"File name: {row['file_name']}")
    print(f"Latent shape: {row['vae_latent_shape']}")
    print(f"Decoded tensor shape: {tuple(samples.shape)}")
    print(f"Frame count: {len(frames)}")
    print(f"FPS: {fps}")
    print(f"Latent format: {args.latent_format}")
    print(f"Overlay actions: {args.overlay_actions}")
    print(f"Model root: {model_root}")


if __name__ == "__main__":
    main()
