from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_DATA_DIR = Path(
    "/mnt/weka/home/hao.zhang/mhuo/traindata_0204_1600/preprocessed"
)
BLOB_KEYS = [
    "vae_latent",
    "clip_feature",
    "first_frame_latent",
    "mouse_cond",
    "keyboard_cond",
]
MOUSE_COLUMNS = [
    "mouse_cond_bytes",
    "mouse_cond_shape",
    "mouse_cond_dtype",
]


def decode_blob(row: pd.Series, prefix: str) -> Optional[np.ndarray]:
    blob = row.get(f"{prefix}_bytes")
    shape = row.get(f"{prefix}_shape")
    dtype = row.get(f"{prefix}_dtype")

    if blob is None or len(blob) == 0 or shape is None or dtype is None:
        return None

    arr = np.frombuffer(blob, dtype=np.dtype(dtype))
    return arr.reshape(tuple(shape))


def print_array_stats(name: str, arr: Optional[np.ndarray]) -> None:
    if arr is None:
        print(f"{name}: empty")
        return

    if np.issubdtype(arr.dtype, np.number):
        print(
            f"{name}: shape={arr.shape}, dtype={arr.dtype}, "
            f"min={arr.min():.4f}, max={arr.max():.4f}"
        )
    else:
        print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")


def is_tail_consistent(
    arr: np.ndarray,
    atol: float = 0.0,
) -> bool:
    if arr.ndim == 0:
        return False

    tail = arr[1:]
    if tail.shape[0] <= 1:
        return True

    ref = np.broadcast_to(tail[0], tail.shape)
    if atol > 0:
        return bool(np.allclose(tail, ref, atol=atol, rtol=0.0, equal_nan=True))
    return bool(np.array_equal(tail, ref))


def count_mouse_consistent_entries(
    data_dir: Path,
    pattern: str,
    atol: float = 0.0,
) -> dict[str, int]:
    parquet_files = sorted(data_dir.rglob(pattern))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    total_entries = 0
    decodable_entries = 0
    consistent_entries = 0
    strict_total_entries = 0
    strict_consistent_entries = 0
    short_tail_entries = 0
    missing_entries = 0
    decode_failed_entries = 0

    print(f"Scanning {len(parquet_files)} parquet files under {data_dir}")
    for idx, parquet_path in enumerate(parquet_files, start=1):
        pf = pq.ParquetFile(parquet_path)
        for row_group_idx in range(pf.num_row_groups):
            table = pf.read_row_group(row_group_idx, columns=MOUSE_COLUMNS)
            data = table.to_pydict()
            blobs = data["mouse_cond_bytes"]
            shapes = data["mouse_cond_shape"]
            dtypes = data["mouse_cond_dtype"]

            for blob, shape, dtype in zip(blobs, shapes, dtypes):
                total_entries += 1
                if (
                    blob is None
                    or len(blob) == 0
                    or shape is None
                    or dtype is None
                ):
                    missing_entries += 1
                    continue

                try:
                    mouse = np.frombuffer(blob, dtype=np.dtype(dtype)).reshape(
                        tuple(shape)
                    )
                except Exception:
                    decode_failed_entries += 1
                    continue

                decodable_entries += 1
                tail_len = max(int(mouse.shape[0]) - 1, 0)
                if tail_len <= 1:
                    short_tail_entries += 1
                else:
                    strict_total_entries += 1

                if is_tail_consistent(mouse, atol=atol):
                    consistent_entries += 1
                    if tail_len > 1:
                        strict_consistent_entries += 1

        print(
            f"[{idx:>3}/{len(parquet_files)}] scanned {parquet_path.name} "
            f"(running total entries={total_entries})"
        )

    return {
        "parquet_files": len(parquet_files),
        "total_entries": total_entries,
        "decodable_entries": decodable_entries,
        "consistent_entries": consistent_entries,
        "strict_total_entries": strict_total_entries,
        "strict_consistent_entries": strict_consistent_entries,
        "short_tail_entries": short_tail_entries,
        "missing_entries": missing_entries,
        "decode_failed_entries": decode_failed_entries,
    }


def inspect_dataframe(
    df: pd.DataFrame,
    row_idx: int,
) -> None:
    if len(df) == 0:
        print("Selected parquet has no rows.")
        return
    if row_idx < 0 or row_idx >= len(df):
        raise IndexError(f"row_idx={row_idx} out of range [0, {len(df) - 1}]")

    row = df.iloc[row_idx]
    print(f"\nInspecting row index: {row_idx}")
    for key in BLOB_KEYS:
        print_array_stats(key, decode_blob(row, key))

    mouse = decode_blob(row, "mouse_cond")
    keyboard = decode_blob(row, "keyboard_cond")

    if mouse is not None:
        print("\nmouse_cond (first 12 rows):")
        print(np.array2string(mouse[:12], precision=3, suppress_small=True))
        print("mouse unique values:", np.unique(np.round(mouse, 3)))
        if mouse.ndim == 2 and mouse.shape[1] == 2:
            mouse_df = pd.DataFrame(mouse, columns=["mouse_dx", "mouse_dy"])
            print(mouse_df.head(20).to_string(index=False))
        else:
            print(pd.DataFrame(mouse).head(20).to_string(index=False))

    if keyboard is not None:
        print("\nkeyboard_cond (first 12 rows):")
        print(np.array2string(keyboard[:12], precision=3, suppress_small=True))
        print("keyboard unique values:", np.unique(np.round(keyboard, 3)))
        key_cols = ["key_w", "key_a", "key_s", "key_d", "mouse_l", "mouse_r"]
        if keyboard.ndim == 2 and keyboard.shape[1] == len(key_cols):
            keyboard_df = pd.DataFrame(keyboard, columns=key_cols)
            print(keyboard_df.head(20).to_string(index=False))
        else:
            print(pd.DataFrame(keyboard).head(20).to_string(index=False))


def resolve_parquet_path(
    data_dir: Path,
    file_idx: int,
    pattern: str,
) -> Path:
    parquet_files = sorted(data_dir.rglob(pattern))
    print(f"Found {len(parquet_files)} parquet files under {data_dir}")
    print("First 5 files:")
    for path in parquet_files[:5]:
        print(" -", path)

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    if file_idx < 0 or file_idx >= len(parquet_files):
        raise IndexError(
            f"file_idx={file_idx} out of range [0, {len(parquet_files) - 1}]"
        )
    return parquet_files[file_idx]


def inspect_parquet_file(
    parquet_path: Path,
    row_idx: int,
    head_rows: int,
) -> None:
    pf = pq.ParquetFile(parquet_path)
    print("\nFILE:", parquet_path)
    print("ROWS:", pf.metadata.num_rows)
    print("ROW_GROUPS:", pf.num_row_groups)
    print("SCHEMA:")
    print(pf.schema)

    df = pq.read_table(parquet_path).to_pandas()
    with pd.option_context("display.max_columns", 50, "display.width", 200):
        print("\nDataFrame preview:")
        print(df.head(head_rows).to_string(index=False))

    inspect_dataframe(df, row_idx=row_idx)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect parquet shards and decode blob fields."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory containing parquet files.",
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=None,
        help="Direct path to a parquet file (overrides --data-dir/--file-idx).",
    )
    parser.add_argument(
        "--file-idx",
        type=int,
        default=0,
        help="Index within sorted parquet files under --data-dir.",
    )
    parser.add_argument(
        "--row-idx",
        type=int,
        default=0,
        help="Row index to decode.",
    )
    parser.add_argument(
        "--head-rows",
        type=int,
        default=3,
        help="Number of rows to print in DataFrame preview.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.parquet",
        help="Filename pattern for parquet discovery under --data-dir.",
    )
    parser.add_argument(
        "--count-mouse-consistent",
        action="store_true",
        help=(
            "Count entries across all matching parquet files where "
            "mouse_cond[1:] is consistent."
        ),
    )
    parser.add_argument(
        "--consistency-atol",
        type=float,
        default=0.0,
        help=(
            "Absolute tolerance used when comparing mouse_cond rows "
            "for consistency."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count_mouse_consistent:
        counts = count_mouse_consistent_entries(
            data_dir=args.data_dir,
            pattern=args.pattern,
            atol=args.consistency_atol,
        )
        print("\nMouse consistency summary (ignoring index 0):")
        print(f"  parquet_files: {counts['parquet_files']}")
        print(f"  total_entries: {counts['total_entries']}")
        print(f"  decodable_entries: {counts['decodable_entries']}")
        print(f"  consistent_entries: {counts['consistent_entries']}")
        print(
            "  strict_consistent_entries (tail length > 1): "
            f"{counts['strict_consistent_entries']}"
        )
        print(
            "  strict_total_entries (tail length > 1): "
            f"{counts['strict_total_entries']}"
        )
        print(
            "  short_tail_entries (tail length <= 1): "
            f"{counts['short_tail_entries']}"
        )
        print(f"  missing_entries: {counts['missing_entries']}")
        print(f"  decode_failed_entries: {counts['decode_failed_entries']}")
        return

    if args.parquet_path is not None:
        parquet_path = args.parquet_path
    else:
        parquet_path = resolve_parquet_path(
            data_dir=args.data_dir,
            file_idx=args.file_idx,
            pattern=args.pattern,
        )
    inspect_parquet_file(
        parquet_path=parquet_path,
        row_idx=args.row_idx,
        head_rows=args.head_rows,
    )


if __name__ == "__main__":
    main()
