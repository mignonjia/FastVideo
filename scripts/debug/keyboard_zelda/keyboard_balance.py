# SPDX-License-Identifier: Apache-2.0
from collections import Counter
from typing import Any

import numpy as np
import pyarrow.parquet as pq

KEYBOARD_BALANCE_CLASS_ORDER_1DP: tuple[tuple[float, ...], ...] = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    (1.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
)


def _round_keyboard_value_1dp(v: np.ndarray) -> tuple[float, ...]:
    return tuple(round(float(x), 1) for x in np.asarray(v).reshape(-1))


def _count_tail_transitions(tail: np.ndarray) -> int:
    n_rows = int(tail.shape[0])
    if n_rows <= 1:
        return 0
    changes = 0
    for i in range(1, n_rows):
        if not np.array_equal(tail[i], tail[i - 1]):
            changes += 1
    return changes


def _majority_keyboard_value_1dp(tail: np.ndarray) -> tuple[float, ...]:
    keys = [_round_keyboard_value_1dp(tail[i]) for i in range(int(tail.shape[0]))]
    counts = Counter(keys)
    max_count = max(counts.values())
    candidates = [k for k, v in counts.items() if v == max_count]

    # Tie-breaker: prefer final value for temporal consistency.
    final_key = keys[-1]
    if final_key in candidates:
        return final_key
    return sorted(candidates)[0]


def build_keyboard_majority_balanced_indices(
    parquet_files: list[str],
    lengths: list[int],
    *,
    max_transitions: int = 2,
    target_per_class: int = 1000,
    seed: int = 42,
    class_order: tuple[tuple[float, ...], ...] = KEYBOARD_BALANCE_CLASS_ORDER_1DP,
) -> tuple[list[int], dict[str, Any]]:
    """
    Build global row indices for training-time postprocess:
      1) keep samples whose keyboard_cond[1:] transitions <= max_transitions
      2) label each sample by majority value in keyboard_cond[1:] (rounded 1dp)
      3) sample each class in class_order to exactly target_per_class.
    """
    class_samples: dict[tuple[float, ...], list[int]] = {
        label: []
        for label in class_order
    }
    skipped_unseen_label = 0
    selected_by_transition = 0
    global_offset = 0

    keyboard_cols = [
        "keyboard_cond_bytes",
        "keyboard_cond_shape",
        "keyboard_cond_dtype",
    ]

    for file_path, file_len in zip(parquet_files, lengths, strict=True):
        parquet_file = pq.ParquetFile(file_path)
        local_offset = 0
        for row_group_idx in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(
                row_group_idx, columns=keyboard_cols).to_pydict()
            n_rows = len(row_group["keyboard_cond_bytes"])
            for i, (blob, shape, dtype) in enumerate(
                    zip(row_group["keyboard_cond_bytes"],
                        row_group["keyboard_cond_shape"],
                        row_group["keyboard_cond_dtype"],
                        strict=True)):
                if (blob is None or len(blob) == 0 or shape is None
                        or dtype is None):
                    continue
                try:
                    keyboard = np.frombuffer(blob,
                                             dtype=np.dtype(dtype)).reshape(
                                                 tuple(shape))
                except Exception:
                    continue

                tail = keyboard[1:]
                if int(tail.shape[0]) == 0:
                    continue
                if _count_tail_transitions(tail) > max_transitions:
                    continue

                selected_by_transition += 1
                label = _majority_keyboard_value_1dp(tail)
                if label not in class_samples:
                    skipped_unseen_label += 1
                    continue
                global_idx = global_offset + local_offset + i
                class_samples[label].append(global_idx)

            local_offset += n_rows
        global_offset += file_len

    before_counts = {label: len(samples) for label, samples in class_samples.items()}
    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    after_counts: dict[tuple[float, ...], int] = {}

    for label in class_order:
        samples = class_samples[label]
        if len(samples) == 0:
            raise ValueError(
                f"No samples found for label {label}. "
                "Cannot build balanced indices.")

        replace = len(samples) < target_per_class
        picked_idx = rng.choice(len(samples),
                                size=target_per_class,
                                replace=replace)
        picked = [samples[int(i)] for i in picked_idx]
        selected_indices.extend(picked)
        after_counts[label] = len(picked)

    # Shuffle final list so class blocks are mixed.
    perm = rng.permutation(len(selected_indices))
    selected_indices = [selected_indices[int(i)] for i in perm]

    stats = {
        "selected_by_transition": selected_by_transition,
        "skipped_unseen_label": skipped_unseen_label,
        "before_counts":
        {str(list(k)): v
         for k, v in before_counts.items()},
        "after_counts":
        {str(list(k)): v
         for k, v in after_counts.items()},
        "total_selected": len(selected_indices),
    }
    return selected_indices, stats
