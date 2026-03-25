#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


DEFAULT_IGNORE_SUBSTRINGS = (
    ".action_model.",
    "condition_embedder.action_embedder",
    "to_out_prope",
)


def should_ignore(name: str, ignore_substrings: tuple[str, ...]) -> bool:
    return any(substr in name for substr in ignore_substrings)


def filter_keys(keys: list[str], ignore_substrings: tuple[str, ...]) -> list[str]:
    return [key for key in keys if not should_ignore(key, ignore_substrings)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two safetensors files and check whether non-action "
            "parameters are exactly the same."
        )
    )
    parser.add_argument("target", type=Path, help="Target safetensors path")
    parser.add_argument("source", type=Path, help="Source safetensors path")
    parser.add_argument(
        "--ignore-substring",
        action="append",
        default=[],
        help=(
            "Additional substring to ignore. Can be provided multiple times. "
            "Defaults ignore action-related parameter families."
        ),
    )
    parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="How many sample entries to print for each mismatch category.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the full summary JSON.",
    )
    args = parser.parse_args()

    ignore_substrings = DEFAULT_IGNORE_SUBSTRINGS + tuple(args.ignore_substring)

    with safe_open(str(args.target), framework="pt", device="cpu") as target_sf:
        target_keys_all = list(target_sf.keys())
        target_keys = filter_keys(target_keys_all, ignore_substrings)
        target_key_set = set(target_keys)

        with safe_open(str(args.source), framework="pt", device="cpu") as source_sf:
            source_keys_all = list(source_sf.keys())
            source_keys = filter_keys(source_keys_all, ignore_substrings)
            source_key_set = set(source_keys)

            common_keys = sorted(target_key_set & source_key_set)
            only_in_target = sorted(target_key_set - source_key_set)
            only_in_source = sorted(source_key_set - target_key_set)

            exact_matches: list[str] = []
            shape_mismatches: list[dict[str, object]] = []
            value_mismatches: list[dict[str, object]] = []

            for key in common_keys:
                target_shape = tuple(target_sf.get_slice(key).get_shape())
                source_shape = tuple(source_sf.get_slice(key).get_shape())
                if target_shape != source_shape:
                    shape_mismatches.append(
                        {
                            "key": key,
                            "target_shape": list(target_shape),
                            "source_shape": list(source_shape),
                        }
                    )
                    continue

                target_tensor = target_sf.get_tensor(key)
                source_tensor = source_sf.get_tensor(key)

                if torch.equal(target_tensor, source_tensor):
                    exact_matches.append(key)
                    continue

                diff = (target_tensor.to(torch.float32) -
                        source_tensor.to(torch.float32)).abs()
                value_mismatches.append(
                    {
                        "key": key,
                        "dtype_target": str(target_tensor.dtype),
                        "dtype_source": str(source_tensor.dtype),
                        "shape": list(target_shape),
                        "max_abs_diff": float(diff.max().item()),
                        "mean_abs_diff": float(diff.mean().item()),
                        "num_diff": int((target_tensor != source_tensor).sum().item()),
                    }
                )

    summary = {
        "target": str(args.target),
        "source": str(args.source),
        "ignore_substrings": list(ignore_substrings),
        "target_total_keys": len(target_keys_all),
        "source_total_keys": len(source_keys_all),
        "target_non_ignored_keys": len(target_keys),
        "source_non_ignored_keys": len(source_keys),
        "common_non_ignored_keys": len(common_keys),
        "exact_match_keys": len(exact_matches),
        "shape_mismatches": len(shape_mismatches),
        "value_mismatches": len(value_mismatches),
        "only_in_target": len(only_in_target),
        "only_in_source": len(only_in_source),
        "sample_only_in_target": only_in_target[:args.show],
        "sample_only_in_source": only_in_source[:args.show],
        "sample_shape_mismatches": shape_mismatches[:args.show],
        "sample_value_mismatches": value_mismatches[:args.show],
    }

    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(
                {
                    **summary,
                    "all_only_in_target": only_in_target,
                    "all_only_in_source": only_in_source,
                    "all_shape_mismatches": shape_mismatches,
                    "all_value_mismatches": value_mismatches,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
