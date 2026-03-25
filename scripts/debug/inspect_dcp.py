"""Inspect a distributed checkpoint (DCP) to see what states it contains."""
import argparse
import os
import pickle
import sys
from collections import defaultdict

import torch
from torch.distributed.checkpoint.metadata import Metadata


def inspect_dcp(dcp_dir: str) -> None:
    metadata_path = os.path.join(dcp_dir, ".metadata")
    if not os.path.exists(metadata_path):
        print(f"ERROR: No .metadata file found in {dcp_dir}")
        sys.exit(1)

    with open(metadata_path, "rb") as f:
        metadata: Metadata = pickle.load(f)

    print(f"DCP directory: {dcp_dir}")
    print(f"Storage files: {len(metadata.storage_data)} shards")
    print()

    # Group keys by top-level state name
    all_keys = sorted(metadata.state_dict_metadata.keys())
    groups: dict[str, list[str]] = defaultdict(list)
    for key in all_keys:
        top = key.split(".")[0]
        groups[top].append(key)

    print(f"Total keys in checkpoint: {len(all_keys)}")
    print()

    for group_name in sorted(groups.keys()):
        keys = groups[group_name]
        print(f"=== {group_name} ({len(keys)} keys) ===")
        for k in keys[:10]:
            meta = metadata.state_dict_metadata[k]
            if hasattr(meta, "size"):
                print(f"  {k}  shape={list(meta.size)}")
            else:
                print(f"  {k}  type={type(meta).__name__}")
        if len(keys) > 10:
            print(f"  ... and {len(keys) - 10} more keys")
        print()

    # Specifically check model keys
    model_keys = groups.get("model", [])
    if model_keys:
        print(f"=== MODEL key analysis ===")
        model_subkeys = [k[len("model."):] for k in model_keys]
        has_blocks = [k for k in model_subkeys if k.startswith("blocks.")]
        has_condition = [
            k for k in model_subkeys
            if k.startswith("condition_embedder.")
        ]
        has_head = [k for k in model_subkeys if k.startswith("head.")]
        has_patch = [
            k for k in model_subkeys if k.startswith("patch_embedding.")
        ]
        has_prope = [k for k in model_subkeys if "to_out_prope" in k]
        has_action = [k for k in model_subkeys if "action" in k.lower()]
        print(f"  blocks.* params:             {len(has_blocks)}")
        print(f"  condition_embedder.* params:  {len(has_condition)}")
        print(f"  head.* params:                {len(has_head)}")
        print(f"  patch_embedding.* params:     {len(has_patch)}")
        print(f"  *to_out_prope* params:        {len(has_prope)}")
        print(f"  *action* params:              {len(has_action)}")
        print()
        print("All model keys:")
        for k in model_subkeys:
            meta = metadata.state_dict_metadata[f"model.{k}"]
            if hasattr(meta, "size"):
                print(f"  {k}  shape={list(meta.size)}")
            else:
                print(f"  {k}  type={type(meta).__name__}")
    else:
        print("WARNING: No 'model' keys found in checkpoint!")
        print("Available top-level groups:", list(groups.keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dcp_dir", help="Path to distributed_checkpoint dir")
    args = parser.parse_args()
    inspect_dcp(args.dcp_dir)
