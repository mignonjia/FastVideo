#!/bin/bash
#SBATCH --job-name=mg-upload-mc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=preprocess_mc_output/upload_%j.out
#SBATCH --error=preprocess_mc_output/upload_%j.err

set -e -x

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  eval "$(/opt/conda/bin/conda shell.bash hook)"
fi
conda activate mhuo-fv

DATA_DIR="/mnt/data/world_model/MC_third_person/preprocessed"
STAGING_DIR="/mnt/data/world_model/MC_third_person/hf_staging"
REPO_ID="mignonjia/mc_third_person"
CHUNK_SIZE="3G"

mkdir -p "$STAGING_DIR"

# Compress each action type into split tar.gz files
for action_dir in "$DATA_DIR"/*/; do
    action_type=$(basename "$action_dir")
    echo "=== Compressing $action_type ==="

    mkdir -p "$STAGING_DIR/$action_type"

    # tar preserves relative paths from DATA_DIR, split into 3GB chunks
    # Output: staging/action_type/action_type.tar.gz.00, .01, ...
    tar -czf - -C "$DATA_DIR" "$action_type" \
        | split -b "$CHUNK_SIZE" -d --suffix-length=3 \
            - "$STAGING_DIR/$action_type/${action_type}.tar.gz."

    echo "  Parts created:"
    ls -lh "$STAGING_DIR/$action_type/"
done

echo "=== All compression done, starting HF upload ==="

python3 - << 'PYEOF'
import os
from huggingface_hub import HfApi

api = HfApi()
repo_id = "mignonjia/mc_third_person"
staging_dir = "/mnt/data/world_model/MC_third_person/hf_staging"

# Create dataset repo if it doesn't exist
try:
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    print(f"Repo {repo_id} ready")
except Exception as e:
    print(f"Repo creation: {e}")

# Upload each part, preserving action_type subfolder structure in the repo
for action_type in sorted(os.listdir(staging_dir)):
    action_dir = os.path.join(staging_dir, action_type)
    if not os.path.isdir(action_dir):
        continue
    parts = sorted(os.listdir(action_dir))
    print(f"\n=== Uploading {action_type}: {len(parts)} parts ===")
    for part in parts:
        local_path = os.path.join(action_dir, part)
        repo_path = f"{action_type}/{part}"
        size_gb = os.path.getsize(local_path) / 1e9
        print(f"  Uploading {part} ({size_gb:.2f} GB) -> {repo_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"  Done: {part}")

print("\nAll uploads complete!")
PYEOF
