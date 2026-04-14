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
    mkdir -p "$STAGING_DIR/$action_type"

    # Skip if staging files already exist (resume support)
    if ls "$STAGING_DIR/$action_type/${action_type}.tar.gz."* &>/dev/null 2>&1; then
        echo "=== Skipping $action_type (staging files already exist) ==="
        ls -lh "$STAGING_DIR/$action_type/"
        continue
    fi

    echo "=== Compressing $action_type ==="

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

# Create/ensure public repo
try:
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.update_repo_visibility(repo_id=repo_id, repo_type="dataset", private=False)
    print(f"Repo {repo_id} ready (public)")
except Exception as e:
    print(f"Repo creation: {e}")

# upload_large_folder batches many files per commit (avoids 128/hr rate limit)
# and automatically skips files already present on the hub (safe to resume).
print(f"\nUploading {staging_dir} -> {repo_id} ...")
api.upload_large_folder(
    folder_path=staging_dir,
    repo_id=repo_id,
    repo_type="dataset",
    ignore_patterns=["map_style_cache/**"],
)

print("\nAll uploads complete!")
PYEOF
