#!/bin/bash
#SBATCH --job-name=mg-preprocess-mc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --exclusive
#SBATCH --time=72:00:00
#SBATCH --output=preprocess_mc_output/mc_%j.out
#SBATCH --error=preprocess_mc_output/mc_%j.err

set -e -x

# Environment Setup
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  eval "$(/opt/conda/bin/conda shell.bash hook)"
fi
conda activate mhuo-fv

export PYTHONPATH="/mnt/home/mhuo/FastVideo-MG:${PYTHONPATH:-}"

# NODE_ID passed as first argument (0-3)
NODE_ID=${1:?NODE_ID required as first argument}
NUM_NODES=4
NUM_GPUS=8
NUM_TOTAL_GPUS=$((NUM_NODES * NUM_GPUS))  # 32

MODEL_PATH="mignonjia/mg_bidirectional_zelda"
DATA_BASE="/mnt/data/world_model/MC_third_person/processed"
OUTPUT_BASE="/mnt/data/world_model/MC_third_person/preprocessed"

mkdir -p preprocess_mc_output

# Collect all merge files across all action types, sorted
mapfile -t ALL_MERGE_FILES < <(find "$DATA_BASE" -name "merge_*.txt" | sort)
TOTAL_FILES=${#ALL_MERGE_FILES[@]}
echo "Node $NODE_ID: found $TOTAL_FILES total merge files"

base_port=$((29500 + NODE_ID * 100))

# Launch 8 GPU workers in parallel; each GPU handles a strided slice of all files
for i in $(seq 0 $((NUM_GPUS - 1))); do
    gpu=$i
    port=$((base_port + i + 1))
    # Global GPU index across all nodes
    global_gpu_idx=$((NODE_ID * NUM_GPUS + i))

    (
        file_idx=$global_gpu_idx
        while [ $file_idx -lt $TOTAL_FILES ]; do
            merge_file="${ALL_MERGE_FILES[$file_idx]}"
            # Derive output dir from merge file path
            action_type=$(basename $(dirname "$merge_file"))
            shard=$(basename "$merge_file" .txt)
            OUTPUT_DIR="${OUTPUT_BASE}/${action_type}/${shard}"

            # Skip if already processed
            if [ -f "${OUTPUT_DIR}/combined_parquet_dataset/worker_0" ]; then
                echo "[Node $NODE_ID GPU $gpu] Skipping $merge_file (already done)"
                file_idx=$((file_idx + NUM_TOTAL_GPUS))
                continue
            fi

            mkdir -p "$OUTPUT_DIR"
            echo "[Node $NODE_ID GPU $gpu] Processing $merge_file -> $OUTPUT_DIR"

            CUDA_VISIBLE_DEVICES=$gpu torchrun \
                --nnodes=1 --nproc_per_node=1 \
                --master_port $port \
                fastvideo/pipelines/preprocess/v1_preprocess.py \
                --model_path $MODEL_PATH \
                --data_merge_path "$merge_file" \
                --preprocess_video_batch_size 4 \
                --seed 42 \
                --max_height 480 \
                --max_width 832 \
                --num_frames 81 \
                --dataloader_num_workers 0 \
                --output_dir "$OUTPUT_DIR" \
                --samples_per_file 8 \
                --train_fps 25 \
                --flush_frequency 8 \
                --preprocess_task matrixgame

            file_idx=$((file_idx + NUM_TOTAL_GPUS))
        done
        echo "[Node $NODE_ID GPU $gpu] Done all assigned shards."
    ) &
done

wait
echo "Node $NODE_ID: all GPU workers finished."
