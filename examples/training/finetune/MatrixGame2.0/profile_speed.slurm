#!/bin/bash
#SBATCH --job-name=mg-profile-speed
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=profile_speed_output/profile_%j.out
#SBATCH --error=profile_speed_output/profile_%j.err
#SBATCH --exclusive

set -e -x

# Environment Setup
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  eval "$(/home/hal-mingjiahuo/miniconda3/bin/conda shell.bash hook)"
fi
conda activate fastvideo

export PYTHONPATH="/home/hal-mingjiahuo/FastVideo:${PYTHONPATH:-}"

export WANDB_MODE="disabled"
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1

# ── Torch profiler: capture training step 2 (skip step 0, warmup step 1) ──────
export FASTVIDEO_TORCH_PROFILER_WAIT_STEPS=1
export FASTVIDEO_TORCH_PROFILER_WARMUP_STEPS=1
export FASTVIDEO_TORCH_PROFILER_ACTIVE_STEPS=1
export FASTVIDEO_TORCH_PROFILE_REGIONS="profiler_region_training_train_one_step"
export FASTVIDEO_TORCH_PROFILER_WITH_STACK=1

# ── Configs ────────────────────────────────────────────────────────────────────
MODEL_PATH="mignonjia/mg_bidirectional_zelda"
DUMMY_DATA_ROOT="profiling/dummy_data"
PROFILE_OUTPUT_ROOT="profiling/results"
NUM_GPUS=4
NUM_NODES=1
NUM_TOTAL_GPUS=$((NUM_GPUS * NUM_NODES))
PROFILE_STEPS=30   # steps per latent-size run (first few are compile/warmup)

mkdir -p "$PROFILE_OUTPUT_ROOT"

# ── Step 1: generate dummy parquet data (skipped if already exists) ────────────
if [ ! -d "${DUMMY_DATA_ROOT}/latent_20" ]; then
  echo "=== Generating dummy profiling data under ${DUMMY_DATA_ROOT} ==="
  python examples/training/finetune/MatrixGame2.0/gen_dummy_profiling_data.py \
    --output_dir "$DUMMY_DATA_ROOT" \
    --num_samples 64
  echo "=== Dummy data ready ==="
else
  echo "=== Dummy data already exists, skipping generation ==="
fi

# ── Parallel arguments (4 nodes, SP=4) ────────────────────────────────────────
parallel_args=(
  --num_gpus $NUM_TOTAL_GPUS
  --sp_size 4
  --tp_size 1
  --hsdp_replicate_dim $NUM_TOTAL_GPUS
  --hsdp_shard_dim 1
)

# ── Model arguments ────────────────────────────────────────────────────────────
model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)

# ── Optimizer arguments ────────────────────────────────────────────────────────
optimizer_args=(
  --learning_rate 2e-5
  --betas "0.9,0.95"
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 99999
  --training_state_checkpointing_steps 99999
  --weight_decay 0
  --max_grad_norm 1.0
)

# ── Miscellaneous arguments ────────────────────────────────────────────────────
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 0
  --best_checkpoint_start_step 99999
  --best_checkpoint_top_k 0
  --training_cfg_rate 0.1
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --ema_start_step 99999
)

# ── Step 2: profile each latent size ──────────────────────────────────────────
for LATENT_T in 20 40 80 120; do
  # (T-1)*4+1 video frames
  NUM_FRAMES=$(( (LATENT_T - 1) * 4 + 1 ))
  DATA_DIR="${DUMMY_DATA_ROOT}/latent_${LATENT_T}"
  OUT_DIR="${PROFILE_OUTPUT_ROOT}/latent_${LATENT_T}"
  TRACE_DIR="$(pwd)/profiling/traces/latent_${LATENT_T}"
  mkdir -p "$TRACE_DIR"
  export FASTVIDEO_TORCH_PROFILER_DIR="$TRACE_DIR"

  echo ""
  echo "============================================================"
  echo " Profiling  num_latent_t=${LATENT_T}  num_frames=${NUM_FRAMES}"
  echo "============================================================"

  torchrun \
    --nnodes $NUM_NODES \
    --nproc_per_node $NUM_GPUS \
    fastvideo/training/matrixgame_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    --data_path "$DATA_DIR" \
    --dataloader_num_workers 1 \
    --tracker_project_name "mg_speed_profile" \
    --output_dir "$OUT_DIR" \
    --wandb_run_name "profile_latent${LATENT_T}" \
    --override_transformer_cls_name "MatrixGameWanModel" \
    --max_train_steps $PROFILE_STEPS \
    --train_batch_size 1 \
    --train_sp_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --action_warmup_steps 0 \
    --num_latent_t $LATENT_T \
    --num_height 480 \
    --num_width 832 \
    --num_frames $NUM_FRAMES \
    --enable_gradient_checkpointing_type "full" \
    --reinit-action-module False \
    "${optimizer_args[@]}" \
    "${miscellaneous_args[@]}"

  echo "=== Done latent_t=${LATENT_T} ==="
done

echo ""
echo "=== All profiling runs complete. Results in ${PROFILE_OUTPUT_ROOT} ==="
