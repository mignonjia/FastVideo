#!/bin/bash
#SBATCH --job-name=mg-finetune
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=finetune_i2v_output/i2v_%j.out
#SBATCH --error=finetune_i2v_output/i2v_%j.err
#SBATCH --exclusive

# 563 without warmup
# 564 with warmup

set -e -x

# Environment Setup
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  eval "$(/opt/conda/bin/conda shell.bash hook)"
fi
conda activate mhuo-fv

export PYTHONPATH="/mnt/home/mhuo/FastVideo-MG:${PYTHONPATH:-}"

# Basic Info
export WANDB_API_KEY="d5b02b05e30d8cb34c7b31c6ae10416fc26dcb66"
export WANDB_MODE="online"
export NCCL_P2P_DISABLE=1
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export MASTER_ADDR=${nodes[0]}
export TOKENIZERS_PARALLELISM=false

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

RUN_NAME=$(date +"%m%d_%H%M")
echo "RUN_NAME: $RUN_NAME"

# Configs
# MODEL_PATH="../mg_models/Solaris-6K"
# MODEL_PATH="../mg_models/Solaris-30K"
MODEL_PATH="FastVideo/Matrix-Game-2.0-Base-Diffusers"
# MODEL_PATH="../mg_models/Matrix-Game-2.0-Base-Diffusers"
DATA_DIR="/mnt/data/world_model/zelda_data/zeldam2-v2"
# DATA_DIR="/mnt/weka/home/hao.zhang/kaiqin/solaris/datasets/train_81f/preprocessed_0308"
# DATA_DIR="/mnt/weka/home/hao.zhang/mhuo/traindata_0208_2000/data/wasd4holdrandview_simple_1key1mouse1/preprocessed"
# VALIDATION_DATASET_FILE="/mnt/weka/home/hao.zhang/kaiqin/FastVideo_clean/examples/training/finetune/MatrixGame2.0/validation_8_vpt_mix.json"
VALIDATION_DATASET_FILE="/mnt/home/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/validation_zelda.json"

# Configs
NUM_GPUS=8
NUM_NODES=4
NUM_TOTAL_GPUS=$((NUM_GPUS * NUM_NODES))

# Training arguments
training_args=(
  --tracker_project_name "wangame_1.3b_zelda"
  --output_dir "/mnt/data/world_model/zelda_runs/mg_finetune_zelda_with_10k_warmup"
  # --wandb_run_name "${RUN_NAME}_bs64_continue_resume_from_8K"
  --wandb_run_name "mg_zelda_with_10k_warmup"
  --override_transformer_cls_name "MatrixGameWanModel"
  --init_weights_from_safetensors "/mnt/data/world_model/zelda_runs/mg_finetune_zelda_warmup/checkpoint-10000/transformer/diffusion_pytorch_model.safetensors"
  --max_train_steps 80000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --action_warmup_steps 0
  --num_latent_t 9
  --num_height 480
  --num_width 832
  --num_frames 33
  # --enable_gradient_checkpointing_type "full"
  --reinit-action-module False
  # --override-keyboard-dim 23
)

# Parallel arguments
parallel_args=(
  --num_gpus $NUM_TOTAL_GPUS
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $NUM_TOTAL_GPUS
)

# Model arguments
model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)

# Dataset arguments
dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 1
)

# Validation arguments
validation_args=(
  --log_validation
  --evaluate_ptlflow
  --ptlflow_dir "/mnt/home/mhuo/ptlflow"
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 500
  --validation_sampling_steps "50"
  --validation_guidance_scale "6.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 2e-5
  --betas "0.9,0.95"
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 10000
  --training_state_checkpointing_steps 10000
  --weight_decay 0
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 10
  --best_checkpoint_start_step 1
  --best_checkpoint_top_k 10
  --training_cfg_rate 0.1
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --ema_start_step 0
)

if [ $NUM_NODES -eq 1 ]; then
  torchrun \
    --nnodes $NUM_NODES \
    --nproc_per_node $NUM_GPUS \
    fastvideo/training/matrixgame_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
else
  srun torchrun \
  --nnodes $SLURM_JOB_NUM_NODES \
  --nproc_per_node 8 \
  --node_rank $SLURM_PROCID \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
      fastvideo/training/matrixgame_training_pipeline.py \
      "${parallel_args[@]}" \
      "${model_args[@]}" \
      "${dataset_args[@]}" \
      "${training_args[@]}" \
      "${optimizer_args[@]}" \
      "${validation_args[@]}" \
      "${miscellaneous_args[@]}"
fi