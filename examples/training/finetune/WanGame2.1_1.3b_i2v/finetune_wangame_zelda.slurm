#!/bin/bash
#SBATCH --job-name=wg-zelda
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=wangame_1.3b_output/wangame_1.3b_%j.out
#SBATCH --error=wangame_1.3b_output/wangame_1.3b_%j.err
#SBATCH --exclusive

# Basic Info
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_DEBUG_SUBSYS=INIT,NET
# different cache dir for different processes
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
export MASTER_PORT=29502
export NODE_RANK=$SLURM_PROCID
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export MASTER_ADDR=${nodes[0]}
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="d5b02b05e30d8cb34c7b31c6ae10416fc26dcb66"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export FASTVIDEO_MAP_STYLE_CACHE_DIR="/mnt/weka/home/hao.zhang/mhuo/FastVideo/map_style_cache"

source ~/conda/miniconda/bin/activate
conda activate mhuo-fv
export HOME="/mnt/weka/home/hao.zhang/mhuo"

# Configs
NUM_GPUS=8
NUM_NODES=4
NUM_TOTAL_GPUS=$((NUM_GPUS * NUM_NODES))
BS_PER_GPU=1
GRADIENT_ACCUMULATION_STEPS=1
WANDB_RUN_NAME="Zelda_MC_no_action"
FREEZE_DIT=False
RUN_DIR="wangame_1.3b_zelda_with_MC_init"
CHECKPOINTING_STEPS=4000
CHECKPOINT_START_STEP=0
BEST_CHECKPOINT_TOP_K=0
ACTION_WARMUP_STEPS=20000
LEARNING_RATE=2e-5
SEED=1000
DMD_DENOISING_STEPS="1000,750,500,250,0"
VAL_ONLY=False

MODEL_PATH="weizhou03/Wan2.1-Game-Fun-1.3B-InP-Diffusers"
CKPT_SAFETENSOR="/mnt/weka/home/hao.zhang/kaiqin/mg_models/Matrix-Game-2.0-Base-Diffusers/transformer/diffusion_pytorch_model.safetensors"
# CKPT_SAFETENSOR="/mnt/weka/home/hao.zhang/mhuo/FastVideo/wangame_1.3b_long_run/checkpoint-best-step-61500/transformer/diffusion_pytorch_model.safetensors"

# Data dirs (use one of the following):
#   - DATA_DIR_ALL: all datasets below combined (comma-separated)
#   - Or a single path / subset; optional ":N" = repeat, ":0" = skip
#
DATA_DIR="/mnt/weka/home/hao.zhang/alex/wm-lab/datas/datasets/zeldam2:1" # Zelda
DATA_DIR="$DATA_DIR,/mnt/weka/home/hao.zhang/mhuo/traindata_0204_2130/preprocessed:1" # Random
DATA_DIR="$DATA_DIR,/mnt/weka/home/hao.zhang/mhuo/traindata_0204_1600/preprocessed:0" # Doom
DATA_DIR="$DATA_DIR,/mnt/weka/home/hao.zhang/mhuo/traindata_0205_1330/data/0_static_plus_w_only/preprocessed:3" # Static + w only
DATA_DIR="$DATA_DIR,/mnt/weka/home/hao.zhang/mhuo/traindata_0205_1330/data/1_wasd_only/preprocessed:3" # w/s/a/d only
DATA_DIR="$DATA_DIR,/mnt/weka/home/hao.zhang/mhuo/traindata_0206_1200/data/wasdonly_alpha1/preprocessed:3" # wasd only
DATA_DIR="$DATA_DIR,/mnt/weka/home/hao.zhang/mhuo/traindata_0206_1200/data/camera/preprocessed:3" # camera l-only and r-only
DATA_DIR="$DATA_DIR,/mnt/weka/home/hao.zhang/mhuo/traindata_0208_2000/data/camera4hold_alpha1/preprocessed:3" # camera only
DATA_DIR="$DATA_DIR,/mnt/weka/home/hao.zhang/mhuo/traindata_0208_2000/data/wasd4holdrandview_simple_1key1mouse1/preprocessed:3" # key_camera_excl_1_action_rand

VALIDATION_DATASET_FILE="examples/training/finetune/WanGame2.1_1.3b_i2v/validation_both.json"

# Single-dir / validation alternatives (comment out DATA_DIR above and uncomment one block):
# MC wasd only:
#   DATA_DIR="/mnt/weka/home/hao.zhang/mhuo/traindata_0205_1330/data/1_wasd_only/preprocessed"
#   VALIDATION_DATASET_FILE="examples/training/finetune/WanGame2.1_1.3b_i2v/validation_wsad.json"
# MC random:
#   DATA_DIR="/mnt/weka/home/hao.zhang/mhuo/traindata_0206_1200/data/wasdonly_alpha1/preprocessed"
#   VALIDATION_DATASET_FILE="examples/training/finetune/WanGame2.1_1.3b_i2v/validation_mc.json"
# Doom:
#   DATA_DIR="/mnt/weka/home/hao.zhang/mhuo/traindata_0204_1600/preprocessed"
#   VALIDATION_DATASET_FILE="examples/training/finetune/WanGame2.1_1.3b_i2v/validation_doom.json"
# Overfit:
#   DATA_DIR="mc_wasd_10/preprocessed/combined_parquet_dataset"
#   VALIDATION_DATASET_FILE="examples/training/finetune/WanGame2.1_1.3b_i2v/validation_overfit.json"


# Training arguments
training_args=(
  --tracker_project_name "wangame_1.3b_zelda"
  --output_dir $RUN_DIR
  --wandb_run_name "$WANDB_RUN_NAME"
  --max_train_steps 20000
  --train_batch_size $BS_PER_GPU
  --train_sp_batch_size $BS_PER_GPU
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS
  --num_latent_t 9
  --num_height 480
  --num_width 832
  --num_frames 33
  --enable_gradient_checkpointing_type "full"
  --train_action_only $FREEZE_DIT
  --action_warmup_steps $ACTION_WARMUP_STEPS
  # --resume_from_checkpoint $RESUME_FROM_CHECKPOINT
  --init_weights_from_safetensors $CKPT_SAFETENSOR
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
  --apply_bot_died_filter False
)

# Validation arguments
validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 500
  --validation_sampling_steps "40"
  --validation_guidance_scale "1.0"
  --validation_num_samples $NUM_TOTAL_GPUS
)

# Optimizer arguments
optimizer_args=(
  --learning_rate $LEARNING_RATE
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps $CHECKPOINTING_STEPS
  --training_state_checkpointing_steps $CHECKPOINTING_STEPS
  --checkpoint_start_step $CHECKPOINT_START_STEP
  --best_checkpoint_top_k $BEST_CHECKPOINT_TOP_K
  --weight_decay 0
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --seed $SEED
  --dmd_denoising_steps "$DMD_DENOISING_STEPS"
  --flow_shift 3
  --val_only $VAL_ONLY
  --checkpoints_total_limit 3
  --training_cfg_rate 0.1
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --ema_start_step 0
)

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t

if [ $NUM_NODES -eq 1 ]; then
  torchrun \
    --nnodes $NUM_NODES \
    --nproc_per_node $NUM_GPUS \
    fastvideo/training/wangame_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
else
  srun torchrun \
    --nnodes $NUM_NODES \
    --nproc_per_node $NUM_GPUS \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --node_rank $SLURM_PROCID \
    fastvideo/training/wangame_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
fi
