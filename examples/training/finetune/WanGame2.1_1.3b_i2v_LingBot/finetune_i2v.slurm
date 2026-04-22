#!/bin/bash
#SBATCH --job-name=wangame_1.3b_lingbot
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/wangame_lingbot/lingbot_%j.out
#SBATCH --error=logs/wangame_lingbot/lingbot_%j.err

source ~/miniconda3/bin/activate
conda activate fastvideo
export HOME="/home/hal-mingjiahuo"

# Basic Info
export TRITON_CACHE_DIR=$HOME/.triton/cache
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="d5b02b05e30d8cb34c7b31c6ae10416fc26dcb66"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_ENTITY="kaiqin_kong_ucsd"
export WANDB_MODE=online
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export FASTVIDEO_MAP_STYLE_CACHE_DIR="map_style_cache"
export NCCL_P2P_DISABLE=1

# ---- torchrun rendezvous (multi-node) ----
# 1. Get the hostname of the first node (Master)
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
MASTER_PORT=29500

# 2. Get the node count automatically
NNODES=$SLURM_NNODES
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
NUM_GPUS=$((NNODES * GPUS_PER_NODE))

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NNODES=$NNODES"

# Use FastVideo repo in /home/hal-mingjiahuo/FastVideo
FASTVIDEO_ROOT="/home/hal-mingjiahuo/FastVideo"
export PYTHONPATH="$FASTVIDEO_ROOT:${PYTHONPATH}"

# Configs
NUM_GPUS=4
NUM_NODES=8
NUM_TOTAL_GPUS=$((NUM_GPUS * NUM_NODES))
BS_PER_GPU=1
GRADIENT_ACCUMULATION_STEPS=1
WANDB_RUN_NAME="MC-wasd-only"
RUN_DIR="/home/hal-mingjiahuo/FastVideo/wangame_1.3b_lingbot"
CHECKPOINTING_STEPS=1000
LEARNING_RATE=1e-5
MAX_TRAIN_STEPS=20000

mkdir -p "$RUN_DIR"

MODEL_PATH="weizhou03/Wan2.1-Game-Fun-1.3B-InP-Diffusers"
DATA_DIR="/home/hal-shared/world_model/MC/mc_wasd_10/preprocessed:0" # overfit
DATA_DIR="$DATA_DIR,/home/hal-shared/world_model/MC/traindata_0205_1330/data/1_wasd_only/preprocessed:1" # w/s/a/d only
DATA_DIR="$DATA_DIR,/home/hal-shared/world_model/MC/traindata_0206_1200/data/wasdonly_alpha1/preprocessed:1" # wasd only
DATA_DIR="$DATA_DIR,/home/hal-shared/world_model/MC/traindata_0206_1200/data/camera/preprocessed:1" # camera l-only and r-only
DATA_DIR="$DATA_DIR,/home/hal-shared/world_model/MC/traindata_0208_2000/data/camera4hold_alpha1/preprocessed:1" # camera only
DATA_DIR="$DATA_DIR,/home/hal-shared/world_model/MC/traindata_0208_2000/data/wasd4holdrandview_simple_1key1mouse1/preprocessed:1" # key_camera_excl_1_action_rand
DATA_DIR="$DATA_DIR,/home/hal-shared/world_model/zelda_overfit2/combined_parquet_dataset:0" # zelda 

VALIDATION_DATASET_FILE="examples/training/finetune/WanGame2.1_1.3b_i2v/validation_random.json"

# Training arguments
training_args=(
  --override-pipeline-cls-name "WanLingBotImageToVideoPipeline"
  --override-transformer-cls-name "WanLingBotTransformer3DModel"
  --tracker_project_name "wangame_lingbot_test"
  --output_dir $RUN_DIR
  --wandb_run_name "$WANDB_RUN_NAME"
  --max_train_steps $MAX_TRAIN_STEPS
  --train_batch_size $BS_PER_GPU
  --train_sp_batch_size $BS_PER_GPU
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS
  --num_latent_t 20
  --num_height 352
  --num_width 640
  --num_frames 77
  --enable_gradient_checkpointing_type "full"
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
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 100
  --validation_sampling_steps "40"
  --validation_guidance_scale "1.0"
  --validation_num_samples $NUM_TOTAL_GPUS
)

# Optimizer arguments
optimizer_args=(
  --learning_rate $LEARNING_RATE
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 1000000
  --training_state_checkpointing_steps $CHECKPOINTING_STEPS
  --weight_decay 1e-4
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
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
    $FASTVIDEO_ROOT/fastvideo/training/wangame_lingbot_training_pipeline.py \
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
    $FASTVIDEO_ROOT/fastvideo/training/wangame_lingbot_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
fi
