#!/bin/bash
#SBATCH --job-name=mg-diag
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=sf_train_output/sf_%j.out
#SBATCH --error=sf_train_output/sf_%j.err
#SBATCH --exclusive

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
MODEL_PATH="FastVideo/Matrix-Game-2.0-Base-Diffusers"
DATA_DIR="/mnt/data/world_model/zelda_data/zeldam2-v2"
VALIDATION_DATASET_FILE="/mnt/home/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/validation_zelda.json"
RUN_NAME="XXXX"
OUTPUT_DIR="/mnt/data/world_model/zelda_runs/checkpoints/$RUN_NAME"
CKPT_PATH="/mnt/data/world_model/zelda_runs/mg_finetune_zelda_no_warmup/checkpoint-best-step-68500-df-init/transformer/diffusion_pytorch_model.safetensors"

# Configs
NUM_GPUS=4
NUM_NODES=1
NUM_TOTAL_GPUS=$((NUM_GPUS * NUM_NODES))

# Training arguments
training_args=(
  --tracker_project_name "wangame_sf"
  --output_dir $OUTPUT_DIR
  --wandb_run_name $RUN_NAME
  --max_train_steps 10000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 4
  --num_latent_t 9
  --num_height 480
  --num_width 832
  --enable_gradient_checkpointing_type "full"
  --log_visualization
  --simulate_generator_forward
  --num_frames 33
  --num_frame_per_block 3  # Frame generation block size for self-forcing
  # --enable_gradient_masking
  # --gradient_mask_last_n_frames 21
  # --use-diagonal-denoising
)

# Parallel arguments
parallel_args=(
  --num_gpus $NUM_TOTAL_GPUS
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $NUM_TOTAL_GPUS
)

model_args=(
  --model_path $GENERATOR_MODEL_PATH  # TODO: check if you can remove this in this script
  --pretrained_model_name_or_path $GENERATOR_MODEL_PATH
  --real_score_model_path $REAL_SCORE_MODEL_PATH
  --fake_score_model_path $FAKE_SCORE_MODEL_PATH
)

dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 4
)

# Validation arguments
validation_args=(
  --log_validation
  --log_visualization
  --visualization-steps 200
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 200
  --validation_sampling_steps "4"
  --validation_guidance_scale "6.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 3e-6
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 400
  --training_state_checkpointing_steps 400
  --weight_decay 0
  --betas "0.9,0.95"
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
  --dit_precision "fp32"
  --flow_shift 5
  --seed 1000
  --use_ema True
  --ema_decay 0.99
  --ema_start_step 200
)

dmd_args=(
  --dmd_denoising_steps '1000,100'
  --diagonal-warmup-mid-steps '750,500'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --dfake_gen_update_ratio 5
  --real_score_guidance_scale 3.0
  --fake_score_learning_rate 3e-7
  --fake_score_betas "0.9,0.95"
  --warp_denoising_step
)

self_forcing_args=(
  --independent_first_frame False  # Whether to treat first frame independently
  --same_step_across_blocks True  # Whether to use same denoising step across all blocks
  --last_step_only False  # Whether to only use the last denoising step
  --context_noise 0  # Amount of noise to add during context caching (0 = no noise)
)

mkdir -p sf_train_output

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node 8 \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/matrixgame_self_forcing_distillation_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${dmd_args[@]}" \
    "${self_forcing_args[@]}"
