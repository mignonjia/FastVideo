#!/bin/bash

export WANDB_API_KEY="7ff8b6e8356924f7a6dd51a0342dd1a422ea9352"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
# export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

# Configs
# MODEL_PATH="../mg_models/Solaris-6K"
MODEL_PATH="FastVideo/Matrix-Game-2.0-Base-Diffusers"
# MODEL_PATH="../mg_models/Matrix-Game-2.0-Base-Diffusers"
DATA_DIR="/mnt/data/world_model/zelda_data/zeldam2-v2"
# DATA_DIR="/mnt/weka/home/hao.zhang/kaiqin/solaris/datasets/train_81f/preprocessed_0308"
# DATA_DIR="/mnt/weka/home/hao.zhang/mhuo/traindata_0208_2000/data/wasd4holdrandview_simple_1key1mouse1/preprocessed"
# VALIDATION_DATASET_FILE="/mnt/weka/home/hao.zhang/kaiqin/FastVideo_clean/examples/training/finetune/MatrixGame2.0/validation_8_vpt_mix.json"
VALIDATION_DATASET_FILE="/mnt/home/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/validation_zelda.json"

NUM_GPUS=4

# Training arguments
training_args=(
  --tracker_project_name "mg_finetune_solaris"
  --output_dir "checkpoints/matrixgame_finetune_${RUN_NAME}"
  # --wandb_run_name "${RUN_NAME}_bs64_continue_resume_from_8K"
  --wandb_run_name "${RUN_NAME}_test"
  --override_transformer_cls_name "MatrixGameWanModel"
  --max_train_steps 30000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 2
  --num_latent_t 20
  --num_height 352
  --num_width 640
  --num_frames 81
  # --enable_gradient_checkpointing_type "full"
  # --reinit-action-module True
  # --override-keyboard-dim 23
)

# Parallel arguments
parallel_args=(
  --num_gpus $NUM_GPUS
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim $NUM_GPUS
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
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 500
  --validation_sampling_steps "50"
  --validation_guidance_scale "6.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 1e-4
  --betas "0.9,0.95"
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 2000
  --training_state_checkpointing_steps 2000
  --weight_decay 0
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --best_checkpoint_start_step 500
  --best_checkpoint_top_k 3
  --training_cfg_rate 0.1
  --multi_phased_distill_schedule "4000-1"
  --not_apply_cfg_solver
  --dit_precision "fp32"
  --num_euler_timesteps 50
  --ema_start_step 0
)

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
torchrun \
  --nnodes 1 \
  --nproc_per_node $NUM_GPUS \
    fastvideo/training/matrixgame_training_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
