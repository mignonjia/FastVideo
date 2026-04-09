# SPDX-License-Identifier: Apache-2.0

import torch

from fastvideo.pipelines.pipeline_batch_info import TrainingBatch
from fastvideo.training.matrixgame_self_forcing_distillation_pipeline import (
    MatrixGameSelfForcingDistillationPipeline,
)


def _make_pipeline() -> MatrixGameSelfForcingDistillationPipeline:
    pipeline = MatrixGameSelfForcingDistillationPipeline.__new__(
        MatrixGameSelfForcingDistillationPipeline
    )
    pipeline.use_action_module = True
    pipeline.keyboard_dim_in = 4
    pipeline.mouse_dim_in = 2
    pipeline.vae_time_compression_ratio = 4
    return pipeline


def test_build_score_model_input_kwargs_slices_to_rollout_window():
    pipeline = _make_pipeline()
    batch_size = 2
    total_latent_frames = 21
    rollout_latent_frames = 9
    action_frames = (total_latent_frames - 1) * 4 + 1
    rollout_action_frames = (rollout_latent_frames - 1) * 4 + 1

    training_batch = TrainingBatch(
        image_embeds=torch.randn(batch_size, 8, 16),
        image_latents=torch.randn(batch_size, 20, total_latent_frames, 1, 1),
        keyboard_cond=torch.randn(batch_size, action_frames, 4),
        mouse_cond=torch.randn(batch_size, action_frames, 2),
    )
    noise_input = torch.randn(batch_size, rollout_latent_frames, 16, 1, 1)
    timestep = torch.tensor([10, 20], dtype=torch.long)

    input_kwargs = pipeline._build_score_model_input_kwargs(
        noise_input,
        timestep,
        training_batch,
    )

    expected_hidden_states = torch.cat(
        [
            noise_input,
            training_batch.image_latents[
                :, :, :rollout_latent_frames, :, :
            ].permute(0, 2, 1, 3, 4),
        ],
        dim=2,
    )

    assert torch.equal(input_kwargs["hidden_states"], expected_hidden_states)
    assert input_kwargs["hidden_states"].shape == (
        batch_size,
        rollout_latent_frames,
        36,
        1,
        1,
    )
    assert input_kwargs["keyboard_cond"].shape == (
        batch_size,
        rollout_action_frames,
        4,
    )
    assert input_kwargs["mouse_cond"].shape == (
        batch_size,
        rollout_action_frames,
        2,
    )
    assert torch.equal(
        input_kwargs["keyboard_cond"],
        training_batch.keyboard_cond[:, :rollout_action_frames, :],
    )
    assert torch.equal(
        input_kwargs["mouse_cond"],
        training_batch.mouse_cond[:, :rollout_action_frames, :],
    )
