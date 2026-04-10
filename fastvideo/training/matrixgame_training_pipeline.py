# SPDX-License-Identifier: Apache-2.0
import os
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange

from fastvideo.configs.sample import SamplingParam
from fastvideo.dataset.dataloader.schema import pyarrow_schema_matrixgame
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler)
from fastvideo.pipelines.basic.matrixgame.matrixgame_i2v_pipeline import (
    MatrixGamePipeline)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, TrainingBatch
from fastvideo.training.ptlflow_validation import (
    PTLFlowValidationHelper,
)
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.training_utils import save_best_checkpoint
from fastvideo.utils import is_vsa_available, shallow_asdict

try:
    vsa_available = is_vsa_available()
except Exception:
    vsa_available = False

logger = init_logger(__name__)


class MatrixGameTrainingPipeline(TrainingPipeline):
    """
    A training pipeline for Matrix-Game-2.0.
    """
    _required_config_modules = ["scheduler", "transformer", "vae"]
    _ACTION_PARAM_PATTERNS = ("action_model", )
    _action_params_frozen_for_warmup: bool = False

    def _set_action_params_grad(self, requires_grad: bool) -> int:
        count = 0
        modules = [self.transformer]
        if getattr(self, "transformer_2", None) is not None:
            modules.append(self.transformer_2)

        for module in modules:
            if module is None:
                continue
            for name, param in module.named_parameters():
                if not any(pattern in name
                           for pattern in self._ACTION_PARAM_PATTERNS):
                    continue
                param.requires_grad_(requires_grad)
                count += 1
        return count

    def _on_train_start(self) -> None:
        action_warmup_steps = max(0,
                                  int(self.training_args.action_warmup_steps))
        if action_warmup_steps <= 0:
            return
        count = self._set_action_params_grad(False)
        self._action_params_frozen_for_warmup = True
        logger.info(
            "Action warmup enabled: freezing %d action parameter tensors for the first %d steps",
            count,
            action_warmup_steps,
        )

    def _before_train_step(self, step: int) -> None:
        if not self._action_params_frozen_for_warmup:
            return
        if int(step) <= int(self.training_args.action_warmup_steps):
            return
        count = self._set_action_params_grad(True)
        self._action_params_frozen_for_warmup = False
        logger.info(
            "Action warmup complete: unfroze %d action parameter tensors at step %d",
            count,
            int(step),
        )

    def _get_expected_action_dim(self, key: str) -> int:
        action_config = getattr(self.transformer, "action_config", {}) or {}
        return int(action_config.get(key, 0))

    def _align_action_feature_dim(
        self,
        action: torch.Tensor | np.ndarray,
        *,
        name: str,
        expected_dim: int,
        context: str,
    ) -> torch.Tensor | np.ndarray:
        if expected_dim <= 0:
            return action

        actual_dim = int(action.shape[-1])
        if actual_dim == expected_dim:
            return action

        if actual_dim > expected_dim:
            extra = action[..., expected_dim:]
            if torch.is_tensor(extra):
                extra_is_zero = bool(torch.count_nonzero(extra).item() == 0)
            else:
                extra_is_zero = bool(np.count_nonzero(extra) == 0)

            if extra_is_zero:
                logger.warning(
                    "%s feature dim mismatch in %s: got=%s expected=%s. "
                    "Truncating trailing all-zero channels.",
                    name,
                    context,
                    actual_dim,
                    expected_dim,
                )
                return action[..., :expected_dim]

            raise ValueError(
                f"{name} feature dim mismatch in {context}: got={actual_dim}, "
                f"expected={expected_dim}, and trailing channels are not all zero."
            )

        raise ValueError(
            f"{name} feature dim mismatch in {context}: got={actual_dim}, "
            f"expected={expected_dim}."
        )

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def create_training_stages(self, training_args: TrainingArgs):
        """
        May be used in future refactors.
        """
        pass

    def set_schemas(self):
        self.train_dataset_schema = pyarrow_schema_matrixgame

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        logger.info("Initializing validation pipeline...")
        args_copy = deepcopy(training_args)

        args_copy.inference_mode = True
        args_copy.dit_cpu_offload = True
        # args_copy.pipeline_config.vae_config.load_encoder = False
        # validation_pipeline = WanImageToVideoValidationPipeline.from_pretrained(
        self.validation_pipeline = MatrixGamePipeline.from_pretrained(
            training_args.model_path,
            args=None,
            inference_mode=True,
            loaded_modules={
                "transformer": self.get_module("transformer"),
            },
            tp_size=training_args.tp_size,
            sp_size=training_args.sp_size,
            num_gpus=training_args.num_gpus,
            dit_cpu_offload=True)
        self._ptlflow_validation = PTLFlowValidationHelper()
        self._last_ptlflow_metric: float | None = None
        self._last_ptlflow_metric_step: int | None = None

    def _get_next_batch(self, training_batch: TrainingBatch) -> TrainingBatch:
        batch = next(self.train_loader_iter, None)  # type: ignore
        if batch is None:
            self.current_epoch += 1
            logger.info("Starting epoch %s", self.current_epoch)
            # Reset iterator for next epoch
            self.train_loader_iter = iter(self.train_dataloader)
            # Get first batch of new epoch
            batch = next(self.train_loader_iter)

        latents = batch['vae_latent']
        latents = latents[:, :, :self.training_args.num_latent_t]
        # encoder_hidden_states = batch['text_embedding']
        # encoder_attention_mask = batch['text_attention_mask']
        clip_features = batch['clip_feature']
        image_latents = batch['first_frame_latent']
        image_latents = image_latents[:, :, :self.training_args.num_latent_t]
        pil_image = batch['pil_image']
        infos = batch['info_list']

        training_batch.latents = latents.to(get_local_torch_device(),
                                            dtype=torch.bfloat16)
        training_batch.encoder_hidden_states = None
        training_batch.encoder_attention_mask = None
        # MatrixGame doesn't use text encoder
        training_batch.preprocessed_image = pil_image.to(
            get_local_torch_device())
        training_batch.image_embeds = clip_features.to(get_local_torch_device())
        training_batch.image_latents = image_latents.to(
            get_local_torch_device())
        training_batch.infos = infos

        # Action conditioning
        if 'mouse_cond' in batch and batch['mouse_cond'].numel() > 0:
            training_batch.mouse_cond = batch['mouse_cond'].to(
                get_local_torch_device(), dtype=torch.bfloat16)
        else:
            training_batch.mouse_cond = None

        if 'keyboard_cond' in batch and batch['keyboard_cond'].numel() > 0:
            training_batch.keyboard_cond = batch['keyboard_cond'].to(
                get_local_torch_device(), dtype=torch.bfloat16)
            training_batch.keyboard_cond = self._align_action_feature_dim(
                training_batch.keyboard_cond,
                name="keyboard_cond",
                expected_dim=self._get_expected_action_dim(
                    "keyboard_dim_in"),
                context="training batch",
            )
        else:
            training_batch.keyboard_cond = None

        return training_batch

    def _prepare_dit_inputs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:
        """Override to properly handle I2V concatenation - call parent first, then concatenate image conditioning."""

        # First, call parent method to prepare noise, timesteps, etc. for video latents
        training_batch = super()._prepare_dit_inputs(training_batch)

        assert isinstance(training_batch.image_latents, torch.Tensor)
        image_latents = training_batch.image_latents.to(
            get_local_torch_device(), dtype=torch.bfloat16)

        temporal_compression_ratio = self.training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        num_frames = (self.training_args.num_latent_t -
                      1) * temporal_compression_ratio + 1
        batch_size, num_channels, _, latent_height, latent_width = image_latents.shape
        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height,
                                   latent_width)
        mask_lat_size[:, :, 1:] = 0

        first_frame_mask = mask_lat_size[:, :, :1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=temporal_compression_ratio)
        mask_lat_size = torch.cat([first_frame_mask, mask_lat_size[:, :, 1:]],
                                  dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1,
                                           temporal_compression_ratio,
                                           latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(
            image_latents.device).to(dtype=torch.bfloat16)

        training_batch.noisy_model_input = torch.cat(
            [training_batch.noisy_model_input, mask_lat_size, image_latents],
            dim=1)

        return training_batch

    def _build_input_kwargs(self,
                            training_batch: TrainingBatch) -> TrainingBatch:

        # Image Embeds for conditioning
        image_embeds = training_batch.image_embeds
        assert torch.isnan(image_embeds).sum() == 0
        image_embeds = image_embeds.to(get_local_torch_device(),
                                       dtype=torch.bfloat16)
        encoder_hidden_states_image = image_embeds

        # NOTE: noisy_model_input already contains concatenated image_latents from _prepare_dit_inputs
        training_batch.input_kwargs = {
            "hidden_states":
            training_batch.noisy_model_input,
            "encoder_hidden_states":
            training_batch.encoder_hidden_states,  # None for MatrixGame
            "timestep":
            training_batch.timesteps.to(get_local_torch_device(),
                                        dtype=torch.bfloat16),
            # "encoder_attention_mask":
            # training_batch.encoder_attention_mask,
            "encoder_hidden_states_image":
            encoder_hidden_states_image,
            # Action conditioning
            "mouse_cond":
            training_batch.mouse_cond,
            "keyboard_cond":
            training_batch.keyboard_cond,
            "return_dict":
            False,
        }
        return training_batch

    def _prepare_validation_batch(self, sampling_param: SamplingParam,
                                  training_args: TrainingArgs,
                                  validation_batch: dict[str, Any],
                                  num_inference_steps: int) -> ForwardBatch:
        sampling_param.prompt = validation_batch['prompt']
        sampling_param.height = training_args.num_height
        sampling_param.width = training_args.num_width
        sampling_param.image_path = validation_batch.get(
            'image_path') or validation_batch.get('video_path')
        sampling_param.num_inference_steps = num_inference_steps
        sampling_param.data_type = "video"
        assert self.seed is not None
        sampling_param.seed = self.seed

        temporal_compression_factor = training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        num_frames = (training_args.num_latent_t -
                      1) * temporal_compression_factor + 1
        sampling_param.num_frames = num_frames
        latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
                        sampling_param.height // 8, sampling_param.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]
        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            latents=None,
            generator=torch.Generator(device="cpu").manual_seed(self.seed),
            n_tokens=n_tokens,
            eta=0.0,
            VSA_sparsity=training_args.VSA_sparsity,
        )
        if "image" in validation_batch and validation_batch["image"] is not None:
            batch.pil_image = validation_batch["image"]

        if "keyboard_cond" in validation_batch and validation_batch[
                "keyboard_cond"] is not None:
            keyboard_cond = validation_batch["keyboard_cond"]
            keyboard_cond = self._align_action_feature_dim(
                keyboard_cond[:sampling_param.num_frames],
                name="keyboard_cond",
                expected_dim=self._get_expected_action_dim(
                    "keyboard_dim_in"),
                context="validation batch",
            )
            keyboard_cond = torch.tensor(
                keyboard_cond,
                dtype=torch.bfloat16,
            )
            keyboard_cond = keyboard_cond.unsqueeze(0)
            batch.keyboard_cond = keyboard_cond

        if "mouse_cond" in validation_batch and validation_batch[
                "mouse_cond"] is not None:
            mouse_cond = validation_batch["mouse_cond"]
            mouse_cond = torch.tensor(
                mouse_cond[:sampling_param.num_frames],
                dtype=torch.bfloat16,
            )
            mouse_cond = mouse_cond.unsqueeze(0)
            batch.mouse_cond = mouse_cond

        return batch

    def _post_process_validation_frames(
            self, frames: list[np.ndarray],
            batch: ForwardBatch) -> list[np.ndarray]:
        from fastvideo.models.dits.matrixgame.utils import (
            overlay_validation_actions_on_frames,
        )

        return overlay_validation_actions_on_frames(
            frames,
            keyboard_cond=getattr(batch, "keyboard_cond", None),
            mouse_cond=getattr(batch, "mouse_cond", None),
        )

    @torch.no_grad()
    def _log_validation(self, transformer, training_args, global_step) -> None:
        super()._log_validation(transformer, training_args, global_step)

    def _after_validation(self, step: int) -> None:
        if self._last_ptlflow_metric_step != int(step):
            return
        save_best_checkpoint(
            self.transformer,
            self.global_rank,
            self.training_args.output_dir,
            step=int(step),
            metric_value=self._last_ptlflow_metric,
            metric_name="mf_angle_err_mean",
            optimizer=self.optimizer,
            dataloader=self.train_dataloader,
            scheduler=self.lr_scheduler,
            noise_generator=self.noise_random_generator,
            start_step=int(
                getattr(self.training_args, "best_checkpoint_start_step", 0)
                or 0),
            top_k=int(
                getattr(self.training_args, "best_checkpoint_top_k", 1) or 1),
            tracker=self.tracker if self.global_rank == 0 else None,
        )


def main(args) -> None:
    logger.info("Starting training pipeline...")

    pipeline = MatrixGameTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("Training pipeline done")


if __name__ == "__main__":
    import sys
    argv = sys.argv
    from fastvideo.fastvideo_args import TrainingArgs
    from fastvideo.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
