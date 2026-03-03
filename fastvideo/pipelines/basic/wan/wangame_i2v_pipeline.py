# SPDX-License-Identifier: Apache-2.0
"""
Wan video diffusion pipeline implementation.

This module contains an implementation of the Wan video diffusion pipeline
using the modular pipeline architecture.
"""

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lora_pipeline import LoRAPipeline

# isort: off
from fastvideo.pipelines.stages import (
    ImageEncodingStage, ConditioningStage, DecodingStage, DenoisingStage,
    ImageVAEEncodingStage, InputValidationStage, LatentPreparationStage,
    TimestepPreparationStage, TextEncodingStage)
# isort: on
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler)

logger = init_logger(__name__)


class WanGameActionImageToVideoPipeline(LoRAPipeline, ComposedPipelineBase):

    _required_config_modules = [
        "vae", "transformer", "scheduler", "image_encoder", "image_processor"
    ]

    def __init__(
        self,
        model_path: str,
        fastvideo_args: FastVideoArgs,
        required_config_modules: list[str] | None = None,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ):
        if required_config_modules is None:
            required_config_modules = list(self._required_config_modules)
            if fastvideo_args.wangame_use_text_module:
                required_config_modules.extend(["text_encoder", "tokenizer"])
        super().__init__(
            model_path=model_path,
            fastvideo_args=fastvideo_args,
            required_config_modules=required_config_modules,
            loaded_modules=loaded_modules,
        )

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        if fastvideo_args.wangame_use_text_module:
            text_encoder = self.get_module("text_encoder")
            tokenizer = self.get_module("tokenizer")
            if text_encoder is None or tokenizer is None:
                raise ValueError(
                    "WanGame text module is enabled, but text_encoder/tokenizer "
                    "is not loaded."
                )
            self.add_stage(stage_name="prompt_encoding_stage",
                           stage=TextEncodingStage(
                               text_encoders=[text_encoder],
                               tokenizers=[tokenizer],
                           ))
        else:
            logger.info(
                "WanGame text module disabled "
                "(--wangame-use-text-module=False); "
                "skipping prompt_encoding_stage.")

        self.add_stage(
            stage_name="image_encoding_stage",
            stage=ImageEncodingStage(
                image_encoder=self.get_module("image_encoder"),
                image_processor=self.get_module("image_processor"),
            ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer")))

        self.add_stage(stage_name="image_latent_preparation_stage",
                       stage=ImageVAEEncodingStage(vae=self.get_module("vae")))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))


class WanLingBotImageToVideoPipeline(WanGameActionImageToVideoPipeline):
    pass


EntryClass = [WanGameActionImageToVideoPipeline, WanLingBotImageToVideoPipeline]
