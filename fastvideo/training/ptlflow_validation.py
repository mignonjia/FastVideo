# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

from fastvideo.fastvideo_args import TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.models.dits.matrixgame.utils import swap_mouse_axes_for_validation

logger = init_logger(__name__)

PTLFLOW_SCALAR_KEYS = (
    "mf_epe_mean",
    "mf_angle_err_mean",
    "mf_cosine_mean",
    "mf_mag_ratio_mean",
    "pixel_epe_mean_mean",
    "px_angle_rmse_mean",
    "fl_all_mean",
    "foe_dist_mean",
    "flow_kl_2d_mean",
)


class PTLFlowValidationHelper:

    def __init__(self) -> None:
        self._flow_eval_init_done = False
        self._flow_eval_ready = False
        self._flow_eval_fn = None

    def initialize(self, training_args: TrainingArgs) -> None:
        if self._flow_eval_init_done:
            return
        self._flow_eval_init_done = True
        self._flow_eval_ready = False

        ptlflow_dir = str(getattr(training_args, "ptlflow_dir", "") or "")
        if not ptlflow_dir:
            logger.warning("PTLFlow evaluation requested but ptlflow_dir is empty.")
            return

        try:
            ptlflow_dir_str = str(Path(ptlflow_dir).resolve())
            if ptlflow_dir_str not in sys.path:
                sys.path.insert(0, ptlflow_dir_str)

            from eval_flow_divergence import evaluate_pair_synthetic

            self._flow_eval_fn = evaluate_pair_synthetic
            self._flow_eval_ready = True
            logger.info(
                "Initialized flow divergence evaluator: %s",
                ptlflow_dir_str,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize flow divergence evaluator: %s",
                exc,
            )

    @property
    def ready(self) -> bool:
        return bool(self._flow_eval_ready)

    def evaluate_video(
        self,
        *,
        video_path: str,
        action_path: str | None,
        global_step: int,
        num_inference_steps: int,
        training_args: TrainingArgs,
    ) -> dict[str, float]:
        self.initialize(training_args)
        if not self._flow_eval_ready:
            raise RuntimeError(
                "ptlflow evaluator is not initialized; cannot compute flow metrics."
            )
        if not isinstance(action_path, str) or not os.path.isfile(action_path):
            raise FileNotFoundError(
                f"Validation sample is missing a valid action_path: {action_path}"
            )

        output_root = training_args.output_dir or os.getcwd()
        eval_output_dir = os.path.join(
            output_root,
            "flow_eval",
            f"step_{global_step}",
            f"inference_steps_{num_inference_steps}",
            Path(video_path).stem,
        )
        ptlflow_dir = str(getattr(training_args, "ptlflow_dir", "") or "")
        default_ckpt = str(Path(ptlflow_dir) / "dpflow-things-2012b5d6.ckpt")
        default_calibration = str(Path(ptlflow_dir) / "calibration.json")
        ptlflow_ckpt = str(getattr(training_args, "ptlflow_ckpt", "") or default_ckpt)
        ptlflow_calibration_path = str(
            getattr(training_args, "ptlflow_calibration_path", "") or default_calibration
        )
        ptlflow_use_depth = bool(getattr(training_args, "ptlflow_use_depth", True))

        aligned_action_path = self._prepare_aligned_action_path(
            action_path=action_path,
            video_path=video_path,
            eval_output_dir=eval_output_dir,
        )
        with torch.inference_mode():
            summary = self._flow_eval_fn(
                gen_video=video_path,
                action_path=aligned_action_path,
                calibration_path=ptlflow_calibration_path,
                output_dir=eval_output_dir,
                model_name="dpflow",
                ckpt=ptlflow_ckpt,
                no_viz=True,
                use_depth=ptlflow_use_depth,
            )
        if not isinstance(summary, dict):
            raise RuntimeError(
                f"ptlflow returned invalid summary type: {type(summary)}"
            )

        metrics: dict[str, float] = {}
        missing_or_invalid: list[str] = []
        for key in PTLFLOW_SCALAR_KEYS:
            value = summary.get(key)
            if not isinstance(value, (float, int, np.floating, np.integer)):
                missing_or_invalid.append(key)
                continue
            value_float = float(value)
            if not np.isfinite(value_float):
                missing_or_invalid.append(key)
                continue
            metrics[key] = value_float

        if missing_or_invalid:
            raise RuntimeError(
                "ptlflow summary missing/invalid metrics: "
                + ", ".join(missing_or_invalid)
            )
        return metrics

    def release_cuda_memory(self) -> None:
        if not torch.cuda.is_available():
            return
        try:
            gc.collect()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    def _count_video_frames(self, video_path: str) -> int:
        reader = None
        try:
            reader = imageio.get_reader(video_path)
            try:
                return int(reader.count_frames())
            except Exception:
                n_frames = 0
                for _ in reader:
                    n_frames += 1
                return int(n_frames)
        except Exception as exc:
            logger.warning(
                "Failed to count frames for validation video %s: %s",
                video_path,
                exc,
            )
            return 0
        finally:
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass

    def _prepare_aligned_action_path(
        self,
        *,
        action_path: str,
        video_path: str,
        eval_output_dir: str,
    ) -> str:
        try:
            actions_obj = np.load(action_path, allow_pickle=True).item()
        except Exception as exc:
            logger.warning(
                "Failed to load action file %s: %s",
                action_path,
                exc,
            )
            return action_path

        if not isinstance(actions_obj, dict):
            logger.warning(
                "Action file %s is not a dict; skipping action alignment.",
                action_path,
            )
            return action_path

        keyboard = actions_obj.get("keyboard")
        mouse = actions_obj.get("mouse")
        if keyboard is None or mouse is None:
            logger.warning(
                "Action file %s is missing 'keyboard' or 'mouse'; skipping action alignment.",
                action_path,
            )
            return action_path

        keyboard_arr = np.asarray(keyboard)
        mouse_arr = np.asarray(mouse)
        if keyboard_arr.ndim < 1 or mouse_arr.ndim < 1:
            logger.warning(
                "Action file %s has invalid shapes keyboard=%s mouse=%s; skipping action alignment.",
                action_path,
                keyboard_arr.shape,
                mouse_arr.shape,
            )
            return action_path

        n_frames = self._count_video_frames(video_path)
        if n_frames <= 0:
            return action_path

        n_keyboard = int(keyboard_arr.shape[0])
        n_mouse = int(mouse_arr.shape[0])
        n_use = min(n_frames, n_keyboard, n_mouse)
        if n_use <= 0:
            return action_path
        swapped_mouse_arr = swap_mouse_axes_for_validation(mouse_arr)
        swapped_mouse_axes = not np.array_equal(
            swapped_mouse_arr[:n_use],
            mouse_arr[:n_use],
        )
        should_save = (
            n_use != n_keyboard
            or n_use != n_mouse
            or swapped_mouse_axes
        )
        if not should_save:
            return action_path

        aligned_actions = dict(actions_obj)
        aligned_actions["keyboard"] = keyboard_arr[:n_use]
        aligned_actions["mouse"] = swapped_mouse_arr[:n_use]

        os.makedirs(eval_output_dir, exist_ok=True)
        aligned_path = os.path.join(eval_output_dir, "actions_aligned.npy")
        np.save(aligned_path, aligned_actions, allow_pickle=True)
        logger.info(
            "Prepared action file for PTLFlow: "
            "video_frames=%d keyboard=%d mouse=%d used=%d "
            "swapped_mouse_axes=%s path=%s",
            n_frames,
            n_keyboard,
            n_mouse,
            n_use,
            swapped_mouse_axes,
            aligned_path,
        )
        return aligned_path
