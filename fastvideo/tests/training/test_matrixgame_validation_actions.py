# SPDX-License-Identifier: Apache-2.0

import numpy as np

from fastvideo.models.dits.matrixgame import utils as matrixgame_utils
from fastvideo.training.ptlflow_validation import PTLFlowValidationHelper
from fastvideo.training import training_utils


def test_overlay_validation_actions_flips_mouse_axes_and_signs(monkeypatch):
    captured: dict[str, float] = {}

    monkeypatch.setattr(matrixgame_utils, "_require_cv2", lambda: True)
    monkeypatch.setattr(
        matrixgame_utils,
        "draw_mouse_on_frame",
        lambda frame, yaw, pitch, top_margin=15: captured.update(
            {"yaw": yaw, "pitch": pitch}
        ),
    )

    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    mouse_cond = np.array([[[1.25, -2.5]]], dtype=np.float32)

    matrixgame_utils.overlay_validation_actions_on_frames(
        frames,
        mouse_cond=mouse_cond,
    )

    assert captured == {"yaw": 2.5, "pitch": -1.25}


def test_ptlflow_action_preparation_flips_mouse_axes_and_signs(
    tmp_path,
    monkeypatch,
):
    helper = PTLFlowValidationHelper()
    monkeypatch.setattr(helper, "_count_video_frames", lambda _: 3)

    original_mouse = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=np.float32,
    )
    original_actions = {
        "keyboard": np.eye(3, 4, dtype=np.float32),
        "mouse": original_mouse,
    }
    action_path = tmp_path / "actions.npy"
    np.save(action_path, original_actions, allow_pickle=True)

    prepared_path = helper._prepare_aligned_action_path(
        action_path=str(action_path),
        video_path="unused.mp4",
        eval_output_dir=str(tmp_path / "flow_eval"),
    )

    assert prepared_path != str(action_path)

    prepared_actions = np.load(prepared_path, allow_pickle=True).item()
    np.testing.assert_allclose(
        prepared_actions["mouse"],
        np.stack(
            (-original_mouse[:, 1], -original_mouse[:, 0]),
            axis=-1,
        ),
    )
    np.testing.assert_allclose(
        prepared_actions["keyboard"],
        original_actions["keyboard"],
    )

    source_actions = np.load(action_path, allow_pickle=True).item()
    np.testing.assert_allclose(source_actions["mouse"], original_mouse)


def test_save_best_distillation_checkpoint_writes_metric_metadata(
    tmp_path,
    monkeypatch,
):
    saved_steps: list[str] = []

    def fake_save_distillation_checkpoint(*args, **kwargs):
        step = kwargs["step"]
        saved_steps.append(step)
        (tmp_path / f"checkpoint-{step}").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        training_utils,
        "save_distillation_checkpoint",
        fake_save_distillation_checkpoint,
    )
    monkeypatch.setattr(training_utils, "_broadcast_int", lambda value, src=0: value)

    saved = training_utils.save_best_distillation_checkpoint(
        generator_transformer=object(),
        fake_score_transformer=object(),
        rank=0,
        output_dir=str(tmp_path),
        step=12,
        metric_value=0.25,
        start_step=1,
        top_k=1,
    )

    assert saved is True
    assert saved_steps == ["best-step-12"]

    metric_path = tmp_path / "checkpoint-best-step-12" / "best_metric.json"
    assert metric_path.is_file()
    metric_meta = metric_path.read_text(encoding="utf-8")
    assert "\"mf_angle_err_mean\": 0.25" in metric_meta
