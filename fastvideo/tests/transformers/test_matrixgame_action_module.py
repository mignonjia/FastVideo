# SPDX-License-Identifier: Apache-2.0

import torch

from fastvideo.models.dits.matrixgame.action_module import ActionModule


def test_scale_keyboard_condition_applies_multiplier():
    module = ActionModule.__new__(ActionModule)
    module.keyboard_value_scale = 10.0
    keyboard = torch.tensor([[[0.0, 1.0, -1.5, 2.0]]], dtype=torch.float32)

    scaled = ActionModule._scale_keyboard_condition(module, keyboard)

    torch.testing.assert_close(scaled, keyboard * 10.0)


def test_scale_keyboard_condition_is_noop_for_default_scale():
    module = ActionModule.__new__(ActionModule)
    module.keyboard_value_scale = 1.0
    keyboard = torch.tensor([[[0.0, 1.0, -1.5, 2.0]]], dtype=torch.float32)

    scaled = ActionModule._scale_keyboard_condition(module, keyboard)

    assert scaled is keyboard
