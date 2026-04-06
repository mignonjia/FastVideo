# SPDX-License-Identifier: Apache-2.0
import os
import importlib
import sys
from pathlib import Path

import pytest


PTLFLOW_DIR = Path("/mnt/home/mhuo/ptlflow")


def test_eval_flow_divergence_import_and_assets() -> None:
    if not PTLFLOW_DIR.exists():
        pytest.skip(f"PTLFlow checkout not found at {PTLFLOW_DIR}")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fastvideo")

    ptlflow_dir_str = str(PTLFLOW_DIR.resolve())
    if ptlflow_dir_str not in sys.path:
        sys.path.insert(0, ptlflow_dir_str)

    module = importlib.import_module("eval_flow_divergence")

    assert hasattr(module, "evaluate_pair_synthetic")
    assert (PTLFLOW_DIR / "dpflow-things-2012b5d6.ckpt").is_file()
    assert (PTLFLOW_DIR / "calibration.json").is_file()
