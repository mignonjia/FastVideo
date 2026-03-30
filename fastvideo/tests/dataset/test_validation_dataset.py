import json

from fastvideo.dataset import validation_dataset as validation_dataset_module
from fastvideo.dataset.validation_dataset import ValidationDataset


def _write_validation_json(path, samples):
    path.write_text(json.dumps({"data": samples}), encoding="utf-8")


def test_validation_dataset_loads_json_data_list(tmp_path, monkeypatch):
    validation_file = tmp_path / "validation.json"
    _write_validation_json(validation_file, [{
        "caption": "sample-0",
        "num_frames": 4,
    }, {
        "caption": "sample-1",
        "num_frames": 4,
    }])

    monkeypatch.setattr(validation_dataset_module, "get_world_rank",
                        lambda: 0)
    monkeypatch.setattr(validation_dataset_module, "get_world_size",
                        lambda: 1)
    monkeypatch.setattr(validation_dataset_module, "get_sp_world_size",
                        lambda: 1)

    dataset = ValidationDataset(str(validation_file))

    assert dataset.original_total_samples == 2
    assert len(dataset) == 2
    assert [sample["prompt"] for sample in dataset] == ["sample-0", "sample-1"]


def test_validation_dataset_balances_samples_across_sp_groups(
        tmp_path, monkeypatch):
    validation_file = tmp_path / "validation.json"
    _write_validation_json(validation_file, [{
        "caption": "sample-0",
        "num_frames": 4,
    }, {
        "caption": "sample-1",
        "num_frames": 4,
    }, {
        "caption": "sample-2",
        "num_frames": 4,
    }])

    monkeypatch.setattr(validation_dataset_module, "get_world_rank",
                        lambda: 2)
    monkeypatch.setattr(validation_dataset_module, "get_world_size",
                        lambda: 4)
    monkeypatch.setattr(validation_dataset_module, "get_sp_world_size",
                        lambda: 2)

    dataset = ValidationDataset(str(validation_file))

    assert dataset.original_total_samples == 3
    assert dataset.total_samples == 4
    assert len(dataset) == 2
    assert [sample["prompt"] for sample in dataset] == ["sample-2", "sample-0"]
