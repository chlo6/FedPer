from __future__ import annotations

import pytest

from scripts.run_kd_sweep import _build_run_config, _validate_values


def _source() -> dict[str, object]:
    return {
        "federated": {
            "result_name": "fedavg_kd",
            "num_rounds": 30,
            "local_epochs": 1,
        },
        "knowledge_distillation": {
            "enabled": True,
            "weight": 0.5,
            "temperature": 2.0,
            "start_round": 2,
        },
        "wandb": {"tags": ["fedavg-kd"]},
    }


def test_build_run_config_changes_only_sweep_values() -> None:
    source = _source()
    payload, result_name = _build_run_config(
        source, "unused", 0.25, 4.0, 5, rounds=60, local_epochs=2
    )

    assert result_name == "fedavg_kd_60r_2e_kdw0p25_t4_sr5"
    assert payload["knowledge_distillation"] == {
        "enabled": True,
        "weight": 0.25,
        "temperature": 4.0,
        "start_round": 5,
    }
    assert payload["federated"]["num_rounds"] == 60
    assert payload["federated"]["local_epochs"] == 2
    assert source["knowledge_distillation"]["weight"] == 0.5
    assert "kd-sweep" in payload["wandb"]["tags"]


def test_invalid_sweep_values_fail_early() -> None:
    with pytest.raises(ValueError):
        _validate_values([0.0], [2.0], [2])
    with pytest.raises(ValueError):
        _validate_values([0.5], [0.0], [2])
    with pytest.raises(ValueError):
        _validate_values([0.5], [2.0], [1])


def test_start_round_cannot_exceed_training_rounds() -> None:
    with pytest.raises(ValueError):
        _build_run_config(_source(), "unused", 0.5, 2.0, 31)
