from __future__ import annotations

from scripts.run_federated_grid import _build_run_config


def test_grid_config_has_unique_result_name_and_wandb_tags() -> None:
    source = {
        "federated": {
            "result_name": "semi_non_iid",
            "num_rounds": 5,
            "local_epochs": 1,
        },
        "wandb": {"tags": ["classification"]},
    }

    generated, result_name = _build_run_config(
        source,
        "ignored",
        rounds=60,
        local_epochs=3,
    )

    assert result_name == "semi_non_iid_60r_3e"
    assert generated["federated"]["num_rounds"] == 60
    assert generated["federated"]["local_epochs"] == 3
    assert generated["wandb"]["tags"] == [
        "classification",
        "grid-search",
        "60-rounds",
        "3-local-epochs",
    ]
    assert source["federated"]["num_rounds"] == 5
    assert source["wandb"]["tags"] == ["classification"]
