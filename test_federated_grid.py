from __future__ import annotations

import json

from scripts.run_federated_grid import (
    _build_run_config,
    _collect_metrics,
    _find_run_summary,
)


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


def test_collects_metrics_from_generated_summary(tmp_path) -> None:
    summary_path = tmp_path / "classification_demo_30r_2e_federated_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "best_val_round": 27,
                "best_val_loss": 0.2,
                "best_val_score": 0.91,
                "test_loss": 0.3,
                "test_score": 0.9,
            }
        )
    )

    found = _find_run_summary(tmp_path, "demo_30r_2e")
    row = _collect_metrics(found, 30, 2, "demo_30r_2e")

    assert row["status"] == "completed"
    assert row["best_val_round"] == 27
    assert row["test_loss"] == 0.3
    assert row["test_score"] == 0.9
