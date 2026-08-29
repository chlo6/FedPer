from scripts.tune_centralized import _best_history_row, _candidate_is_better


def test_classification_epoch_tie_uses_lower_validation_loss() -> None:
    history = [
        {"val_score": 1.0, "val_loss": 0.20},
        {"val_score": 1.0, "val_loss": 0.05},
    ]
    assert _best_history_row("classification", history) == history[1]


def test_classification_candidate_tie_uses_lower_validation_loss() -> None:
    current = {"best_val_score": 1.0, "best_val_loss": 0.20}
    candidate = {"best_val_score": 1.0, "best_val_loss": 0.05}
    assert _candidate_is_better("classification", candidate, current)


def test_accuracy_still_has_priority_over_loss() -> None:
    current = {"best_val_score": 0.99, "best_val_loss": 0.01}
    candidate = {"best_val_score": 1.0, "best_val_loss": 0.50}
    assert _candidate_is_better("classification", candidate, current)
