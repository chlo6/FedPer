from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.artifact_contract import load_validated_artifact
from redo_by_sara.config import load_config
from redo_by_sara.training import create_loaders, create_model, fit, run_epoch


def _best_history_row(task: str, history: list[dict[str, float]]) -> dict[str, float]:
    if task == "regression":
        return min(history, key=lambda row: row["val_score"])
    return max(history, key=lambda row: row["val_score"])


def _candidate_is_better(
    task: str,
    candidate: dict[str, Any],
    current_best: dict[str, Any] | None,
) -> bool:
    if current_best is None:
        return True
    if task == "regression":
        return float(candidate["best_val_score"]) < float(current_best["best_val_score"])
    return float(candidate["best_val_score"]) > float(current_best["best_val_score"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune a centralized CNN using validation only, then evaluate the "
            "single selected model once on the held-out test split."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-variants", nargs="+", default=["simple", "deep"])
    parser.add_argument(
        "--learning-rates", nargs="+", type=float, default=[0.001, 0.0005]
    )
    parser.add_argument(
        "--weight-decays", nargs="+", type=float, default=[0.0001, 0.00001]
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32])
    parser.add_argument("--epochs", nargs="+", type=int, default=[15, 30])
    args = parser.parse_args()

    config = load_config(args.config)
    artifact = load_validated_artifact(config.artifact_path, config)
    task = config.training.task
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None

    candidates = itertools.product(
        args.model_variants,
        args.learning_rates,
        args.weight_decays,
        args.batch_sizes,
        args.epochs,
    )
    for candidate_id, (
        model_variant,
        learning_rate,
        weight_decay,
        batch_size,
        epochs,
    ) in enumerate(candidates, start=1):
        model, history, test_result = fit(
            artifact=artifact,
            task=task,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_workers=config.training.num_workers,
            seed=config.seed,
            model_variant=model_variant,
            evaluate_test=False,
        )
        if test_result is not None:
            raise RuntimeError("Tuning candidates must not evaluate the test split.")
        best_history = _best_history_row(task, history)
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "model_variant": model_variant,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
            "best_epoch": int(best_history["epoch"]),
            "best_val_loss": float(best_history["val_loss"]),
            "best_val_score": float(best_history["val_score"]),
        }
        rows.append(row)
        print(json.dumps(row))
        if _candidate_is_better(task, row, best_row):
            best_row = row
            best_state = copy.deepcopy(model.state_dict())

    if best_row is None or best_state is None:
        raise RuntimeError("No centralized tuning candidates were evaluated.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = create_model(
        artifact,
        task,
        model_variant=str(best_row["model_variant"]),
    ).to(device)
    best_model.load_state_dict(best_state)
    _, _, test_loader = create_loaders(
        artifact,
        task,
        batch_size=int(best_row["batch_size"]),
        num_workers=config.training.num_workers,
    )
    criterion: nn.Module = (
        nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()
    )
    test_result = run_epoch(
        best_model,
        test_loader,
        criterion,
        optimizer=None,
        device=device,
        task=task,
    )

    result_stem = f"centralized_{task}_tuning"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = config.output_dir / f"{result_stem}.csv"
    model_path = config.output_dir / f"{result_stem}_best.pt"
    summary_path = config.output_dir / f"{result_stem}_summary.json"

    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(best_model.state_dict(), model_path)

    summary = {
        "algorithm": "centralized",
        "task": task,
        "score_name": "rmse" if task == "regression" else "accuracy",
        "selection_rule": "validation only; test evaluated once after selection",
        "seed": config.seed,
        "num_candidates": len(rows),
        "best_hyperparameters": best_row,
        "test_loss": float(test_result.loss),
        "test_score": float(test_result.score),
        "model_path": str(model_path),
        "candidate_table_path": str(table_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
