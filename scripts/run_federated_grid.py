from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a federated experiment grid over communication rounds and "
            "local epochs without modifying the source YAML configuration."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/run_flower.py"),
        help=(
            "Federated entry point. Use scripts/run_flower_iid.py for the "
            "whole-run IID pipeline."
        ),
    )
    parser.add_argument("--rounds", nargs="+", type=int, default=[30, 60])
    parser.add_argument(
        "--local-epochs",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next combination if an experiment fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the six commands and generated names without training.",
    )
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _validate_positive(values: list[int], label: str) -> None:
    if not values or any(value < 1 for value in values):
        raise ValueError(f"{label} must contain positive integers.")


def _tag_for_local_epochs(local_epochs: int) -> str:
    suffix = "epoch" if local_epochs == 1 else "epochs"
    return f"{local_epochs}-local-{suffix}"


def _build_run_config(
    source: dict[str, Any],
    source_stem: str,
    rounds: int,
    local_epochs: int,
) -> tuple[dict[str, Any], str]:
    payload = copy.deepcopy(source)
    if "federated" not in payload:
        raise ValueError("The configuration does not contain a federated section.")

    federated = payload["federated"]
    federated["num_rounds"] = rounds
    federated["local_epochs"] = local_epochs

    base_name = federated.get("result_name") or source_stem
    result_name = f"{base_name}_{rounds}r_{local_epochs}e"
    federated["result_name"] = result_name

    wandb = payload.setdefault("wandb", {})
    tags = [str(tag) for tag in wandb.get("tags", [])]
    for tag in (
        "grid-search",
        f"{rounds}-rounds",
        _tag_for_local_epochs(local_epochs),
    ):
        if tag not in tags:
            tags.append(tag)
    wandb["tags"] = tags
    return payload, result_name


def _find_run_summary(output_dir: Path, result_name: str) -> Path:
    matches = list(output_dir.glob(f"*{result_name}*federated_summary.json"))
    if not matches:
        raise FileNotFoundError(
            f"No federated summary was found for result_name={result_name}."
        )
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _collect_metrics(
    summary_path: Path,
    rounds: int,
    local_epochs: int,
    result_name: str,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    return {
        "result_name": result_name,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "status": "completed",
        "best_val_round": summary.get("best_val_round"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_val_score": summary.get("best_val_score"),
        "best_val_r2": summary.get("best_val_r2"),
        "test_loss": summary.get("test_loss"),
        "test_score": summary.get("test_score"),
        "test_r2": summary.get("test_r2"),
        "summary_path": str(summary_path),
    }


def _metric_text(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _print_final_table(rows: list[dict[str, Any]], task: str) -> None:
    score_label = "test_accuracy" if task == "classification" else "test_rmse"
    columns = [
        ("rounds", "rounds"),
        ("epochs", "local_epochs"),
        ("status", "status"),
        ("val_loss", "best_val_loss"),
        ("val_score", "best_val_score"),
        ("test_loss", "test_loss"),
        (score_label, "test_score"),
    ]
    if task == "regression":
        columns.append(("test_r2", "test_r2"))

    widths = {
        label: max(
            len(label),
            *(len(_metric_text(row.get(key))) for row in rows),
        )
        for label, key in columns
    }
    print("\nFINAL GRID RESULTS")
    print("  ".join(label.ljust(widths[label]) for label, _ in columns))
    print("  ".join("-" * widths[label] for label, _ in columns))
    for row in rows:
        print(
            "  ".join(
                _metric_text(row.get(key)).ljust(widths[label])
                for label, key in columns
            )
        )


def _save_grid_summary(
    rows: list[dict[str, Any]],
    output_dir: Path,
    base_name: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = output_dir / f"{base_name}_grid_summary_{timestamp}"
    json_path = stem.with_suffix(".json")
    csv_path = stem.with_suffix(".csv")
    json_path.write_text(json.dumps(rows, indent=2))

    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, json_path


def main() -> None:
    args = _parse_args()
    _validate_positive(args.rounds, "--rounds")
    _validate_positive(args.local_epochs, "--local-epochs")

    config_path = _resolve_project_path(args.config)
    runner_path = _resolve_project_path(args.runner)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    if not runner_path.is_file():
        raise FileNotFoundError(f"Runner not found: {runner_path}")

    source: dict[str, Any] = yaml.safe_load(config_path.read_text())
    task = str(source.get("training", {}).get("task", "unknown"))
    output_dir_raw = Path(str(source.get("output_dir", "artifacts")))
    output_dir = (
        output_dir_raw
        if output_dir_raw.is_absolute()
        else (ROOT / output_dir_raw).resolve()
    )
    base_name = (
        source.get("federated", {}).get("result_name") or config_path.stem
    )
    combinations = [
        (rounds, local_epochs)
        for rounds in args.rounds
        for local_epochs in args.local_epochs
    ]
    print(
        f"Running {len(combinations)} configurations with "
        f"{runner_path.relative_to(ROOT)}"
    )

    rows: list[dict[str, Any]] = []
    pending_error: str | None = None
    for index, (rounds, local_epochs) in enumerate(combinations, start=1):
        payload, result_name = _build_run_config(
            source,
            config_path.stem,
            rounds,
            local_epochs,
        )
        print(
            f"[{index}/{len(combinations)}] rounds={rounds}, "
            f"local_epochs={local_epochs}, result_name={result_name}",
            flush=True,
        )
        if args.dry_run:
            print(
                f"  {sys.executable} {runner_path.relative_to(ROOT)} "
                f"--config <generated-config>"
            )
            continue

        # Keep the temporary YAML directly inside configs/. load_config() uses
        # the config location to resolve project-relative paths.
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix=f".grid_{result_name}_",
            dir=config_path.parent,
            delete=False,
        ) as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
            generated_path = Path(handle.name)

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(runner_path),
                    "--config",
                    str(generated_path),
                ],
                cwd=ROOT,
                check=False,
            )
        finally:
            generated_path.unlink(missing_ok=True)

        if completed.returncode != 0:
            message = (
                f"Experiment failed for {rounds} rounds and "
                f"{local_epochs} local epochs (exit {completed.returncode})."
            )
            if args.continue_on_error:
                print(f"WARNING: {message}", file=sys.stderr)
            else:
                pending_error = message
            rows.append(
                {
                    "result_name": result_name,
                    "rounds": rounds,
                    "local_epochs": local_epochs,
                    "status": "failed",
                    "best_val_round": None,
                    "best_val_loss": None,
                    "best_val_score": None,
                    "best_val_r2": None,
                    "test_loss": None,
                    "test_score": None,
                    "test_r2": None,
                    "summary_path": None,
                }
            )
            if pending_error is not None:
                break
            continue

        try:
            summary_path = _find_run_summary(output_dir, result_name)
            rows.append(
                _collect_metrics(
                    summary_path,
                    rounds,
                    local_epochs,
                    result_name,
                )
            )
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            rows.append(
                {
                    "result_name": result_name,
                    "rounds": rounds,
                    "local_epochs": local_epochs,
                    "status": "summary-missing",
                    "best_val_round": None,
                    "best_val_loss": None,
                    "best_val_score": None,
                    "best_val_r2": None,
                    "test_loss": None,
                    "test_score": None,
                    "test_r2": None,
                    "summary_path": None,
                }
            )
            print(f"WARNING: {exc}", file=sys.stderr)

    if not args.dry_run and rows:
        _print_final_table(rows, task)
        csv_path, json_path = _save_grid_summary(rows, output_dir, str(base_name))
        print(f"\nSaved grid CSV:  {csv_path}")
        print(f"Saved grid JSON: {json_path}")

    if pending_error is not None:
        raise SystemExit(pending_error)


if __name__ == "__main__":
    main()
