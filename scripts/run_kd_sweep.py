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

CLASSIFICATION_DEFAULT_WEIGHTS = [0.1, 0.25, 0.5, 1.0]
CLASSIFICATION_DEFAULT_TEMPERATURES = [1.0, 2.0, 4.0]
CLASSIFICATION_DEFAULT_START_ROUNDS = [2]

# Regression distillation compares scalar teacher/student predictions with
# MSE, so softmax temperature has no effect. Sweep when distillation starts
# instead, as well as a slightly wider range of KD weights.
REGRESSION_DEFAULT_WEIGHTS = [0.1, 0.25, 0.5, 1.0, 2.0]
REGRESSION_DEFAULT_TEMPERATURES = [1.0]
REGRESSION_DEFAULT_START_ROUNDS = [2, 5, 10]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep historical self-distillation parameters while keeping "
            "the federated rounds and local epochs from the source config fixed."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/run_flower.py"),
        help="Use run_flower_iid.py for IID or run_flower.py for non-IID.",
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help="KD weights. Defaults depend on the task in the config.",
    )
    parser.add_argument(
        "--temperatures", nargs="+", type=float, default=None,
        help="KD temperatures. Regression is fixed at 1 because it uses MSE KD.",
    )
    parser.add_argument(
        "--start-rounds", nargs="+", type=int, default=None,
        help="Rounds at which KD begins. Defaults depend on the task.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        help="Override num_rounds from the YAML for every sweep run.",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        help="Override local_epochs from the YAML for every sweep run.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after a failed combination.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show combinations without training.",
    )
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _validate_values(
    weights: list[float], temperatures: list[float], start_rounds: list[int]
) -> None:
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("--weights must contain positive values.")
    if not temperatures or any(temperature <= 0 for temperature in temperatures):
        raise ValueError("--temperatures must contain positive values.")
    if not start_rounds or any(start_round < 2 for start_round in start_rounds):
        raise ValueError("--start-rounds must be at least 2; round 1 has no teacher.")


def _resolve_sweep_values(
    task: str,
    weights: list[float] | None,
    temperatures: list[float] | None,
    start_rounds: list[int] | None,
) -> tuple[list[float], list[float], list[int]]:
    if task == "classification":
        return (
            weights or CLASSIFICATION_DEFAULT_WEIGHTS,
            temperatures or CLASSIFICATION_DEFAULT_TEMPERATURES,
            start_rounds or CLASSIFICATION_DEFAULT_START_ROUNDS,
        )
    if task == "regression":
        resolved_temperatures = temperatures or REGRESSION_DEFAULT_TEMPERATURES
        if any(temperature != 1.0 for temperature in resolved_temperatures):
            raise ValueError(
                "Regression KD uses MSE between scalar predictions, so "
                "temperature has no effect. Use --temperatures 1."
            )
        return (
            weights or REGRESSION_DEFAULT_WEIGHTS,
            resolved_temperatures,
            start_rounds or REGRESSION_DEFAULT_START_ROUNDS,
        )
    raise ValueError(f"Unsupported training task for KD sweep: {task!r}.")


def _number_token(value: float) -> str:
    return format(value, "g").replace("-", "m").replace(".", "p")


def _build_run_config(
    source: dict[str, Any],
    source_stem: str,
    weight: float,
    temperature: float,
    start_round: int,
    rounds: int | None = None,
    local_epochs: int | None = None,
) -> tuple[dict[str, Any], str]:
    payload = copy.deepcopy(source)
    if "federated" not in payload:
        raise ValueError("The configuration does not contain a federated section.")

    kd = payload.setdefault("knowledge_distillation", {})
    kd.update(
        {
            "enabled": True,
            "weight": weight,
            "temperature": temperature,
            "start_round": start_round,
        }
    )

    federated = payload["federated"]
    if rounds is not None:
        federated["num_rounds"] = rounds
    if local_epochs is not None:
        federated["local_epochs"] = local_epochs
    num_rounds = int(federated["num_rounds"])
    num_local_epochs = int(federated["local_epochs"])
    if start_round > num_rounds:
        raise ValueError(
            f"KD start round {start_round} exceeds num_rounds={num_rounds}."
        )

    base_name = federated.get("result_name") or source_stem
    suffix = (
        f"{num_rounds}r_{num_local_epochs}e"
        f"_kdw{_number_token(weight)}"
        f"_t{_number_token(temperature)}"
        f"_sr{start_round}"
    )
    result_name = f"{base_name}_{suffix}"
    federated["result_name"] = result_name

    wandb = payload.setdefault("wandb", {})
    tags = [str(tag) for tag in wandb.get("tags", [])]
    for tag in (
        "kd-sweep",
        f"kd-weight-{format(weight, 'g')}",
        f"kd-temperature-{format(temperature, 'g')}",
        f"kd-start-round-{start_round}",
    ):
        if tag not in tags:
            tags.append(tag)
    wandb["tags"] = tags
    return payload, result_name


def _find_run_summary(output_dir: Path, result_name: str) -> Path:
    matches = list(output_dir.glob(f"*{result_name}*federated_summary.json"))
    if not matches:
        raise FileNotFoundError(
            f"No federated summary found for result_name={result_name}."
        )
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _empty_row(
    result_name: str,
    weight: float,
    temperature: float,
    start_round: int,
    status: str,
) -> dict[str, Any]:
    return {
        "result_name": result_name,
        "kd_weight": weight,
        "kd_temperature": temperature,
        "kd_start_round": start_round,
        "status": status,
        "best_val_round": None,
        "best_val_loss": None,
        "best_val_score": None,
        "best_val_r2": None,
        "test_loss": None,
        "test_score": None,
        "test_r2": None,
        "summary_path": None,
    }


def _collect_metrics(
    summary_path: Path,
    result_name: str,
    weight: float,
    temperature: float,
    start_round: int,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    row = _empty_row(
        result_name, weight, temperature, start_round, "completed"
    )
    row.update(
        {
            "best_val_round": summary.get("best_val_round"),
            "best_val_loss": summary.get("best_val_loss"),
            "best_val_score": summary.get("best_val_score"),
            "best_val_r2": summary.get("best_val_r2"),
            "test_loss": summary.get("test_loss"),
            "test_score": summary.get("test_score"),
            "test_r2": summary.get("test_r2"),
            "summary_path": str(summary_path),
        }
    )
    return row


def _metric_text(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _print_final_table(rows: list[dict[str, Any]], task: str) -> None:
    score_label = "test_accuracy" if task == "classification" else "test_rmse"
    columns = [
        ("weight", "kd_weight"),
        ("temp", "kd_temperature"),
        ("start", "kd_start_round"),
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
            len(label), *(len(_metric_text(row.get(key))) for row in rows)
        )
        for label, key in columns
    }
    print("\nFINAL KD SWEEP RESULTS")
    print("  ".join(label.ljust(widths[label]) for label, _ in columns))
    print("  ".join("-" * widths[label] for label, _ in columns))
    for row in rows:
        print(
            "  ".join(
                _metric_text(row.get(key)).ljust(widths[label])
                for label, key in columns
            )
        )

    completed = [row for row in rows if row["status"] == "completed"]
    if not completed:
        return
    candidates = [row for row in completed if row["best_val_score"] is not None]
    if not candidates:
        return
    best = (
        min(candidates, key=lambda row: float(row["best_val_score"]))
        if task == "regression"
        else max(candidates, key=lambda row: float(row["best_val_score"]))
    )
    print(
        "\nBEST BY VALIDATION SCORE: "
        f"weight={best['kd_weight']}, temperature={best['kd_temperature']}, "
        f"start_round={best['kd_start_round']}, "
        f"val_score={_metric_text(best['best_val_score'])}"
    )


def _save_summary(
    rows: list[dict[str, Any]], output_dir: Path, base_name: str
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = output_dir / f"{base_name}_kd_sweep_summary_{timestamp}"
    json_path = stem.with_suffix(".json")
    csv_path = stem.with_suffix(".csv")
    json_path.write_text(json.dumps(rows, indent=2))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, json_path


def main() -> None:
    args = _parse_args()
    if args.rounds is not None and args.rounds < 1:
        raise ValueError("--rounds must be positive.")
    if args.local_epochs is not None and args.local_epochs < 1:
        raise ValueError("--local-epochs must be positive.")
    config_path = _resolve_project_path(args.config)
    runner_path = _resolve_project_path(args.runner)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    if not runner_path.is_file():
        raise FileNotFoundError(f"Runner not found: {runner_path}")

    source: dict[str, Any] = yaml.safe_load(config_path.read_text())
    task = str(source.get("training", {}).get("task", "unknown"))
    weights, temperatures, start_rounds = _resolve_sweep_values(
        task, args.weights, args.temperatures, args.start_rounds
    )
    _validate_values(weights, temperatures, start_rounds)
    federated = source.get("federated", {})
    rounds = (
        args.rounds
        if args.rounds is not None
        else int(federated.get("num_rounds", 0))
    )
    local_epochs = (
        args.local_epochs
        if args.local_epochs is not None
        else int(federated.get("local_epochs", 0))
    )
    if rounds < 1 or local_epochs < 1:
        raise ValueError("The config must define positive rounds and local epochs.")

    output_dir_raw = Path(str(source.get("output_dir", "artifacts")))
    output_dir = (
        output_dir_raw
        if output_dir_raw.is_absolute()
        else (ROOT / output_dir_raw).resolve()
    )
    base_name = federated.get("result_name") or config_path.stem
    combinations = [
        (weight, temperature, start_round)
        for weight in weights
        for temperature in temperatures
        for start_round in start_rounds
    ]
    print(
        f"Running {len(combinations)} {task} KD configurations with "
        f"{runner_path.relative_to(ROOT)}; rounds={rounds}, "
        f"local_epochs={local_epochs}"
    )

    rows: list[dict[str, Any]] = []
    pending_error: str | None = None
    for index, (weight, temperature, start_round) in enumerate(
        combinations, start=1
    ):
        payload, result_name = _build_run_config(
            source,
            config_path.stem,
            weight,
            temperature,
            start_round,
            rounds=rounds,
            local_epochs=local_epochs,
        )
        print(
            f"[{index}/{len(combinations)}] weight={weight:g}, "
            f"temperature={temperature:g}, start_round={start_round}, "
            f"result_name={result_name}",
            flush=True,
        )
        if args.dry_run:
            print(
                f"  {sys.executable} {runner_path.relative_to(ROOT)} "
                "--config <generated-config>"
            )
            continue

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix=f".kd_sweep_{result_name}_",
            dir=config_path.parent,
            delete=False,
        ) as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
            generated_path = Path(handle.name)

        try:
            completed = subprocess.run(
                [sys.executable, str(runner_path), "--config", str(generated_path)],
                cwd=ROOT,
                check=False,
            )
        finally:
            generated_path.unlink(missing_ok=True)

        if completed.returncode != 0:
            row = _empty_row(
                result_name, weight, temperature, start_round, "failed"
            )
            rows.append(row)
            message = (
                f"KD experiment failed for weight={weight:g}, "
                f"temperature={temperature:g}, start_round={start_round} "
                f"(exit {completed.returncode})."
            )
            if args.continue_on_error:
                print(f"WARNING: {message}", file=sys.stderr)
                continue
            pending_error = message
            break

        try:
            summary_path = _find_run_summary(output_dir, result_name)
            rows.append(
                _collect_metrics(
                    summary_path,
                    result_name,
                    weight,
                    temperature,
                    start_round,
                )
            )
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            rows.append(
                _empty_row(
                    result_name,
                    weight,
                    temperature,
                    start_round,
                    "summary-missing",
                )
            )
            print(f"WARNING: {exc}", file=sys.stderr)

    if not args.dry_run and rows:
        _print_final_table(rows, task)
        csv_path, json_path = _save_summary(rows, output_dir, str(base_name))
        print(f"\nSaved KD sweep CSV:  {csv_path}")
        print(f"Saved KD sweep JSON: {json_path}")

    if pending_error is not None:
        raise SystemExit(pending_error)


if __name__ == "__main__":
    main()
