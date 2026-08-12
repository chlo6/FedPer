from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import tempfile
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
    combinations = [
        (rounds, local_epochs)
        for rounds in args.rounds
        for local_epochs in args.local_epochs
    ]
    print(
        f"Running {len(combinations)} configurations with "
        f"{runner_path.relative_to(ROOT)}"
    )

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
                raise SystemExit(message)


if __name__ == "__main__":
    main()
