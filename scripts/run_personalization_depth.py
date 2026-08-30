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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare 1-, 2-, and 3-block CNNs at every valid FedPer boundary."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/run_flower.py"),
        help="Use scripts/run_flower_iid.py for IID experiments.",
    )
    parser.add_argument(
        "--conv-blocks",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Total convolution blocks to test (valid values: 1, 2, 3).",
    )
    parser.add_argument(
        "--shared-blocks",
        nargs="+",
        type=int,
        help=(
            "Optional shared depths to consider. By default, every valid "
            "depth from 1 through the total block count is tested."
        ),
    )
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--local-epochs", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def find_summary(output_dir: Path, result_name: str) -> Path:
    matches = list(output_dir.glob(f"*{result_name}*federated_summary.json"))
    if not matches:
        raise FileNotFoundError(f"No summary found for {result_name}.")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def main() -> None:
    args = parse_args()
    if not args.conv_blocks or any(depth not in (1, 2, 3) for depth in args.conv_blocks):
        raise ValueError("--conv-blocks accepts only 1, 2, and/or 3.")
    if args.shared_blocks is not None and (
        not args.shared_blocks or any(depth not in (1, 2, 3) for depth in args.shared_blocks)
    ):
        raise ValueError("--shared-blocks accepts only 1, 2, and/or 3.")
    if args.rounds is not None and args.rounds < 1:
        raise ValueError("--rounds must be positive.")
    if args.local_epochs is not None and args.local_epochs < 1:
        raise ValueError("--local-epochs must be positive.")

    config_path = project_path(args.config)
    runner_path = project_path(args.runner)
    source: dict[str, Any] = yaml.safe_load(config_path.read_text())
    if "federated" not in source:
        raise ValueError("Configuration is missing its federated section.")

    output_raw = Path(str(source.get("output_dir", "artifacts")))
    output_dir = output_raw if output_raw.is_absolute() else ROOT / output_raw
    base_name = source["federated"].get("result_name") or config_path.stem
    rows: list[dict[str, Any]] = []
    combinations = [
        (total_blocks, shared_blocks)
        for total_blocks in args.conv_blocks
        for shared_blocks in (
            args.shared_blocks
            if args.shared_blocks is not None
            else range(1, total_blocks + 1)
        )
        if shared_blocks <= total_blocks
    ]
    if not combinations:
        raise ValueError("No valid architecture/split combinations were selected.")

    print(f"Running {len(combinations)} architecture/split combinations")
    for index, (total_blocks, shared_blocks) in enumerate(combinations, start=1):
        payload = copy.deepcopy(source)
        fed = payload["federated"]
        if args.rounds is not None:
            fed["num_rounds"] = args.rounds
        if args.local_epochs is not None:
            fed["local_epochs"] = args.local_epochs
        result_name = (
            f"{base_name}_{total_blocks}conv_share{shared_blocks}_"
            f"{fed['num_rounds']}r_{fed['local_epochs']}e"
        )
        fed["num_conv_blocks"] = total_blocks
        fed["shared_conv_blocks"] = shared_blocks
        fed["result_name"] = result_name

        wandb = payload.setdefault("wandb", {})
        tags = [str(tag) for tag in wandb.get("tags", [])]
        for tag in (
            "personalization-depth-study",
            f"{total_blocks}-conv-blocks",
            f"shared-{shared_blocks}-conv-blocks",
            f"local-{total_blocks - shared_blocks}-conv-blocks",
            f"{fed['num_rounds']}-rounds",
            f"{fed['local_epochs']}-local-epochs",
        ):
            if tag not in tags:
                tags.append(tag)
        wandb["tags"] = tags

        print(
            f"[{index}/{len(combinations)}] num_conv_blocks={total_blocks}, "
            f"shared_conv_blocks={shared_blocks}, "
            f"rounds={fed['num_rounds']}, local_epochs={fed['local_epochs']}"
        )
        if args.dry_run:
            rows.append(
                {
                    "num_conv_blocks": total_blocks,
                    "shared_conv_blocks": shared_blocks,
                    "local_conv_blocks": total_blocks - shared_blocks,
                    "status": "dry-run",
                }
            )
            continue

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix=".personalization_depth_",
            dir=config_path.parent,
            delete=False,
        ) as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
            temporary_config = Path(handle.name)
        try:
            completed = subprocess.run(
                [sys.executable, str(runner_path), "--config", str(temporary_config)],
                cwd=ROOT,
                check=False,
            )
        finally:
            temporary_config.unlink(missing_ok=True)

        if completed.returncode != 0:
            rows.append(
                {
                    "num_conv_blocks": total_blocks,
                    "shared_conv_blocks": shared_blocks,
                    "local_conv_blocks": total_blocks - shared_blocks,
                    "rounds": fed["num_rounds"],
                    "local_epochs": fed["local_epochs"],
                    "status": "failed",
                }
            )
            if not args.continue_on_error:
                break
            continue

        summary_path = find_summary(output_dir, result_name)
        summary = json.loads(summary_path.read_text())
        rows.append(
            {
                "num_conv_blocks": total_blocks,
                "shared_conv_blocks": shared_blocks,
                "local_conv_blocks": total_blocks - shared_blocks,
                "rounds": fed["num_rounds"],
                "local_epochs": fed["local_epochs"],
                "status": "completed",
                "best_val_loss": summary.get("best_val_loss"),
                "best_val_score": summary.get("best_val_score"),
                "best_val_r2": summary.get("best_val_r2"),
                "test_loss": summary.get("test_loss"),
                "test_score": summary.get("test_score"),
                "test_r2": summary.get("test_r2"),
                "summary_path": str(summary_path),
            }
        )

    print("\nPERSONALIZATION DEPTH RESULTS")
    for row in rows:
        print(json.dumps(row, sort_keys=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = output_dir / f"{base_name}_personalization_depth_{timestamp}"
    json_path = stem.with_suffix(".json")
    csv_path = stem.with_suffix(".csv")
    json_path.write_text(json.dumps(rows, indent=2))
    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nSaved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
