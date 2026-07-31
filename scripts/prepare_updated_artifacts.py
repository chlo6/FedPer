from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _sample_digest(sample: torch.Tensor) -> bytes:
    return hashlib.sha256(sample.contiguous().numpy().tobytes()).digest()


def add_background_class(
    classification: dict[str, object],
    regression: dict[str, object],
) -> tuple[dict[str, object], int]:
    """Label classification-only windows as background/person 0.

    The updated dataset contract defines the 1,399 walking windows as the
    regression set and adds 625 no-walking windows only to classification.
    Exact signal matching identifies those classification-only windows without
    relying on their currently incorrect subject metadata.
    """
    classification_samples = classification["samples"]
    regression_samples = regression["samples"]
    remaining = Counter(_sample_digest(sample) for sample in regression_samples)
    background_indices: list[int] = []
    for index, sample in enumerate(classification_samples):
        digest = _sample_digest(sample)
        if remaining[digest] > 0:
            remaining[digest] -= 1
        else:
            background_indices.append(index)

    unmatched_regression = sum(remaining.values())
    if unmatched_regression:
        raise ValueError(
            f"{unmatched_regression} regression windows were not found in classification."
        )
    if not background_indices:
        if "0" in classification["subject_to_class"]:
            return classification, 0
        raise ValueError("No classification-only windows were found for background class 0.")

    old_mapping = {
        str(subject): int(class_id)
        for subject, class_id in classification["subject_to_class"].items()
    }
    if "0" in old_mapping:
        raise ValueError(
            "Artifact already maps subject 0 but still has unmatched windows; "
            "refusing to guess their labels."
        )

    repaired = copy.deepcopy(classification)
    new_mapping = {"0": 0}
    for subject, _ in sorted(old_mapping.items(), key=lambda item: item[1]):
        new_mapping[subject] = len(new_mapping)

    targets = repaired["classification_targets"].clone().long() + 1
    targets[torch.tensor(background_indices, dtype=torch.long)] = 0
    repaired["classification_targets"] = targets
    repaired["subject_to_class"] = new_mapping

    for index in background_indices:
        repaired["metadata"][index]["background_source_subject_id"] = str(
            repaired["metadata"][index]["subject_id"]
        )
        repaired["metadata"][index]["subject_id"] = "0"

    summary = repaired["summary"]
    summary["subjects"] = list(new_mapping)
    summary["num_classes"] = len(new_mapping)
    summary["subject_to_class"] = new_mapping
    summary["background_subject_id"] = "0"
    summary["num_background_examples"] = len(background_indices)
    repaired["background_indices"] = torch.tensor(background_indices, dtype=torch.long)
    return repaired, len(background_indices)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and repair the updated task-specific window artifacts."
    )
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument(
        "--expected-background-windows",
        type=int,
        default=625,
    )
    args = parser.parse_args()

    classification_path = args.artifact_dir / "raw_windows_classification.pt"
    regression_path = args.artifact_dir / "raw_windows_regression.pt"
    classification = torch.load(classification_path, map_location="cpu", weights_only=False)
    regression = torch.load(regression_path, map_location="cpu", weights_only=False)

    repaired, count = add_background_class(classification, regression)
    if count and count != args.expected_background_windows:
        raise ValueError(
            f"Found {count} background windows, expected "
            f"{args.expected_background_windows}; no file was changed."
        )

    if count:
        backup_path = classification_path.with_suffix(".before_background_fix.pt")
        if not backup_path.exists():
            classification_path.replace(backup_path)
        torch.save(repaired, classification_path)
        print(f"Repaired {classification_path} with {count} person-0 windows.")
        print(f"Original saved as {backup_path}.")
    else:
        print(f"{classification_path} already contains class 0; no change needed.")

    print(
        f"classification={tuple(repaired['samples'].shape)}, "
        f"classes={repaired['subject_to_class']}"
    )
    print(f"regression={tuple(regression['samples'].shape)}")


if __name__ == "__main__":
    main()
