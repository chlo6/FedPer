"""Run genuine IID FedAvg with historical client self-distillation."""

from __future__ import annotations

from run_flower import main


if __name__ == "__main__":
    main(force_iid=True)
