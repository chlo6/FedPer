"""Run genuine IID FedPer using the shared-base/personal-head implementation."""

from __future__ import annotations

from run_flower import main


if __name__ == "__main__":
    main(force_iid=True)
