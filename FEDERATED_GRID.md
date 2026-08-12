# Federated round/local-epoch grid

`scripts/run_federated_grid.py` runs all six default combinations sequentially:

- 30 rounds with 1, 2, and 3 local epochs
- 60 rounds with 1, 2, and 3 local epochs

It creates a temporary YAML file for each experiment and leaves the source
configuration unchanged. Every experiment receives a unique `result_name` and
the W&B tags `grid-search`, `<N>-rounds`, and `<N>-local-epoch(s)`.

Subject-partitioned example:

```bash
python scripts/run_federated_grid.py \
  --config configs/fl_classification.yaml
```

Genuine IID example:

```bash
python scripts/run_federated_grid.py \
  --config configs/fl_iid_classification.yaml \
  --runner scripts/run_flower_iid.py
```

Preview the commands without training:

```bash
python scripts/run_federated_grid.py \
  --config configs/fl_classification.yaml \
  --dry-run
```

Custom values can be supplied with `--rounds` and `--local-epochs`. By default,
the grid stops on the first failed experiment. Add `--continue-on-error` to run
the remaining combinations after a failure.
