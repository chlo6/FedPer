# FedAvg with historical client self-distillation

This branch implements federated learning with knowledge distillation and no
FedPer personalization. The server sends and averages the complete model,
including all convolutional and linear layers. No client-specific head is kept.

Each client does retain one private historical full model for use as a frozen
teacher. Round 1 uses ordinary supervised training and saves the trained local
model. From round 2 onward, the current global model is the student and the
same client's previous local model is the teacher. The complete trained student
is returned to the server for FedAvg aggregation and is also saved locally as
that client's next teacher.

Classification uses temperature-scaled KL divergence:

```text
total = cross_entropy(student, labels)
        + weight * temperature^2 * KL(teacher || student)
```

Regression uses MSE between student and teacher predictions as its distillation
term. The initial configuration is `weight: 0.5`, `temperature: 2.0`, and
`start_round: 2`.

Run subject-partitioned experiments with:

```bash
python scripts/run_flower.py --config configs/fl_classification.yaml
```

Run genuine whole-run IID experiments with:

```bash
python scripts/run_flower_iid.py --config configs/fl_iid_classification.yaml
```

W&B logs `train_supervised_loss`, `train_kd_loss`, `train_total_loss`, and
`kd_active` for each client and at the federated level.
