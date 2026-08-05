# FedPer with self-knowledge distillation

This branch extends FedPer with a historical model teacher for every client.

During round 1, each client trains normally because no historical teacher exists.
At the end of the round, the complete trained local model is saved. In later
rounds, the server's newly averaged base is joined to the client's retained head
to form the student, while the complete model saved from the preceding round is
loaded as a frozen teacher. Both models receive the same local input.

For classification, the objective is:

```text
CE(student_logits, true_label)
+ weight * temperature^2 * KL(teacher_probabilities || student_probabilities)
```

For regression, the distillation term is the MSE between the teacher and student
predictions. Gradients update the student's shared base and local head. The
teacher is frozen, and only the student's shared base is returned to the server.

The four FedPer configuration files enable KD with an initial setting of
`weight: 0.5`, `temperature: 2.0`, and `start_round: 2`. These are tunable
hyperparameters, not claimed optimal values.

Run an experiment as usual, for example:

```bash
python scripts/run_flower.py --config configs/fl_classification.yaml
```

For genuine IID FedPerKD, use the IID entry point. It assigns whole training
runs from every subject across clients while leaving validation and test runs
globally held out:

```bash
python scripts/run_flower_iid.py --config configs/fl_iid_classification.yaml
```

W&B records `train_supervised_loss`, `train_kd_loss`, `train_total_loss`, and
`kd_active` at both the client and federated levels. `train_loss` remains the
ordinary supervised evaluation loss so comparisons with the FedPer baseline
remain interpretable.
