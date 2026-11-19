
import csv
from pathlib import Path
from collections import Counter, defaultdict

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

if not (DATA / "train.csv").exists():
    DATA = ROOT.parent / "data"

H = 32
EMBED_DIM = 16
BATCH_SIZE = 2048

WORKLOADS = {
    0: "MCF",
    1: "GCC",
    2: "Perlbench",
    3: "X264",
    4: "Leela",
    5: "Deepsjeng",
    6: "XZ",
    7: "OMNeT++",
    8: "Xalancbmk",
    9: "Exchange2",
    10: "WRF",
    11: "Imagick"
}

# Recreate the architecture used during training
class BranchMLP(nn.Module):
    def __init__(self, num_pcs):
        super().__init__()

        self.embedding = nn.Embedding(
            num_pcs + 1, EMBED_DIM, padding_idx=0
        )

        self.layers = nn.Sequential(
            nn.Linear(EMBED_DIM + H, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, pc, history):
        features = torch.cat(
            [self.embedding(pc), history], dim=1
        )
        return self.layers(features).squeeze(1)


# Load the trained model
checkpoint = torch.load(
    ROOT / "checkpoints/mlp_12workloads.pt",
    map_location="cpu"
)

pc_to_id = checkpoint["pc_to_id"]
model = BranchMLP(len(pc_to_id))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Train-set-only PC-majority reference
counts = defaultdict(Counter)
global_counts = Counter()

with open(DATA / "train.csv") as f:
    for row in csv.DictReader(f):
        pc = int(row["pc"])
        taken = int(row["taken"])
        counts[pc][taken] += 1
        global_counts[taken] += 1

default_prediction = int(
    global_counts[1] >= global_counts[0]
)

with open(DATA / "test.csv") as f:
    test_rows = list(csv.DictReader(f))

# Results: total, MLP correct, PC-majority correct
results = defaultdict(lambda: [0, 0, 0])

with torch.no_grad():
    for start in range(0, len(test_rows), BATCH_SIZE):
        batch = test_rows[start:start + BATCH_SIZE]

        pc = torch.tensor([
            pc_to_id.get(int(r["pc"]), 0)
            for r in batch
        ], dtype=torch.long)

        history = torch.tensor([
            [int(r[f"h{i}"]) for i in range(H)]
            for r in batch
        ], dtype=torch.float32)

        predictions = (model(pc, history) >= 0).int().tolist()

        for row, prediction in zip(batch, predictions):
            pc_key = int(row["pc"])
            actual = int(row["taken"])
            workload = WORKLOADS[int(row["workload_id"])]

            if pc_key in counts:
                pc_prediction = int(
                    counts[pc_key][1] >= counts[pc_key][0]
                )
            else:
                pc_prediction = default_prediction

            results[workload][0] += 1
            results[workload][1] += int(prediction == actual)
            results[workload][2] += int(pc_prediction == actual)

print("\nPER-WORKLOAD TEST RESULTS")
print(f"{'Workload':<12} {'Samples':>9} {'MLP Acc':>11} {'PC Acc':>11}")

for workload in WORKLOADS.values():
    n, mlp_correct, pc_correct = results[workload]
    print(
        f"{workload:<12} {n:>9} "
        f"{100 * mlp_correct / n:>10.2f}% "
        f"{100 * pc_correct / n:>10.2f}%"
    )

total = [sum(r[i] for r in results.values()) for i in range(3)]
n, mlp_correct, pc_correct = total

print("\nOVERALL")
print(f"MLP accuracy: {100 * mlp_correct / n:.2f}%")
print(f"MLP misprediction rate: {100 * (n - mlp_correct) / n:.2f}%")
print(f"PC-majority accuracy: {100 * pc_correct / n:.2f}%")
