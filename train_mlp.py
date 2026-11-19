
import csv
from pathlib import Path

import torch 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

H = 32
EMBED_DIM = 16
EPOCHS = 6
BATCH_SIZE = 1024

# Locate our preprocessed data
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

if not (DATA / "train.csv").exists():
    DATA = ROOT.parent / "data"

def read_csv(filename):
    with open(DATA / filename) as f:
        reader = csv.reader(f)
        next(reader)
        return [list(map(int, row)) for row in reader]

train_rows = read_csv("train.csv")
val_rows = read_csv("val.csv")
test_rows = read_csv("test.csv")

# Assign a unique integer ID to each training PC.
# ID 0 is reserved for unseen PCs.
unique_pcs = sorted(set(row[0] for row in train_rows))
pc_to_id = {pc: i + 1 for i, pc in enumerate(unique_pcs)}

def create_dataset(rows):
    pc = torch.tensor(
        [pc_to_id.get(row[0], 0) for row in rows],
        dtype=torch.long
    )

    history = torch.tensor(
        [row[1:1+H] for row in rows],
        dtype=torch.float32
    )

    labels = torch.tensor(
        [row[-1] for row in rows],
        dtype=torch.float32
    )

    return TensorDataset(pc, history, labels)

train_data = create_dataset(train_rows)
val_data = create_dataset(val_rows)
test_data = create_dataset(test_rows)

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE
)
test_loader = DataLoader(
    test_data, batch_size=BATCH_SIZE
)

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
        pc_features = self.embedding(pc)
        features = torch.cat(
            [pc_features, history], dim=1
        )
        return self.layers(features).squeeze(1)

model = BranchMLP(len(unique_pcs))

