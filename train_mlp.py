
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

optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)
criterion = nn.BCEWithLogitsLoss()

def evaluate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for pc, history, labels in loader:
            logits = model(pc, history)

            loss = criterion(logits, labels)
            predictions = (logits >= 0).float()

            count = len(labels)
            total_loss += loss.item() * count
            correct += (predictions == labels).sum().item()
            total += count

    return total_loss / total, 100 * correct / total

print("Training samples:", len(train_data))
print("Validation samples:", len(val_data))
print("Testing samples:", len(test_data))
print("Unique training PCs:", len(unique_pcs))
print("Model parameters:",
      sum(p.numel() for p in model.parameters()))

best_val_loss = float("inf")
checkpoint_dir = ROOT / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)
checkpoint_path = checkpoint_dir / "mlp_mcf.pt"

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for pc, history, labels in train_loader:
        optimizer.zero_grad()

        logits = model(pc, history)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(labels)

    train_loss = running_loss / len(train_data)
    val_loss, val_accuracy = evaluate(val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val accuracy: {val_accuracy:.2f}%"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "pc_to_id": pc_to_id,
            "history_length": H,
            "embedding_dim": EMBED_DIM
        }, checkpoint_path)

# Test only the model selected using validation data
checkpoint = torch.load(
    checkpoint_path, map_location="cpu"
)
model.load_state_dict(checkpoint["model_state_dict"])

test_loss, test_accuracy = evaluate(test_loader)

print("\nFinal held-out test results")
print(f"Test accuracy: {test_accuracy:.2f}%")
print(f"Misprediction rate: {100-test_accuracy:.2f}%")
print("Model saved to:", checkpoint_path)

