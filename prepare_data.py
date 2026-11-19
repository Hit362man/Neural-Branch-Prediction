
import csv
from collections import deque
from pathlib import Path

H = 32
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Read the branch records extracted by ChampSim
with open("champsim/conditional_branches.csv") as f:
    branches = [
        (int(pc), int(taken))
        for _, pc, taken in csv.reader(f)
    ]

n = len(branches)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

history = deque(maxlen=H)

files = {
    "train": open(data_dir / "train.csv", "w"),
    "val": open(data_dir / "val.csv", "w"),
    "test": open(data_dir / "test.csv", "w"),
}
writers = {k: csv.writer(f) for k, f in files.items()}

header = ["pc"] + [f"h{i}" for i in range(H)] + ["taken"]
for writer in writers.values():
    writer.writerow(header)

counts = {"train": 0, "val": 0, "test": 0}

for i, (pc, taken) in enumerate(branches):
    if len(history) == H:
        split = "train" if i < train_end else (
            "val" if i < val_end else "test"
        )
        writers[split].writerow([pc] + list(history) + [taken])
        counts[split] += 1

    # Update history AFTER creating the sample
    history.append(taken)

for f in files.values():
    f.close()

print("Total branch records:", n)
print("Training samples:", counts["train"])
print("Validation samples:", counts["val"])
print("Testing samples:", counts["test"])
