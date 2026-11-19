
import csv
from collections import deque
from pathlib import Path

H = 32

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"

WORKLOADS = {
    "mcf": 0,
    "gcc": 1,
    "perlbench": 2,
    "x264": 3,
}

DATA.mkdir(exist_ok=True)

files = {
    split: open(DATA / f"{split}.csv", "w", newline="")
    for split in ("train", "val", "test")
}

writers = {
    split: csv.writer(f)
    for split, f in files.items()
}

header = (
    ["pc"]
    + [f"h{i}" for i in range(H)]
    + ["workload_id", "raw_pc", "taken"]
)

for writer in writers.values():
    writer.writerow(header)

totals = {"train": 0, "val": 0, "test": 0}

for workload, workload_id in WORKLOADS.items():

    path = DATA / f"{workload}_1m_branches.csv"

    with open(path) as f:
        n = sum(1 for _ in csv.reader(f))

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    history = deque(maxlen=H)
    counts = {"train": 0, "val": 0, "test": 0}

    with open(path) as f:
        reader = csv.reader(f)

        for i, row in enumerate(reader):
            instr_id, pc, taken = map(int, row)

            if taken not in (0, 1):
                raise ValueError("Invalid branch outcome")

            if not (0 <= pc < 2**64):
                raise ValueError("PC outside 64-bit range")

            # Unique branch identity across workloads
            branch_key = (workload_id << 64) | pc

            if len(history) == H:

                if i < train_end:
                    split = "train"
                elif i < val_end:
                    split = "val"
                else:
                    split = "test"

                writers[split].writerow(
                    [branch_key]
                    + list(history)
                    + [workload_id, pc, taken]
                )

                counts[split] += 1
                totals[split] += 1

            # Current outcome is added AFTER sample creation
            history.append(taken)

    print(f"\n{workload.upper()}")
    print(f"  Total branches: {n}")
    print(f"  Training: {counts['train']}")
    print(f"  Validation: {counts['val']}")
    print(f"  Testing: {counts['test']}")

for f in files.values():
    f.close()

print("\nCOMBINED DATASET")
for split, count in totals.items():
    print(f"{split}: {count}")

print("Total samples:", sum(totals.values()))
