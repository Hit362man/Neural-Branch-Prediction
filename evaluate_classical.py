
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"

WORKLOADS = [
    "mcf", "gcc", "perlbench", "x264",
    "leela", "deepsjeng", "xz", "omnetpp",
    "xalancbmk", "exchange2", "wrf", "imagick"
]

TABLE_SIZE = 16384
HISTORY_BITS = 14
HISTORY_MASK = (1 << HISTORY_BITS) - 1


def evaluate(workload):
    path = DATA / f"{workload}_1m_branches.csv"

    with open(path) as f:
        n = sum(1 for _ in f)

    test_start = int(n * 0.85)

    # 2-bit counters, initialized weakly not-taken
    bimodal = [1] * TABLE_SIZE
    gshare = [1] * TABLE_SIZE

    history = 0
    correct_b = 0
    correct_g = 0
    total = 0

    with open(path) as f:
        for i, row in enumerate(csv.reader(f)):
            _, pc, taken = map(int, row)

            # Indices
            b_idx = pc % 16381
            g_idx = (pc ^ history) & HISTORY_MASK

            # Predict BEFORE updating
            pred_b = int(bimodal[b_idx] >= 2)
            pred_g = int(gshare[g_idx] >= 2)

            if i >= test_start:
                total += 1
                correct_b += pred_b == taken
                correct_g += pred_g == taken

            # Online counter updates
            delta = 1 if taken else -1

            bimodal[b_idx] = max(
                0, min(3, bimodal[b_idx] + delta)
            )
            gshare[g_idx] = max(
                0, min(3, gshare[g_idx] + delta)
            )

            # Update global history AFTER prediction
            history = ((history << 1) | taken) & HISTORY_MASK

    return total, correct_b, correct_g


totals = [0, 0, 0]

print(f"{'Workload':<14} {'Bimodal':>10} {'Gshare':>10}")

for workload in WORKLOADS:
    n, b, g = evaluate(workload)

    totals[0] += n
    totals[1] += b
    totals[2] += g

    print(
        f"{workload:<14}"
        f"{100*b/n:>9.2f}%"
        f"{100*g/n:>10.2f}%"
    )

n, b, g = totals

print("\nOVERALL")
print(f"Bimodal: {100*b/n:.2f}%")
print(f"Gshare:  {100*g/n:.2f}%")
print("MLP:     95.31%")
