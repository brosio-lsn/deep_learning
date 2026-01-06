import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(".")  # script is in models/
pes = ["cot1", "abs_2d_learned", "rel_2d_bias", "abs_1d_learned", "abs_2d_sin+rel_2d_bias", "abs_1d_sinusoidal", "abs_2d_sinusoidal", "cot2"]

THRESH = 0.9
MAX_EPOCHS = 10
METRICS_FILE = "history.json"

def epochs_to_threshold(acc_curve, thresh=THRESH, max_epochs=MAX_EPOCHS):
    for i, acc in enumerate(acc_curve):
        if acc >= thresh:
            return i + 1  # epochs are 1-indexed
    return max_epochs

pe_mean_epochs = []
pe_std_epochs = []

plt.figure()

labels = []

for pe in pes:
    pe_dir = root / pe
    times = []

    if not pe_dir.exists():
        pe_mean_epochs.append(np.nan)
        pe_std_epochs.append(0.0)
        labels.append(pe)
        continue

    for run_dir in pe_dir.iterdir():
        if not run_dir.is_dir():
            continue

        metrics_path = run_dir / METRICS_FILE
        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        acc_curve = metrics.get("val_acc", [])
        if not acc_curve:
            continue

        t = epochs_to_threshold(acc_curve)
        times.append(t)

    if len(times) == 0:
        pe_mean_epochs.append(np.nan)
        pe_std_epochs.append(0.0)
    else:
        pe_mean_epochs.append(np.mean(times))
        pe_std_epochs.append(np.std(times))

    labels.append(pe)

x = np.arange(len(labels))

plt.bar(x, pe_mean_epochs, yerr=pe_std_epochs, capsize=5)
plt.xticks(x, labels, rotation=25, ha="right")
plt.ylabel("Mean epochs to reach val_acc ≥ 0.9")
plt.title("Learning speed by positional encoding")
plt.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("pe_speed_acc09-final.png", dpi=300, bbox_inches="tight")
plt.show()
