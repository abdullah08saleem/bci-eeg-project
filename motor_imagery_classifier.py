
import csv
from pathlib import Path

import mne
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import welch
from scipy.integrate import trapezoid

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load the motor imagery recording
raw = mne.io.read_raw_edf(
    "S006R04.edf",
    preload=True,
    verbose=False
)

channels = ["C3..", "Cz..", "C4.."]
raw.pick(channels)

# Filter EEG signals
raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)

sfreq = raw.info["sfreq"]
data = raw.get_data()

# Extract Mu and Beta power from each labelled trial
bands = {
    "Mu": (8, 13),
    "Beta": (13, 30)
}

features = []
labels = []

for onset, duration, event in zip(
    raw.annotations.onset,
    raw.annotations.duration,
    raw.annotations.description
):
    if event not in ("T1", "T2"):
        continue

    start = int(onset * sfreq)
    stop = int((onset + duration) * sfreq)
    epoch = data[:, start:stop]

    if epoch.shape[1] < 256:
        continue

    trial_features = []

    for signal in epoch:
        freqs, psd = welch(
            signal,
            fs=sfreq,
            nperseg=min(512, len(signal))
        )

        for low, high in bands.values():
            mask = (freqs >= low) & (freqs <= high)
            power = trapezoid(psd[mask], freqs[mask])
            trial_features.append(power)

    features.append(trial_features)
    labels.append(0 if event == "T1" else 1)

X = np.asarray(features)
y = np.asarray(labels)

if len(np.unique(y)) != 2 or min(np.bincount(y)) < 2:
    raise ValueError("Not enough left/right trials for classification.")

# Leave-one-trial-out validation
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=2000)
)

predictions = cross_val_predict(
    model,
    X,
    y,
    cv=LeaveOneOut()
)

accuracy = accuracy_score(y, predictions)
report = classification_report(
    y,
    predictions,
    labels=[0, 1],
    target_names=["Left imagery", "Right imagery"],
    zero_division=0
)

output = Path("outputs")
output.mkdir(exist_ok=True)

# Save predictions
with open(output / "motor_imagery_predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Trial", "Actual", "Predicted"])
    for i, (actual, predicted) in enumerate(zip(y, predictions), 1):
        writer.writerow([
            i,
            "Left" if actual == 0 else "Right",
            "Left" if predicted == 0 else "Right"
        ])

# Save confusion matrix
cm = confusion_matrix(y, predictions, labels=[0, 1])
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Left", "Right"]
)

fig, ax = plt.subplots(figsize=(6, 5))
display.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Motor Imagery: Left vs Right")
fig.tight_layout()
fig.savefig(output / "motor_imagery_confusion_matrix.png", dpi=150)
plt.close(fig)

# Save written report
with open(output / "motor_imagery_classification_report.txt", "w") as f:
    f.write("MOTOR IMAGERY CLASSIFICATION\n")
    f.write("============================\n")
    f.write(f"Recording: S006R04.edf\n")
    f.write(f"Channels: {', '.join(channels)}\n")
    f.write(f"Number of trials: {len(y)}\n")
    f.write("Validation: Leave-One-Trial-Out\n")
    f.write(f"Accuracy: {accuracy:.2%}\n\n")
    f.write(report)

print("Motor imagery classification completed!")
print("Trials:", len(y))
print("Left trials:", int(np.sum(y == 0)))
print("Right trials:", int(np.sum(y == 1)))
print(f"Leave-one-out accuracy: {accuracy:.2%}")
print("\n", report)
print("Saved: outputs/motor_imagery_predictions.csv")
print("Saved: outputs/motor_imagery_confusion_matrix.png")
print("Saved: outputs/motor_imagery_classification_report.txt")
