
import mne
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

# 1. Load the motor imagery recording
raw = mne.io.read_raw_edf(
    "S006R04.edf",
    preload=True,
    verbose=False
)

# 2. Select motor cortex channels
channels = ["C3..", "Cz..", "C4.."]

# 3. Extract original signals
original = raw.copy().pick(channels)

# 4. Apply band-pass filtering (1–40 Hz)
filtered = original.copy()
filtered.filter(
    l_freq=1.0,
    h_freq=40.0,
    verbose=False
)

# 5. Convert signals from volts to microvolts
times = filtered.times
original_data = original.get_data() * 1e6
filtered_data = filtered.get_data() * 1e6

# 6. Create output folder
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# 7. Plot the first 35 seconds
fig, axes = plt.subplots(
    3, 1,
    figsize=(14, 9),
    sharex=True
)

for i, ax in enumerate(axes):

    ax.plot(
        times,
        filtered_data[i],
        linewidth=0.7,
        label="Filtered EEG"
    )

    # Mark left and right imagery periods
    for onset, duration, description in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description
    ):
        if description in ["T1", "T2"]:
            color = "blue" if description == "T1" else "red"

            ax.axvspan(
                onset,
                onset + duration,
                alpha=0.12,
                color=color
            )

            if onset < 35:
                ax.text(
                    onset + duration / 2,
                    0.90,
                    description,
                    transform=ax.get_xaxis_transform(),
                    ha="center"
                )

    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(channels[i])
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 35)

axes[-1].set_xlabel("Time (seconds)")

fig.suptitle(
    "Motor Imagery EEG — C3, Cz and C4",
    fontsize=15
)

fig.tight_layout()
fig.savefig(
    output_dir / "motor_imagery_filtered.png",
    dpi=150
)

plt.close(fig)

# 8. Create a raw vs filtered comparison for C3
fig, ax = plt.subplots(figsize=(13, 5))

ax.plot(
    times,
    original_data[0],
    label="Original EEG",
    alpha=0.65,
    linewidth=0.7
)

ax.plot(
    times,
    filtered_data[0],
    label="Filtered EEG (1–40 Hz)",
    linewidth=0.8
)

ax.set_xlim(0, 15)
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Amplitude (µV)")
ax.set_title("C3 — Original vs Filtered EEG")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(
    output_dir / "motor_imagery_filter_comparison.png",
    dpi=150
)

plt.close(fig)

print("Motor imagery visualization completed!")
print("Channels:", channels)
print("Filter: 1–40 Hz")
print("Saved: outputs/motor_imagery_filtered.png")
print("Saved: outputs/motor_imagery_filter_comparison.png")
