
import mne
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import csv
from pathlib import Path
from scipy.signal import welch
from scipy.integrate import trapezoid

# 1. Load EEG data
raw = mne.io.read_raw_edf(
    "S006R04.edf",
    preload=True,
    verbose=False
)

# 2. Select motor cortex channels
channels = ["C3..", "Cz..", "C4.."]
raw.pick(channels)

# 3. Apply 1-40 Hz filtering
raw.filter(
    l_freq=1.0,
    h_freq=40.0,
    verbose=False
)

sfreq = raw.info["sfreq"]
data = raw.get_data()

# 4. Define frequency bands
bands = {
    "Mu (8-13 Hz)": (8, 13),
    "Beta (13-30 Hz)": (13, 30)
}

# 5. Define event names
event_names = {
    "T0": "Rest",
    "T1": "Left imagery",
    "T2": "Right imagery"
}

results = []

# 6. Calculate power for each event
for onset, duration, description in zip(
    raw.annotations.onset,
    raw.annotations.duration,
    raw.annotations.description
):

    if description not in event_names:
        continue

    start = int(onset * sfreq)
    stop = int((onset + duration) * sfreq)

    epoch = data[:, start:stop]

    if epoch.shape[1] < 256:
        continue

    for channel_index, channel in enumerate(channels):

        signal = epoch[channel_index]

        frequencies, psd = welch(
            signal,
            fs=sfreq,
            nperseg=512
        )

        for band_name, (low, high) in bands.items():

            mask = (
                (frequencies >= low) &
                (frequencies <= high)
            )

            power = trapezoid(
                psd[mask],
                frequencies[mask]
            ) * 1e12

            results.append({
                "event": event_names[description],
                "channel": channel,
                "band": band_name,
                "power": power
            })

# 7. Calculate average power for each class
classes = ["Rest", "Left imagery", "Right imagery"]

averages = {}

for channel in channels:
    averages[channel] = {}

    for band_name in bands:
        averages[channel][band_name] = []

        for event in classes:

            values = [
                r["power"] for r in results
                if r["channel"] == channel
                and r["band"] == band_name
                and r["event"] == event
            ]

            averages[channel][band_name].append(
                np.mean(values) if values else 0
            )

# 8. Create output directory
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# 9. Save numerical results
with open(
    output_dir / "motor_imagery_band_power.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Channel",
        "Event",
        "Band",
        "Average power (uV^2)"
    ])

    for channel in channels:
        for band_name in bands:
            for event, power in zip(
                classes,
                averages[channel][band_name]
            ):
                writer.writerow([
                    channel,
                    event,
                    band_name,
                    round(power, 4)
                ])

# 10. Plot frequency-band comparison
fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 5)
)

x = np.arange(len(classes))
width = 0.35

for ax, channel in zip(axes, channels):

    mu = averages[channel]["Mu (8-13 Hz)"]
    beta = averages[channel]["Beta (13-30 Hz)"]

    ax.bar(
        x - width / 2,
        mu,
        width,
        label="Mu"
    )

    ax.bar(
        x + width / 2,
        beta,
        width,
        label="Beta"
    )

    ax.set_title(channel)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Rest", "Left", "Right"],
        rotation=15
    )
    ax.set_ylabel("Average power (µV²)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "Motor Imagery: Mu and Beta Band Power",
    fontsize=15
)

fig.tight_layout()

fig.savefig(
    output_dir / "motor_imagery_band_power.png",
    dpi=150
)

plt.close(fig)

print("Motor imagery frequency analysis completed!")
print("Channels:", channels)
print("Events:", classes)
print("Bands:", list(bands.keys()))
print("Saved: outputs/motor_imagery_band_power.png")
print("Saved: outputs/motor_imagery_band_power.csv")
