import mne
import numpy as np
import matplotlib.pyplot as plt
import os

# Load EEG recording
raw = mne.io.read_raw_edf(
    "S006R01.edf",
    preload=True,
    verbose=False
)

# Select the first 4 EEG channels
channels = raw.ch_names[:4]

# Get the first 10 seconds of EEG data
sfreq = raw.info["sfreq"]
data = raw.get_data(
    picks=channels,
    start=0,
    stop=int(10 * sfreq)
)

# Convert volts into microvolts
data = data * 1e6

# Create time axis
times = np.arange(data.shape[1]) / sfreq

# Create four waveform plots
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

for i, ax in enumerate(axes):
    ax.plot(times, data[i], linewidth=0.7)
    ax.set_ylabel("µV")
    ax.set_title(channels[i])
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (seconds)")

fig.suptitle("EEG Signal Waveforms - First 10 Seconds", fontsize=15)
plt.tight_layout()

# Save graph
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/eeg_waveform.png", dpi=150)
plt.close()

print("EEG waveform graph generated successfully!")
print("Channels plotted:", channels)
print("Saved to: outputs/eeg_waveform.png")
