import mne
import numpy as np
import matplotlib.pyplot as plt
import os

# Load original EEG recording
raw = mne.io.read_raw_edf(
    "S006R01.edf",
    preload=True,
    verbose=False
)

# Select first EEG channel
channel = raw.ch_names[0]

# Copy original signal
original = raw.copy()

# Apply band-pass filter: 1–40 Hz
filtered = raw.copy()
filtered.filter(
    l_freq=1.0,
    h_freq=40.0,
    picks="eeg",
    verbose=False
)

# Extract first 10 seconds
sfreq = raw.info["sfreq"]
start = 0
stop = int(10 * sfreq)

original_data = original.get_data(
    picks=[channel],
    start=start,
    stop=stop
)[0] * 1e6

filtered_data = filtered.get_data(
    picks=[channel],
    start=start,
    stop=stop
)[0] * 1e6

times = np.arange(len(original_data)) / sfreq

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Compare original and filtered signals
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(times, original_data, linewidth=0.7)
axes[0].set_title("Original EEG Signal")
axes[0].set_ylabel("Amplitude (µV)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(times, filtered_data, linewidth=0.7)
axes[1].set_title("Filtered EEG Signal (1–40 Hz)")
axes[1].set_ylabel("Amplitude (µV)")
axes[1].set_xlabel("Time (seconds)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/eeg_filter_comparison.png", dpi=150)
plt.close()

# Calculate signal statistics
print("EEG filtering completed successfully!")
print("Channel:", channel)
print("Filter: 1–40 Hz")
print("Original standard deviation:", np.std(original_data), "µV")
print("Filtered standard deviation:", np.std(filtered_data), "µV")
print("Saved: outputs/eeg_filter_comparison.png")
