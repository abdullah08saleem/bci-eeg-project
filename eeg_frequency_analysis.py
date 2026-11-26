import mne
import numpy as np
import matplotlib.pyplot as plt
import os

# Load actual EEG recording
raw = mne.io.read_raw_edf(
    "S006R01.edf",
    preload=True,
    verbose=False
)

# Select first EEG channel
channel = raw.ch_names[0]
signal = raw.get_data(picks=[channel])[0]

# Calculate Power Spectral Density
sfreq = raw.info["sfreq"]

psd, freqs = mne.time_frequency.psd_array_welch(
    signal,
    sfreq=sfreq,
    fmin=0.5,
    fmax=45,
    n_fft=1024,
    verbose=False
)

# Frequency bands
bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 45)
}

# Calculate average power in each band
band_power = {}

for name, (low, high) in bands.items():
    mask = (freqs >= low) & (freqs < high)
    band_power[name] = np.mean(psd[mask])

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Plot frequency spectrum
plt.figure(figsize=(11, 5))
plt.plot(freqs, 10 * np.log10(psd + 1e-20))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.title(f"EEG Frequency Spectrum - {channel}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/eeg_frequency_spectrum.png", dpi=150)
plt.close()

# Plot frequency band power
plt.figure(figsize=(9, 5))
plt.bar(band_power.keys(), band_power.values())
plt.xlabel("EEG Frequency Bands")
plt.ylabel("Average Power (V²/Hz)")
plt.title(f"EEG Band Power - {channel}")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/eeg_band_power.png", dpi=150)
plt.close()

print("Frequency analysis completed!")
print("Channel:", channel)

for name, power in band_power.items():
    print(f"{name}: {power:.8e} V²/Hz")

print("Saved graphs in outputs/")
