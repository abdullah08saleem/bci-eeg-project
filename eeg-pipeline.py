"""
Title: EEG Signal Processing & BCI Foundations Pipeline
Author: Abdullah Saleem (abdullahsalim1408@gmail.com)
Course Code: ANPP-OP | Diploma in Cloud Cyber Security
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne

# 1. Dataset Acquisition: Generate simulated EEG locally
print("[*] Generating simulated EEG data (no download required)...")

import numpy as np
import mne

sfreq = 256
duration = 30
n_samples = sfreq * duration

channel_names = [
    "Fp1", "Fp2", "F3", "F4",
    "C3", "C4", "O1", "O2"
]

times = np.arange(n_samples) / sfreq
rng = np.random.default_rng(42)
data = []

for i, name in enumerate(channel_names):
    signal = (
        10e-6 * np.sin(2 * np.pi * 10 * times + i * 0.2)
        + 5e-6 * np.sin(2 * np.pi * 6 * times)
        + 3e-6 * np.sin(2 * np.pi * 20 * times)
        + 2e-6 * np.sin(2 * np.pi * 2 * times)
        + 2e-6 * rng.standard_normal(n_samples)
    )
    data.append(signal)

data = np.asarray(data)

info = mne.create_info(
    ch_names=channel_names,
    sfreq=sfreq,
    ch_types=["eeg"] * len(channel_names)
)

raw = mne.io.RawArray(data, info, verbose=False)
raw.crop(tmin=0.0, tmax=29.9)

print(
    f"[+] Loaded simulated EEG: {len(raw.ch_names)} channels, "
    f"Sampling Rate: {raw.info['sfreq']} Hz"
)

# 2. Signal Processing: Bandpass Filtering (1 Hz to 40 Hz)
# Filters out low-frequency drifts (sweat/movement artifacts) and high-frequency noise (line noise).
print("[*] Applying Bandpass Filter (1.0 - 40.0 Hz)...")
filtered_raw = raw.copy().filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', verbose=False)

# 3. Visualization: Plot Raw vs Filtered Signals & Save Output
os.makedirs('outputs', exist_ok=True)

fig_raw = raw.plot(duration=5.0, n_channels=5, scalings='auto', title='Raw EEG Signals', show=False)
fig_raw.savefig('outputs/eeg_raw_signal.png', dpi=300)
plt.close(fig_raw)

fig_filtered = filtered_raw.plot(duration=5.0, n_channels=5, scalings='auto', title='Filtered EEG Signals (1-40 Hz)', show=False)
fig_filtered.savefig('outputs/eeg_filtered_signal.png', dpi=300)
plt.close(fig_filtered)
print("[+] Visualizations saved in 'outputs/' directory.")

# 4. Feature Extraction: Compute Power Spectral Density (PSD)
print("[*] Extracting Power Spectral Density (PSD) features...")
psds, freqs = mne.time_frequency.psd_array_multitaper(
    filtered_raw.get_data(), 
    sfreq=filtered_raw.info['sfreq'], 
    fmin=1.0, 
    fmax=40.0, 
    verbose=False
)

# Calculate mean power per frequency band
# Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz)
mean_psd = np.mean(psds, axis=0)

print("\n--- Feature Extraction Summary (Mean Power per Band) ---")
print(f"Delta Band (1-4 Hz):   {np.mean(mean_psd[(freqs >= 1) & (freqs <= 4)]):.4f} uV^2/Hz")
print(f"Theta Band (4-8 Hz):   {np.mean(mean_psd[(freqs >= 4) & (freqs <= 8)]):.4f} uV^2/Hz")
print(f"Alpha Band (8-13 Hz):  {np.mean(mean_psd[(freqs >= 8) & (freqs <= 13)]):.4f} uV^2/Hz")
print(f"Beta Band (13-30 Hz):  {np.mean(mean_psd[(freqs >= 13) & (freqs <= 30)]):.4f} uV^2/Hz")
print("\n[+] Pipeline execution completed successfully!")
