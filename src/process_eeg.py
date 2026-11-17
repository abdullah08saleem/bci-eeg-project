
from pathlib import Path

import mne
import numpy as np
import matplotlib

# Save graphs to files; no graphical display required
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mne.set_log_level("WARNING")


def main():
    print("=" * 50)
    print("BCI EEG Signal Processing Project")
    print("=" * 50)

    # 1. Load a public baseline EEG recording
    print("\n[1/6] Loading EEG dataset...")

    runs = [6, 10, 14]

    eeg_files = mne.datasets.eegbci.load_data(
    subjects=1,
    runs=runs
)

    raw = mne.io.read_raw_edf(
        eeg_files[0],
        preload=True,
        verbose=False
    )

    # Select EEG channels only
    raw.pick("eeg")

    sampling_rate = float(raw.info["sfreq"])
    duration = raw.n_times / sampling_rate

    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Number of EEG channels: {len(raw.ch_names)}")
    print(f"Recording duration: {duration:.2f} seconds")

    # 2. Filter the EEG signal
    print("\n[2/6] Applying 1-40 Hz band-pass filter...")

    filtered = raw.copy()
    filtered.filter(
        l_freq=1.0,
        h_freq=40.0,
        picks="eeg",
        verbose=False
    )

    # 3. Extract the first channel for the first 10 seconds
    print("\n[3/6] Preparing signal visualization...")

    channel = raw.ch_names[0]
    number_of_samples = min(
        int(10 * sampling_rate),
        raw.n_times
    )

    times = np.arange(number_of_samples) / sampling_rate

    original_signal = raw.get_data(
        picks=[channel],
        start=0,
        stop=number_of_samples
    )[0] * 1e6

    filtered_signal = filtered.get_data(
        picks=[channel],
        start=0,
        stop=number_of_samples
    )[0] * 1e6

    # 4. Plot original and filtered signals
    print("\n[4/6] Creating EEG comparison graph...")

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 7),
        sharex=True
    )

    axes[0].plot(times, original_signal, linewidth=0.7)
    axes[0].set_title(f"Original EEG Signal: {channel}")
    axes[0].set_ylabel("Amplitude (µV)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, filtered_signal, linewidth=0.7)
    axes[1].set_title("Filtered EEG Signal (1-40 Hz)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude (µV)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "raw_vs_filtered.png",
        dpi=150
    )
    plt.close(fig)

    # 5. Calculate and plot the power spectrum
    print("\n[5/6] Calculating power spectrum...")

    # Convert microvolts back to volts
    frequencies, power = welch(
        filtered_signal / 1e6,
        fs=sampling_rate,
        nperseg=min(1024, len(filtered_signal))
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(frequencies, power)
    ax.set_xlim(0, 45)
    ax.set_title(f"EEG Power Spectrum: {channel}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (V²/Hz)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "power_spectrum.png",
        dpi=150
    )
    plt.close(fig)

    # 6. Calculate simple signal features
    rms = float(np.sqrt(np.mean(filtered_signal ** 2)))
    variance = float(np.var(filtered_signal))

    alpha_mask = (
        (frequencies >= 8) &
        (frequencies <= 13)
    )

    alpha_power = (
        float(np.trapezoid(
            power[alpha_mask],
            frequencies[alpha_mask]
        ))
        if np.any(alpha_mask)
        else 0.0
    )

    report_path = OUTPUT_DIR / "feature_report.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write("BCI EEG Signal Analysis Report\n")
        report.write("=" * 35 + "\n")
        report.write("Dataset: EEGBCI, subject 1, run 1\n")
        report.write(f"Channel: {channel}\n")
        report.write(f"Sampling rate: {sampling_rate} Hz\n")
        report.write(f"Number of EEG channels: {len(raw.ch_names)}\n")
        report.write(f"Recording duration: {duration:.2f} seconds\n")
        report.write("Filter: 1-40 Hz band-pass\n")
        report.write("Feature window: first 10 seconds\n")
        report.write(f"RMS amplitude: {rms:.4f} µV\n")
        report.write(f"Variance: {variance:.6f} µV²\n")
        report.write(
            f"Alpha-band power (8-13 Hz): {alpha_power:.6e} V²\n"
        )
        report.write(
            "\nThese are signal measurements, not a diagnosis "
            "or proof of thoughts or intentions.\n"
        )

    print("\nSUCCESS: EEG processing completed!")
    print("Generated files:")

    for filename in [
        "raw_vs_filtered.png",
        "power_spectrum.png",
        "feature_report.txt"
    ]:
        print(f"  outputs/{filename}")

    print("\nRMS amplitude:", f"{rms:.4f} µV")
    print("Variance:", f"{variance:.6f} µV²")


if __name__ == "__main__":
    main()

