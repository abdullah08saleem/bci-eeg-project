import os
from common import load_raw

os.makedirs('outputs', exist_ok=True)
raw = load_raw()

print(raw.info)
print('Sampling rate (Hz):', raw.info['sfreq'])
print('Number of channels:', len(raw.ch_names))
print('Duration (seconds):', round(raw.times[-1], 1))

# Graph 1: raw signal, 10 seconds, 10 channels
fig1 = raw.plot(duration=10, n_channels=10, scalings=dict(eeg=100e-6), show=False)
fig1.savefig('outputs/01_raw_signal.png')

# Graph 2: where the electrodes sit on the head
fig2 = raw.plot_sensors(show_names=True, show=False)
fig2.savefig('outputs/01_sensor_layout.png')

# Graph 3: power spectrum (how much energy at each frequency)
fig3 = raw.compute_psd(fmax=60).plot(show=False)
fig3.savefig('outputs/01_psd_raw.png')

print('Saved 3 figures in outputs/')
