import matplotlib
matplotlib.use('Agg')  # save graphs as PNG files instead of opening windows
import mne
from mne.datasets import eegbci

mne.viz.set_browser_backend('matplotlib')


def load_raw(subject=1, runs=(4, 8, 12)):
    # Download (first time only) and read the EDF recordings
    files = eegbci.load_data(subject, list(runs))
    raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in files]
    raw = mne.concatenate_raws(raws)
    # Clean up channel names (e.g. 'C3..' becomes 'C3') and add 3D positions
    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage('standard_1005'))
    return raw


def preprocess(raw):
    # Technique used: filtering
    out = raw.copy()
    out.notch_filter(freqs=[50], verbose=False)         # remove 50 Hz mains hum (Pakistan)
    out.filter(l_freq=1.0, h_freq=40.0, verbose=False)  # keep 1-40 Hz brain activity
    return out
