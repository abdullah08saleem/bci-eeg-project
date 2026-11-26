# BCI EEG Signal Processing and Motor Imagery Classification

## Project Overview
This project demonstrates EEG signal acquisition, preprocessing,
visualization, frequency analysis, and basic motor-imagery classification
using Python and open-source tools.

## Dataset
PhysioNet EEG Motor Movement/Imagery Dataset:
https://physionet.org/content/eegmmidb/1.0.0/

Recording used:
- Subject: 6
- Run: 4
- File: S006R04.edf
- Channels: 64
- Sampling frequency: 160 Hz
- Duration: approximately 123 seconds

The EDF dataset is not included in Git because of its size and data
distribution considerations. Download it separately from PhysioNet.

## Technologies
- Python
- MNE-Python
- NumPy
- SciPy
- Matplotlib
- Scikit-learn
- Git and GitHub
- CentOS Stream 10

## Processing Workflow
1. Load EEG data from an EDF recording.
2. Identify T0 (rest), T1 (left-fist imagery), and T2 (right-fist imagery).
3. Select C3, Cz, and C4 channels.
4. Apply a 1–40 Hz band-pass filter.
5. Visualize original and filtered EEG signals.
6. Calculate Mu (8–13 Hz) and Beta (13–30 Hz) band power.
7. Extract features from left- and right-imagery trials.
8. Train and evaluate a logistic-regression classifier.

## Project Outputs
The outputs directory contains:
- EEG waveform and filtering plots
- Motor-imagery signal plots
- Mu and Beta band-power comparison
- Band-power CSV data
- Classification confusion matrix
- Classification predictions and report

## Setup on Linux
Clone the repository and enter the project directory:

    git clone YOUR_GITHUB_REPOSITORY_URL
    cd bci-eeg-project

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

Download S006R04.edf from PhysioNet and place it in the project root.

Run the analysis:

    python motor_imagery_visualization.py
    python motor_imagery_frequency.py
    python motor_imagery_classifier.py

## Classification Note
The classifier distinguishes left-fist imagery from right-fist imagery.
It uses leave-one-trial-out cross-validation.

This recording contains only a small number of trials. Therefore,
classification results are preliminary and should not be interpreted
as proof of clinical or real-world BCI accuracy.

## Academic Purpose
Developed as an educational project to demonstrate the foundations of
Brain-Computer Interfaces and EEG signal processing.
