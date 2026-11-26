
5. Visualize original and filtered EEG signals.
6. Calculate Mu (8–13 Hz) and Beta (13–30 Hz) band power.
7. Extract features from left- and right-imagery trials.
# BCI EEG Signal Processing and Motor Imagery Classification

## 1. Project Overview

This project demonstrates the foundations of Brain-Computer Interfaces (BCIs) through EEG signal processing, visualization, feature extraction, and basic machine-learning classification.

It uses a publicly available EEG dataset and open-source Python libraries to investigate left- and right-fist motor imagery.

## 2. Dataset

**Source:** PhysioNet EEG Motor Movement/Imagery Dataset

https://physionet.org/content/eegmmidb/1.0.0/

Recording details:

- Subject: S006
- Recording: S006R04.edf
- Channels: 64
- Sampling frequency: 160 Hz
- Duration: approximately 123 seconds
- Format: EDF

The dataset is downloaded separately and is not committed to this repository.

## 3. Technologies and Libraries

- CentOS Stream 10
- Python 3.12
- MNE-Python
- NumPy
- SciPy
- Matplotlib
- Pandas
- Scikit-learn
- Git and GitHub

## 4. Processing Workflow

1. Load the EEG recording using MNE.
2. Read event annotations:
   - T0: Rest
   - T1: Left-fist motor imagery
   - T2: Right-fist motor imagery
3. Select C3, Cz, and C4 EEG channels.
4. Apply a 1–40 Hz band-pass filter.
5. Visualize raw and filtered EEG signals.
6. Calculate Mu (8–13 Hz) and Beta (13–30 Hz) frequency-band power.
7. Extract features from motor imagery trials.
8. Train a Logistic Regression classifier.
9. Evaluate the model using Leave-One-Out Cross-Validation.

## 5. Classification Results

The preliminary classification experiment used 15 trials:

- Left imagery trials: 8
- Right imagery trials: 7
- Cross-validation accuracy: 53.33%

The result is preliminary because the dataset contains a small number of trials. It does not establish reliable real-world BCI performance.

## 6. Project Outputs

The `outputs/` directory contains:

- Raw EEG signal visualization
- Filtered EEG signal visualization
- Frequency spectrum
- EEG band-power plots
- Motor imagery filtering comparison
- Motor imagery band-power comparison
- Classification confusion matrix
- Classification predictions in CSV format
- Classification performance report

## 7. Installation on Linux

Clone the repository:

```bash
git clone https://github.com/abdullah08saleem/bci-eeg-project.git
cd bci-eeg-project
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download `S006R04.edf` from PhysioNet and place it in the project root.

## 8. Run the Project

Run the motor imagery visualization:

```bash
python src/motor_imagery/visualization.py
```

Run frequency analysis:

```bash
python src/motor_imagery/frequency.py
```

Run classification:

```bash
python src/motor_imagery/classifier.py
```

Generated results are saved in the `outputs/` directory.

## 9. Academic Purpose

This project was developed for educational purposes to demonstrate BCI foundations, EEG signal acquisition and processing, feature extraction, visualization, and basic machine-learning classification.

## 10. Limitations and Future Improvements

- Increase the number of EEG recordings and trials.
- Test the model on additional subjects.
- Explore additional EEG features and classification algorithms.
- Improve evaluation using subject-independent testing.
- Investigate ethical and privacy considerations for neural data.

## 11. Author

Developed as an academic project for the Diploma in Cloud Cyber Security.
