import os

# Data directories
DATA_DIR = 'data'

# Configuration settings
SERIAL_PORT = '/dev/cu.usbmodem11'  # Default serial port for OpenBCI Ganglion
BOARD_ID = 1  # BrainFlow board ID for OpenBCI Ganglion
SAMPLING_RATE = 200  # Hz
CHANNELS = 4  # Number of channels (Ganglion has 4)

# Preprocessing settings
BANDPASS_LOW = 1.0  # Hz
BANDPASS_HIGH = 50.0  # Hz
NOTCH_FREQ = 50.0  # Hz for power line noise

# Feature extraction
WINDOW_SIZE = 2.0  # seconds for real-time windows
OVERLAP = 0.5  # overlap fraction

# Model settings
MODEL_PATH = 'models/emotion_classifier.pkl'
CLASSES = ['Happy', 'Neutral', 'Frustrated']

# Training protocol settings
SAMPLING_DURATION = 15.0  # seconds of sampling per emotion stimulus
REST_DURATION = 11.0  # seconds of rest between stimuli
TRAINING_ROUNDS = 3  # number of rounds per emotion for more data

# Advanced preprocessing settings
ICA_COMPONENTS = 4  # Number of ICA components for artifact removal
WAVELET = 'db4'     # Wavelet type for CWT
WAVELET_SCALES = 'auto'  # Scales for wavelet transform

# Deep learning settings
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Tuning settings
TUNING_TRIALS = 20  # Number of Optuna trials for hyperparameter tuning

# Logging
LOG_FILE = 'logs/eeg_data.csv'
LOG_HEADERS = ['timestamp', 'ch1', 'ch2', 'ch3', 'ch4', 'features', 'ground_truth', 'predicted', 'confidence']

# Visualization
PLOT_UPDATE_INTERVAL = 1.0  # seconds