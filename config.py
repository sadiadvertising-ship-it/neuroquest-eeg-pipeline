# Board
SERIAL_PORT = '/dev/cu.usbmodem11'  # BLED112 dongle of the OpenBCI Ganglion on macOS
BOARD_ID = 1  # BrainFlow BoardIds.GANGLION_BOARD
SAMPLING_RATE = 200  # Hz
CHANNELS = 4
CHANNEL_NAMES = ['Fp1', 'Fp2', 'T3', 'T4']

# Preprocessing (thesis, Figure Arq-5)
BANDPASS_LOW = 0.5  # Hz
BANDPASS_HIGH = 45.0  # Hz
NOTCH_FREQ = 60.0  # Hz, power line in Colombia
ARTIFACT_UV = 100.0  # a window is rejected if any channel goes beyond +-100 uV
FLAT_UV = 0.5  # ...or if any channel is flat or saturated (filtered std below 0.5 uV)
FILTER_PAD = 3.0  # seconds of past signal filtered along with each window to avoid edge transients

# Feature extraction: log band power per channel -> 5 bands x 4 channels = 20 features
WINDOW_SIZE = 2.0  # seconds
OVERLAP = 0.5  # fraction, used when slicing recordings into windows
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}

# Model
MODEL_PATH = 'models/emotion_classifier.pkl'
CLASSES = ['Happy', 'Neutral', 'Frustrated']
MIN_CLASS_WINDOWS = 20  # training stops if a class has fewer clean windows than this

# Recording protocol
SAMPLING_DURATION = 15.0  # seconds of each emotion stimulus
REST_DURATION = 11.0  # seconds between stimuli, the last CUE_SECONDS of them announce the next emotion
CUE_SECONDS = 3.0
TRAINING_ROUNDS = 3  # rounds per session, each round shows every emotion once
SHUFFLE_ORDER = True  # new emotion order each round, so the label is not tied to time in the session
ONSET_MARGIN = 2.0  # seconds after a stimulus starts that stay out of training (a person presents it)
REST_LABEL = 'Rest'
CUE_LABEL = 'Cue'
LIVE_LABEL = 'Live'
PARTICIPANT_PATTERN = r'^[A-Za-z0-9_-]{1,16}$'  # codes like P03, never names

# Folders
SESSIONS_DIR = 'data/sessions'
SYNTHETIC_DIR = 'data/synthetic'  # anything recorded with --synthetic, kept away from training
LIVE_DIR = 'data/live'
LOGS_DIR = 'logs'

# Real-time
PREDICT_INTERVAL = 1.0  # seconds
NO_DATA_SECONDS = 1.0  # without new samples for this long the signal counts as lost

# Quality checks
GAP_SECONDS = 0.1  # a pause between samples longer than this counts as a stall
