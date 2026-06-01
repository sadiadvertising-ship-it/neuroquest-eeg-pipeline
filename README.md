# EEG Emotion Classification Project

This project implements a machine learning pipeline for classifying emotions (Happy, Neutral, Frustrated) from EEG data collected via OpenBCI Ganglion board.

## Features

- Real-time EEG data collection from OpenBCI Ganglion (4 channels)
- Data preprocessing with filtering and artifact removal
- Feature extraction (power spectral density, statistical features)
- Machine learning model training (SVM/Random Forest/CNN)
- Real-time emotion classification with confidence scores
- Logging to CSV with timestamps and predictions
- Visualization of EEG signals and emotion probabilities
- Command-line interface for configuration

## Setup

### Prerequisites

- Python 3.8+
- OpenBCI Ganglion board
- macOS/Linux/Windows with serial port access

### Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Connect your OpenBCI Ganglion board and note the serial port (e.g., `/dev/cu.usbmodem11`)

### Usage

#### Training Mode

Collect training data and train the model:

```bash
python main.py --mode train --port /dev/cu.usbmodem11 --duration 300
```

#### Real-Time Classification

Run real-time emotion classification:

```bash
python main.py --mode realtime --port /dev/cu.usbmodem11 --model models/emotion_classifier.pkl
```

#### Command-Line Arguments

- `--mode`: 'train' or 'realtime'
- `--port`: Serial port for OpenBCI board (default: /dev/cu.usbmodem11)
- `--model`: Path to trained model (for realtime mode)
- `--duration`: Training duration in seconds (default: 300)
- `--config`: Path to config file (default: config.py)

## Project Structure

- `main.py`: Entry point with CLI
- `data_acquisition.py`: EEG data collection from OpenBCI
- `preprocessing.py`: Signal filtering, artifact removal, feature extraction
- `model_training.py`: Model training and evaluation
- `real_time_classification.py`: Live inference
- `utils.py`: Logging and visualization utilities
- `config.py`: Configuration settings
- `requirements.txt`: Python dependencies

## Data Format

Logged data is saved to `logs/eeg_data.csv` with columns:
- timestamp
- ch1, ch2, ch3, ch4 (raw EEG values)
- features (extracted features as JSON)
- ground_truth (during training)
- predicted (emotion class)
- confidence (prediction confidence)

## Troubleshooting

- Ensure the OpenBCI board is properly connected and the serial port is correct
- Check permissions for serial port access (may need `sudo` on Linux)
- For training, ensure labeled stimuli are prepared for each emotion class
- Model accuracy may vary; retrain with more data if needed

## License

This project is open-source. Use at your own risk.