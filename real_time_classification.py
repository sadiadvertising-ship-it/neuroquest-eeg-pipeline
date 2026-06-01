import time
import threading
import numpy as np
from collections import deque
import json
from data_acquisition import DataAcquisition
from preprocessing import Preprocessor
from model_training import EmotionClassifier
from utils import Logger
import config

class RealTimeClassifier:
    def __init__(self, model_path=config.MODEL_PATH, serial_port=config.SERIAL_PORT):
        self.model_path = model_path
        self.serial_port = serial_port
        self.classifier = EmotionClassifier()
        self.preprocessor = Preprocessor()
        self.logger = Logger()
        self.data_acquisition = None
        self.is_running = False
        self.buffer = deque(maxlen=int(config.SAMPLING_RATE * config.WINDOW_SIZE))

    def load_model(self):
        """Load the trained model."""
        try:
            self.classifier.load_model(self.model_path)
            print("Model loaded successfully")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load model: {e}")

    def start_classification(self):
        """Start real-time classification."""
        self.load_model()

        with DataAcquisition(self.serial_port) as daq:
            self.data_acquisition = daq
            daq.start_streaming()
            self.is_running = True

            print("Real-time classification started. Press Ctrl+C to stop.")

            try:
                while self.is_running:
                    self.process_window()
                    time.sleep(config.PLOT_UPDATE_INTERVAL)

            except KeyboardInterrupt:
                print("Stopping classification...")
            finally:
                self.is_running = False

    def process_window(self):
        """Process a window of EEG data."""
        # Get current data
        current_data = self.data_acquisition.get_current_data(int(config.SAMPLING_RATE * config.WINDOW_SIZE))

        if current_data is None or current_data.size == 0:
            return

        # Transpose to (samples, channels)
        current_data = current_data.T

        # Add to buffer
        for sample in current_data:
            self.buffer.append(sample)

        # Check if we have enough data for a window
        if len(self.buffer) < int(config.SAMPLING_RATE * config.WINDOW_SIZE):
            return

        # Convert buffer to numpy array
        window_data = np.array(list(self.buffer))

        # Preprocess window
        try:
            features = self.preprocessor.preprocess_window(window_data)
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return

        # Classify
        try:
            prediction, confidence, probabilities = self.classifier.predict(features)
        except Exception as e:
            print(f"Classification error: {e}")
            return

        # Display result
        self.display_prediction(prediction, confidence, probabilities)

        # Log data
        timestamp = time.time()
        raw_values = window_data[-1]  # Last sample
        self.logger.log_data(timestamp, raw_values, features, None, prediction, confidence)

    def display_prediction(self, prediction, confidence, probabilities):
        """Display classification result."""
        print(f"\rPredicted Emotion: {prediction} | Confidence: {confidence:.2f}", end='', flush=True)

        # Optional: Print probabilities
        prob_str = " | ".join([f"{cls}: {prob:.2f}" for cls, prob in zip(config.CLASSES, probabilities)])
        print(f" | Probabilities: {prob_str}")

    def stop_classification(self):
        """Stop real-time classification."""
        self.is_running = False
        if self.data_acquisition:
            self.data_acquisition.stop_streaming()

def run_real_time_classification(model_path=config.MODEL_PATH, serial_port=config.SERIAL_PORT):
    """Main function to run real-time classification."""
    rtc = RealTimeClassifier(model_path, serial_port)
    rtc.start_classification()