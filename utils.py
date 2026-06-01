import os
import csv
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import config

class Logger:
    def __init__(self, log_file=config.LOG_FILE):
        self.log_file = log_file
        self.ensure_log_directory()
        self.initialize_log_file()

    def ensure_log_directory(self):
        """Ensure the log directory exists."""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def initialize_log_file(self):
        """Initialize the log file with headers if it doesn't exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(config.LOG_HEADERS)

    def log_data(self, timestamp, raw_channels, features, ground_truth=None, predicted=None, confidence=None):
        """Log a single data entry."""
        row = [
            timestamp,
            raw_channels[0] if len(raw_channels) > 0 else '',
            raw_channels[1] if len(raw_channels) > 1 else '',
            raw_channels[2] if len(raw_channels) > 2 else '',
            raw_channels[3] if len(raw_channels) > 3 else '',
            json.dumps(features.tolist()) if isinstance(features, np.ndarray) else str(features),
            ground_truth,
            predicted,
            confidence
        ]

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def load_log_data(self):
        """Load logged data into a pandas DataFrame."""
        return pd.read_csv(self.log_file)

class Visualizer:
    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size
        self.eeg_buffer = deque(maxlen=buffer_size)
        self.time_buffer = deque(maxlen=buffer_size)
        self.emotion_buffer = deque(maxlen=buffer_size)

    def update_buffers(self, timestamp, eeg_data, emotion=None):
        """Update visualization buffers."""
        self.time_buffer.append(timestamp)
        self.eeg_buffer.append(eeg_data)
        if emotion:
            self.emotion_buffer.append(emotion)

    def plot_eeg_signals(self, save_path=None):
        """Plot real-time EEG signals."""
        if not self.eeg_buffer:
            return

        fig, axes = plt.subplots(config.CHANNELS, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('Real-time EEG Signals')

        eeg_data = np.array(list(self.eeg_buffer))
        time_data = np.array(list(self.time_buffer))

        for ch in range(config.CHANNELS):
            axes[ch].plot(time_data, eeg_data[:, ch])
            axes[ch].set_ylabel(f'Ch {ch+1} (V)')
            axes[ch].grid(True)

        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_emotion_probabilities(self, probabilities, save_path=None):
        """Plot emotion probability bars."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(config.CLASSES, probabilities, color=['red', 'blue', 'green'])
        ax.set_ylabel('Probability')
        ax.set_title('Emotion Classification Probabilities')
        ax.set_ylim(0, 1)

        # Add probability values on bars
        for i, prob in enumerate(probabilities):
            ax.text(i, prob + 0.01, f'{prob:.2f}', ha='center', va='bottom')

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def animate_eeg_plot(self):
        """Create animated EEG plot."""
        fig, axes = plt.subplots(config.CHANNELS, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('Real-time EEG Signals')

        lines = []
        for ch in range(config.CHANNELS):
            line, = axes[ch].plot([], [], 'b-')
            axes[ch].set_ylabel(f'Ch {ch+1} (V)')
            axes[ch].set_xlim(0, self.buffer_size / config.SAMPLING_RATE)
            axes[ch].set_ylim(-100e-6, 100e-6)  # Typical EEG range
            axes[ch].grid(True)
            lines.append(line)

        axes[-1].set_xlabel('Time (s)')

        def animate(frame):
            if len(self.eeg_buffer) > 0:
                eeg_data = np.array(list(self.eeg_buffer))
                time_data = np.array(list(self.time_buffer))
                for ch in range(config.CHANNELS):
                    lines[ch].set_data(time_data, eeg_data[:, ch])
            return lines

        ani = animation.FuncAnimation(fig, animate, interval=1000, blit=True)
        plt.show()

def create_directories():
    """Create necessary directories."""
    directories = ['models', 'logs', 'plots']
    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

# Initialize directories on import
create_directories()