import os
import time

import numpy as np

import config
from data_acquisition import DataAcquisition
from model_training import EmotionClassifier
from preprocessing import Preprocessor
from sessions import SessionRecorder, board_meta, predictions_path, recording_dir
from utils import Logger, RollingBuffer


class RealTimeClassifier:
    """
    Classifies the live stream every PREDICT_INTERVAL seconds. The raw signal is recorded to
    data/live (data/synthetic with --synthetic) and every prediction to logs/, so the dashboard can
    follow it and review it later.
    """

    def __init__(self, model_path=config.MODEL_PATH, serial_port=config.SERIAL_PORT, board_id=config.BOARD_ID,
                 participant=None, synthetic=False):
        self.model_path = model_path
        self.serial_port = serial_port
        self.board_id = board_id
        self.participant = participant
        self.synthetic = synthetic
        self.classifier = EmotionClassifier()
        self.is_running = False

    def start_classification(self):
        """Start real-time classification."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)
        daq = DataAcquisition(self.serial_port, self.board_id)
        self.classifier.load_model(self.model_path, sampling_rate=daq.sampling_rate)

        with daq:
            daq.start_streaming()
            prep = Preprocessor(daq.sampling_rate)
            buffer = RollingBuffer(prep.window + prep.pad, len(daq.eeg_channels), reset_after=config.NO_DATA_SECONDS)
            recorder = SessionRecorder(daq.sampling_rate, recording_dir(self.synthetic, live=True), prefix='live',
                                       participant=self.participant,
                                       board='synthetic' if self.synthetic else 'ganglion', protocol='live',
                                       **board_meta(daq))
            recorder.mark(config.LIVE_LABEL, 0, 0)
            logger = Logger(predictions_path(recorder.path), self.classifier.classes, self.model_path)
            print(f"Recording to {recorder.path}, predictions to {logger.path}")
            print("Real-time classification started. Press Ctrl+C to stop.")

            status = None
            self.is_running = True
            next_prediction = time.time() + config.PREDICT_INTERVAL
            try:
                while self.is_running:
                    time.sleep(0.05)
                    timestamps, package, samples = daq.read_new()
                    recorder.write(timestamps, samples, package)
                    buffer.push(timestamps, samples)
                    now = time.time()
                    if now >= next_prediction:
                        next_prediction = max(next_prediction + config.PREDICT_INTERVAL, now)
                        self.process_window(prep, buffer, logger)
                status = 'completed'
            except KeyboardInterrupt:
                print("\nStopping classification...")
                status = 'stopped'
            finally:
                self.is_running = False
                recorder.mark_end()
                recorder.close(status)

    def process_window(self, prep, buffer, logger):
        silent = buffer.silent_for()
        if silent > config.NO_DATA_SECONDS:
            # Never classify a frozen buffer again: the board stopped sending
            print(f"\rNo data for {silent:.0f}s: check the headset and the dongle".ljust(80), end='', flush=True)
            logger.log(time.time(), 'no_data', label=config.LIVE_LABEL, block=0)
            return
        if not buffer.full:
            print(f"\rBuffering {buffer.received}/{len(buffer.data)} samples...", end='', flush=True)
            return

        features, artifact, reason, peaks = prep.analyze_window(buffer.data)
        if artifact:
            where = f" on {config.CHANNEL_NAMES[int(np.argmax(peaks))]}" if reason == 'amplitude' else ''
            print(f"\rArtifact ({reason}{where}), window skipped".ljust(80), end='', flush=True)
            logger.log(buffer.last_timestamp, 'artifact', label=config.LIVE_LABEL, block=0)
            return
        prediction, confidence, probabilities = self.classifier.predict(features)
        probs = " | ".join(f"{cls}: {p:.2f}" for cls, p in zip(self.classifier.classes, probabilities))
        print(f"\rPredicted: {prediction} ({confidence:.2f}) | {probs}".ljust(80), end='', flush=True)
        logger.log(buffer.last_timestamp, 'ok', prediction, confidence, probabilities, label=config.LIVE_LABEL, block=0)


def run_real_time_classification(model_path=config.MODEL_PATH, serial_port=config.SERIAL_PORT,
                                 board_id=config.BOARD_ID, participant=None, synthetic=False):
    """Main function to run real-time classification."""
    RealTimeClassifier(model_path, serial_port, board_id, participant, synthetic).start_classification()
