#!/usr/bin/env python3
"""
GUI for EEG acquisition, visualization (Time Series + FFT), protocol-driven data collection to CSV,
and optional live classification using a previously trained model.

Defaults:
- macOS serial port from config.SERIAL_PORT
- Board: config.BOARD_ID (Ganglion)
- Sessions saved to data/sessions/session_YYYYmmdd_HHMMSS.csv
"""

import os
import sys
import csv
import time
import math
import logging
from datetime import datetime

import numpy as np

from pyqtgraph.Qt import QtCore, QtWidgets  # type: ignore
import pyqtgraph as pg  # type: ignore

from brainflow.board_shim import BoardShim, BrainFlowInputParams  # type: ignore
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations  # type: ignore

import config
from preprocessing import Preprocessor
from model_training import EmotionClassifier


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class AcquisitionManager(QtCore.QObject):
    """
    Wraps BrainFlow BoardShim for continuous data access.
    Provides methods to get the most recent window of EXG data.
    """
    def __init__(self, serial_port: str, parent=None):
        super().__init__(parent)
        self.serial_port = serial_port
        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port

        self.board_id = config.BOARD_ID if hasattr(config, "BOARD_ID") else 1
        self.board_shim = BoardShim(self.board_id, self.params)

        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)

        # Window samples based on config
        self.window_sec = getattr(config, "WINDOW_SIZE", 2.0)
        self.num_points = int(self.window_sec * self.sampling_rate)

        self._is_running = False

    def start(self):
        BoardShim.enable_dev_board_logger()
        logging.info("Preparing BrainFlow session on port: %s", self.serial_port)
        self.board_shim.prepare_session()
        # Increase buffer to reduce packet drops
        self.board_shim.start_stream(450000, "")
        self._is_running = True
        logging.info("BrainFlow stream started")

    def stop(self):
        try:
            if self._is_running:
                self.board_shim.stop_stream()
            if self.board_shim.is_prepared():
                self.board_shim.release_session()
        except Exception:
            logging.exception("Error while stopping BoardShim")
        finally:
            self._is_running = False
            logging.info("BrainFlow session released")

    def get_window(self) -> np.ndarray:
        """
        Returns most recent window [samples, channels] of EXG data.
        Applies detrend and notch/bandpass filters inline for visualization and features.
        """
        data = self.board_shim.get_current_board_data(self.num_points)
        if data.shape[1] == 0:
            # No data yet
            return np.zeros((self.num_points, len(self.exg_channels)))

        # data shape: [num_rows, num_samples]; EXG channels are indices in rows
        # We need [samples, channels]
        window = []
        for ch in self.exg_channels:
            series = np.copy(data[ch, :])  # shape [num_points]
            # Inline filtering similar to user's working code
            DataFilter.detrend(series, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(series, self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(series, self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(series, self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            window.append(series)
        window = np.stack(window, axis=1)  # [samples, channels]
        return window


class CSVWriter:
    """
    Writes feature vectors and labels to CSV. Header written once on first write.
    """
    def __init__(self, base_dir: str = "data/sessions"):
        ensure_dir(base_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(base_dir, f"session_{ts}.csv")
        self._file = open(self.path, "w", newline="")
        self._writer = None
        self._header_written = False
        logging.info("Writing session CSV to %s", self.path)

    def write_row(self, features: np.ndarray, label: str):
        if self._writer is None:
            self._writer = csv.writer(self._file)
        if not self._header_written:
            header = [f"feature_{i:04d}" for i in range(features.shape[0])] + ["label"]
            self._writer.writerow(header)
            self._header_written = True
        row = list(np.asarray(features).ravel()) + [label]
        self._writer.writerow(row)

    def close(self):
        try:
            if self._file:
                self._file.close()
        except Exception:
            pass


class ProtocolController(QtCore.QObject):
    """
    Drives the protocol:
      For each round in config.TRAINING_ROUNDS:
        For each emotion in config.CLASSES:
          - Measure SAMPLING_DURATION seconds (collect labeled windows)
          - Rest REST_DURATION seconds
    Emits signals for UI updates.
    """
    status_changed = QtCore.pyqtSignal(str)
    phase_changed = QtCore.pyqtSignal(str, str)  # phase, emotion_or_blank
    round_progress = QtCore.pyqtSignal(int, int)  # current_round, total_rounds
    tick = QtCore.pyqtSignal(int)  # remaining seconds in current phase

    def __init__(self, parent=None):
        super().__init__(parent)
        self.emotions = list(config.CLASSES)
        self.rounds = int(getattr(config, "TRAINING_ROUNDS", 3))
        self.measure_sec = float(getattr(config, "SAMPLING_DURATION", 15.0))
        self.rest_sec = float(getattr(config, "REST_DURATION", 11.0))

        self._active = False
        self._current_round = 0
        self._idx_emotion = 0
        self._phase = "idle"
        self._remaining = 0

        self._timer = QtCore.QTimer()
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._on_tick)

    def start(self):
        if self._active:
            return
        self._active = True
        self._current_round = 1
        self._idx_emotion = 0
        self._enter_measure()

    def stop(self):
        self._active = False
        self._phase = "idle"
        self._timer.stop()
        self.status_changed.emit("Protocol stopped")
        self.phase_changed.emit("idle", "")

    def _enter_measure(self):
        if not self._active:
            return
        if self._current_round > self.rounds:
            self.stop()
            self.status_changed.emit("Protocol completed")
            return

        if self._idx_emotion >= len(self.emotions):
            # Next round
            self._idx_emotion = 0
            self._current_round += 1
            self._enter_measure()
            return

        emotion = self.emotions[self._idx_emotion]
        self._phase = "measure"
        self._remaining = int(self.measure_sec)
        self.round_progress.emit(self._current_round, self.rounds)
        self.phase_changed.emit("measure", emotion)
        self.status_changed.emit(f"Measuring {emotion} for {self._remaining}s")
        self._timer.start()

    def _enter_rest(self):
        if not self._active:
            return
        self._phase = "rest"
        self._remaining = int(self.rest_sec)
        self.phase_changed.emit("rest", "")
        self.status_changed.emit(f"Rest for {self._remaining}s")
        self._timer.start()

    def _on_tick(self):
        self._remaining -= 1
        self.tick.emit(max(self._remaining, 0))
        if self._remaining <= 0:
            self._timer.stop()
            if self._phase == "measure":
                # finished measuring this emotion
                self._idx_emotion += 1
                # Rest unless we just finished the last emotion and last round
                if not (self._current_round == self.rounds and self._idx_emotion >= len(self.emotions)):
                    self._enter_rest()
                else:
                    # Completed protocol
                    self.stop()
            elif self._phase == "rest":
                # Go to next emotion measure
                self._enter_measure()


class EEGGui(QtWidgets.QMainWindow):
    """
    Main GUI window:
      - Time Series per channel
      - FFT per channel
      - Status panel (phase, emotion, timers, rounds)
      - Buttons: Start/Stop protocol, Toggle Live Classification, Load/Unload model
    """
    def __init__(self, acq: AcquisitionManager, parent=None):
        super().__init__(parent)
        self.acq = acq
        self.setWindowTitle("EEG GUI - Time Series & FFT")
        self.resize(1100, 800)

        # Core components
        self.prep = Preprocessor()
        # Use model without deep extractor for stable runtime integration
        self.classifier = EmotionClassifier(model_type="hybrid")
        self.classifier_loaded = False

        self.writer = None  # CSVWriter
        self.protocol = ProtocolController()
        self.protocol.status_changed.connect(self.on_status)
        self.protocol.phase_changed.connect(self.on_phase)
        self.protocol.round_progress.connect(self.on_round)
        self.protocol.tick.connect(self.on_tick)

        # UI setup
        self._init_ui()

        # Timers
        self.update_ms = 50
        self.timer_ui = QtCore.QTimer()
        self.timer_ui.setInterval(self.update_ms)
        self.timer_ui.timeout.connect(self.on_update)
        self.timer_ui.start()

        # Window feature writer cadence (every WINDOW_SIZE sec) during measure
        self.win_timer = QtCore.QTimer()
        self.win_timer.setInterval(int(getattr(config, "WINDOW_SIZE", 2.0) * 1000))
        self.win_timer.timeout.connect(self.on_window_capture)

        # Live classification timer (once per second)
        self.live_timer = QtCore.QTimer()
        self.live_timer.setInterval(1000)
        self.live_timer.timeout.connect(self.on_live_predict)

        # Colors for channels
        self.colors = ['#FFFF00', '#7FFF00', '#00BFFF', '#DA70D6', '#FFA500', '#00FF7F', '#FF69B4', '#1E90FF']
        self.colors = self.colors[:len(self.acq.exg_channels)]

    def _init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Controls
        ctrl_layout = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start Protocol")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.chk_live = QtWidgets.QCheckBox("Live Classification")
        self.btn_load_model = QtWidgets.QPushButton("Load Model")
        self.btn_unload_model = QtWidgets.QPushButton("Unload Model")
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.chk_live)
        ctrl_layout.addWidget(self.btn_load_model)
        ctrl_layout.addWidget(self.btn_unload_model)
        layout.addLayout(ctrl_layout)

        # Status
        self.lbl_status = QtWidgets.QLabel("Status: Idle")
        self.lbl_phase = QtWidgets.QLabel("Phase: -")
        self.lbl_emotion = QtWidgets.QLabel("Emotion: -")
        self.lbl_round = QtWidgets.QLabel("Round: 0/0")
        self.lbl_timer = QtWidgets.QLabel("Time: 0")
        self.lbl_predict = QtWidgets.QLabel("Prediction: - (conf -)")
        status_layout = QtWidgets.QHBoxLayout()
        for w in [self.lbl_status, self.lbl_phase, self.lbl_emotion, self.lbl_round, self.lbl_timer, self.lbl_predict]:
            status_layout.addWidget(w)
            status_layout.addSpacing(12)
        layout.addLayout(status_layout)

        # Tabs: Time Series and FFT
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        # Time Series Tab
        self.ts_widget = pg.GraphicsLayoutWidget(show=False)
        self.plots_ts = []
        self.curves_ts = []
        for i, color in enumerate(self.colors):
            p = self.ts_widget.addPlot(row=i, col=0)
            p.showAxis('left', True)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('Time Series')
            p.setLabel('left', f'Ch {i+1}')
            p.showGrid(x=True, y=True, alpha=0.3)
            curve = p.plot(pen=pg.mkPen(color=color, width=2))
            self.plots_ts.append(p)
            self.curves_ts.append(curve)
        self.tabs.addTab(self.ts_widget, "Time Series")

        # FFT Tab
        self.fft_widget = pg.GraphicsLayoutWidget(show=False)
        self.plots_fft = []
        self.curves_fft = []
        for i, color in enumerate(self.colors):
            p = self.fft_widget.addPlot(row=i, col=0)
            p.showAxis('left', True)
            p.setMenuEnabled('left', False)
            if i == 0:
                p.setTitle('FFT (Magnitude)')
            p.setLabel('left', f'Ch {i+1}')
            p.setLabel('bottom', 'Frequency (Hz)')
            p.showGrid(x=True, y=True, alpha=0.3)
            curve = p.plot(pen=pg.mkPen(color=color, width=2))
            self.plots_fft.append(p)
            self.curves_fft.append(curve)
        self.tabs.addTab(self.fft_widget, "FFT")

        # Wire controls
        self.btn_start.clicked.connect(self.on_start_protocol)
        self.btn_stop.clicked.connect(self.on_stop_protocol)
        self.btn_load_model.clicked.connect(self.on_load_model)
        self.btn_unload_model.clicked.connect(self.on_unload_model)
        self.chk_live.stateChanged.connect(self.on_toggle_live)

    # UI event handlers
    def on_start_protocol(self):
        if self.writer is None:
            self.writer = CSVWriter(base_dir="data/sessions")
        self.protocol.start()
        self.win_timer.start()

    def on_stop_protocol(self):
        self.protocol.stop()
        self.win_timer.stop()
        if self.writer:
            self.writer.close()
            self.writer = None

    def on_toggle_live(self, state):
        if state == QtCore.Qt.Checked:
            self.live_timer.start()
        else:
            self.live_timer.stop()

    def on_load_model(self):
        try:
            self.classifier.load_model(config.MODEL_PATH)
            self.classifier_loaded = True
            self.on_status(f"Model loaded: {config.MODEL_PATH}")
        except Exception as e:
            self.on_status(f"Failed to load model: {e}")

    def on_unload_model(self):
        self.classifier = EmotionClassifier(model_type="hybrid")
        self.classifier_loaded = False
        self.on_status("Model unloaded")

    def on_status(self, text: str):
        self.lbl_status.setText(f"Status: {text}")

    def on_phase(self, phase: str, emotion: str):
        self.lbl_phase.setText(f"Phase: {phase}")
        self.lbl_emotion.setText(f"Emotion: {emotion if emotion else '-'}")

    def on_round(self, current: int, total: int):
        self.lbl_round.setText(f"Round: {current}/{total}")

    def on_tick(self, remaining: int):
        self.lbl_timer.setText(f"Time: {remaining}")

    def on_update(self):
        # UI update: Time Series and FFT
        window = self.acq.get_window()  # [samples, channels]
        if window.shape[0] == 0:
            return

        # Time Series curves
        for i, curve in enumerate(self.curves_ts):
            curve.setData(window[:, i])

        # FFT curves (rfft over window)
        # Frequency axis
        n = window.shape[0]
        sr = self.acq.sampling_rate
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        for i, curve in enumerate(self.curves_fft):
            sig = window[:, i]
            spec = np.fft.rfft(sig)
            mag = np.abs(spec)
            curve.setData(freqs, mag)

    def on_window_capture(self):
        # Called every WINDOW_SIZE seconds; if measuring, write labeled features
        if self.protocol._phase != "measure" or self.writer is None:
            return
        window = self.acq.get_window()
        if window.shape[0] == 0:
            return
        # Compute features
        feats = self.prep.preprocess_window(window)  # 1D vector
        # Current emotion from label
        emotion = self.lbl_emotion.text().replace("Emotion: ", "")
        if emotion and emotion != "-":
            self.writer.write_row(feats, emotion)

    def on_live_predict(self):
        # Predict current emotion from window using loaded model
        if not self.classifier_loaded:
            return
        window = self.acq.get_window()
        if window.shape[0] == 0:
            return
        feats = self.prep.preprocess_window(window)
        try:
            pred, conf, _ = self.classifier.predict(feats)
            self.lbl_predict.setText(f"Prediction: {pred} (conf {conf:.2f})")
        except Exception as e:
            self.lbl_predict.setText(f"Prediction error: {e}")


def run_gui(serial_port: str = None):
    """
    Launch the GUI:
      - Initializes BrainFlow acquisition
      - Starts Qt event loop
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    port = serial_port or getattr(config, "SERIAL_PORT", "/dev/cu.usbmodem11")
    acq = AcquisitionManager(port)
    acq.start()

    win = EEGGui(acq)
    win.show()

    ret = 0
    try:
        ret = app.exec_()
    finally:
        acq.stop()
    sys.exit(ret)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_gui()