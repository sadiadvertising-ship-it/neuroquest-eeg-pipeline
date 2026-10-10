#!/usr/bin/env python3
"""
GUI for EEG acquisition, visualization (Time Series + FFT), protocol-driven recording of the raw
signal to CSV, and optional live classification using a previously trained model.

Defaults:
- macOS serial port from config.SERIAL_PORT
- Board: config.BOARD_ID (Ganglion)
- Sessions saved to data/sessions/session_<participant>_YYYYmmdd_HHMMSS.csv (data/synthetic with --synthetic)
- Live predictions logged to logs/
"""

import logging
import os
import re
import signal
import sys
import time
from datetime import datetime

from pyqtgraph.Qt import QtCore, QtWidgets  # type: ignore
import pyqtgraph as pg  # type: ignore
import numpy as np

import config
import analysis
from data_acquisition import DataAcquisition
from model_training import EmotionClassifier
from preprocessing import Preprocessor
from sessions import SessionRecorder, predictions_path, protocol_meta, protocol_steps, recording_dir
from utils import Logger, RollingBuffer


class ProtocolController(QtCore.QObject):
    """
    Walks through sessions.protocol_steps() with a precise 1 s timer. step_started fires when every
    step begins, so the recorder can mark it; finished carries 'completed' or 'stopped'.
    """
    status_changed = QtCore.pyqtSignal(str)
    phase_changed = QtCore.pyqtSignal(str, str)  # phase, emotion (or the next emotion during a cue)
    round_progress = QtCore.pyqtSignal(int, int)  # current_round, total_rounds
    tick = QtCore.pyqtSignal(int)  # remaining seconds in current phase
    step_started = QtCore.pyqtSignal(str, int, int)  # label, round, block
    finished = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.steps = []
        self.block = -1
        self._remaining = 0
        self._timer = QtCore.QTimer()
        self._timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._on_tick)

    @property
    def active(self):
        return 0 <= self.block < len(self.steps)

    @property
    def label(self):
        return self.steps[self.block].label if self.active else ''

    def start(self, steps):
        if self.active:
            return
        self.steps = steps
        self.block = -1
        self._next()
        self._timer.start()

    def stop(self):
        if self.active:
            self._end('stopped', "Protocol stopped")

    def _end(self, status, text):
        self._timer.stop()
        self.block = -1
        self.phase_changed.emit("idle", "")
        self.status_changed.emit(text)
        self.finished.emit(status)

    def _next(self):
        self.block += 1
        if not self.active:
            self._end('completed', "Protocol completed")
            return
        step = self.steps[self.block]
        self._remaining = int(round(step.seconds))
        self.step_started.emit(step.label, step.round, self.block)
        self.round_progress.emit(step.round, config.TRAINING_ROUNDS)
        if step.label == config.CUE_LABEL:
            self.phase_changed.emit("cue", step.next)
            self.status_changed.emit(f"Next: {step.next} in {self._remaining}s, get the stimulus ready")
        elif step.label == config.REST_LABEL:
            self.phase_changed.emit("rest", "")
            self.status_changed.emit(f"Rest for {self._remaining}s")
        else:
            self.phase_changed.emit("measure", step.label)
            self.status_changed.emit(f"Show {step.label} now ({self._remaining}s)")
        self.tick.emit(self._remaining)

    def _on_tick(self):
        self._remaining -= 1
        self.tick.emit(max(self._remaining, 0))
        if self._remaining <= 0:
            self._next()


class EEGGui(QtWidgets.QMainWindow):
    """
    Main GUI window:
      - Time Series per channel (filtered like the classifier sees it)
      - FFT per channel
      - Status panel (phase, emotion, timers, rounds, signal quality)
      - Participant code, Start/Stop protocol, Live Classification, Load/Unload model
    """
    def __init__(self, daq: DataAcquisition, model_path=config.MODEL_PATH, participant=None, synthetic=False,
                 parent=None):
        super().__init__(parent)
        self.daq = daq
        self.model_path = model_path
        self.synthetic = synthetic
        self.setWindowTitle("EEG GUI - Time Series & FFT" + (" (synthetic board)" if synthetic else ""))
        self.resize(1100, 800)

        self.prep = Preprocessor(daq.sampling_rate)
        self.classifier = EmotionClassifier()
        self.classifier_loaded = False

        # Rolling buffer of the raw signal: filter pad + window, enough for plots and features
        self.buffer = RollingBuffer(self.prep.pad + self.prep.window, len(daq.eeg_channels),
                                    reset_after=config.NO_DATA_SECONDS)
        self.line_warning = ''  # 60 Hz noise, refreshed once per second
        self.ticks = 0

        self.recorder = None
        self.session_logger = None  # predictions made while recording, next to the session in logs/
        self.idle_logger = None  # predictions made outside a recording
        self.protocol = ProtocolController()
        self.protocol.status_changed.connect(self.on_status)
        self.protocol.phase_changed.connect(self.on_phase)
        self.protocol.round_progress.connect(self.on_round)
        self.protocol.tick.connect(self.on_tick)
        self.protocol.step_started.connect(self.on_step)
        self.protocol.finished.connect(self.on_finished)

        self.colors = ['#FFFF00', '#7FFF00', '#00BFFF', '#DA70D6'][:len(daq.eeg_channels)]
        self._init_ui(participant)

        self.timer_ui = QtCore.QTimer()
        self.timer_ui.setInterval(50)
        self.timer_ui.timeout.connect(self.on_update)
        self.timer_ui.start()

        self.live_timer = QtCore.QTimer()
        self.live_timer.setInterval(int(config.PREDICT_INTERVAL * 1000))
        self.live_timer.timeout.connect(self.on_live_predict)

        if os.path.exists(model_path):
            self.on_load_model()

    def _init_ui(self, participant):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        ctrl_layout = QtWidgets.QHBoxLayout()
        self.txt_participant = QtWidgets.QLineEdit(participant or '')
        self.txt_participant.setPlaceholderText("Participant code, e.g. P03" + (" (optional)" if self.synthetic else ""))
        self.txt_participant.setMaximumWidth(220)
        self.btn_start = QtWidgets.QPushButton("Start Protocol")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.chk_live = QtWidgets.QCheckBox("Live Classification")
        self.btn_load_model = QtWidgets.QPushButton("Load Model")
        self.btn_unload_model = QtWidgets.QPushButton("Unload Model")
        ctrl_layout.addWidget(self.txt_participant)
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.chk_live)
        ctrl_layout.addWidget(self.btn_load_model)
        ctrl_layout.addWidget(self.btn_unload_model)
        layout.addLayout(ctrl_layout)

        self.lbl_status = QtWidgets.QLabel("Status: Idle")
        self.lbl_phase = QtWidgets.QLabel("Phase: -")
        self.lbl_emotion = QtWidgets.QLabel("Emotion: -")
        self.lbl_round = QtWidgets.QLabel("Round: 0/0")
        self.lbl_timer = QtWidgets.QLabel("Time: 0")
        self.lbl_signal = QtWidgets.QLabel("Signal: -")
        self.lbl_predict = QtWidgets.QLabel("Prediction: - (conf -)")
        status_layout = QtWidgets.QHBoxLayout()
        for w in [self.lbl_status, self.lbl_phase, self.lbl_emotion, self.lbl_round,
                  self.lbl_timer, self.lbl_signal, self.lbl_predict]:
            status_layout.addWidget(w)
            status_layout.addSpacing(12)
        layout.addLayout(status_layout)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        self.ts_widget = pg.GraphicsLayoutWidget(show=False)
        self.curves_ts = []
        for i, color in enumerate(self.colors):
            p = self.ts_widget.addPlot(row=i, col=0)
            p.showAxis('bottom', False)
            p.setMenuEnabled('left', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle(f'Time Series ({config.BANDPASS_LOW}-{config.BANDPASS_HIGH} Hz, uV)')
            p.setLabel('left', config.CHANNEL_NAMES[i])
            p.showGrid(x=True, y=True, alpha=0.3)
            self.curves_ts.append(p.plot(pen=pg.mkPen(color=color, width=2)))
        self.tabs.addTab(self.ts_widget, "Time Series")

        self.fft_widget = pg.GraphicsLayoutWidget(show=False)
        self.curves_fft = []
        for i, color in enumerate(self.colors):
            p = self.fft_widget.addPlot(row=i, col=0)
            p.setMenuEnabled('left', False)
            if i == 0:
                p.setTitle('FFT (Magnitude)')
            p.setLabel('left', config.CHANNEL_NAMES[i])
            p.setLabel('bottom', 'Frequency (Hz)')
            p.showGrid(x=True, y=True, alpha=0.3)
            self.curves_fft.append(p.plot(pen=pg.mkPen(color=color, width=2)))
        self.tabs.addTab(self.fft_widget, "FFT")

        self.txt_participant.textChanged.connect(self.on_participant_changed)
        self.btn_start.clicked.connect(self.on_start_protocol)
        self.btn_stop.clicked.connect(self.protocol.stop)
        self.btn_load_model.clicked.connect(self.on_load_model)
        self.btn_unload_model.clicked.connect(self.on_unload_model)
        self.chk_live.stateChanged.connect(self.on_toggle_live)
        self.on_participant_changed(self.txt_participant.text())

    # UI event handlers
    def participant(self):
        return self.txt_participant.text().strip()

    def on_participant_changed(self, text):
        valid = bool(re.match(config.PARTICIPANT_PATTERN, text.strip()))
        self.btn_start.setEnabled(valid or (self.synthetic and not text.strip()))

    def on_start_protocol(self):
        if self.protocol.active:
            return
        participant = self.participant()
        valid = bool(re.match(config.PARTICIPANT_PATTERN, participant))
        if not valid and (participant or not self.synthetic):
            self.on_status("Enter a participant code such as P03 (never a name) before starting")
            return
        steps = protocol_steps()
        try:
            self.recorder = SessionRecorder(self.daq.sampling_rate, recording_dir(self.synthetic),
                                            participant=participant or None,
                                            board='synthetic' if self.synthetic else 'ganglion',
                                            **protocol_meta(steps))
        except OSError as e:
            self.on_status(f"Could not create the session file: {e}")
            return
        self.session_logger = None
        self.txt_participant.setEnabled(False)
        self.protocol.start(steps)

    def on_step(self, label, round_num, block):
        if self.recorder is not None:
            self.recorder.mark(label, round_num, block)

    def on_finished(self, status):
        self.txt_participant.setEnabled(True)
        if self.recorder is None:
            return
        path = self.recorder.path
        self.recorder.mark_end()
        self.pull()  # samples acquired before the end still belong to the last block
        if self.recorder is None:
            return  # writing failed while pulling; recording_failed already told the operator
        recorder, self.recorder = self.recorder, None
        self.session_logger = None
        try:
            recorder.close(status)
        except OSError as e:
            logging.exception("Could not close %s", path)
            self.on_status(f"Could not close {path} ({e}); training will treat it as incomplete")
            return
        self.on_status(f"Session saved ({status}): {path}")

    def recording_failed(self, error):
        logging.error("Recording failed: %s", error)
        recorder, self.recorder = self.recorder, None
        self.session_logger = None
        try:
            recorder.close('stopped')
        except OSError:
            pass
        self.protocol.stop()
        self.on_status(f"Recording failed ({error}); the protocol was stopped")

    def on_toggle_live(self, state):
        if state == QtCore.Qt.Checked:
            self.live_timer.start()
        else:
            self.live_timer.stop()

    def on_load_model(self):
        try:
            self.classifier.load_model(self.model_path, sampling_rate=self.daq.sampling_rate)
            self.classifier_loaded = True
            self.session_logger = self.idle_logger = None
            self.on_status(f"Model loaded: {self.model_path}")
        except Exception as e:
            self.on_status(f"Failed to load model: {e}")

    def on_unload_model(self):
        self.classifier = EmotionClassifier()
        self.classifier_loaded = False
        self.session_logger = self.idle_logger = None
        self.lbl_predict.setText("Prediction: - (conf -)")
        self.on_status("Model unloaded")

    def on_status(self, text: str):
        self.lbl_status.setText(f"Status: {text}")

    def on_phase(self, phase: str, emotion: str):
        self.lbl_phase.setText(f"Phase: {phase}")
        if phase == 'cue':
            self.lbl_emotion.setText(f"Next: {emotion}")
        else:
            self.lbl_emotion.setText(f"Emotion: {emotion if emotion else '-'}")

    def on_round(self, current: int, total: int):
        self.lbl_round.setText(f"Round: {current}/{total}")

    def on_tick(self, remaining: int):
        self.lbl_timer.setText(f"Time: {remaining}")

    def pull(self):
        """Read what the board sent since the last call: record it if recording, keep it for the plots."""
        timestamps, package, samples = self.daq.read_new()
        if self.recorder is not None and len(timestamps):
            try:
                self.recorder.write(timestamps, samples, package)
            except OSError as e:
                self.recording_failed(e)
        self.buffer.push(timestamps, samples)

    def signal_text(self, window):
        peaks = np.abs(window).max(axis=0)
        loud = [ch for ch, p in zip(config.CHANNEL_NAMES, peaks) if p > config.ARTIFACT_UV]
        flat = [ch for ch, s in zip(config.CHANNEL_NAMES, window.std(axis=0)) if s < config.FLAT_UV]
        if loud:
            return f"Signal: ARTIFACT on {', '.join(loud)}" + self.line_warning
        if flat:
            return f"Signal: FLAT on {', '.join(flat)}" + self.line_warning
        return "Signal: OK" + self.line_warning

    def update_line_warning(self):
        """A channel with much more 60 Hz noise than the others usually has a loose electrode."""
        fs = self.daq.sampling_rate
        line = analysis.line_noise(self.buffer.data[-int(2 * fs):], fs)
        noisy = [ch for i, ch in enumerate(config.CHANNEL_NAMES)
                 if line[i] >= analysis.LINE_MIN_UV and line[i] >= analysis.LINE_RATIO * max(np.median(np.delete(line, i)), 1e-6)]
        self.line_warning = f" | 60 Hz noise on {', '.join(noisy)}: check that electrode" if noisy else ''

    def on_update(self):
        # Exceptions inside Qt slots abort PyQt5, so a bad packet must not take the session down
        try:
            self.pull()
            silent = self.buffer.silent_for()
            if silent > config.NO_DATA_SECONDS:
                self.lbl_signal.setText("Signal: waiting for data" if silent == float('inf')
                                        else f"Signal: NO DATA for {silent:.0f}s, check the headset")
                return
            if not self.buffer.full:
                return
            window = self.prep.clean_window(self.buffer.data)
            self.ticks += 1
            if self.ticks % 20 == 1:
                self.update_line_warning()
            self.lbl_signal.setText(self.signal_text(window))
            for i, curve in enumerate(self.curves_ts):
                curve.setData(window[:, i])
            freqs = np.fft.rfftfreq(len(window), d=1.0 / self.daq.sampling_rate)
            mags = np.abs(np.fft.rfft(window * np.hamming(len(window))[:, None], axis=0))
            for i, curve in enumerate(self.curves_fft):
                curve.setData(freqs, mags[:, i])
        except Exception:
            logging.exception("UI update failed")

    def prediction_logger(self):
        if self.recorder is not None:
            if self.session_logger is None:
                self.session_logger = Logger(predictions_path(self.recorder.path), self.classifier.classes,
                                             self.model_path)
            return self.session_logger
        if self.idle_logger is None:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.idle_logger = Logger(os.path.join(config.LOGS_DIR, f'gui_{stamp}_predictions.csv'),
                                      self.classifier.classes, self.model_path)
        return self.idle_logger

    def on_live_predict(self):
        if not self.classifier_loaded:
            return
        try:
            label, block = (self.protocol.label, self.protocol.block) if self.protocol.active else ('', '')
            if self.buffer.silent_for() > config.NO_DATA_SECONDS:
                self.lbl_predict.setText("Prediction: - (no data)")
                self.prediction_logger().log(time.time(), 'no_data', label=label, block=block)
                return
            if not self.buffer.full:
                return
            features, artifact, _, _ = self.prep.analyze_window(self.buffer.data)
            if artifact:
                self.lbl_predict.setText("Prediction: - (artifact)")
                self.prediction_logger().log(self.buffer.last_timestamp, 'artifact', label=label, block=block)
                return
            pred, conf, probs = self.classifier.predict(features)
            self.lbl_predict.setText(f"Prediction: {pred} (conf {conf:.2f})")
            self.prediction_logger().log(self.buffer.last_timestamp, 'ok', pred, conf, probs, label=label, block=block)
        except Exception as e:
            self.lbl_predict.setText(f"Prediction error: {e}")

    def closeEvent(self, event):
        self.protocol.stop()  # a recording in progress is closed and marked as stopped
        self.timer_ui.stop()
        self.live_timer.stop()
        super().closeEvent(event)


def run_gui(serial_port: str = None, board_id: int = config.BOARD_ID, model_path: str = config.MODEL_PATH,
            participant: str = None, synthetic: bool = False):
    """
    Launch the GUI:
      - Initializes BrainFlow acquisition
      - Starts Qt event loop
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    daq = DataAcquisition(serial_port or config.SERIAL_PORT, board_id)
    try:
        daq.initialize_board()
    except BaseException:
        daq.close()  # Ctrl+C or an error halfway through prepare_session
        raise
    ret = 0
    try:
        daq.start_streaming()
        win = EEGGui(daq, model_path, participant, synthetic)
        win.show()
        # Ctrl+C in the terminal closes the window first (so a recording is marked as stopped), then
        # quits; the 50 ms UI timer gives Python the chance to run the handler.
        signal.signal(signal.SIGINT, lambda *_: (app.closeAllWindows(), app.quit()))
        ret = app.exec_()
    finally:
        try:
            daq.stop_streaming()
        finally:
            daq.close()
    sys.exit(ret)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_gui()
