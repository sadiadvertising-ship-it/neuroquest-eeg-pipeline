import csv
import os
import time

import numpy as np


class RollingBuffer:
    """
    Last n samples of the stream, plus when the newest one arrived (to notice a silent board). After a
    silence longer than reset_after seconds it counts as empty again, so a window never mixes the signal
    from before a cut with the first samples after it.
    """

    def __init__(self, n_samples, n_channels, reset_after=None):
        self.data = np.zeros((n_samples, n_channels))
        self.received = 0
        self.last_timestamp = None
        self.last_arrival = None
        self.reset_after = reset_after

    def push(self, timestamps, samples):
        n = len(samples)
        if not n:
            return
        if self.reset_after is not None and self.silent_for() > self.reset_after:
            self.received = 0
        if n >= len(self.data):
            self.data[:] = samples[-len(self.data):]
        else:
            self.data = np.roll(self.data, -n, axis=0)
            self.data[-n:] = samples
        self.received += n
        self.last_timestamp = float(timestamps[-1])
        self.last_arrival = time.time()

    @property
    def full(self):
        return self.received >= len(self.data)

    def silent_for(self, now=None):
        """Seconds since the last sample arrived (infinite before the first one)."""
        if self.last_arrival is None:
            return float('inf')
        return (time.time() if now is None else now) - self.last_arrival


class Logger:
    """Appends one row per live prediction: ok, artifact or no_data."""

    def __init__(self, path, classes, model_path=''):
        self.path = path
        self.classes = list(classes)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                f.write(f'# model={model_path}\n')
                csv.writer(f).writerow(['timestamp', 'status', 'label', 'block', 'predicted', 'confidence',
                                        *[f'p_{c}' for c in self.classes]])

    def log(self, timestamp, status, predicted=None, confidence=None, probabilities=None, label='', block=''):
        probs = [f'{p:.4f}' for p in probabilities] if probabilities is not None else [''] * len(self.classes)
        conf = f'{confidence:.4f}' if confidence is not None else ''
        with open(self.path, 'a', newline='') as f:
            csv.writer(f).writerow([f'{timestamp:.3f}', status, label, block, predicted or '', conf, *probs])
