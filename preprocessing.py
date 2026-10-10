import numpy as np
from scipy import signal
import config


def feature_names():
    return [f'{ch}_{band}' for ch in config.CHANNEL_NAMES for band in config.BANDS]


class Preprocessor:
    """
    Pipeline of thesis Figure Arq-5:
    band-pass 0.5-45 Hz + notch 60 Hz -> artifact flag at +-100 uV -> Hamming FFT -> log band power.
    """

    def __init__(self, sampling_rate=config.SAMPLING_RATE):
        self.sampling_rate = sampling_rate
        self.window = int(config.WINDOW_SIZE * sampling_rate)
        self.pad = int(config.FILTER_PAD * sampling_rate)
        bandpass = signal.butter(4, [config.BANDPASS_LOW, config.BANDPASS_HIGH], btype='band',
                                 fs=sampling_rate, output='sos')
        notch = signal.tf2sos(*signal.iirnotch(config.NOTCH_FREQ, Q=30, fs=sampling_rate))
        self.sos = np.vstack([bandpass, notch])
        self.zi = signal.sosfilt_zi(self.sos)

    def apply_filters(self, data):
        """
        Causal band-pass + notch along time. data: [samples, channels].
        Zero-phase filtfilt leaves transients at the newest edge of the window, which is the part
        that matters in real time, so the filter runs forward only and starts from the first
        sample's level (no step from the DC offset of the Ganglion).
        """
        filtered, _ = signal.sosfilt(self.sos, data, axis=0, zi=self.zi[:, :, None] * data[0])
        return filtered

    def clean_window(self, raw):
        """Filter the raw tail (pad + window) and keep only the last window."""
        return self.apply_filters(raw)[-self.window:]

    def is_artifact(self, window):
        """Same rule as analyze_window, on an already clean window."""
        return bool(np.any(np.abs(window) > config.ARTIFACT_UV) or np.any(window.std(axis=0) < config.FLAT_UV))

    def analyze_window(self, raw):
        """
        raw: last pad + window samples [samples, channels].
        Returns (features, artifact, reason, peaks). reason is 'amplitude' when a channel goes beyond
        +-ARTIFACT_UV, 'flat' when a channel is flat or saturated, '' otherwise; peaks is the largest
        absolute uV of each channel in the clean window.
        """
        window = self.clean_window(raw)
        peaks = np.abs(window).max(axis=0)
        if np.any(peaks > config.ARTIFACT_UV):
            reason = 'amplitude'
        elif np.any(window.std(axis=0) < config.FLAT_UV):
            reason = 'flat'
        else:
            reason = ''
        return self.extract_features(window), bool(reason), reason, peaks

    def extract_features(self, window):
        """Log10 band power per channel, ordered as feature_names()."""
        freqs, psd = signal.periodogram(window, fs=self.sampling_rate, window='hamming', axis=0)
        df = freqs[1] - freqs[0]
        power = np.array([psd[(freqs >= lo) & (freqs < hi)].sum(axis=0) * df
                          for lo, hi in config.BANDS.values()])
        return np.log10(power.T.ravel() + 1e-12)

    def preprocess_window(self, raw):
        """raw: last pad + window samples [samples, channels]. Returns (features, is_artifact)."""
        features, artifact, _, _ = self.analyze_window(raw)
        return features, artifact
