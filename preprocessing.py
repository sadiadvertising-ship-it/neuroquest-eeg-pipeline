import numpy as np
from scipy import signal
from scipy.signal import welch
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
try:
    import mne  # type: ignore[reportMissingImports]
    from mne.preprocessing import ICA  # type: ignore[reportMissingImports]
    MNE_AVAILABLE = True
except Exception:
    MNE_AVAILABLE = False
    mne = None  # type: ignore[assignment]
    ICA = None  # type: ignore[assignment]
from concurrent.futures import ThreadPoolExecutor, as_completed
import config

class Preprocessor:
    def __init__(self, sampling_rate=config.SAMPLING_RATE):
        self.sampling_rate = sampling_rate

    def apply_filters(self, data):
        """Apply bandpass and notch filters to EEG data."""
        # Bandpass filter 1-50 Hz
        b, a = signal.butter(4, [config.BANDPASS_LOW, config.BANDPASS_HIGH], btype='band', fs=self.sampling_rate)
        filtered_data = signal.filtfilt(b, a, data, axis=0)

        # Notch filter at 50 Hz
        b_notch, a_notch = signal.iirnotch(config.NOTCH_FREQ, Q=30, fs=self.sampling_rate)
        filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data, axis=0)

        return filtered_data

    def remove_artifacts_ica(self, data):
        """Advanced artifact removal using ICA."""
        if not MNE_AVAILABLE:
            return data  # Return original data if MNE not available

        # Basic sanity checks to avoid ICA errors
        n_samples, n_channels = data.shape
        if n_channels < 2 or n_samples < max(10, getattr(config, "ICA_COMPONENTS", 1)):
            return data

        # Create MNE Raw object
        info = mne.create_info(ch_names=[f'ch{i}' for i in range(n_channels)],
                               sfreq=self.sampling_rate, ch_types='eeg')
        raw = mne.io.RawArray(data.T, info)

        # Apply ICA with safe number of components
        safe_components = max(1, min(getattr(config, "ICA_COMPONENTS", 1), n_channels - 1))
        ica = ICA(n_components=safe_components, random_state=42)
        ica.fit(raw)
        ica.apply(raw, exclude=[0])  # Exclude first component as example; tune based on inspection

        return raw.get_data().T

    def extract_wavelet_features(self, data):
        """Extract wavelet features using Continuous Wavelet Transform."""
        if not PYWT_AVAILABLE:
            return np.array([])  # Return empty if pywt not available
        wavelet_features = []
        # Use scale indices directly for CWT and pass sampling_period to map to Hz
        scales = np.arange(1, 128)
        for ch in range(data.shape[1]):
            coeffs, freqs = pywt.cwt(
                data[:, ch],
                scales,
                config.WAVELET,
                sampling_period=1.0 / self.sampling_rate
            )
            wavelet_features.extend(np.mean(np.abs(coeffs), axis=1))  # Mean absolute coefficients per scale
        return np.array(wavelet_features)

    def extract_features(self, data):
        """Extract enhanced features including statistical, PSD, and wavelet."""
        features = {}

        # Statistical features
        features['mean'] = np.mean(data, axis=0)
        features['variance'] = np.var(data, axis=0)
        features['std'] = np.std(data, axis=0)
        features['rms'] = np.sqrt(np.mean(data**2, axis=0))

        # Power spectral density features
        freqs, psd = welch(data, fs=self.sampling_rate, nperseg=256, axis=0)

        # Alpha band (8-12 Hz)
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        features['alpha_power'] = np.mean(psd[alpha_mask], axis=0)

        # Beta band (13-30 Hz)
        beta_mask = (freqs >= 13) & (freqs <= 30)
        features['beta_power'] = np.mean(psd[beta_mask], axis=0)

        # Theta band (4-7 Hz)
        theta_mask = (freqs >= 4) & (freqs <= 7)
        features['theta_power'] = np.mean(psd[theta_mask], axis=0)

        # Alpha/Beta ratio
        features['alpha_beta_ratio'] = features['alpha_power'] / (features['beta_power'] + 1e-10)

        # Wavelet features
        wavelet_feats = self.extract_wavelet_features(data)
        features['wavelet'] = wavelet_feats

        # Flatten features
        feature_vector = np.concatenate([v.flatten() for v in features.values()])

        return feature_vector

    def preprocess_window(self, data_window):
        """Enhanced preprocessing pipeline with ICA and wavelets."""
        # Apply filters
        filtered = self.apply_filters(data_window)

        # Advanced artifact removal
        clean_data = self.remove_artifacts_ica(filtered)

        # Extract enhanced features
        features = self.extract_features(clean_data)

        return features

    def preprocess_batch(self, data_batch, max_workers=4):
        """Preprocess a batch of data windows in parallel."""
        features_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_window = {executor.submit(self.preprocess_window, window): window for window in data_batch}
            for future in as_completed(future_to_window):
                features_list.append(future.result())
        return np.array(features_list)