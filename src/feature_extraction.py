import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
from src.config import FFT_LENGTH, LOW_FREQ_RATIO

def decimate_low_frequency(x):
    sos = signal.cheby1(8, 0.05, 0.8 / LOW_FREQ_RATIO, output="sos")
    zi = signal.sosfilt_zi(sos)
    y, _ = signal.sosfilt(sos, x, zi=zi * x[0])
    return y[::LOW_FREQ_RATIO]


def welch_max_hold(x):
    overlap = FFT_LENGTH // 2
    step = FFT_LENGTH - overlap
    power = np.zeros(FFT_LENGTH // 2 + 1)

    for start in range(0, len(x) + 1, step):
        frame = x[start:start + FFT_LENGTH]
        fft_values = np.abs(np.fft.rfft(frame, n=FFT_LENGTH))
        power = np.maximum(power, fft_values ** 2 / FFT_LENGTH)

    return power


def extract_spectral_features(x):
    power = welch_max_hold(x)

    features = [
        np.sqrt(np.mean(x ** 2)),
        skew(x),
        kurtosis(x),
        skew(power),
        kurtosis(power),
    ]

    power = np.log10(np.where(power == 0, 1e-10, power))
    features.extend(power[1:])

    return np.asarray(features, dtype=np.float32)


def extract_features_from_window(window):
    window = np.asarray(window, dtype=np.float32)

    normal_features = []
    low_frequency_features = []

    for i in range(window.shape[1]):
        x = window[:, i]
        x = x - np.mean(x)

        normal_features.extend(extract_spectral_features(x))

        x_low = decimate_low_frequency(x)
        x_low = x_low - np.mean(x_low)

        low_frequency_features.extend(extract_spectral_features(x_low))

    features = np.asarray(normal_features + low_frequency_features, dtype=np.float32)

    expected_features = 42 * window.shape[1]
    assert features.shape[0] == expected_features, f"Expected {expected_features} features, got {features.shape[0]}"

    return features


def extract_features(windows):
    return np.vstack([
        extract_features_from_window(window["data"])
        for window in windows
    ])