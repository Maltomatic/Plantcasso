"""
Plant bioelectric signal -> 3D embedding -> (a) 5-DOF servo control, (b) K-means clustering
=============================================================================================

Assumes: ~30,000 samples @ 100 Hz (~5 min), range 0-3.3V, baseline ~1.4-1.6V.

Pipeline:
  raw signal
    -> filter (lowpass + optional mains notch)
    -> windowing (overlapping, 1s windows by default)
    -> 9-feature extraction per window
    -> standardize -> PCA to 3D   (fit once, reused for both downstream uses)
    -> (a) 3D point + 2 raw features -> 5 servo angles (smoothed)
    -> (b) K-means on the 3D embedding

Replace `load_signal()` with your real data loader, and `send_to_servos()` with
your actual hardware driver call (Adafruit ServoKit, pyserial to an Arduino/
PCA9685 board, RPi.GPIO PWM, etc.) -- everything else runs end to end as-is.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FS = 100                  # Hz
WINDOW_SEC = 1.0          # 1s windows -> more observations from a short recording,
                          # and low enough latency for live servo control
WINDOW_LEN = int(FS * WINDOW_SEC)
OVERLAP = 0.5
STEP = int(WINDOW_LEN * (1 - OVERLAP))

LOWPASS_HZ = 30           # plant signal content of interest is well under this
NOTCH_HZ = None           # set to 60 (US) or 50 (EU/UK/etc.) if you see mains hum


# ---------------------------------------------------------------------------
# 1. Data loading (placeholder)
# ---------------------------------------------------------------------------
def load_signal(n_samples=30_000, seed=0):
    """Synthetic stand-in matching the stated spec: baseline ~1.5V, occasional
    spikes, slow drift, noise, clipped to 0-3.3V."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / FS
    baseline = 1.5 + 0.08 * np.sin(2 * np.pi * t / 60)        # slow ~1 min drift
    spikes = np.zeros(n_samples)
    for _ in range(n_samples // 500):
        idx = rng.integers(0, n_samples - 30)
        spikes[idx:idx + 30] += rng.uniform(0.15, 0.5) * np.hanning(30)
    noise = rng.normal(0, 0.02, n_samples)
    raw = baseline + spikes + noise
    return np.clip(raw, 0.0, 3.3)


# ---------------------------------------------------------------------------
# 2. Filtering
# ---------------------------------------------------------------------------
def filter_signal(raw, fs=FS, lowpass_hz=LOWPASS_HZ, notch_hz=NOTCH_HZ, notch_q=20):
    x = np.asarray(raw, dtype=float)
    if notch_hz is not None:
        b, a = iirnotch(notch_hz, notch_q, fs)
        x = filtfilt(b, a, x)
    b, a = butter(4, lowpass_hz / (fs / 2), btype="low")
    x = filtfilt(b, a, x)
    return x


# ---------------------------------------------------------------------------
# 3. Windowing
# ---------------------------------------------------------------------------
def window_signal(signal, window_len=WINDOW_LEN, step=STEP):
    return np.array([signal[i:i + window_len]
                      for i in range(0, len(signal) - window_len + 1, step)])


# ---------------------------------------------------------------------------
# 4. Feature extraction -- 9 features
# ---------------------------------------------------------------------------
def hjorth_complexity(x):
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x, var_dx, var_ddx = np.var(x), np.var(dx), np.var(ddx)
    mobility = np.sqrt(var_dx / var_x) if var_x > 0 else 0.0
    mobility_dx = np.sqrt(var_ddx / var_dx) if var_dx > 0 else 0.0
    return mobility_dx / mobility if mobility > 0 else 0.0


def spectral_features(x, fs=FS, split_hz=5.0):
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    power = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    low = power[freqs < split_hz].sum()
    high = power[freqs >= split_hz].sum()
    ratio = high / (low + 1e-9)
    p = power / (power.sum() + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p))  # normalized 0-1
    return ratio, entropy


def extract_features(x, fs=FS):
    x = np.asarray(x, dtype=float)
    idx = np.arange(len(x))
    slope, _ = np.polyfit(idx, x, 1)
    detrended = x - np.polyval(np.polyfit(idx, x, 1), idx)
    zero_crossings = np.sum(np.diff(np.sign(detrended)) != 0)

    peaks, _ = find_peaks(np.abs(detrended), height=2 * x.std(), distance=fs * 0.05)
    spike_count = len(peaks)

    band_power_ratio, spectral_entropy = spectral_features(x, fs)

    return {
        "mean": x.mean(),
        "std": x.std(),
        "slope": slope,
        "ptp": x.max() - x.min(),
        "zero_crossings": zero_crossings,
        "spike_count": spike_count,
        "hjorth_complexity": hjorth_complexity(x),
        "band_power_ratio": band_power_ratio,
        "spectral_entropy": spectral_entropy,
    }


def build_feature_matrix(windows, fs=FS):
    rows = [extract_features(w, fs) for w in windows]
    names = list(rows[0].keys())
    X = np.array([[r[k] for k in names] for r in rows])
    return X, names


# ---------------------------------------------------------------------------
# 5. Fit scaler + PCA (do this once, on the full historical dataset)
# ---------------------------------------------------------------------------
def fit_embedding(X):
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    pca = PCA(n_components=3, random_state=0).fit(X_scaled)
    emb = pca.transform(X_scaled)
    return scaler, pca, emb


# ---------------------------------------------------------------------------
# 6a. K-means clustering on the 3D embedding
# ---------------------------------------------------------------------------
def cluster(embedding, k_range=range(2, 6)):
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(embedding)
        scores[k] = silhouette_score(embedding, labels)
    best_k = max(scores, key=scores.get)
    labels = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit_predict(embedding)
    return labels, best_k, scores


# ---------------------------------------------------------------------------
# 6b. 3D embedding (+2 raw features) -> 5 servo angles
# ---------------------------------------------------------------------------
def fit_servo_calibration(emb, X, feature_names, lo_pct=5, hi_pct=95):
    """Robust min/max (5th/95th percentile) per driving signal, computed once
    on the historical dataset, used to scale live values into servo range."""
    spike_idx = feature_names.index("spike_count")
    bpr_idx = feature_names.index("band_power_ratio")
    calib = {}
    for i, name in enumerate(["pc1", "pc2", "pc3"]):
        calib[name] = tuple(np.percentile(emb[:, i], [lo_pct, hi_pct]))
    calib["spike_count"] = tuple(np.percentile(X[:, spike_idx], [lo_pct, hi_pct]))
    calib["band_power_ratio"] = tuple(np.percentile(X[:, bpr_idx], [lo_pct, hi_pct]))
    return calib


def _scale(value, lo, hi, angle_min=0, angle_max=180):
    value = np.clip(value, lo, hi)
    if hi - lo < 1e-9:
        return (angle_min + angle_max) / 2
    return angle_min + (value - lo) / (hi - lo) * (angle_max - angle_min)


def embedding_to_servo_angles(pc, spike_count, band_power_ratio, calib):
    """pc = [pc1, pc2, pc3]. Joints 0-2 follow the smooth PCA embedding
    (overall 'posture' / slow environmental state). Joints 3-4 follow raw
    spike/chaos features directly, so transient plant activity reads as
    sharp, distinct arm motion rather than being smoothed into the average."""
    s0 = _scale(pc[0], *calib["pc1"])
    s1 = _scale(pc[1], *calib["pc2"])
    s2 = _scale(pc[2], *calib["pc3"])
    s3 = _scale(spike_count, *calib["spike_count"])          # wrist: spikes
    s4 = _scale(band_power_ratio, *calib["band_power_ratio"])  # gripper: chaos
    return [s0, s1, s2, s3, s4]


class ServoSmoother:
    """Exponential smoothing per joint. Slower alpha = smoother/slower
    motion (good for the PCA-driven 'posture' joints), faster alpha = snappier
    response (good for the spike/chaos-driven joints)."""

    def __init__(self, alphas=(0.15, 0.15, 0.15, 0.6, 0.5)):
        self.alphas = alphas
        self.state = None

    def update(self, angles):
        if self.state is None:
            self.state = list(angles)
        else:
            self.state = [a * v + (1 - a) * s
                          for a, v, s in zip(self.alphas, angles, self.state)]
        return list(self.state)


# ---------------------------------------------------------------------------
# 7. Hardware stub -- replace with your actual servo driver call
# ---------------------------------------------------------------------------
def send_to_servos(angles):
    # e.g. with Adafruit ServoKit:
    #   for i, angle in enumerate(angles): kit.servo[i].angle = angle
    # or write a serial command to an Arduino/PCA9685 board.
    print(f"servo angles -> {[round(a, 1) for a in angles]}")


# ---------------------------------------------------------------------------
# 8. Live processing step (what you'd call on each new buffered window)
# ---------------------------------------------------------------------------
def process_window(raw_window, scaler, pca, calib, smoother, fs=FS):
    filtered = filter_signal(raw_window, fs)
    feats = extract_features(filtered, fs)
    names = list(feats.keys())
    x = np.array([[feats[n] for n in names]])
    x_scaled = scaler.transform(x)
    pc = pca.transform(x_scaled)[0]
    raw_angles = embedding_to_servo_angles(
        pc, feats["spike_count"], feats["band_power_ratio"], calib)
    smoothed = smoother.update(raw_angles)
    return smoothed, pc, feats


# ---------------------------------------------------------------------------
# Run end to end
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    raw = load_signal()
    filtered = filter_signal(raw)
    windows = window_signal(filtered)
    print(f"Raw samples: {len(raw)} -> {len(windows)} windows of {WINDOW_LEN} samples")

    X, feature_names = build_feature_matrix(windows)
    print("Features:", feature_names)

    scaler, pca, emb = fit_embedding(X)
    print(f"PCA explained variance (3 comps): {pca.explained_variance_ratio_} "
          f"(total {pca.explained_variance_ratio_.sum():.2%})")

    labels, best_k, sil_scores = cluster(emb)
    print(f"Silhouette by k: {sil_scores}  ->  chosen k={best_k}")
    print(f"Cluster sizes: {np.bincount(labels)}")

    calib = fit_servo_calibration(emb, X, feature_names)
    smoother = ServoSmoother()

    print("\n--- live processing demo on last 5 windows ---")
    for w in windows[-5:]:
        angles, pc, feats = process_window(w, scaler, pca, calib, smoother)
        send_to_servos(angles)
