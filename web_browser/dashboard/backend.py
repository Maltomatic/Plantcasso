import numpy as np
import pandas as pd
from numpy import random
from scipy.signal import iirnotch, filtfilt, butter, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from firebase_admin import credentials, firestore
import firebase_admin
from time import sleep

# ----------------------------
# Configuration constants
# ----------------------------
FS = 100.0              # sampling frequency (Hz) - adjust to your real sensor
LOWPASS_HZ = 10.0       # lowpass cutoff
NOTCH_HZ = 50.0         # notch frequency (e.g. power line)
NOTCH_Q = 20

WINDOW_LEN = 50         # window size
STEP = 25               # step between windows

CSV_PATH = "plant_data.csv"
FIREBASE_KEY_PATH = "plant_service_key.json"

# ----------------------------
# 1. Firebase setup
# ----------------------------
cred = credentials.Certificate("plant-81856-firebase-adminsdk-fbsvc-3ef0631cfd.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def save_anomaly_result(window_start, feats, pc, anomaly_score, is_anomaly, cluster_label=None):
    doc = {
        "window_start": window_start,
        "mean": feats["mean"],
        "std": feats["std"],
        "slope": feats["slope"],
        "ptp": feats["ptp"],
        "zero_crossings": feats["zero_crossings"],
        "spike_count": feats["spike_count"],
        "hjorth_complexity": feats["hjorth_complexity"],
        "band_power_ratio": feats["band_power_ratio"],
        "spectral_entropy": feats["spectral_entropy"],
        "pc1": pc[0],
        "pc2": pc[1],
        "pc3": pc[2],
        "anomaly_score": anomaly_score,
        "is_anomaly": is_anomaly,
        "cluster_label": cluster_label if cluster_label is not None else -1,
        "timestamp": firestore.SERVER_TIMESTAMP,
    }
    db.collection("plant_anomalies").add(doc)

# ----------------------------
# 2. Data loading (use CSV or replace with serial)
# ----------------------------
def load_signal_from_csv(path=CSV_PATH, voltage_col="voltage"):
    df = pd.read_csv(path)
    if voltage_col not in df.columns:
        raise ValueError(f"Missing column: {voltage_col}. Found: {list(df.columns)}")
    return df[voltage_col].astype(float).to_numpy()

# ----------------------------
# 3. Filtering
# ----------------------------
def filter_signal(raw, fs=FS, lowpass_hz=LOWPASS_HZ, notch_hz=NOTCH_HZ, notch_q=NOTCH_Q):
    x = np.asarray(raw, dtype=float)
    if notch_hz is not None:
        b, a = iirnotch(notch_hz, notch_q, fs)
        x = filtfilt(b, a, x)
    b, a = butter(4, lowpass_hz / (fs / 2), btype="low")
    x = filtfilt(b, a, x)
    return x

# ----------------------------
# 4. Windowing
# ----------------------------
def window_signal(signal, window_len=WINDOW_LEN, step=STEP):
    return np.array([signal[i:i + window_len]
                     for i in range(0, len(signal) - window_len + 1, step)])

# ----------------------------
# 5. Feature extraction (9 features)
# ----------------------------
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

# ----------------------------
# 6. Fit scaler + PCA (once on historical data)
# ----------------------------
def fit_embedding(X):
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    pca = PCA(n_components=3, random_state=0).fit(X_scaled)
    emb = pca.transform(X_scaled)
    return scaler, pca, emb

# ----------------------------
# 7. K-means clustering on 3D embedding
# ----------------------------
def cluster(emb, k_range=range(2, 6)):
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(emb)
        scores[k] = silhouette_score(emb, labels)
    best_k = max(scores, key=scores.get)
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(emb)
    labels = kmeans.predict(emb)
    return labels, best_k, kmeans

# ----------------------------
# 8. Live processing step
# ----------------------------
def process_window(raw_window, scaler, pca, kmeans, threshold, fs=FS):
    filtered = filter_signal(raw_window, fs)
    feats = extract_features(filtered, fs)
    names = list(feats.keys())
    x = np.array([[feats[n] for n in names]])

    x_scaled = scaler.transform(x)
    pc = pca.transform(x_scaled)[0]

    # K-means distance to nearest centroid
    dists = kmeans.transform(pc.reshape(1, -1))
    anomaly_score = np.min(dists, axis=1)[0]
    is_anomaly = anomaly_score > threshold

    # Cluster label
    cluster_label = int(kmeans.predict(pc.reshape(1, -1))[0])

    return feats, pc, anomaly_score, is_anomaly, cluster_label

# ----------------------------
# 9. Main pipeline: train + live loop
# ----------------------------
if __name__ == "__main__":
    # Load data
    raw = load_signal_from_csv(CSV_PATH)
    filtered = filter_signal(raw)
    windows = window_signal(filtered)

    print(f"Raw samples: {len(raw)} -> {len(windows)} windows of {WINDOW_LEN} samples")

    # Build feature matrix
    X, feature_names = build_feature_matrix(windows)
    print("Features:", feature_names)

    # Fit scaler + PCA
    scaler, pca, emb = fit_embedding(X)
    print(f"PCA explained variance (3 comps): {pca.explained_variance_ratio_} "
          f"(total {pca.explained_variance_ratio_.sum():.2%})")

    # K-means clustering
    labels, best_k, kmeans = cluster(emb)
    print(f"Silhouette by k: {silhouette_score(emb, labels)}  ->  chosen k={best_k}")
    print(f"Cluster sizes: {np.bincount(labels)}")

    # Compute anomaly threshold (99th percentile of distances)
    dists_all = kmeans.transform(emb)
    distances = np.min(dists_all, axis=1)
    threshold = np.percentile(distances, 99)
    print(f"Anomaly threshold (99th percentile): {threshold:.4f}")

    # Live processing loop (simulate on last 10 windows)
    print("\n--- live processing demo on last 10 windows ---")
    for w in windows[-10:]:
        window_start = np.where(
            (np.arange(0, len(filtered) - WINDOW_LEN + 1, STEP) ==
             np.arange(0, len(filtered) - WINDOW_LEN + 1, STEP)[
                 np.searchsorted(np.arange(0, len(filtered) - WINDOW_LEN + 1, STEP),
                                 np.where(np.arange(0, len(filtered) - WINDOW_LEN + 1, STEP) <= np.flatnonzero(np.concatenate([[0], [1]]))[0] if len(np.flatnonzero(np.concatenate([[0], [1]])) > 0 else [0]))[0] if len(np.arange(0, len(filtered) - WINDOW_LEN + 1, STEP)) > 0 else 0))[0] if len(np.arange(0, len(filtered) - WINDOW_LEN + 1, STEP)) > 0 else 0))[0] if len(np.arange(0, len(filtered) - WINDOW_LEN + 1, STEP)) > 0 else 0

        # Simplified: just approximate window start as index * step
        idx = np.where(w == windows[-10:])[0][0] if len(w) > 0 else 0
        window_start = idx * STEP

        feats, pc, anomaly_score, is_anomaly, cluster_label = process_window(
            w, scaler, pca, kmeans, threshold, fs=FS)

        save_anomaly_result(
            window_start=int(window_start),
            feats=feats,
            pc=pc,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            cluster_label=cluster_label
        )

        print(f"window_start={window_start}, score={anomaly_score:.4f}, "
              f"is_anomaly={is_anomaly}, cluster={cluster_label}")

        sleep(0.1)  # slow down

    print("Done.")