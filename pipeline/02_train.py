"""
02_train.py
===========
Load features.csv → StandardScaler → PCA(3D) → K-means (k chosen by
silhouette score) → save model.pkl and generate plant_inference/model_params.h

Run:
    python pipeline/02_train.py

Outputs
-------
pipeline/out/model.pkl              — Python model (scaler + PCA + KMeans + calib)
pipeline/out/pca_coords.csv         — 3D coords + cluster label per window
pipeline/out/features_clustered.csv — features.csv with cluster column appended
plant_inference/model_params.h      — C header for ESP32-S3 real-time inference

Servo calibration
-----------------
Joints 0-2 are driven by PC1/PC2/PC3 (slow, smooth 'posture' — reflects overall
plant state). Joints 3-4 are driven directly by spike_count and hjorth_complexity
(fast, expressive — transients cause sharp visible motion rather than blending
into an averaged embedding).

Calibration ranges are computed as [5th, 95th] percentiles on training data so
the servos sweep most of their range in normal operation without saturating.
"""

import json
import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR     = ROOT / "pipeline" / "out"
HEADER_PATH = ROOT / "plant_inference" / "model_params.h"

FEATURE_NAMES = [
    "mean", "std", "ptp", "slope",
    "zcr", "spike_count",
    "hjorth_mobility", "hjorth_complexity", "rms_first_diff",
]

N_PCA   = 3
K_RANGE = range(3, 6)
N_INIT  = 30          # KMeans random restarts per k

# ── Per-feature PCA weights (applied AFTER StandardScaler, BEFORE PCA) ────────
# StandardScaler forces every feature to unit variance, so raw-space scaling
# has no effect.  Multiplying here directly controls each feature's contribution
# to the PCA eigenvectors.  1.0 = normal weight.  Exported to model_params.h.
# Index order matches FEATURE_NAMES above.
FEATURE_WEIGHTS = np.array(
    [1.0, 1.2, 1.2, 1.0,   # mean, std, ptp, slope
     1.2, 1.0,             # zcr, spike_count
     1.2, 1.5, 1.0],       # hjorth_mobility, hjorth_complexity, rms_first_diff
    dtype=np.float32,
)


# ── K selection ───────────────────────────────────────────────────────────────
def pick_k(X_pca: np.ndarray) -> int:
    best_k, best_score = 2, -1.0
    for k in K_RANGE:
        km     = KMeans(n_clusters=k, random_state=42, n_init=N_INIT)
        labels = km.fit_predict(X_pca)
        score  = silhouette_score(X_pca, labels)
        marker = "  ←" if score > best_score else ""
        print(f"    k={k}  silhouette={score:.4f}{marker}")
        if score > best_score:
            best_score, best_k = score, k
    print(f"  → Best k={best_k}  (silhouette={best_score:.4f})")
    return best_k


# ── Servo calibration ─────────────────────────────────────────────────────────
def servo_calibration(
    X_pca: np.ndarray, X_raw: np.ndarray,
    lo_pct: int = 5,  hi_pct: int = 95,
    # Tighter bounds for chaos joints → small real variations sweep full servo range
    chaos_lo_pct: int = 20, chaos_hi_pct: int = 80,
) -> dict:
    """Robust percentile ranges for each of the 5 servo drivers.
    Posture joints (PC1-3) use wide [lo_pct, hi_pct].
    Chaos joints (spike_count, hjorth_complexity) use tighter [chaos_lo_pct, chaos_hi_pct]
    so that typical transient activity fills the full servo arc."""
    calib: dict = {}
    for i, key in enumerate(["pc1", "pc2", "pc3"]):
        calib[key] = (
            float(np.percentile(X_pca[:, i], lo_pct)),
            float(np.percentile(X_pca[:, i], hi_pct)),
        )
    spike_idx = FEATURE_NAMES.index("spike_count")
    chaos_idx = FEATURE_NAMES.index("hjorth_complexity")
    calib["spike_count"] = (
        float(np.percentile(X_raw[:, spike_idx], chaos_lo_pct)),
        float(np.percentile(X_raw[:, spike_idx], chaos_hi_pct)),
    )
    calib["hjorth_complexity"] = (
        float(np.percentile(X_raw[:, chaos_idx], chaos_lo_pct)),
        float(np.percentile(X_raw[:, chaos_idx], chaos_hi_pct)),
    )
    return calib


# ── C header helpers ──────────────────────────────────────────────────────────
def _f(v: float) -> str:
    return f"{v:.8f}f"


def arr1d_to_c(name: str, arr: np.ndarray) -> str:
    vals = ", ".join(_f(float(v)) for v in arr.flat)
    return f"static const float {name}[{arr.size}] = {{{vals}}};"


def arr2d_to_c(name: str, arr: np.ndarray) -> str:
    r, c = arr.shape
    rows = [
        "    {" + ", ".join(_f(float(v)) for v in arr[i]) + "}"
        for i in range(r)
    ]
    inner = ",\n".join(rows)
    return f"static const float {name}[{r}][{c}] = {{\n{inner}\n}};"


# ── C header export ───────────────────────────────────────────────────────────
def export_header(
    scaler: StandardScaler,
    pca: PCA,
    km: KMeans,
    sos_export: np.ndarray,
    calib: dict,
    weights: np.ndarray,
    path: pathlib.Path,
) -> None:
    K   = km.n_clusters
    n_f = len(FEATURE_NAMES)
    n_s = int(sos_export.shape[0])

    # Servo calibration arrays — order: PC1, PC2, PC3, spike_count, hjorth_complexity
    servo_keys = ["pc1", "pc2", "pc3", "spike_count", "hjorth_complexity"]
    servo_lo   = np.array([calib[k][0] for k in servo_keys], dtype=np.float32)
    servo_hi   = np.array([calib[k][1] for k in servo_keys], dtype=np.float32)

    lines = [
        "// Auto-generated by pipeline/02_train.py — do not edit manually.",
        "// Re-run 01_filter_extract.py then 02_train.py to regenerate.",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"static const int N_FEATURES  = {n_f};",
        f"static const int N_PCA       = {N_PCA};",
        f"static const int N_CLUSTERS  = {K};",
        f"static const int WINDOW_SIZE = 100;   // samples (1 s @ 100 Hz)",
        f"static const int HOP_SIZE    = 30;    // 70 % overlap",
        f"static const int N_SOS       = {n_s}; // biquad sections",
        "",
        "// ── StandardScaler ──────────────────────────────────────────────────",
        "// scaled[i] = (raw[i] - SCALER_MEAN[i]) / SCALER_STD[i]",
        arr1d_to_c("SCALER_MEAN", scaler.mean_.astype(np.float32)),
        arr1d_to_c("SCALER_STD",  scaler.scale_.astype(np.float32)),
        "",
        "// ── PCA projection [N_PCA][N_FEATURES] ─────────────────────────────",
        "// pca[p] = dot(PCA_COMPONENTS[p], scaled_features)",
        arr2d_to_c("PCA_COMPONENTS", pca.components_.astype(np.float32)),
        "",
        "// ── K-means centroids [N_CLUSTERS][N_PCA] ──────────────────────────",
        arr2d_to_c("KMEANS_CENTROIDS", km.cluster_centers_.astype(np.float32)),
        "",
        "// ── Biquad IIR filter [N_SOS][5]: {b0, b1, b2, a1, a2} ─────────────",
        "// Direct Form II:  w = x - a1*w1 - a2*w2;  y = b0*w + b1*w1 + b2*w2",
        "// (a0 = 1 is normalised out by scipy)",
        arr2d_to_c("BIQUAD_SOS", sos_export.astype(np.float32)),
        "",
        "// ── Servo calibration [5]: {PC1, PC2, PC3, spike_count, hjorth_complexity}",
        "// map_to_servo(v, SERVO_LO[j], SERVO_HI[j]) → angle in [SERVO_MIN, SERVO_MAX]",
        arr1d_to_c("SERVO_LO", servo_lo),
        arr1d_to_c("SERVO_HI", servo_hi),
        "",
        "// ── Per-feature PCA weights [N_FEATURES] ───────────────────────────────",
        "// Applied AFTER StandardScaler, BEFORE PCA dot-product.",
        "// weighted[i] = scaled[i] * FEATURE_WEIGHTS[i]",
        arr1d_to_c("FEATURE_WEIGHTS", weights.astype(np.float32)),
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  C header → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    feat_path = OUT_DIR / "features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"{feat_path} not found — run 01_filter_extract.py first."
        )
    df = pd.read_csv(feat_path)
    X  = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    print(f"Loaded {X.shape[0]} windows × {X.shape[1]} features")

    # ── Baseline-subtract the mean feature (index 0) ─────────────────────
    # Use a causal N-window rolling mean as the baseline — identical to the
    # ring-buffer approach in plant_inference.ino (same N_MEAN_WINDOWS).
    # Feature 0 becomes *deviation from recent local mean* rather than
    # absolute voltage:  captures genuine state shifts while rejecting drift.
    N_MEAN_WINDOWS = 10
    baseline = (
        pd.Series(X[:, 0])
        .rolling(window=N_MEAN_WINDOWS, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float32)
    )
    X = X.copy()
    X[:, 0] -= baseline
    print(f"  mean \u2192 deviation from {N_MEAN_WINDOWS}-window rolling mean  "
          f"(range: {X[:, 0].min():.4f} … {X[:, 0].max():.4f} V)")

    # ── Scale ─────────────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Apply per-feature weights (post-scale, pre-PCA) ───────────────────
    # This is the correct place: StandardScaler already removed scale differences;
    # multiplying now directly inflates a feature's contribution to eigenvectors.
    X_weighted = X_scaled * FEATURE_WEIGHTS
    print("  Feature weights: " +
          ", ".join(f"{n}={w:.1f}" for n, w in zip(FEATURE_NAMES, FEATURE_WEIGHTS)
                   if w != 1.0))

    # ── PCA ───────────────────────────────────────────────────────────────
    pca   = PCA(n_components=N_PCA, random_state=42)
    X_pca = pca.fit_transform(X_weighted).astype(np.float32)
    ev    = pca.explained_variance_ratio_
    print(
        f"PCA explained variance: PC1={ev[0]:.2%}  PC2={ev[1]:.2%}  "
        f"PC3={ev[2]:.2%}  total={ev.sum():.2%}"
    )

    # ── K-means ───────────────────────────────────────────────────────────
    print("Silhouette sweep …")
    k      = pick_k(X_pca)
    km     = KMeans(n_clusters=k, random_state=42, n_init=N_INIT)
    labels = km.fit_predict(X_pca)
    print(f"Cluster sizes: {np.bincount(labels)}")

    # ── Servo calibration ─────────────────────────────────────────────────
    calib = servo_calibration(X_pca, X)
    print("Servo calibration ranges:")
    for key, (lo, hi) in calib.items():
        print(f"  {key:22s}  [{lo:+.4f}, {hi:+.4f}]")

    # ── Save Python model ─────────────────────────────────────────────────
    model = {
        "scaler": scaler, "pca": pca, "kmeans": km,
        "labels": labels, "calib": calib,
        "feature_names": FEATURE_NAMES,
    }
    model_path = OUT_DIR / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model → {model_path}")

    # ── Load SOS and export C header ──────────────────────────────────────
    sos_meta   = json.load((OUT_DIR / "filter_sos.json").open())
    sos_export = np.array(sos_meta["sos"], dtype=np.float64)
    export_header(scaler, pca, km, sos_export, calib, FEATURE_WEIGHTS, HEADER_PATH)

    # ── Save auxiliary CSVs for visualiser ────────────────────────────────
    pca_df = pd.DataFrame(X_pca, columns=["pc1", "pc2", "pc3"])
    pca_df["cluster"] = labels
    pca_df.to_csv(OUT_DIR / "pca_coords.csv", index=False)

    df_out = df.copy()
    df_out["cluster"] = labels
    df_out.to_csv(OUT_DIR / "features_clustered.csv", index=False)
    print(f"  Auxiliary CSVs → {OUT_DIR}")


if __name__ == "__main__":
    main()
