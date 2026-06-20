"""
01_filter_extract.py
====================
Load plant signal CSV → 4th-order Butterworth lowpass (30 Hz) →
extract 9 FFT-free features in 1 s / 50 % overlap sliding windows →
save  pipeline/out/features.csv  +  pipeline/out/filter_sos.json

Run:
    python pipeline/01_filter_extract.py
    python pipeline/01_filter_extract.py --csv data/data_100hz.csv

The 9 features are chosen to be:
  • computationally cheap (O(N) passes, no FFT)
  • implementable identically in Python and C++ on the ESP32-S3
  • collectively covering DC level, variability, spike transients,
    oscillation frequency, and waveform shape complexity

Index  Name               Captures
─────  ─────────────────  ────────────────────────────────────────────
  0    mean               DC / baseline voltage level
  1    std                overall amplitude variability
  2    ptp                peak-to-peak range (spike amplitude)
  3    slope              linear drift within the window
  4    zcr                zero-crossing rate (oscillation speed proxy)
  5    spike_count        rate of peaks above mean + 2 σ
  6    hjorth_mobility    waveform 'speed'  (√(var(dx)/var(x)))
  7    hjorth_complexity  waveform irregularity (mobility(dx)/mobility(x))
  8    rms_first_diff     √(mean(Δx²))  — high-frequency content proxy
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

# ── Constants ─────────────────────────────────────────────────────────────────
FS         = 100      # Hz
WINDOW     = 100      # samples  (1 s at 100 Hz)
HOP        = 50       # 50 % overlap → step of 0.5 s
LOWPASS_HZ = 30.0     # Hz — plant signal content lives well below this
FILT_ORDER = 4        # Butterworth order

# ── Spike detection knobs ────────────────────────────────────────────────────
# These three constants are mirrored in plant_inference.ino — keep in sync.
SPIKE_SIGMA      = 1.5  # peak must exceed mean + SPIKE_SIGMA * std
SPIKE_SHARP      = 0.5  # peak must drop by SPIKE_SHARP * std within HALF_WIDTH
                        # samples on each side  (rejects broad fluctuations)
SPIKE_HALF_WIDTH = 3    # samples — 30 ms at 100 Hz; raise to allow slower spikes

ROOT    = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pipeline" / "out"

FEATURE_NAMES = [
    "mean", "std", "ptp", "slope",
    "zcr", "spike_count",
    "hjorth_mobility", "hjorth_complexity", "rms_first_diff",
]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_signal(csv_path: str) -> np.ndarray:
    """Load a single-column voltage CSV (trailing comma tolerated)."""
    df = pd.read_csv(csv_path, header=None, usecols=[0])
    return df[0].to_numpy(dtype=np.float32)


# ── Filter ────────────────────────────────────────────────────────────────────
def design_filter() -> np.ndarray:
    """Return 4th-order Butterworth lowpass as second-order sections (float64)."""
    return butter(FILT_ORDER, LOWPASS_HZ, fs=FS, btype="low", output="sos")


def apply_filter(signal: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Zero-phase forward-backward filter — offline training only."""
    return sosfiltfilt(sos, signal).astype(np.float32)


# ── Feature extraction — single window ───────────────────────────────────────
def extract_window(w: np.ndarray) -> np.ndarray:
    """
    Extract 9 features from a 1-D window w of length WINDOW.

    Implementation note: every operation here has a direct C++ counterpart
    in plant_inference.ino::extract_features(). Keep them in sync.
    """
    n    = len(w)
    mean = float(np.mean(w))
    std  = float(np.std(w, ddof=0))             # population std (ddof=0)

    # slope — least-squares fit to indices 0 … n-1
    x_idx = np.arange(n, dtype=np.float64)
    slope = float(np.polyfit(x_idx, w.astype(np.float64), 1)[0])

    ptp = float(w.max() - w.min())

    # zero-crossing rate on the demeaned signal
    dm  = w - mean
    zcr = float(np.sum(np.diff(np.sign(dm)) != 0)) / n

    # spike count: local maxima that are (a) above SPIKE_SIGMA*std and
    # (b) drop by at least SPIKE_SHARP*std within HALF_WIDTH samples on each
    # side — rejects slow fluctuations that merely crest above the threshold.
    thr = mean + SPIKE_SIGMA * std
    hw  = SPIKE_HALF_WIDTH
    spikes = 0
    for i in range(hw, n - hw):
        if (w[i] > w[i-1] and w[i] > w[i+1] and w[i] > thr
                and w[i] - w[i - hw] > SPIKE_SHARP * std
                and w[i] - w[i + hw] > SPIKE_SHARP * std):
            spikes += 1
    spike_count = float(spikes) / n

    # Hjorth parameters — all use population variance (ddof=0) to match C++
    dx   = np.diff(w.astype(np.float64))
    dx2  = np.diff(dx)
    var_x   = np.var(w,   ddof=0) + 1e-10
    var_dx  = np.var(dx,  ddof=0) + 1e-10
    var_dx2 = np.var(dx2, ddof=0) + 1e-10
    mobility   = float(np.sqrt(var_dx  / var_x))
    complexity = float(np.sqrt(var_dx2 / var_dx) / mobility)

    rms_diff = float(np.sqrt(np.mean(dx ** 2)))

    return np.array(
        [mean, std, ptp, slope, zcr, spike_count, mobility, complexity, rms_diff],
        dtype=np.float32,
    )


# ── Sliding-window batch extraction ──────────────────────────────────────────
def extract_all(signal: np.ndarray) -> pd.DataFrame:
    rows = [
        extract_window(signal[s : s + WINDOW])
        for s in range(0, len(signal) - WINDOW + 1, HOP)
    ]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Filter and extract features from plant signal")
    parser.add_argument("--csv", default=str(ROOT / "data" / "data_100hz.csv"),
                        help="Path to voltage CSV file")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv} …")
    sig = load_signal(args.csv)
    print(f"  {len(sig):,} samples  ({len(sig) / FS:.1f} s @ {FS} Hz)")

    sos = design_filter()
    filtered = apply_filter(sig, sos)
    print(f"  Applied order-{FILT_ORDER} Butterworth lowpass at {LOWPASS_HZ} Hz")

    # Export SOS without the a0=1 column → each row: [b0, b1, b2, a1, a2]
    sos_export = np.hstack([sos[:, :3], sos[:, 4:6]])
    sos_meta   = {
        "sos":        sos_export.tolist(),
        "fs":         FS,
        "cutoff_hz":  LOWPASS_HZ,
        "order":      FILT_ORDER,
        "n_sections": int(sos_export.shape[0]),
    }
    sos_path = OUT_DIR / "filter_sos.json"
    json.dump(sos_meta, sos_path.open("w"), indent=2)
    print(f"  Filter SOS ({sos_export.shape[0]} sections) → {sos_path}")

    print("Extracting features …")
    df = extract_all(filtered)
    feat_path = OUT_DIR / "features.csv"
    df.to_csv(feat_path, index=False)
    print(f"  {len(df)} windows × {len(FEATURE_NAMES)} features → {feat_path}")
    print()
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
