# PlantCasso

Turning a plant's bioelectric signals into expressive robot-arm motion.
Built at the **Berkeley AI Hackathon 2026**.

An electrode on a plant is sampled by an ESP32-S3 ADC. The signal is filtered,
reduced to 9 cheap time-domain features per window, projected to 3D with PCA and
clustered with K-means. The resulting embedding (plus two raw features) drives a
5-DOF servo arm so the plant's electrical "mood" becomes visible movement.

```
plant electrode → ESP32 ADC → lowpass filter → 9 features → StandardScaler
              → PCA (3D) → K-means cluster → servo angles (smoothed) → arm
```

## Repository layout

| Path | What it is |
|------|------------|
| `pipeline/` | Offline Python ML pipeline (filter → features → PCA/K-means → C header) |
| `pipeline/out/` | Generated artifacts (figures, model, CSVs). **Git-ignored** — regenerate by running the pipeline |
| `plant_inference/` | ESP32-S3 firmware that runs real-time inference + servo control |
| `plant_inference/model_params.h` | Auto-generated C header (scaler/PCA/K-means/filter) — produced by `pipeline/02_train.py` |
| `POC_electrode_reader/` | Minimal Arduino sketch that streams raw ADC voltage over serial |
| `data/` | Recorded voltage datasets (`data_5hz.csv`, `data_100hz.csv`, `data_unhealthy.csv`) |
| `data.csv` | Raw capture from a collection session |
| `data_collection.py` | Logs serial voltage samples from the ESP32 to a CSV |
| `Base.3mf` | 3D-printable arm base model |
| `ref_plantsignal.md`, `ref_plantsignal_converter.py` | Design notes / reference end-to-end script |
| `AGENTS.md` | Embedded C++ coding guidelines for the firmware |

## Quick start

### 1. Python environment

This project uses [`uv`](https://docs.astral.sh/uv/) (see `pyproject.toml` /
`.python-version`):

```bash
uv sync
```

Or with plain pip:

```bash
pip install numpy pandas scipy scikit-learn matplotlib pyserial
```

### 2. Run the pipeline (in order)

```bash
python pipeline/01_filter_extract.py --csv data/data_100hz.csv  # → pipeline/out/features.csv, filter_sos.json
python pipeline/02_train.py                                     # → model.pkl, plant_inference/model_params.h
python pipeline/03_visualize.py                                 # → pipeline/out/fig1..3.png
```

`python main.py` prints this sequence as a reminder.

### 3. Flash the firmware

1. Open `plant_inference/plant_inference.ino` in the Arduino IDE (Arduino-ESP32
   core ≥ 2.0) with `model_params.h` alongside it.
2. Install the **Adafruit PWM Servo Driver Library** (pulls in **Adafruit BusIO**).
   The 5 servos are driven through a **PCA9685** over I2C.
3. Wire the PCA9685: `SDA/SCL` → `PIN_I2C_SDA`/`PIN_I2C_SCL`, `V+` → a dedicated
   5–6 V servo supply (not the 3.3 V rail), `GND` common with the ESP32.
4. Set `PIN_PLANT`, the I2C pins, `SERVO_CH`, and the `SERVO_US_MIN/MAX` pulse
   range to match your wiring/servos, then upload.

To just capture data, flash `POC_electrode_reader/` instead and run
`python data_collection.py --port <your-port>`.

## How it works

**9 features per 1 s window** (`pipeline/01_filter_extract.py`): `mean`, `std`,
`ptp`, `slope`, `zcr`, `spike_count`, `hjorth_mobility`, `hjorth_complexity`,
`rms_first_diff`. They are intentionally FFT-free so the exact same math runs in
Python (training) and C++ on the ESP32 (inference) — keep
`extract_window()` and `extract_features()` in sync.

**Servo mapping** (`pipeline/02_train.py` → firmware): joints 0–2 follow the 3
PCA axes (slow, smooth "posture"), while joints 3–4 follow `spike_count` and
`hjorth_complexity` directly (snappy, expressive transients).

> **Note on windowing:** the training window hop is set in
> `pipeline/01_filter_extract.py` (`HOP = 50`), while the firmware's
> `HOP_SIZE` is the on-device inference cadence and is configured independently
> in the generated header.

See `ref_plantsignal.md` for the design rationale behind the feature choices,
window size, and PCA-vs-autoencoder decision.

## Firmware conventions

C++ for the ESP32 follows the guidelines in [`AGENTS.md`](AGENTS.md) — C++23,
`float`-only math (no hardware `double`), no heap allocation on hot paths, and
fixed compile-time buffer sizes.
