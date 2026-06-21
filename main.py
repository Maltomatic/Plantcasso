"""
PlantCasso entry point.

This is a thin convenience launcher. The real work lives in the numbered
pipeline scripts, which are designed to run in order:

    python pipeline/01_filter_extract.py   # raw CSV → filtered → features.csv
    python pipeline/02_train.py            # features → PCA + K-means → model_params.h
    python pipeline/03_visualize.py        # clustering figures in pipeline/out/

Firmware that consumes the generated model lives in plant_inference/.
"""

PIPELINE_STEPS = [
    ("pipeline/01_filter_extract.py", "Filter signal and extract 9 windowed features"),
    ("pipeline/02_train.py", "Scale → PCA(3D) → K-means, export model_params.h"),
    ("pipeline/03_visualize.py", "Render clustering / time-series / correlation figures"),
]


def main() -> None:
    print("PlantCasso — plant bioelectric signals → 5-DOF servo arm\n")
    print("Run the pipeline in order:")
    for script, desc in PIPELINE_STEPS:
        print(f"  python {script:<32s} {desc}")


if __name__ == "__main__":
    main()
