"""
data_collection.py
==================
Stream comma-terminated voltage samples from the ESP32 (running
POC_electrode_reader/) over serial and append them to a CSV.

The microcontroller prints one value per line (e.g. ``1.7092,``); send the
line ``stop`` to end collection.

Run:
    python data_collection.py                       # defaults: COM3 @ 115200 → data.csv
    python data_collection.py --port /dev/ttyUSB0 --out data/session.csv
"""

import argparse
import csv

import serial


def main() -> None:
    parser = argparse.ArgumentParser(description="Log serial voltage samples to CSV")
    parser.add_argument("--port", default="COM3", help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--out", default="data.csv", help="Output CSV path (appended to)")
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud)
    try:
        with open(args.out, "a", newline="") as f:
            writer = csv.writer(f)
            while True:
                line = ser.readline().decode().strip()
                if line.lower() == "stop":
                    break
                row = line.split(",")
                writer.writerow(row)
                print(f"Data collected: {row}")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
