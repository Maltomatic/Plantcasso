from flask import Flask, render_template
from flask_socketio import SocketIO
import serial
import json
import time
import random
import threading

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app)

# ESP32 serial port - adjust to your system
# On macOS: /dev/cu.usbserial-...
# On Windows: COM3, COM4, etc.
SERIAL_PORT = "/dev/cu.usbmodem101"
SERIAL_BAUDRATE = 115200

ser = None

def connect_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
        print(f"Connected to serial on {SERIAL_PORT}")
    except Exception as e:
        print(f"Serial connection failed: {e}")
        ser = None

def parse_esp32_message(line):
    """
    Parse ESP32 string:
    C:0  S0:166  S1:98  S2:10  S3:33  S4:75    mean:-0.000  spk:0.0200  chaos:1.382
      volts:0.1725
    """
    try:
        # Extract cluster from "C:0"
        cluster = 0
        if "C:" in line:
            cluster_str = line.split("C:")[1].split()[0]
            cluster = int(cluster_str)
        
        # Extract mean from "mean:-0.000"
        mean = 0.0
        if "mean:" in line:
            mean_str = line.split("mean:")[1].split()[0]
            mean = float(mean_str)
        
        # Extract std from "spk:0.0200"
        std = 0.0
        if "spk:" in line:
            std_str = line.split("spk:")[1].split()[0]
            std = float(std_str)
        
        # Extract hjorth from "chaos:1.382"
        hjorth = 0.0
        if "chaos:" in line:
            hjorth_str = line.split("chaos:")[1].split()[0]
            hjorth = float(hjorth_str)
        
        # Spike count: set to 0 or extract from S0-S4
        # For now, set to 0 since you don't have a direct spike count
        spike_count = 0
        
        return {
            "mean": mean,
            "std": std,
            "spike_count": spike_count,
            "hjorth": hjorth,
            "cluster": cluster
        }
    except Exception as e:
        print(f"Parse error: {e}, line: {line}")
        return None
    
# def serial_read_loop():
#     """
#     Background thread that reads serial and emits data to the browser.
#     """
#     global ser
#     connect_serial()
#     if ser is None:
#         return

#     while True:
#         try:
#             if ser.in_waiting > 0:
#                 line = ser.readline().decode("utf-8", errors="ignore").strip()
#                 if not line:
#                     continue
#                 data = parse_esp32_message(line)
#                 if data:
#                     data["timestamp"] = time.time()
#                     socketio.emit("new_data", data)
#         except Exception as e:
#             print(f"Serial read error: {e}")
#             time.sleep(1)


#test func while usb is being used / no live data

def serial_read_loop():
    """
    Simulate ESP32 data for testing.
    Replace this with real serial reading when ESP32 is connected.
    """
    while True:
        data = {
            "mean": 1.5 + random.uniform(-0.1, 0.1),
            "std": random.uniform(0.01, 0.03),
            "spike_count": random.randint(0, 3),
            "hjorth": random.uniform(0.7, 1.1),
            "cluster": random.randint(0, 2),
            "timestamp": time.time(),
        }
        socketio.emit("new_data", data)
        time.sleep(0.5)  # send every 0.5 seconds

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    # Start serial reading in a background thread
    t = threading.Thread(target=serial_read_loop, daemon=True)
    t.start()

    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
    