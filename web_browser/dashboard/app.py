from flask import Flask, render_template, jsonify
from firebase_admin import credentials, firestore
import firebase_admin
import plotly.graph_objects as go
import plotly.utils
import json
import time

# ----------------------------
# 1. Firebase setup
# ----------------------------

cred = credentials.Certificate("plant-81856-firebase-adminsdk-fbsvc-3ef0631cfd.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ----------------------------
# 2. Flask app
# ----------------------------

app = Flask(__name__)

def get_recent_anomalies(limit=50):
    docs = db.collection("plant_anomalies") \
        .order_by("timestamp") \
        .limit(limit) \
        .get()

    timestamps = []
    voltage_means = []
    scores = []
    anomalies = []

    for doc in docs:
        data = doc.to_dict()
        ts = data.get("timestamp")
        if ts is None:
            continue
        # Convert Firestore timestamp to something usable
        if hasattr(ts, "nano"):
           timestamps.append(ts.timestamp())
        else:
            timestamps.append(time.time())

        voltage_means.append(data["voltage_mean"])
        scores.append(data["anomaly_score"])
        anomalies.append(data["is_anomaly"])

    return timestamps, voltage_means, scores, anomalies

@app.route("/")
def index():
    timestamps, voltage_means, scores, anomalies = get_recent_anomalies()
    return render_template("index.html")

@app.route("/data")
def data():
    timestamps, voltage_means, scores, anomalies = get_recent_anomalies()
    return jsonify({
        "timestamps": timestamps,
        "voltage_means": voltage_means,
        "scores": scores,
        "anomalies": anomalies,
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)