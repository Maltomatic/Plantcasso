import firebase_admin
from firebase_admin import credentials, firestore

# Connect using the service account key
cred = credentials.Certificate("plant-81856-firebase-adminsdk-fbsvc-3ef0631cfd.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# Write a test document
doc = {
    "message": "Hello from Python",
    "timestamp": firestore.SERVER_TIMESTAMP,
}
db.collection("test_collection").add(doc)

print("Test document written to Firestore!")