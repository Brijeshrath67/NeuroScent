"""
Train simulated electronic nose model.
Sniffs each smell 200 times with simulated sensor noise and ambient humidity,
then trains a Random Forest classifier to identify the smell from the 8 sensor channels.
"""

import json
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sensor import GasSensorArray

def train_nose(samples_path="samples.json", model_output_path="nose_model.joblib", n_sniffs=200):
    if not os.path.exists(samples_path):
        # Fallback if called from parent folder
        samples_path = os.path.join(os.path.dirname(__file__), samples_path)

    with open(samples_path, "r") as f:
        samples = json.load(f)

    sensor = GasSensorArray()
    X = []
    y = []
    smell_names = list(samples.keys())

    print(f"Generating synthetic sniffing dataset ({n_sniffs} sniffs per smell)...")
    for smell_name, molecules in samples.items():
        for _ in range(n_sniffs):
            # Vary humidity (30% to 70%) and noise
            humidity = float(np.random.uniform(0.30, 0.70))
            reading = sensor.sample_response(molecules, humidity=humidity, noise_level=0.04)
            X.append(reading)
            y.append(smell_name)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training Random Forest on {len(X_train)} sniffs...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nNose Accuracy on unseen sniffs: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    model_bundle = {
        "model": clf,
        "classes": clf.classes_.tolist(),
        "sensor_names": sensor.sensor_names,
        "accuracy": float(accuracy)
    }

    joblib.dump(model_bundle, model_output_path)
    print(f"Saved trained electronic nose model to {model_output_path}")
    return model_bundle

def load_nose(model_path="nose_model.joblib"):
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), model_path)
    return joblib.load(model_path)

def predict_nose(sensor_readings, model_bundle=None):
    if model_bundle is None:
        model_bundle = load_nose()
    clf = model_bundle["model"]
    classes = model_bundle["classes"]

    readings = np.asarray(sensor_readings).reshape(1, -1)
    probs = clf.predict_proba(readings)[0]
    best_idx = np.argmax(probs)
    best_class = classes[best_idx]
    confidence = float(probs[best_idx])

    prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
    return best_class, confidence, prob_dict

if __name__ == "__main__":
    train_nose()

