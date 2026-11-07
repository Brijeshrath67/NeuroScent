"""
Smell Model: Predicts odor descriptors from chemical molecules.
Turns each molecule into a Morgan fingerprint of chemical fragments,
then learns which fragments correspond with 138 olfactory descriptors.
"""

import os
import urllib.request
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

DATA_FILENAME = "curated_GS_LF_merged_4983.csv"
DATA_URL = "https://raw.githubusercontent.com/BioMachineLearning/neuroscent/main/neuroscent/data/curated_datasets/curated_GS_LF_merged_4983.csv"
MODEL_FILENAME = "smell_model.joblib"

def ensure_dataset():
    """Locate or download the 4,983 curated smell dataset."""
    search_paths = [
        DATA_FILENAME,
        os.path.join(os.path.dirname(__file__), DATA_FILENAME),
        os.path.join(os.path.dirname(__file__), "..", "neuroscent", "data", "curated_datasets", DATA_FILENAME),
        os.path.join("neuroscent", "data", "curated_datasets", DATA_FILENAME)
    ]
    for path in search_paths:
        if os.path.exists(path):
            return os.path.abspath(path)

    target_path = os.path.join(os.path.dirname(__file__), DATA_FILENAME)
    print(f"Downloading NeuroScent curated dataset to {target_path}...")
    urllib.request.urlretrieve(DATA_URL, target_path)
    return target_path

def train_smell_model(save_path=MODEL_FILENAME):
    data_path = ensure_dataset()
    print(f"Loading curated dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # 138 odor descriptor columns
    descriptor_cols = [c for c in df.columns if c not in ["nonStereoSMILES", "descriptors"]]
    print(f"Dataset has {len(df)} molecules and {len(descriptor_cols)} odor descriptors.")

    print("Generating Morgan fingerprints (radius=2, 1024 bits)...")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fps = []
    valid_indices = []

    for idx, smiles in enumerate(df["nonStereoSMILES"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fps.append(gen.GetFingerprintAsNumPy(mol))
            valid_indices.append(idx)

    X = np.array(fps, dtype=np.int8)
    Y = df.loc[valid_indices, descriptor_cols].values

    print(f"Valid molecules parsed: {len(X)}.")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print("Training multi-label smell model (Logistic Regression)...")
    base_clf = LogisticRegression(max_iter=250, C=0.5, solver="liblinear", random_state=42)
    model = MultiOutputClassifier(base_clf, n_jobs=1)
    model.fit(X_train, Y_train)

    print("Evaluating ROC-AUC on unseen molecules...")
    preds = np.zeros((len(X_test), len(descriptor_cols)), dtype=np.float32)
    for i, est in enumerate(model.estimators_):
        if len(est.classes_) > 1:
            preds[:, i] = est.predict_proba(X_test)[:, 1]

    aucs = []
    for col_idx in range(len(descriptor_cols)):
        col_true = Y_test[:, col_idx]
        if len(np.unique(col_true)) > 1:
            aucs.append(roc_auc_score(col_true, preds[:, col_idx]))

    mean_auc = float(np.mean(aucs))
    print(f"--> Smell model ROC-AUC on unseen test molecules: {mean_auc:.4f} (0.5=guessing, 1.0=perfect)")

    # Retrain on full data for maximum production accuracy
    print("Fitting model on full dataset for maximum accuracy...")
    full_model = MultiOutputClassifier(base_clf, n_jobs=1)
    full_model.fit(X, Y)

    save_target = os.path.join(os.path.dirname(__file__), save_path)
    bundle = {
        "model": full_model,
        "descriptors": descriptor_cols,
        "fp_size": 1024,
        "radius": 2,
        "test_roc_auc": mean_auc
    }
    joblib.dump(bundle, save_target)
    print(f"Saved trained smell model bundle to {save_target}")
    return bundle

def load_smell_model(path=MODEL_FILENAME):
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(path):
        print("Trained model not found. Training now...")
        return train_smell_model(path)
    return joblib.load(path)

def predict_molecule_descriptors(smiles, model_bundle=None, top_n=5):
    """Predict top smell words for a single molecule SMILES."""
    if model_bundle is None:
        model_bundle = load_smell_model()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=model_bundle.get("radius", 2),
        fpSize=model_bundle.get("fp_size", 1024)
    )
    fp = gen.GetFingerprintAsNumPy(mol).reshape(1, -1)
    model = model_bundle["model"]
    descriptors = model_bundle["descriptors"]

    probs = np.array([
        est.predict_proba(fp)[0, 1] if len(est.classes_) > 1 else 0.0
        for est in model.estimators_
    ])

    top_indices = np.argsort(probs)[::-1][:top_n]
    return [(descriptors[i], float(probs[i])) for i in top_indices]

def predict_mixture_descriptors(molecule_list, model_bundle=None, top_n=5):
    """
    Predict top smell words for a blend/mixture of molecules.
    molecule_list: list of dicts with 'smiles' and optional 'weight'
    """
    if model_bundle is None:
        model_bundle = load_smell_model()

    descriptors = model_bundle["descriptors"]
    total_probs = np.zeros(len(descriptors), dtype=np.float64)
    total_weight = 0.0

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=model_bundle.get("radius", 2),
        fpSize=model_bundle.get("fp_size", 1024)
    )
    model = model_bundle["model"]

    for item in molecule_list:
        smiles = item.get("smiles", "")
        weight = float(item.get("weight", 1.0))
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        fp = gen.GetFingerprintAsNumPy(mol).reshape(1, -1)
        probs = np.array([
            est.predict_proba(fp)[0, 1] if len(est.classes_) > 1 else 0.0
            for est in model.estimators_
        ])
        total_probs += probs * weight
        total_weight += weight

    if total_weight > 0:
        total_probs /= total_weight

    top_indices = np.argsort(total_probs)[::-1][:top_n]
    return [(descriptors[i], float(total_probs[i])) for i in top_indices]

if __name__ == "__main__":
    train_smell_model()

