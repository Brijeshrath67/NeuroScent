"""
Automated Test Suite for Smell AI:
Verifies the sensor simulator, nose model, and molecular smell model end-to-end.
"""

import os
import json
import numpy as np

def test_pipeline():
    print("=" * 60)
    print("RUNNING SMELL-AI VERIFICATION TESTS")
    print("=" * 60)

    # 1. Test samples.json
    print("\n[1/5] Checking samples.json...")
    samples_path = os.path.join(os.path.dirname(__file__), "samples.json")
    assert os.path.exists(samples_path), "samples.json not found!"
    with open(samples_path, "r") as f:
        samples = json.load(f)
    expected_smells = {"coffee", "vanilla candle", "mint", "lemon", "banana", "wood smoke"}
    assert set(samples.keys()) == expected_smells, f"Expected smells: {expected_smells}, got: {set(samples.keys())}"
    print(f"✓ Found all {len(samples)} sample smells with chemical formulas.")

    # 2. Test sensor simulation
    print("\n[2/5] Testing 8-channel GasSensorArray...")
    from sensor import GasSensorArray, sniff_sample
    sensor = GasSensorArray()
    test_reading = sensor.sample_response(samples["mint"], humidity=0.5, noise_level=0.03)
    assert len(test_reading) == 8, f"Expected 8 sensor channels, got {len(test_reading)}"
    assert np.all(test_reading > 0), "Sensor readings should all be positive"
    print(f"✓ 8 sensor channels successfully simulated: {np.round(test_reading, 3)}")

    # 3. Test Electronic Nose Model
    print("\n[3/5] Testing Electronic Nose Model...")
    from train_nose import train_nose, load_nose, predict_nose
    nose_model_path = os.path.join(os.path.dirname(__file__), "nose_model.joblib")
    if not os.path.exists(nose_model_path):
        print("Training nose model...")
        nose_bundle = train_nose(samples_path=samples_path, model_output_path=nose_model_path)
    else:
        nose_bundle = load_nose(nose_model_path)

    # Sniff test on coffee and mint
    coffee_reading = sensor.sample_response(samples["coffee"], humidity=0.45)
    guess, conf, _ = predict_nose(coffee_reading, nose_bundle)
    print(f"✓ Sniffed Coffee -> Nose classified as: '{guess}' (Confidence: {conf*100:.1f}%)")
    assert guess == "coffee", f"Expected coffee, got {guess}"
    assert conf >= 0.80, f"Expected high confidence, got {conf}"

    mint_reading = sensor.sample_response(samples["mint"], humidity=0.55)
    guess_m, conf_m, _ = predict_nose(mint_reading, nose_bundle)
    print(f"✓ Sniffed Mint   -> Nose classified as: '{guess_m}' (Confidence: {conf_m*100:.1f}%)")
    assert guess_m == "mint", f"Expected mint, got {guess_m}"

    # 4. Test Smell Model
    print("\n[4/6] Testing Baseline Molecular Smell Model...")
    from smell_model import load_smell_model, predict_molecule_descriptors, predict_mixture_descriptors
    smell_model_path = os.path.join(os.path.dirname(__file__), "smell_model.joblib")
    smell_bundle = load_smell_model(smell_model_path)

    # Test single molecule (Menthol: CC1CCC(C(C1)O)C(C)C)
    menthol_smiles = "CC1CCC(C(C1)O)C(C)C"
    m_descriptors = predict_molecule_descriptors(menthol_smiles, smell_bundle, top_n=5)
    m_words = [d[0] for d in m_descriptors]
    print(f"✓ Menthol predicted descriptors: {m_descriptors}")
    assert any(w in m_words for w in ["mint", "cooling", "herbal"]), "Menthol should predict minty/cooling descriptors"

    # Test coffee mixture descriptors
    coffee_descriptors = predict_mixture_descriptors(samples["coffee"], smell_bundle, top_n=6)
    c_words = [d[0] for d in coffee_descriptors]
    print(f"✓ Coffee mixture predicted descriptors: {coffee_descriptors}")
    assert any(w in c_words for w in ["roasted", "coffee", "phenolic", "smoky", "caramellic", "sweet"]), "Coffee descriptors missing"

    # 5. Test Deep Molecular GNN (MPNN)
    print("\n[5/6] Testing Deep Molecular Graph Neural Network (PyTorch Geometric GNN)...")
    from gnn_model import load_or_train_gnn, predict_gnn_smell
    gnn_model, gnn_descs, gnn_auc = load_or_train_gnn(device="cpu")
    print(f"✓ GNN Checkpoint Loaded. Benchmark Multi-Task Test ROC-AUC: {gnn_auc:.4f}")
    
    # Test Vanillin with GNN
    vanillin_smiles = "COc1cc(C=O)ccc1O"
    gnn_preds_v, pom_emb_v = predict_gnn_smell(vanillin_smiles, gnn_model, gnn_descs, top_n=5)
    v_words = [d[0] for d in gnn_preds_v]
    print(f"✓ Vanillin GNN Predictions: {gnn_preds_v}")
    print(f"✓ Extracted 256-D Principal Odor Map (POM) Latent Vector: Shape {pom_emb_v.shape}")
    assert any(w in v_words for w in ["sweet", "vanilla", "phenolic"]), "Vanillin should predict sweet/vanilla descriptors"
    assert pom_emb_v.shape == (256,), "POM embedding should be 256-dimensional"

    # 6. Full End-to-End Simulation
    print("\n[6/6] Full End-to-End Pipeline Execution:")
    print("-" * 50)
    for smell_name in samples.keys():
        readings = sensor.sample_response(samples[smell_name], humidity=0.5, noise_level=0.03)
        pred_smell, confidence, _ = predict_nose(readings, nose_bundle)
        descriptors = predict_mixture_descriptors(samples[pred_smell], smell_bundle, top_n=4)
        desc_str = ", ".join([f"{w} ({prob*100:.0f}%)" for w, prob in descriptors])
        print(f"Sample: {smell_name.title():<15} -> Nose: {pred_smell.title()} ({confidence*100:.1f}%) | Words: {desc_str}")

    print("-" * 50)
    print("\n🎉 ALL TESTS PASSED! The AI nose and smell model pipeline is ready.\n")

if __name__ == "__main__":
    test_pipeline()

