# ⚡ NeuroScent-GNN & NeuroScent: Deep Molecular Olfactory Intelligence

> Autonomous In Silico Molecular Olfactory Screening & Physical Electronic Gas Sensor Simulation Platform  
> Powered by **PyTorch Geometric**, **Message Passing Graph Neural Networks (GINEConv)**, **RDKit**, and **Altair**.  
> Engineered for real-time inference on Apple Silicon M4 Series (10-Core GPU via MPS) and standard CPU environments.

---

## Technical Overview

**NeuroScent-GNN** is an end-to-end Chemoinformatics and Deep Learning platform designed to predict perceptual olfactory qualities directly from molecular graph topology. 

Rather than relying on static bitwise fingerprints or text tokens, NeuroScent-GNN models chemical structures as **attributed molecular graphs** ($G = (V, E)$), executing deep message passing across atoms and bonds to project molecules into a continuous **256-dimensional Principal Odor Map (POM)** bottleneck, simultaneously predicting **138 human olfactory descriptors** (*fruity, floral, minty, woody, sweet, roasted, etc.*).

The platform also integrates an **8-channel physical electronic gas sensor array (e-nose)** simulator that models volatile organic compound interactions under variable ambient humidity and Gaussian noise drift, coupled to a calibrated electronic nose classifier.

---

## Architectural Highlights

- **Molecular Graph Attributed Tensors**:
  - **Atom Node Tensor ($X \in \mathbb{R}^{N \times 33}$)**: Atomic numbers, degree, hybridization, formal charge, aromaticity, and hydrogen bonding capacity.
  - **Bond Edge Tensor ($E \in \mathbb{R}^{M \times 10}$)**: Single, double, triple, aromatic bonds, conjugation, ring membership, and stereochemistry.
- **Deep Message Passing GNN (`MolecularPOMGNN`)**:
  - 4 layers of **`GINEConv`** (Graph Isomorphism Networks with Edge features) and residual skip-connections.
  - Dual global readout pooling (`mean + max`) projecting into a **256-dimensional POM bottleneck**.
  - Multi-task classification head with dropout regularization predicting **138 simultaneous odor tasks**.
- **Hardware Acceleration**:
  - Accelerated via **Apple Silicon Metal Performance Shaders (`mps`)** on M4 Series 10-Core GPUs with seamless CPU fallback.
  - Sub-20ms inference latency per compound.
- **Validated Benchmarks**:
  - Curated on **4,983 GoodScents & Leffingwell fragrance molecules**.
  - **Multi-Task Test ROC-AUC**: `0.8583` on completely unseen held-out molecules.
  - **Electronic Nose Classification Accuracy**: `100.0%` on benchmark aroma blends.

---

## Repository Structure

```
neuroscent-main/
├── README.md                          # Platform documentation
├── requirements.txt                   # Universal Python package dependencies
├── setup.py                           # NeuroScent Python package configuration
├── LICENSE                            # Apache 2.0 Open Source License
├── smell-ai/                          # Interactive Web Platform & Models
│   ├── app.py                         # Streamlit application with shadcn/ui dashboard
│   ├── gnn_model.py                   # PyTorch Geometric GNN architecture & featurizer
│   ├── gnn_pom_model.pt               # Trained GNN model weights checkpoint
│   ├── sensor.py                      # 8-channel physical electronic gas sensor array
│   ├── train_nose.py                  # Electronic nose classifier trainer & predictor
│   ├── nose_model.joblib              # Trained electronic nose classifier weights
│   ├── smell_model.py                 # Baseline molecular ECFP4 fingerprint model
│   ├── smell_model.joblib             # Trained baseline model weights
│   ├── curated_GS_LF_merged_4983.csv  # 4,983 compounds × 138 olfactory tasks dataset
│   ├── samples.json                   # Benchmark aroma mixtures & chemical constituent weights
│   ├── test_app.py                    # End-to-end verification regression test suite
│   ├── requirements.txt               # smell-ai environment dependencies
│   └── .streamlit/                    # Streamlit server and theme configurations
├── neuroscent/                           # NeuroScent Chemoinformatics Library
│   ├── data/                          # Dataset curation scripts & raw benchmarks
│   ├── feat/                          # Graph featurization utilities
│   ├── models/                        # MPNN POM reference architectures
│   ├── layers/                        # Neural network layers & pooling modules
│   └── utils/                         # Loss functions & optimizer utilities
└── examples/                          # Research notebooks & hyperparameter trials
```

---

## Application Interface (4 Structured Tabs)

1. **In Silico Molecular Screener (Tab 1)**:
   - Unified 1-card header displaying SMILES input, preset selection, rendered 2D structure, and 6 physicochemical descriptors (*Mol Wt, LogP, TPSA, H-Bond D/A, Graph Nodes, Graph Edges*).
   - Primary Odor Predictions with confidence-coded badges and Altair probability bar charts.
   - Interactive 256-D Principal Odor Map latent vector with cursor pointing crosshairs, zoom/pan controls, and one-click JSON report export.
   - Deep inspection sub-tab with live search across all 138 odor tasks and PyTorch Geometric Graph Tensor Inspector ($X \in \mathbb{R}^{N \times 33}, E \in \mathbb{R}^{M \times 10}$).

2. **Principal Odor Map: Latent Perceptual Space (Tab 2)**:
   - Live 2D PCA projection of the 256-D GNN bottleneck coordinate manifold.
   - Deep, high-contrast categorical palette with active query highlight (`size: 250`).
   - Real-time Euclidean distance computation displaying the 5 nearest olfactory analogs in latent space.

3. **Electronic Gas Sensor Array Simulator (Tab 3)**:
   - Physical 8-channel chemical gas sensor array simulating differential voltage reactions ($\Delta V$).
   - Interactive atmospheric controls for Relative Humidity (%) and Gaussian sensor noise.
   - Live electronic nose classification meter with probability distribution tables and composite GNN mixture note synthesis.

4. **User Guide & Live Model Audit (Tab 4)**:
   - Consolidated 1-page architecture audit with zero vertical scrolling.
   - Collapsible step-by-step user guide walk-through.
   - Real-time parameter introspection (trainable weights, ROC-AUC, label frequencies).
   - Live software and hardware environment telemetry (*PyTorch, PyG, RDKit, Apple Silicon M4 10-Core GPU*).

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- macOS (Apple Silicon M-Series or Intel), Linux, or Windows

### Step 1: Clone or Navigate to the Workspace
```bash
cd /path/to/neuroscent-main
```

### Step 2: Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Platform

To launch the interactive NeuroScent-GNN web application:

```bash
cd smell-ai
streamlit run app.py
```

Once launched, navigate to `http://localhost:8501` in your browser.

---

## Running the Test Suite

To execute the 6-step end-to-end verification suite:

```bash
cd smell-ai
python test_app.py
```

Expected output:
```
============================================================
RUNNING SMELL-AI VERIFICATION TESTS
============================================================
[1/5] Checking samples.json... ✓ Found all 6 sample smells.
[2/5] Testing 8-channel GasSensorArray... ✓ 8 channels simulated.
[3/5] Testing Electronic Nose Model... ✓ Classification 100.0%.
[4/6] Testing Baseline Smell Model... ✓ Descriptors predicted.
[5/6] Testing Deep Molecular GNN... ✓ ROC-AUC: 0.8583 | Latent: Shape (256,).
[6/6] Full End-to-End Pipeline Execution:
--------------------------------------------------
Sample: Coffee          -> Nose: Coffee (100.0%)
Sample: Vanilla Candle  -> Nose: Vanilla Candle (100.0%)
Sample: Mint            -> Nose: Mint (100.0%)
Sample: Lemon           -> Nose: Lemon (100.0%)
Sample: Banana          -> Nose: Banana (100.0%)
Sample: Wood Smoke      -> Nose: Wood Smoke (100.0%)
--------------------------------------------------
ALL TESTS PASSED! The AI nose and smell model pipeline is ready.
```

---

## Programmatic Usage Example

```python
from gnn_model import load_or_train_gnn, predict_gnn_smell
from sensor import GasSensorArray
from train_nose import load_nose, predict_nose

# 1. Load GNN Model
model, descriptors, auc = load_or_train_gnn(device="mps")

# 2. Predict Smell Profile and 256-D Latent Coordinates
smiles = "COc1cc(C=O)ccc1O"  # Vanillin
predictions, latent_vector = predict_gnn_smell(smiles, model, descriptors, top_n=5)
print("Top Odor Notes:", predictions)
print("POM Latent Vector Shape:", latent_vector.shape)

# 3. Simulate Electronic Nose Reaction
sensor = GasSensorArray()
readings = sensor.sample_response([{"smiles": smiles, "weight": 1.0}], humidity=0.5)
nose_bundle = load_nose("nose_model.joblib")
smell_id, confidence, _ = predict_nose(readings, nose_bundle)
print(f"Electronic Nose Identification: {smell_id} ({confidence*100:.1f}%)")
```

---

## License

This project is licensed under the Apache 2.0 License.

Developed by **Brijesh Rath** • All Rights Reserved © 2026
