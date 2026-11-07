# ⚡ NeuroScent-GNN: Deep Molecular Graph Olfactory Intelligence

> **In Silico Molecular Olfactory Screening & Electronic Gas Sensor Simulation Platform**  
> Powered by **PyTorch Geometric**, **Message Passing Graph Neural Networks (GINEConv)**, **RDKit**, and **Altair**.  
> Inspired by Google Research & Osmo's landmark *Science* (2023) Principal Odor Map paper.  
> Optimized for Apple Silicon M4 Series (10-Core GPU via MPS) & standard CPU environments.

---

## Overview

**NeuroScent-GNN** is an end-to-end Chemoinformatics and Deep Learning platform designed to predict perceptual olfactory qualities directly from molecular structures. 

Instead of traditional text tokens or simple bitwise fingerprints, NeuroScent-GNN models molecules as **attributed graphs** ($G = (V, E)$), using multi-layer graph convolutions to learn continuous representations on a **256-dimensional Principal Odor Map (POM)** latent manifold, simultaneously predicting **138 human odor descriptors** (*fruity, minty, roasted, floral, woody, sweet, etc.*).

The platform also couples this molecular perception engine to an **8-channel electronic gas sensor array (e-nose)** that fingerprints volatile vapors in real time under noisy, humid ambient conditions.

---

## Key Technical Highlights

- **Molecular Graph Attributed Tensors**:
  - **Atom Nodes ($V \in \mathbb{R}^{N \times 33}$)**: Atomic numbers, degree, hybridization, formal charge, aromaticity, and hydrogen bonding capacity.
  - **Bond Edges ($E \in \mathbb{R}^{M \times 10}$)**: Single, double, triple, aromatic bonds, conjugation, ring membership, and stereochemistry.
- **Deep Message Passing GNN (`MolecularPOMGNN`)**:
  - 4 layers of **`GINEConv`** (Graph Isomorphism Networks with Edge features) and residual skip-connections.
  - Dual global readout pooling (`mean + max`) projecting into a **256-dimensional POM bottleneck**.
  - Multi-task classification head with dropout regularization predicting **138 simultaneous odor tasks**.
- **Rigorous Benchmarks**:
  - Trained on the curated **4,983 GoodScents & Leffingwell fragrance molecules**.
  - **Multi-Task Test ROC-AUC: 0.8583** on completely unseen molecules.
  - **Electronic Nose Classification Accuracy: 100.0%** on unseen noisy sniff cycles.
- **Sub-20ms Real-Time Inference**: Runs locally on CPU or Apple Silicon (MPS) with zero cloud dependencies or external API keys.

---

## Platform Interface (4 Organized Tabs)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ⚡ NeuroScent-GNN: Deep Molecular Olfactory Intelligence                   │
│ Active Engine | Apple Silicon M4 10-Core GPU | GINEConv | ROC-AUC: 0.8583   │
├───────────────────┬───────────────────┬───────────────────┬─────────────────┤
│ 1. In Silico      │ 2. Latent Scent   │ 3. Electronic     │ 4. User Guide   │
│    Screener       │    Space (2D)     │    Nose Array     │    & Model Audit│
└───────────────────┴───────────────────┴───────────────────┴─────────────────┘
```

### 1. In Silico Molecular Screener
- **Live SMILES Input**: Type any arbitrary chemical structure or pick benchmark presets (*Vanillin, Menthol, Limonene, Damascenone, Geosmin*).
- **2D Chemical Structure**: High-resolution 2D chemical drawing rendered in real time via RDKit.
- **Physicochemical Properties**: Real-time calculation of Molecular Weight, LogP, Polar Surface Area (TPSA), and H-Bond Donors/Acceptors.
- **Graph Tensor Inspector**: Inspect live Atom Node tensors ($X$) and Bond Edge tensors ($E$).
- **Top Odor Predictions**: Multi-task probability bars with confidence-coded badges.
- **138-Descriptor Live Search**: Query any specific descriptor (*e.g., "chocolate", "rose", "tobacco"*).
- **256-D POM Latent Activation**: Continuous activation plot with dominant feature index and L2 norm.
- **One-Click JSON Export**: Download complete screening reports with a single click.

### 2. Principal Odor Map (2D Latent Manifold)
- **Live 2D PCA Projection**: Projects the 256-D GNN bottleneck into a 2D perceptual map.
- **Deep High-Contrast Colors**: Custom palette ensuring points pop clearly in Light Mode, with active query scaled to `size: 250`.
- **Explained Variance**: Real-time computation of $PC_1$ and $PC_2$ explained variance ratios.
- **Cluster Visualizer**: Shows where your molecule lands relative to fragrance clusters (*Sweet/Vanilla, Mint/Cooling, Citrus, Fruity, Woody/Smoky, Roasted/Coffee, Spicy*).
- **256-D Nearest Neighbors**: Computes Euclidean distance to benchmark compounds, identifying the 5 closest olfactory analogs.

### 3. Electronic Gas Sensor Array
- **Dual Mode**: Choose between curated benchmark samples or the **Custom Aroma Blend Studio**.
- **Custom Blend Studio**: Combine multiple chemical compounds with concentration sliders.
- **Physicochemical Sensor Array**: 8 channels reacting dynamically to molecular weight, LogP, TPSA, oxygen, sulfur, and nitrogen.
- **Electronic Nose Classification**: Random Forest classification with live confidence meter and full probability breakdown table.
- **GNN Mixture Profile**: Runs the multi-compound blend through `predict_gnn_mixture()` to predict the composite scent profile.

### 4. User Guide & Live Model Audit
- **Zero Placeholders**: Everything introspected directly from disk and model weights:
  - **Trainable Parameters**: `867,466` parameters.
  - **Dataset Metrics**: `4,983` compounds, `138` tasks, `24,312` labels, `4.88` labels/molecule.
  - **Descriptor Frequency**: Top 10 most common descriptors across the dataset.
  - **Runtime Environment**: Exact versions of PyTorch, PyTorch Geometric, RDKit, and hardware acceleration status (`Apple Silicon M4 Series 10-Core GPU / MPS`).
- **One-Page Layout**: Collapsible step-by-step user guide and consolidated audit metrics requiring zero vertical scrolling.

---

## Quickstart

### Prerequisites
- Python 3.10 or newer (tested on Python 3.10 - 3.13 on macOS / Linux / Windows).

### 1. Setup Virtual Environment
```bash
cd smell-ai
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python test_app.py
```

### 4. Launch the Web Platform
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## Repository Structure

```
smell-ai/
├── app.py                         # Interactive Streamlit frontend (shadcn/ui dashboard)
├── gnn_model.py                   # Molecular GNN (GINEConv, 256-D POM, multi-task head)
├── gnn_pom_model.pt               # Trained GNN PyTorch checkpoint (ROC-AUC: 0.8583)
├── sensor.py                      # 8-channel physical electronic gas sensor simulator
├── train_nose.py                  # Random Forest e-nose classifier training script
├── nose_model.joblib              # Trained electronic nose classifier checkpoint
├── smell_model.py                 # Baseline ECFP4 fingerprint model
├── smell_model.joblib             # Trained baseline model bundle
├── samples.json                   # Curated aroma blends with chemical formulas & weights
├── curated_GS_LF_merged_4983.csv  # 4,983-molecule expert-annotated fragrance dataset
├── requirements.txt               # Project dependencies
├── test_app.py                    # End-to-end regression test suite
└── README.md                      # Comprehensive documentation
```

---

## Citations & References

- **Principal Odor Map**: Brian K. Lee, Emily J. Mayhew, Benjamin Sanchez-Lengeling, et al. *"A principal odor map unifies diverse tasks in olfactory perception."* **Science** 381, 999-1006 (2023). [DOI: 10.1126/science.ade4401](https://doi.org/10.1126/science.ade4401)
- **NeuroScent**: [BioMachineLearning/neuroscent](https://github.com/BioMachineLearning/neuroscent) (Apache 2.0 License).
- **PyTorch Geometric**: Fey, M., & Lenssen, J. E. *"Fast Graph Representation Learning with PyTorch Geometric."* (ICLR 2019).

---

## License

Developed by **Brijesh Rath** • All Rights Reserved © 2026
