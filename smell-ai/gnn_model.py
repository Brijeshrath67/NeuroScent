"""
Production Molecular Graph Neural Network (GNN) for Olfactory Perception (NeuroScent).
Converts SMILES molecules to molecular graphs (nodes=atoms, edges=bonds) and uses
Message Passing Neural Networks (MPNN) to predict 138 perceptual odor descriptors.
Extracts a 256-dimensional Principal Odor Map (POM) latent representation.
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Constants
DATA_FILENAME = "curated_GS_LF_merged_4983.csv"
DATA_URL = "https://raw.githubusercontent.com/BioMachineLearning/neuroscent/main/neuroscent/data/curated_datasets/curated_GS_LF_merged_4983.csv"
GNN_CHECKPOINT_PATH = "gnn_pom_model.pt"

# Atom feature vocabularies
ALLOWED_ATOMS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H, B, C, N, O, F, Si, P, S, Cl, Br, I
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

def atom_features(atom):
    """Compute dense 32-dim atom feature vector."""
    atomic_num = atom.GetAtomicNum()
    atom_type_enc = [1.0 if atomic_num == a else 0.0 for a in ALLOWED_ATOMS]
    if not any(atom_type_enc):
        atom_type_enc.append(1.0)
    else:
        atom_type_enc.append(0.0)

    degree = [1.0 if atom.GetTotalDegree() == d else 0.0 for d in range(6)]
    formal_charge = [atom.GetFormalCharge() / 4.0]
    num_h = [1.0 if atom.GetTotalNumHs() == h else 0.0 for h in range(5)]
    is_aromatic = [1.0 if atom.GetIsAromatic() else 0.0]
    in_ring = [1.0 if atom.IsInRing() else 0.0]
    
    hyb = atom.GetHybridization()
    hyb_enc = [1.0 if hyb == h else 0.0 for h in HYBRIDIZATIONS]
    if not any(hyb_enc):
        hyb_enc.append(1.0)
    else:
        hyb_enc.append(0.0)

    feat = atom_type_enc + degree + formal_charge + num_h + is_aromatic + in_ring + hyb_enc
    return np.array(feat, dtype=np.float32)

def bond_features(bond):
    """Compute dense 10-dim bond feature vector."""
    b_type = bond.GetBondType()
    b_enc = [
        1.0 if b_type == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if b_type == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if b_type == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if b_type == Chem.rdchem.BondType.AROMATIC else 0.0,
    ]
    is_conjugated = [1.0 if bond.GetIsConjugated() else 0.0]
    in_ring = [1.0 if bond.IsInRing() else 0.0]
    stereo = bond.GetStereo()
    stereo_enc = [
        1.0 if stereo == Chem.rdchem.BondStereo.STEREONONE else 0.0,
        1.0 if stereo == Chem.rdchem.BondStereo.STEREOANY else 0.0,
        1.0 if stereo == Chem.rdchem.BondStereo.STEREOZ else 0.0,
        1.0 if stereo == Chem.rdchem.BondStereo.STEREOE else 0.0,
    ]
    return np.array(b_enc + is_conjugated + in_ring + stereo_enc, dtype=np.float32)

def mol_to_graph(smiles_or_mol):
    """Converts a SMILES string to a PyTorch Geometric Data object."""
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    else:
        mol = smiles_or_mol

    if mol is None:
        return None

    atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(np.array(atom_feats), dtype=torch.float32)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        b_feat = bond_features(bond)
        # Bidirectional graph
        edge_indices.append([i, j])
        edge_attrs.append(b_feat)
        edge_indices.append([j, i])
        edge_attrs.append(b_feat)

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 10), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = Chem.MolToSmiles(mol) if isinstance(smiles_or_mol, str) else ""
    return data

class MolecularPOMGNN(nn.Module):
    """
    Message Passing Neural Network (MPNN) architecture for multi-task
    olfactory prediction.
    Outputs:
      - 256-dim POM Latent Embedding
      - 138 Odor Descriptor probabilities (logits)
    """
    def __init__(self, in_atom_dim=33, in_bond_dim=10, hidden_dim=128, pom_dim=256, n_tasks=138, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.atom_encoder = nn.Linear(in_atom_dim, hidden_dim)
        self.bond_encoder = nn.Linear(in_bond_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Latent Principal Odor Map (POM) projection
        self.pom_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, pom_dim),
            nn.BatchNorm1d(pom_dim),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        # Multi-task classification head
        self.classifier = nn.Sequential(
            nn.Linear(pom_dim, 384),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, n_tasks)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_encoder(x)
        if edge_attr.size(0) > 0:
            e = self.bond_encoder(edge_attr)
        else:
            e = torch.empty((0, h.size(1)), device=x.device)

        # Message Passing with Residuals
        for conv, bn in zip(self.convs, self.batch_norms):
            h_new = conv(h, edge_index, e)
            h = h + F.relu(bn(h_new))

        # Global Readout: Concatenate mean and max pooling
        pool_mean = global_mean_pool(h, batch)
        pool_max = global_max_pool(h, batch)
        graph_repr = torch.cat([pool_mean, pool_max], dim=-1)

        # Principal Odor Map Latent Vector (256-D)
        pom_embedding = self.pom_projection(graph_repr)

        # Odor Logits (138 Tasks)
        logits = self.classifier(pom_embedding)
        return logits, pom_embedding

def load_or_train_gnn(device="cpu", epochs=25, batch_size=64):
    """Loads existing trained GNN weights or trains the MPNN on 4,983 molecules."""
    ckpt_path = os.path.join(os.path.dirname(__file__), GNN_CHECKPOINT_PATH)
    data_path = os.path.join(os.path.dirname(__file__), DATA_FILENAME)

    if not os.path.exists(data_path):
        search_paths = [
            os.path.join(os.path.dirname(__file__), "..", "neuroscent", "data", "curated_datasets", DATA_FILENAME),
            os.path.join("neuroscent", "data", "curated_datasets", DATA_FILENAME)
        ]
        for p in search_paths:
            if os.path.exists(p):
                data_path = p
                break
        else:
            urllib.request.urlretrieve(DATA_URL, data_path)

    df = pd.read_csv(data_path)
    descriptor_cols = [c for c in df.columns if c not in ["nonStereoSMILES", "descriptors"]]
    n_tasks = len(descriptor_cols)

    # Initialize model
    model = MolecularPOMGNN(in_atom_dim=33, in_bond_dim=10, hidden_dim=128, pom_dim=256, n_tasks=n_tasks)
    model.to(device)

    if os.path.exists(ckpt_path):
        print(f"Loading trained GNN weights from {ckpt_path}...")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        model.eval()
        return model, descriptor_cols, float(state["test_roc_auc"])

    print("Building molecular graphs for 4,983 compounds...")
    graph_list = []
    labels_list = []
    for idx, row in df.iterrows():
        g = mol_to_graph(row["nonStereoSMILES"])
        if g is not None:
            g.y = torch.tensor(row[descriptor_cols].values.astype(np.float32), dtype=torch.float32).unsqueeze(0)
            graph_list.append(g)

    train_graphs, test_graphs = train_test_split(graph_list, test_size=0.15, random_state=42)
    print(f"Dataset split: {len(train_graphs)} train, {len(test_graphs)} test.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    print("Training Molecular MPNN Graph Neural Network...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        # Shuffle batches
        np.random.shuffle(train_graphs)
        for i in range(0, len(train_graphs), batch_size):
            batch_slice = train_graphs[i:i + batch_size]
            batch = Batch.from_data_list(batch_slice).to(device)

            optimizer.zero_grad()
            logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_slice)

        if epoch % 5 == 0 or epoch == epochs:
            avg_loss = total_loss / len(train_graphs)
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f}")

    # Evaluate on unseen test compounds
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i in range(0, len(test_graphs), batch_size):
            batch_slice = test_graphs[i:i + batch_size]
            batch = Batch.from_data_list(batch_slice).to(device)
            logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(batch.y.cpu().numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    aucs = []
    for col in range(n_tasks):
        if len(np.unique(targets[:, col])) > 1:
            aucs.append(roc_auc_score(targets[:, col], preds[:, col]))

    mean_auc = float(np.mean(aucs))
    print(f"=== GNN Multi-Task Test ROC-AUC: {mean_auc:.4f} ===")

    torch.save({
        "model_state": model.state_dict(),
        "descriptors": descriptor_cols,
        "test_roc_auc": mean_auc
    }, ckpt_path)
    print(f"Saved trained GNN model to {ckpt_path}")
    return model, descriptor_cols, mean_auc

def predict_gnn_smell(smiles, model, descriptors, device="cpu", top_n=6):
    """Inference for a single molecule: returns top descriptors and 256-D POM embedding."""
    g = mol_to_graph(smiles)
    if g is None:
        return [], None

    batch = Batch.from_data_list([g]).to(device)
    model.eval()
    with torch.no_grad():
        logits, pom_embedding = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        embedding = pom_embedding[0].cpu().numpy()

    top_idx = np.argsort(probs)[::-1][:top_n]
    results = [(descriptors[i], float(probs[i])) for i in top_idx]
    return results, embedding

def predict_gnn_mixture(molecule_list, model, descriptors, device="cpu", top_n=6):
    """
    Inference for a mixture/blend of molecules using the GNN.
    Aggregates multi-task predictions and 256-D POM embeddings weighted by concentration.
    """
    total_probs = np.zeros(len(descriptors), dtype=np.float64)
    total_embedding = np.zeros(256, dtype=np.float64)
    total_weight = 0.0

    for item in molecule_list:
        smiles = item.get("smiles", "")
        weight = float(item.get("weight", 1.0))
        g = mol_to_graph(smiles)
        if g is None:
            continue

        batch = Batch.from_data_list([g]).to(device)
        model.eval()
        with torch.no_grad():
            logits, pom_emb = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            emb = pom_emb[0].cpu().numpy()

        total_probs += probs * weight
        total_embedding += emb * weight
        total_weight += weight

    if total_weight > 0:
        total_probs /= total_weight
        total_embedding /= total_weight

    top_idx = np.argsort(total_probs)[::-1][:top_n]
    results = [(descriptors[i], float(total_probs[i])) for i in top_idx]
    return results, total_embedding

