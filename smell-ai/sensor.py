"""
Simulated 8-channel electronic nose gas sensor array.
Each sensor reacts differently to molecular size (MolWt), lipophilicity (LogP),
oxygen content, sulfur content, and functional groups.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

class GasSensorArray:
    def __init__(self, n_sensors=8, random_state=None):
        self.n_sensors = n_sensors
        self.rng = np.random.default_rng(random_state)
        self.sensor_names = [
            "S1_Volatiles",
            "S2_Lipophilic",
            "S3_Oxygenated",
            "S4_Sulfur",
            "S5_Aromatic",
            "S6_Terpenic",
            "S7_Esters",
            "S8_Broadband"
        ]

    def molecule_properties(self, smiles_or_mol):
        """Extract physical-chemical properties relevant to gas sensors."""
        if isinstance(smiles_or_mol, str):
            mol = Chem.MolFromSmiles(smiles_or_mol)
        else:
            mol = smiles_or_mol

        if mol is None:
            return {"mw": 100.0, "logp": 1.0, "n_o": 0, "n_s": 0, "n_arom": 0, "n_rot": 0}

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        n_o = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
        n_s = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'S')
        n_n = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
        n_arom = Lipinski.NumAromaticRings(mol)
        n_rot = Lipinski.NumRotatableBonds(mol)

        return {
            "mw": mw,
            "logp": logp,
            "tpsa": tpsa,
            "n_o": n_o,
            "n_s": n_s,
            "n_n": n_n,
            "n_arom": n_arom,
            "n_rot": n_rot
        }

    def molecule_raw_response(self, props):
        """Calculate clean sensor response (8 channels) for a single molecule."""
        mw = props["mw"]
        logp = props["logp"]
        tpsa = props["tpsa"]
        n_o = props["n_o"]
        n_s = props["n_s"]
        n_n = props.get("n_n", 0)
        n_arom = props["n_arom"]

        # 8 channels derived from physical quantities:
        # S1: Small volatile molecules (lower MW -> higher volatility)
        s1 = np.exp(-mw / 180.0) * (1.2 + 0.3 * np.maximum(0, logp))
        # S2: Lipophilic / oil-loving (high logP, low polarity)
        s2 = np.clip(logp / 2.5, 0.1, 3.0) + (0.2 if (n_o == 0 and n_n == 0) else 0.0)
        # S3: Oxygenated (alcohols, esters, carbonyls, polar surface area)
        s3 = n_o * 0.95 + 0.3 * (mw / 120.0) + 0.01 * tpsa
        # S4: Sulfur-sensitive (thiols, mercaptans, sulfides)
        s4 = n_s * 4.2 + 0.05 * s1
        # S5: Aromatic / Nitrogenous / Phenolic rings (pyrazines, pyridines, phenols)
        s5 = n_arom * 1.6 + n_n * 1.2 + (0.8 if n_o > 0 and n_arom > 0 else 0.0)
        # S6: Terpenic / cyclic hydrocarbons
        s6 = (1.5 if (n_arom == 0 and logp > 2.0) else 0.2) + 0.2 * logp
        # S7: Esters and medium MW oxygenates
        s7 = (1.4 if (n_o >= 2 and n_s == 0) else 0.3) + 0.15 * (mw / 100.0)
        # S8: Broadband general VOC response
        s8 = 0.5 + 0.2 * (mw / 100.0) + 0.15 * abs(logp) + 0.3 * n_o + 0.8 * n_s + 0.5 * n_n

        return np.array([s1, s2, s3, s4, s5, s6, s7, s8], dtype=np.float32)

    def sample_response(self, molecule_list, humidity=0.5, noise_level=0.04):
        """
        Simulate sensor response to a sample mixture of molecules.
        molecule_list: list of dicts with 'smiles' and optional 'weight'
        humidity: ambient humidity (0.0 to 1.0)
        noise_level: sensor noise standard deviation
        """
        response = np.zeros(self.n_sensors, dtype=np.float32)
        total_weight = 0.0

        for item in molecule_list:
            smiles = item.get("smiles", "")
            weight = float(item.get("weight", 1.0))
            props = self.molecule_properties(smiles)
            resp = self.molecule_raw_response(props)
            response += resp * weight
            total_weight += weight

        if total_weight > 0:
            response = response / total_weight

        # Humidity affects baseline and polar channels
        humidity_drift = (humidity - 0.5) * np.array([0.08, -0.04, 0.18, 0.02, 0.05, -0.03, 0.12, 0.10], dtype=np.float32)
        response += humidity_drift

        # Add Gaussian electronic noise
        noise = self.rng.normal(0, noise_level, size=self.n_sensors).astype(np.float32)
        response = np.maximum(0.01, response + noise)

        return response

def sniff_sample(molecules, humidity=None, noise_level=0.04):
    """Convenience function to take a sniff."""
    sensor = GasSensorArray()
    if humidity is None:
        humidity = np.random.uniform(0.3, 0.7)
    return sensor.sample_response(molecules, humidity=humidity, noise_level=noise_level)

