"""
TransChemo-Repurpose: Chemoinformatics & Molecular Descriptors Module
Provides canonical SMILES handling, 2048-bit ECFP4 Morgan fingerprints, MACCS keys,
and Lipinski / ADMET physicochemical descriptor calculations.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit import RDLogger

# Silence noisy RDKit chemistry logs
RDLogger.DisableLog('rdApp.*')


class MolecularFeatureExtractor:
    """Extracts topological, fingerprint, and physicochemical features from SMILES."""

    def __init__(self, fp_radius: int = 2, fp_nbits: int = 2048):
        self.fp_radius = fp_radius
        self.fp_nbits = fp_nbits

    def validate_smiles(self, smiles: str) -> Optional[Chem.Mol]:
        """Validates and canonicalizes a SMILES string."""
        if not smiles or not isinstance(smiles, str):
            return None
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            return mol
        except Exception:
            return None

    def canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """Returns canonical SMILES string if valid, else None."""
        mol = self.validate_smiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return None

    def get_morgan_fingerprint(self, smiles_or_mol: Union[str, Chem.Mol]) -> np.ndarray:
        """Computes 2048-bit Morgan Fingerprint (ECFP4 equivalent)."""
        mol = smiles_or_mol if isinstance(smiles_or_mol, Chem.Mol) else self.validate_smiles(smiles_or_mol)
        if mol is None:
            return np.zeros(self.fp_nbits, dtype=np.float32)
        
        bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.fp_radius, nBits=self.fp_nbits)
        arr = np.zeros((self.fp_nbits,), dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(bit_vect, arr)
        return arr

    def get_maccs_keys(self, smiles_or_mol: Union[str, Chem.Mol]) -> np.ndarray:
        """Computes 166-bit MACCS structural keys."""
        mol = smiles_or_mol if isinstance(smiles_or_mol, Chem.Mol) else self.validate_smiles(smiles_or_mol)
        if mol is None:
            return np.zeros(166, dtype=np.float32)
        
        maccs_vect = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        arr = np.zeros((167,), dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(maccs_vect, arr)
        return arr[1:]  # Return the 166 defined MACCS keys (ignoring bit 0)

    def get_physicochemical_descriptors(self, smiles_or_mol: Union[str, Chem.Mol]) -> Dict[str, float]:
        """
        Calculates key drug-likeness & ADMET physicochemical properties:
        - Molecular Weight (MW)
        - LogP (Lipophilicity)
        - Topological Polar Surface Area (TPSA)
        - Hydrogen Bond Donors (HBD)
        - Hydrogen Bond Acceptors (HBA)
        - Rotatable Bonds (RotB)
        - Aromatic Rings (AromRings)
        - Heavy Atom Count (HeavyAtoms)
        - Quantitative Estimate of Drug-Likeness (QED)
        - Lipinski Rule of 5 Violations
        """
        mol = smiles_or_mol if isinstance(smiles_or_mol, Chem.Mol) else self.validate_smiles(smiles_or_mol)
        if mol is None:
            return {
                "mw": 0.0, "logp": 0.0, "tpsa": 0.0, "hbd": 0.0, "hba": 0.0,
                "rotb": 0.0, "arom_rings": 0.0, "heavy_atoms": 0.0, "qed": 0.0,
                "lipinski_violations": 0.0, "fraction_csp3": 0.0
            }
        
        mw = float(Descriptors.MolWt(mol))
        logp = float(Descriptors.MolLogP(mol))
        tpsa = float(Descriptors.TPSA(mol))
        hbd = float(Lipinski.NumHDonors(mol))
        hba = float(Lipinski.NumHAcceptors(mol))
        rotb = float(Lipinski.NumRotatableBonds(mol))
        arom_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
        heavy_atoms = float(mol.GetNumHeavyAtoms())
        qed_score = float(QED.qed(mol))
        frac_csp3 = float(Lipinski.FractionCSP3(mol))

        # Calculate Lipinski violations (Rule of 5)
        violations = sum([
            mw > 500.0,
            logp > 5.0,
            hbd > 5,
            hba > 10
        ])

        return {
            "mw": round(mw, 2),
            "logp": round(logp, 2),
            "tpsa": round(tpsa, 2),
            "hbd": float(hbd),
            "hba": float(hba),
            "rotb": float(rotb),
            "arom_rings": float(arom_rings),
            "heavy_atoms": float(heavy_atoms),
            "qed": round(qed_score, 3),
            "lipinski_violations": float(violations),
            "fraction_csp3": round(frac_csp3, 3)
        }

    def get_full_feature_vector(self, smiles_or_mol: Union[str, Chem.Mol]) -> np.ndarray:
        """
        Concatenates Morgan Fingerprints (2048-d) and normalized physicochemical descriptors
        into a unified chemical feature vector.
        """
        fp = self.get_morgan_fingerprint(smiles_or_mol)
        props = self.get_physicochemical_descriptors(smiles_or_mol)
        
        # Selected scaled descriptor vector for ML stability
        desc_vec = np.array([
            props["mw"] / 500.0,
            props["logp"] / 5.0,
            props["tpsa"] / 150.0,
            props["hbd"] / 5.0,
            props["hba"] / 10.0,
            props["rotb"] / 10.0,
            props["arom_rings"] / 5.0,
            props["heavy_atoms"] / 50.0,
            props["qed"],
            props["lipinski_violations"] / 4.0,
            props["fraction_csp3"]
        ], dtype=np.float32)
        
        return np.concatenate([fp, desc_vec])
