"""
TransChemo-Repurpose: Reversal Scoring & Virtual Screening Engine
Orchestrates virtual screening of compound libraries against user-provided or pre-calculated
disease expression signatures using the trained multi-modal architecture.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch

from src.chemoinformatics.descriptors import MolecularFeatureExtractor
from src.transcriptomics.signature_engine import TranscriptomicSignatureEngine, CURATED_DRUG_LIBRARY, CANCER_PROFILES
from src.transcriptomics.pathways import PathwayAnalysisEngine
from src.models.multimodal_net import TransChemoNet


class DrugRepurposingEngine:
    """End-to-end inference and screening orchestrator."""

    def __init__(
        self,
        model: Optional[TransChemoNet] = None,
        feature_extractor: Optional[MolecularFeatureExtractor] = None,
        signature_engine: Optional[TranscriptomicSignatureEngine] = None,
        device: str = "cpu"
    ):
        self.device = device
        self.feature_extractor = feature_extractor or MolecularFeatureExtractor()
        self.signature_engine = signature_engine or TranscriptomicSignatureEngine()
        self.pathway_engine = PathwayAnalysisEngine(self.signature_engine.landmark_genes)
        
        if model is None:
            self.model = TransChemoNet(
                trans_dim=self.signature_engine.num_genes,
                chem_dim=2059,
                latent_dim=128
            ).to(self.device)
            self.model.eval()
        else:
            self.model = model.to(self.device)
            self.model.eval()

    def screen_library(
        self,
        disease_profile_or_vector: Union[str, np.ndarray],
        custom_compounds: Optional[List[Dict[str, any]]] = None,
        top_k: int = 15
    ) -> pd.DataFrame:
        """
        Screens small molecules against a target disease expression signature.
        
        Args:
            disease_profile_or_vector: Either name of cancer cohort or custom float array of length 100
            custom_compounds: Optional custom list of drug dicts containing 'name', 'smiles', 'moa', etc.
            top_k: Number of top candidate repositioning hits to return
            
        Returns:
            pd.DataFrame containing ranked candidates, predicted Reversal Score, pIC50, efficacy probability,
            and ADMET drug-likeness metrics.
        """
        if isinstance(disease_profile_or_vector, str):
            disease_vec = self.signature_engine.get_disease_vector(disease_profile_or_vector)
            indication_label = disease_profile_or_vector
        else:
            disease_vec = np.asarray(disease_profile_or_vector, dtype=np.float32)
            indication_label = "Custom Patient Signature"

        compounds = custom_compounds or CURATED_DRUG_LIBRARY
        results = []

        # Prepare disease tensor
        trans_tensor = torch.tensor(disease_vec, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            for drug in compounds:
                smiles = drug.get("smiles", "")
                name = drug.get("name", "Unknown Compound")
                moa = drug.get("moa", "N/A")
                target = drug.get("target", "N/A")
                fda_approved = drug.get("fda_approved", False)
                
                chem_feats = self.feature_extractor.get_full_feature_vector(smiles)
                props = self.feature_extractor.get_physicochemical_descriptors(smiles)
                
                chem_tensor = torch.tensor(chem_feats, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Multi-modal model forward pass
                out = self.model(trans_tensor, chem_tensor)
                pred_srs = float(out["srs"].item())
                pred_pic50 = float(out["pic50"].item())
                pred_prob = float(out["reversal_prob"].item())
                
                # Calculate direct cosine ground-truth approximation from perturbation vector
                drug_pert_vec = self.signature_engine.get_drug_perturbation_vector(drug)
                direct_srs = self.signature_engine.compute_signature_reversion_score(disease_vec, drug_pert_vec)
                
                # Composite Score (combines AI SRS prediction, bioactivity pIC50, and QED drug-likeness)
                composite_score = round(
                    0.55 * ((pred_srs + 1.0) / 2.0) +
                    0.25 * (np.clip(pred_pic50, 4.0, 9.0) / 9.0) +
                    0.20 * props["qed"],
                    4
                )

                results.append({
                    "Drug Name": name,
                    "SMILES": smiles,
                    "Mechanism of Action": moa,
                    "Primary Target": target,
                    "FDA Status": "Approved" if fda_approved else "Investigational",
                    "Predicted SRS (Reversal)": round(pred_srs, 3),
                    "Direct Signature Reversal": round(direct_srs, 3),
                    "Predicted pIC50": round(pred_pic50, 2),
                    "Efficacy Probability": round(pred_prob * 100.0, 1),
                    "Composite Repurposing Score": composite_score,
                    "MW (g/mol)": props["mw"],
                    "LogP": props["logp"],
                    "TPSA (Å²)": props["tpsa"],
                    "QED": props["qed"],
                    "Lipinski Violations": int(props["lipinski_violations"])
                })

        df = pd.DataFrame(results)
        df = df.sort_values(by="Composite Repurposing Score", ascending=False).reset_index(drop=True)
        return df.head(top_k)
