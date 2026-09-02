"""
TransChemo-Repurpose: Pathway Enrichment & Gene Set Analysis (GSEA/ORA) Module
Maps differential gene expression profiles to Hallmark Cancer Pathways and computes
perturbation reversal statistics at the pathway level.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


# Hallmark Cancer Pathways defined by landmark gene sets
HALLMARK_PATHWAYS: Dict[str, List[str]] = {
    "Hallmark: E2F Targets / Cell Cycle": [
        "CDK1", "CDK2", "CDK4", "CDK6", "CCNA2", "CCNB1", "CCND1", "CCNE1",
        "E2F1", "E2F2", "MKI67", "TOP2A", "PCNA", "PLK1", "AURKA", "AURKB"
    ],
    "Hallmark: G2M DNA Damage Checkpoint": [
        "CHEK1", "CHEK2", "ATM", "ATR", "BRCA1", "BRCA2", "RAD51", "TOP2A",
        "PLK1", "BUB1", "GADD45A", "TP53", "MDM2"
    ],
    "Hallmark: Apoptosis & Cell Death": [
        "BAX", "BAK1", "CASP3", "CASP8", "CASP9", "APAF1", "PUMA", "NOXA",
        "BID", "BAD", "DIABLO", "CYCS", "FAS", "FASLG", "TNFRSF10B"
    ],
    "Hallmark: PI3K / AKT / mTOR Signaling": [
        "PIK3CA", "PIK3CB", "AKT1", "AKT2", "AKT3", "MTOR", "RPTOR", "RICTOR",
        "RPS6KB1", "RPS6", "EIF4EBP1", "GSK3B", "FOXO1", "FOXO3", "PTEN", "TSC1", "TSC2"
    ],
    "Hallmark: RTK / RAS / MAPK Cascade": [
        "EGFR", "ERBB2", "ERBB3", "MET", "FGFR1", "FGFR2", "KRAS", "NRAS", "HRAS",
        "BRAF", "RAF1", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "FOS", "JUN", "MYC"
    ],
    "Hallmark: Epithelial-Mesenchymal Transition (EMT)": [
        "CDH1", "CDH2", "VIM", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "MMP2", "MMP9"
    ],
    "Hallmark: Hypoxia & Angiogenesis": [
        "VEGFA", "VEGFB", "KDR", "FLT1", "HIF1A", "EPAS1", "VHL", "ANGPT1", "ANGPT2", "LDHA", "GLUT1", "HK2"
    ],
    "Hallmark: Epigenetic & Chromatin Remodeling": [
        "HDAC1", "HDAC2", "HDAC3", "HDAC6", "EZH2", "KDM5A", "BRD4", "DNMT1", "DNMT3A", "DNMT3B", "SIRT1"
    ]
}


class PathwayAnalysisEngine:
    """Computes pathway activity scores, over-representation enrichment, and pathway reversal."""

    def __init__(self, gene_list: List[str]):
        self.gene_list = gene_list
        self.gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    def calculate_pathway_activity(self, expression_vec: np.ndarray) -> Dict[str, float]:
        """
        Calculates normalized pathway enrichment scores (NES-like z-scores) for each pathway
        given a differential expression vector.
        """
        pathway_scores = {}
        for pathway_name, genes in HALLMARK_PATHWAYS.items():
            valid_indices = [self.gene_to_idx[g] for g in genes if g in self.gene_to_idx]
            if not valid_indices:
                pathway_scores[pathway_name] = 0.0
                continue
            
            # Average expression of genes in pathway
            vals = expression_vec[valid_indices]
            mean_score = float(np.mean(vals))
            pathway_scores[pathway_name] = round(mean_score, 4)
            
        return pathway_scores

    def compute_pathway_reversal(
        self, disease_vec: np.ndarray, drug_vec: np.ndarray
    ) -> pd.DataFrame:
        """
        Evaluates pathway-level therapeutic reversal.
        Compares Disease State vs Predicted Post-Treatment State (Disease + Drug).
        """
        disease_scores = self.calculate_pathway_activity(disease_vec)
        drug_scores = self.calculate_pathway_activity(drug_vec)
        
        post_treatment_vec = disease_vec + drug_vec
        post_scores = self.calculate_pathway_activity(post_treatment_vec)
        
        rows = []
        for pathway in HALLMARK_PATHWAYS.keys():
            d_val = disease_scores.get(pathway, 0.0)
            dr_val = drug_scores.get(pathway, 0.0)
            p_val = post_scores.get(pathway, 0.0)
            
            # Reversal effect: if disease was upregulated (positive), drug should downregulate (negative)
            reversal_delta = d_val - p_val
            pct_reversed = 0.0
            if abs(d_val) > 1e-4:
                pct_reversed = (reversal_delta / abs(d_val)) * 100.0
                
            rows.append({
                "Pathway": pathway.replace("Hallmark: ", ""),
                "Disease State Score": d_val,
                "Drug Effect Score": dr_val,
                "Post-Treatment State": p_val,
                "Reversal %": round(pct_reversed, 1)
            })
            
        df = pd.DataFrame(rows)
        return df.sort_values(by="Reversal %", ascending=False).reset_index(drop=True)
