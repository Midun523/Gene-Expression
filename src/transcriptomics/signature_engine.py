"""
TransChemo-Repurpose: Transcriptomics Signature Engine
Handles LINCS L1000 landmark gene space mapping, TCGA/GEO cancer disease profiles,
and chemical perturbation signatures.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# Curated core representative subset of LINCS L1000 landmark genes covering major oncogenic pathways
LINCS_LANDMARK_GENES = [
    # Cell Cycle & DNA Damage
    "CDK1", "CDK2", "CDK4", "CDK6", "CCNA2", "CCNB1", "CCND1", "CCNE1", "CHEK1", "CHEK2",
    "ATM", "ATR", "BRCA1", "BRCA2", "TP53", "MDM2", "RB1", "E2F1", "E2F2", "AURKA", "AURKB",
    "PLK1", "BUB1", "TOP2A", "PCNA", "MKI67", "PARP1", "RAD51", "ERCC1", "GADD45A",
    
    # Apoptosis & Survival
    "BCL2", "BCL2L1", "BAX", "BAK1", "MCL1", "CASP3", "CASP8", "CASP9", "APAF1", "BIRC5",
    "XIAP", "FAS", "FASLG", "TNFRSF10B", "BID", "BAD", "PUMA", "NOXA", "DIABLO", "CYCS",
    
    # MAPK / ERK & RTK Signaling
    "EGFR", "ERBB2", "ERBB3", "MET", "FGFR1", "FGFR2", "FGFR3", "PDGFRA", "PDGFRB", "KIT",
    "KRAS", "NRAS", "HRAS", "BRAF", "RAF1", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "MAPK8",
    "MAPK14", "FOS", "JUN", "MYC", "MYCN", "ELK1", "DUSP1", "DUSP6", "SPRY2", "ETV4",
    
    # PI3K / AKT / mTOR Signaling
    "PIK3CA", "PIK3CB", "PIK3CD", "PTEN", "AKT1", "AKT2", "AKT3", "MTOR", "RPTOR", "RICTOR",
    "RPS6KB1", "RPS6", "EIF4EBP1", "GSK3B", "FOXO1", "FOXO3", "TSC1", "TSC2", "INPP4B", "PDK1",
    
    # Epigenetic Modifiers & Transcription
    "HDAC1", "HDAC2", "HDAC3", "HDAC6", "KAT2A", "EP300", "CREBBP", "EZH2", "KDM5A", "BRD4",
    "DNMT1", "DNMT3A", "DNMT3B", "SIRT1", "ARID1A", "SMARCA4", "STAT3", "STAT5A", "NFKB1", "RELA",
    
    # Angiogenesis & Hypoxia
    "VEGFA", "VEGFB", "KDR", "FLT1", "HIF1A", "EPAS1", "VHL", "ANGPT1", "ANGPT2", "TEK",
    
    # Epithelial-Mesenchymal Transition (EMT) & Invasion
    "CDH1", "CDH2", "VIM", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "MMP2", "MMP9",
    
    # Immuno-Oncology & Metabolism
    "CD274", "PDCD1", "CTLA4", "TGFB1", "TGFBR1", "IL6", "IL1B", "TNF", "CXCL8", "LDHA",
    "PKM", "GLUT1", "HK2", "IDH1", "IDH2", "G6PD", "ACACA", "FASN", "SREBF1", "PPARGC1A"
]

NUM_LANDMARK_GENES = len(LINCS_LANDMARK_GENES)  # 100 benchmark cancer signaling nodes


# Comprehensive Curated Disease Profiles (TCGA / GEO cohorts)
CANCER_PROFILES: Dict[str, Dict[str, any]] = {
    "Triple-Negative Breast Cancer (TNBC)": {
        "description": "Basal-like aggressive breast cancer lacking ER, PR, and HER2 expression. Characterized by elevated MYC, CCNE1, PARP1, EGFR, and suppressed PTEN/RB1.",
        "indication": "Oncology - Breast",
        "hallmarks": ["E2F Targets", "G2M Checkpoint", "MYC Targets", "DNA Repair Deficiency"],
        "upregulated": ["MYC", "EGFR", "CCNE1", "PARP1", "MKI67", "TOP2A", "AURKA", "PLK1", "VEGFA", "MMP9", "LDHA", "SNAI1", "CDK1", "RAD51", "BIRC5"],
        "downregulated": ["PTEN", "RB1", "CDH1", "FOXO3", "BAX", "DUSP6", "TSC1", "TSC2", "ATM", "CASP3"]
    },
    "Glioblastoma Multiforme (GBM)": {
        "description": "Lethal primary brain tumor with extensive RTK/EGFR amplification, PI3K/AKT activation, loss of PTEN, and prominent hypoxia-driven angiogenesis.",
        "indication": "Oncology - Neuro",
        "hallmarks": ["Hypoxia", "Angiogenesis", "PI3K/AKT/mTOR Signaling", "EMT"],
        "upregulated": ["EGFR", "MET", "VEGFA", "KDR", "HIF1A", "AKT1", "MTOR", "MMP2", "VIM", "ZEB1", "STAT3", "BIRC5", "CDK4", "CCND1", "HDAC1"],
        "downregulated": ["PTEN", "TP53", "CDH1", "FOXO1", "DUSP1", "EPAS1", "BRCA1", "CASP9", "BAK1"]
    },
    "Lung Adenocarcinoma (LUAD)": {
        "description": "Non-small cell lung cancer with prevalent KRAS/EGFR oncogene addiction, MAPK activation, and metabolic reprogramming.",
        "indication": "Oncology - Thoracic",
        "hallmarks": ["KRAS Signaling", "Glycolysis", "Cell Cycle", "Immune Evasion"],
        "upregulated": ["KRAS", "EGFR", "BRAF", "MAPK1", "MYC", "HK2", "LDHA", "GLUT1", "CD274", "CDK2", "CCNA2", "AURKB", "MKI67", "FASN"],
        "downregulated": ["DUSP6", "PTEN", "SPRY2", "CDH1", "FOXO3", "TGFBR1", "ATM", "CASP8", "BAX"]
    },
    "Colorectal Adenocarcinoma (CRC)": {
        "description": "Colorectal malignancy driven by Wnt/beta-catenin, EGFR/KRAS, and epigenetic dysregulation with high metastatic potential.",
        "indication": "Oncology - GI",
        "hallmarks": ["Wnt Signaling", "MYC Targets", "Epithelial-Mesenchymal Transition"],
        "upregulated": ["MYC", "EGFR", "KRAS", "EZH2", "DNMT1", "HDAC2", "VEGFA", "MMP9", "CD44", "SNAI2", "CCND1", "BIRC5", "MCL1"],
        "downregulated": ["APC", "TP53", "PTEN", "CDH1", "SMAD4", "CASP3", "BAX", "FOXO1"]
    }
}


# Curated FDA-Approved & Investigational Small-Molecule Perturbagens
CURATED_DRUG_LIBRARY: List[Dict[str, any]] = [
    {
        "name": "Doxorubicin",
        "smiles": "CC1C(C(CC(O1)OC2CC(CC3=C(C4=C(C(=C23)O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O",
        "moa": "Topoisomerase II Inhibitor / DNA Intercalator",
        "target": "TOP2A, TOP2B",
        "fda_approved": True,
        "primary_indication": "Breast Cancer, Lymphoma",
        "induced_up": ["TP53", "BAX", "PUMA", "NOXA", "CDKN1A", "CASP3", "CASP9", "GADD45A"],
        "induced_down": ["TOP2A", "MKI67", "CCNB1", "CDK1", "MYC", "BIRC5", "PLK1", "AURKA"]
    },
    {
        "name": "Lapatinib",
        "smiles": "CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)OCC5=CC(=CC=C5)F)Cl",
        "moa": "Dual EGFR / HER2 Tyrosine Kinase Inhibitor",
        "target": "EGFR, ERBB2",
        "fda_approved": True,
        "primary_indication": "HER2+ Breast Cancer",
        "induced_up": ["DUSP6", "FOXO3", "BAX", "PTEN", "CASP8"],
        "induced_down": ["EGFR", "ERBB2", "MAPK1", "MAPK3", "AKT1", "CCND1", "MYC", "VEGFA"]
    },
    {
        "name": "Olaparib",
        "smiles": "O=C(C1CC1)N2CCN(CC2)C(=O)C3=C(CC4=NNC(=O)C5=CC=CC=C45)C=CC(=C3)F",
        "moa": "PARP Inhibitor (Synthetic Lethality in HR Deficiency)",
        "target": "PARP1, PARP2",
        "fda_approved": True,
        "primary_indication": "BRCA-mutated Breast & Ovarian Cancer",
        "induced_up": ["ATM", "CHEK1", "GADD45A", "BAX", "CASP3"],
        "induced_down": ["PARP1", "RAD51", "MKI67", "TOP2A", "CCNE1", "BIRC5"]
    },
    {
        "name": "Vorinostat (SAHA)",
        "smiles": "O=C(CCCCCCC(=O)NO)NC1=CC=CC=C1",
        "moa": "Pan-HDAC Class I & II Inhibitor (Epigenetic Re-activator)",
        "target": "HDAC1, HDAC2, HDAC3, HDAC6",
        "fda_approved": True,
        "primary_indication": "Cutaneous T-cell Lymphoma",
        "induced_up": ["CDKN1A", "CDH1", "BAX", "CASP3", "FOXO1", "DUSP1", "PTEN"],
        "induced_down": ["HDAC1", "HDAC2", "EZH2", "MYC", "CCND1", "MMP9", "BIRC5", "MCL1"]
    },
    {
        "name": "Everolimus",
        "smiles": "CC1CCC2CC(=O)C(=CC=CC=CC(CC(C(=O)C(C(C(=CC(C(=O)CC(OC(=O)C3CCCCN3C(=O)C(=O)C1(O2)O)C(C)CC4CCC(C(C4)OC)OCCO)C)C)O)OC)C)C)C",
        "moa": "mTORC1 Allosteric Inhibitor",
        "target": "MTOR, FKBP1A",
        "fda_approved": True,
        "primary_indication": "Breast, Renal, Neuroendocrine Tumors",
        "induced_up": ["AKT1", "FOXO3", "TSC1", "TSC2", "CDKN1A", "BAX"],
        "induced_down": ["MTOR", "RPS6KB1", "RPS6", "EIF4EBP1", "HIF1A", "VEGFA", "CCND1", "MYC"]
    },
    {
        "name": "Palbociclib",
        "smiles": "CC1=C(C(=O)N(C2=NC(=NC=C12)NC3=NC=C(C=C3)N4CCNCC4)C5CCCC5)C(=O)C",
        "moa": "Selective CDK4/6 Cell Cycle Inhibitor",
        "target": "CDK4, CDK6",
        "fda_approved": True,
        "primary_indication": "HR+/HER2- Breast Cancer",
        "induced_up": ["RB1", "CDKN1A", "FOXO1", "DUSP6"],
        "induced_down": ["CDK4", "CDK6", "CCND1", "E2F1", "E2F2", "MKI67", "TOP2A", "PLK1", "AURKA"]
    },
    {
        "name": "Gefitinib",
        "smiles": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
        "moa": "EGFR Tyrosine Kinase Inhibitor",
        "target": "EGFR",
        "fda_approved": True,
        "primary_indication": "EGFR-Mutant Non-Small Cell Lung Cancer",
        "induced_up": ["DUSP6", "FOXO3", "PTEN", "BAX", "SPRY2"],
        "induced_down": ["EGFR", "MAPK1", "MAPK3", "AKT1", "MYC", "CCND1", "VEGFA"]
    },
    {
        "name": "Trametinib",
        "smiles": "CC1=C(C(=O)N(C(=O)N1C2=C(C=C(C=C2)I)F)C3=C(C=C(C=C3)NC(=O)C4CC4)F)NC5=CC=CC(=C5)F",
        "moa": "Allosteric MEK1/2 Inhibitor",
        "target": "MAP2K1, MAP2K2",
        "fda_approved": True,
        "primary_indication": "BRAF-mutant Melanoma & NSCLC",
        "induced_up": ["DUSP6", "SPRY2", "FOXO3", "BAX"],
        "induced_down": ["MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "MYC", "CCND1", "FOS", "JUN"]
    },
    {
        "name": "Metformin (Repurposing)",
        "smiles": "CN(C)C(=N)NC(=N)N",
        "moa": "AMPK Activator / Complex I Inhibitor (Repurposing Candidate)",
        "target": "PRKAA1, MTOR",
        "fda_approved": True,
        "primary_indication": "Type 2 Diabetes / Investigational Oncology",
        "induced_up": ["PRKAA1", "FOXO3", "TSC2", "PTEN", "CDKN1A"],
        "induced_down": ["MTOR", "MYC", "LDHA", "GLUT1", "FASN", "SREBF1", "HIF1A"]
    },
    {
        "name": "Disulfiram (Repurposing)",
        "smiles": "CCN(CC)C(=S)SSC(=S)N(CC)CC",
        "moa": "ALDH / Proteasome Inhibitor + Copper Chelation",
        "target": "ALDH1A1, PSMD4",
        "fda_approved": True,
        "primary_indication": "Alcoholism / Investigational Cancer Stem Cell Inhibitor",
        "induced_up": ["GADD45A", "PUMA", "NOXA", "CASP3", "BAX"],
        "induced_down": ["BIRC5", "MCL1", "SNAI1", "ZEB1", "MMP9", "NFKB1", "RELA"]
    },
    {
        "name": "Niclosamide (Repurposing)",
        "smiles": "C1=CC(=C(C=C1[N+](=O)[O-])Cl)NC(=O)C2=C(C=CC(=C2)Cl)O",
        "moa": "STAT3 / Wnt-beta-catenin / Oxidative Phosphorylation Inhibitor",
        "target": "STAT3, WNT signaling",
        "fda_approved": True,
        "primary_indication": "Anthelmintic / Investigational Cancer Repurposing",
        "induced_up": ["CDH1", "PTEN", "BAX", "CASP3", "CDKN1A"],
        "induced_down": ["STAT3", "MYC", "CCND1", "SNAI1", "VIM", "BIRC5", "MCL1", "VEGFA"]
    },
    {
        "name": "Bortezomib",
        "smiles": "CC(C)CC(NC(=O)C(CC1=CC=CC=C1)NC(=O)C2=NC=CN=C2)B(O)O",
        "moa": "26S Proteasome Catalytic Subunit Inhibitor",
        "target": "PSMB5",
        "fda_approved": True,
        "primary_indication": "Multiple Myeloma & Mantle Cell Lymphoma",
        "induced_up": ["PUMA", "NOXA", "GADD45A", "CASP3", "CASP9", "BAX"],
        "induced_down": ["NFKB1", "RELA", "MCL1", "BIRC5", "CCND1", "MYC", "VEGFA"]
    },
    {
        "name": "Sorafenib",
        "smiles": "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F",
        "moa": "Multi-Kinase (VEGFR, PDGFR, RAF) Inhibitor",
        "target": "KDR, FLT1, BRAF, RAF1",
        "fda_approved": True,
        "primary_indication": "Hepatocellular & Renal Cell Carcinoma",
        "induced_up": ["DUSP1", "FOXO3", "BAX", "PTEN"],
        "induced_down": ["KDR", "VEGFA", "BRAF", "MAPK1", "MCL1", "MMP2", "HIF1A"]
    },
    {
        "name": "Celecoxib (Repurposing)",
        "smiles": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
        "moa": "COX-2 (PTGS2) Inhibitor / Anti-inflammatory Chemoprevention",
        "target": "PTGS2, AKT1",
        "fda_approved": True,
        "primary_indication": "NSAID / FAP Colon Polyp Chemoprevention",
        "induced_up": ["BAX", "CASP3", "PTEN"],
        "induced_down": ["VEGFA", "MMP9", "AKT1", "BIRC5", "IL6", "TNF"]
    },
    {
        "name": "Imatinib",
        "smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        "moa": "BCR-ABL, KIT & PDGFR Tyrosine Kinase Inhibitor",
        "target": "ABL1, KIT, PDGFRA",
        "fda_approved": True,
        "primary_indication": "CML, GIST",
        "induced_up": ["FOXO3", "PTEN", "BAX", "CASP3"],
        "induced_down": ["KIT", "PDGFRA", "AKT1", "MAPK1", "MYC", "CCND1"]
    }
]


class TranscriptomicSignatureEngine:
    """Handles expression vector formatting, signature reversal calculation, and synthetic datasets."""

    def __init__(self, landmark_genes: List[str] = LINCS_LANDMARK_GENES):
        self.landmark_genes = landmark_genes
        self.gene_to_idx = {g: i for i, g in enumerate(landmark_genes)}
        self.num_genes = len(landmark_genes)

    def get_disease_vector(self, disease_name: str, noise_level: float = 0.05, seed: Optional[int] = None) -> np.ndarray:
        """
        Generates a normalized differential expression z-score vector for a disease phenotype.
        Positive values = upregulated in disease; Negative values = downregulated in disease.
        """
        if seed is not None:
            np.random.seed(seed)
            
        profile = CANCER_PROFILES.get(disease_name)
        if profile is None:
            raise ValueError(f"Unknown disease: {disease_name}. Choose from {list(CANCER_PROFILES.keys())}")
            
        vec = np.random.normal(0, noise_level, size=(self.num_genes,)).astype(np.float32)
        
        for gene in profile["upregulated"]:
            if gene in self.gene_to_idx:
                idx = self.gene_to_idx[gene]
                vec[idx] += np.random.uniform(1.8, 3.5)
                
        for gene in profile["downregulated"]:
            if gene in self.gene_to_idx:
                idx = self.gene_to_idx[gene]
                vec[idx] -= np.random.uniform(1.5, 3.0)
                
        # Unit normalization for robust inner-product / cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec

    def get_drug_perturbation_vector(self, drug_dict: Dict[str, any], noise_level: float = 0.05, seed: Optional[int] = None) -> np.ndarray:
        """
        Generates a drug perturbation z-score signature vector (post-treatment vs vehicle control).
        Positive values = induced expression; Negative values = repressed expression.
        """
        if seed is not None:
            np.random.seed(seed)
            
        vec = np.random.normal(0, noise_level, size=(self.num_genes,)).astype(np.float32)
        
        for gene in drug_dict.get("induced_up", []):
            if gene in self.gene_to_idx:
                idx = self.gene_to_idx[gene]
                vec[idx] += np.random.uniform(1.6, 3.2)
                
        for gene in drug_dict.get("induced_down", []):
            if gene in self.gene_to_idx:
                idx = self.gene_to_idx[gene]
                vec[idx] -= np.random.uniform(1.6, 3.5)
                
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec

    def compute_signature_reversion_score(
        self, disease_vec: np.ndarray, drug_perturbation_vec: np.ndarray
    ) -> float:
        """
        Computes the Signature Reversion Score (SRS).
        SRS is defined as the negative cosine similarity (or anti-correlation) between
        the disease signature vector and the drug perturbation vector:
            SRS = - (v_disease · v_drug) / (||v_disease|| * ||v_drug||)
        
        SRS range: [-1.0, +1.0]
        - +1.0: Perfect signature reversal (optimal therapeutic candidate).
        -  0.0: No transcriptomic alignment / orthogonal effect.
        - -1.0: Mimics or exacerbates the disease transcriptomic phenotype.
        """
        norm_d = np.linalg.norm(disease_vec)
        norm_p = np.linalg.norm(drug_perturbation_vec)
        if norm_d < 1e-6 or norm_p < 1e-6:
            return 0.0
            
        cos_sim = float(np.dot(disease_vec, drug_perturbation_vec) / (norm_d * norm_p))
        srs = -cos_sim  # Invert sign: positive means disease reversal
        return float(np.clip(srs, -1.0, 1.0))
