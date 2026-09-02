"""
TransChemo-Repurpose: Comprehensive Test Suite
Validates chemoinformatics extraction, transcriptomics processing, multi-modal neural network,
scaffold splitting, and XAI explainers.
"""

import pytest
import numpy as np
import torch
import pandas as pd

from src.chemoinformatics.descriptors import MolecularFeatureExtractor
from src.chemoinformatics.scaffold_split import generate_bemis_murcko_scaffold, scaffold_split
from src.transcriptomics.signature_engine import (
    LINCS_LANDMARK_GENES,
    CANCER_PROFILES,
    CURATED_DRUG_LIBRARY,
    TranscriptomicSignatureEngine
)
from src.transcriptomics.pathways import PathwayAnalysisEngine, HALLMARK_PATHWAYS
from src.models.multimodal_net import (
    TranscriptomicsEncoder,
    ChemoinformaticsEncoder,
    CrossModalAttentionFusion,
    TransChemoNet
)
from src.models.reversal_scorer import DrugRepurposingEngine
from src.xai.shap_explainer import ModelExplainer
from src.xai.visualizer import render_molecule_svg
from src.evaluation.metrics import (
    compute_regression_metrics,
    compute_classification_metrics,
    compute_enrichment_factor
)


# --- CHEMOINFORMATICS TESTS ---
def test_molecular_feature_extractor():
    extractor = MolecularFeatureExtractor(fp_nbits=2048)
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    
    # Validation & Canonicalization
    assert extractor.validate_smiles(smiles) is not None
    assert extractor.validate_smiles("INVALID_SMILES_XYZ") is None
    
    # Fingerprints
    fp = extractor.get_morgan_fingerprint(smiles)
    assert fp.shape == (2048,)
    assert np.sum(fp) > 0
    
    maccs = extractor.get_maccs_keys(smiles)
    assert maccs.shape == (166,)
    
    # Descriptors
    props = extractor.get_physicochemical_descriptors(smiles)
    assert "mw" in props
    assert "logp" in props
    assert "tpsa" in props
    assert "qed" in props
    assert props["mw"] > 170.0 and props["mw"] < 190.0
    
    # Full concatenated vector
    full_vec = extractor.get_full_feature_vector(smiles)
    assert full_vec.shape == (2059,)  # 2048 + 11 descriptors


def test_scaffold_splitting():
    smiles_list = [
        "CC1=CC=CC=C1",
        "CC1=CC=CC=C1C",
        "CC1=CC=CC=C1Cl",
        "C1CCCCC1",
        "C1CCCC1",
        "C1CCNCC1",
        "CC(=O)NC1=CC=CC=C1",
        "C1=CN=CC=N1"
    ]
    train_idx, val_idx, test_idx = scaffold_split(smiles_list, train_frac=0.6, val_frac=0.2, test_frac=0.2)
    
    # Assert disjoint splits
    assert len(set(train_idx).intersection(set(val_idx))) == 0
    assert len(set(train_idx).intersection(set(test_idx))) == 0
    assert len(set(val_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(smiles_list)


# --- TRANSCRIPTOMICS TESTS ---
def test_transcriptomics_engine():
    engine = TranscriptomicSignatureEngine()
    assert engine.num_genes == len(LINCS_LANDMARK_GENES)
    
    # Disease vector
    vec_tnbc = engine.get_disease_vector("Triple-Negative Breast Cancer (TNBC)")
    assert vec_tnbc.shape == (engine.num_genes,)
    assert np.isclose(np.linalg.norm(vec_tnbc), 1.0, atol=1e-3)
    
    # Drug perturbation vector
    drug = CURATED_DRUG_LIBRARY[0]
    vec_drug = engine.get_drug_perturbation_vector(drug)
    assert vec_drug.shape == (engine.num_genes,)
    
    # Signature Reversion Score
    srs = engine.compute_signature_reversion_score(vec_tnbc, vec_drug)
    assert -1.0 <= srs <= 1.0


def test_pathway_engine():
    engine = PathwayAnalysisEngine(LINCS_LANDMARK_GENES)
    dummy_expr = np.random.normal(0, 1, size=len(LINCS_LANDMARK_GENES))
    activities = engine.calculate_pathway_activity(dummy_expr)
    
    assert len(activities) == len(HALLMARK_PATHWAYS)
    for p, score in activities.items():
        assert isinstance(score, float)
        
    df_reversal = engine.compute_pathway_reversal(dummy_expr, -dummy_expr)
    assert not df_reversal.empty
    assert "Reversal %" in df_reversal.columns


# --- MULTI-MODAL NEURAL NETWORK TESTS ---
def test_transchemo_net_forward():
    batch_size = 4
    trans_dim = 100
    chem_dim = 2059
    latent_dim = 128
    
    model = TransChemoNet(trans_dim=trans_dim, chem_dim=chem_dim, latent_dim=latent_dim)
    model.eval()
    
    x_trans = torch.randn(batch_size, trans_dim)
    x_chem = torch.randn(batch_size, chem_dim)
    
    out = model(x_trans, x_chem)
    
    assert "srs" in out
    assert "pic50" in out
    assert "reversal_prob" in out
    assert "fused_embedding" in out
    assert "attn_weights" in out
    
    assert out["srs"].shape == (batch_size,)
    assert out["pic50"].shape == (batch_size,)
    assert out["reversal_prob"].shape == (batch_size,)
    assert out["fused_embedding"].shape == (batch_size, latent_dim)
    
    # Test output bounds
    assert torch.all(out["srs"] >= -1.0) and torch.all(out["srs"] <= 1.0)
    assert torch.all(out["reversal_prob"] >= 0.0) and torch.all(out["reversal_prob"] <= 1.0)


# --- REPURPOSING SCREENING & XAI TESTS ---
def test_repurposing_engine_screen():
    engine = DrugRepurposingEngine()
    df_results = engine.screen_library("Triple-Negative Breast Cancer (TNBC)", top_k=5)
    
    assert len(df_results) == 5
    assert "Drug Name" in df_results.columns
    assert "Composite Repurposing Score" in df_results.columns
    assert "Predicted SRS (Reversal)" in df_results.columns
    assert df_results["Composite Repurposing Score"].is_monotonic_decreasing


def test_explainer_and_visualizer():
    explainer = ModelExplainer()
    dummy_chem = np.random.randn(2059)
    dummy_trans = np.random.randn(100)
    
    exp = explainer.explain_instance(dummy_chem, dummy_trans, LINCS_LANDMARK_GENES)
    assert "top_positive_genes" in exp
    assert "chemical_attribution" in exp
    assert len(exp["top_positive_genes"]) > 0
    
    svg = render_molecule_svg("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert "<svg" in svg
    assert "</svg>" in svg


# --- METRICS & BENCHMARKS TESTS ---
def test_evaluation_metrics():
    y_true_reg = np.array([0.5, 0.8, -0.2, 0.1, 0.9])
    y_pred_reg = np.array([0.45, 0.75, -0.15, 0.12, 0.85])
    reg_metrics = compute_regression_metrics(y_true_reg, y_pred_reg)
    assert reg_metrics["RMSE"] < 0.1
    assert reg_metrics["Pearson r"] > 0.95
    
    y_true_cls = np.array([1, 1, 0, 0, 1])
    y_prob_cls = np.array([0.9, 0.8, 0.1, 0.2, 0.85])
    cls_metrics = compute_classification_metrics(y_true_cls, y_prob_cls)
    assert cls_metrics["AUROC"] == 1.0
    
    ef = compute_enrichment_factor(y_true_cls, y_prob_cls, top_fraction=0.4)
    assert ef >= 1.0
