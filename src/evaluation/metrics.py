"""
TransChemo-Repurpose: Evaluation & Benchmarking Module
Computes regression (RMSE, Pearson r, Spearman rho, R2), classification (AUROC, AUPRC, F1),
and cheminformatics screening metrics (Enrichment Factor EF, BedROC).
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score
)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes standard continuous evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    p_corr, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0.0, 0.0)
    s_corr, _ = spearmanr(y_true, y_pred) if len(y_true) > 1 else (0.0, 0.0)
    
    return {
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "R2 Score": round(float(r2), 4),
        "Pearson r": round(float(p_corr), 4),
        "Spearman rho": round(float(s_corr), 4)
    }


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes binary classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = 0.5
        
    try:
        auprc = average_precision_score(y_true, y_prob)
    except Exception:
        auprc = float(np.mean(y_true))
        
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "AUROC": round(float(auroc), 4),
        "AUPRC": round(float(auprc), 4),
        "F1 Score": round(float(f1), 4),
        "Accuracy": round(float(acc), 4),
        "Balanced Accuracy": round(float(bal_acc), 4)
    }


def compute_enrichment_factor(y_true: np.ndarray, y_scores: np.ndarray, top_fraction: float = 0.1) -> float:
    """
    Computes Enrichment Factor (EF_x%) for virtual screening hit recovery.
    EF = (Actives in top x%) / (Total Actives * x%)
    """
    n_samples = len(y_true)
    n_top = max(1, int(n_samples * top_fraction))
    
    # Sort descending by predicted score
    order = np.argsort(y_scores)[::-1]
    top_labels = y_true[order[:n_top]]
    
    total_actives = np.sum(y_true)
    if total_actives == 0:
        return 1.0
        
    actives_in_top = np.sum(top_labels)
    ef = (actives_in_top / n_top) / (total_actives / n_samples)
    return round(float(ef), 3)


def generate_benchmark_summary(
    results_dict: Dict[str, Dict[str, any]]
) -> pd.DataFrame:
    """Converts multi-model benchmark results into a clean comparison DataFrame."""
    rows = []
    for model_name, metrics in results_dict.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)
        
    df = pd.DataFrame(rows)
    return df
