"""
TransChemo-Repurpose: Baseline Comparative Models Module
Implements classical machine learning baselines and traditional CMap heuristic
for rigorous benchmarking against the Multi-Modal TransChemoNet.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
import xgboost as xgb


class BaselineSuite:
    """Benchmark suite providing training, inference, and evaluation for classical models."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1)
        self.rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1)
        self.xgb_regressor = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=random_state)
        self.xgb_classifier = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=random_state)
        self.ridge_regressor = Ridge(alpha=1.0)
        self.log_reg = LogisticRegression(max_iter=500, random_state=random_state)

    def train_all(self, X_train: np.ndarray, y_srs_train: np.ndarray, y_bin_train: np.ndarray):
        """Fits all baseline models on concatenated features (Transcriptomics + Chemoinformatics)."""
        self.rf_regressor.fit(X_train, y_srs_train)
        self.rf_classifier.fit(X_train, y_bin_train)
        
        self.xgb_regressor.fit(X_train, y_srs_train)
        self.xgb_classifier.fit(X_train, y_bin_train)
        
        self.ridge_regressor.fit(X_train, y_srs_train)
        self.log_reg.fit(X_train, y_bin_train)

    def predict_all(self, X_test: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Generates predictions across all baselines."""
        return {
            "Random Forest": {
                "srs_pred": self.rf_regressor.predict(X_test),
                "prob_pred": self.rf_classifier.predict_proba(X_test)[:, 1]
            },
            "XGBoost": {
                "srs_pred": self.xgb_regressor.predict(X_test),
                "prob_pred": self.xgb_classifier.predict_proba(X_test)[:, 1]
            },
            "Ridge / Logistic Reg": {
                "srs_pred": self.ridge_regressor.predict(X_test),
                "prob_pred": self.log_reg.predict_proba(X_test)[:, 1]
            }
        }


def classical_cmap_score(disease_sig: np.ndarray, drug_sig: np.ndarray) -> float:
    """
    Simulates the standard non-parametric Connectivity Map (CMap) reversal score
    via negative normalized Spearman rank correlation.
    """
    from scipy.stats import spearmanr
    corr, _ = spearmanr(disease_sig, drug_sig)
    if np.isnan(corr):
        return 0.0
    return float(-corr)
