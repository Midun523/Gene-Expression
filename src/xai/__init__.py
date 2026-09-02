"""
Explainable AI (XAI) module for TransChemo-Repurpose.
"""

from .shap_explainer import ModelExplainer
from .visualizer import render_molecule_svg

__all__ = [
    "ModelExplainer",
    "render_molecule_svg"
]
