"""
TransChemo-Repurpose: Multi-Modal Cross-Attention Deep Neural Network
Unifies high-dimensional transcriptomics signatures with molecular cheminformatics features
to predict drug reversal scores, bioactivity (pIC50), and therapeutic efficacy.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TranscriptomicsEncoder(nn.Module):
    """Encodes gene expression differential vectors into a latent embedding space."""

    def __init__(self, input_dim: int = 100, latent_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, latent_dim)
        self.ln2 = nn.LayerNorm(latent_dim)
        self.residual = nn.Linear(input_dim, latent_dim) if input_dim != latent_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        h = self.act(self.ln1(self.fc1(x)))
        h = self.dropout(h)
        h = self.ln2(self.fc2(h))
        return self.act(h + res)


class ChemoinformaticsEncoder(nn.Module):
    """Encodes molecular fingerprints and physicochemical descriptors into a latent embedding."""

    def __init__(self, input_dim: int = 2059, latent_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, latent_dim)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = self.act(self.bn2(self.fc2(h)))
        h = self.dropout(h)
        h = self.act(self.bn3(self.fc3(h)))
        return h


class CrossModalAttentionFusion(nn.Module):
    """Multi-Head Cross-Modal Attention Layer fusing biology and chemistry representations."""

    def __init__(self, latent_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.mha = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )
        self.fusion_fc = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, h_trans: torch.Tensor, h_chem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Stack as sequence: [batch_size, 2, latent_dim]
        seq = torch.stack([h_trans, h_chem], dim=1)
        attn_out, attn_weights = self.mha(seq, seq, seq)
        attn_out = self.norm(attn_out + seq)
        
        t_attended = attn_out[:, 0, :]
        c_attended = attn_out[:, 1, :]
        
        # Gated fusion
        concat = torch.cat([t_attended, c_attended], dim=-1)
        g = self.gate(concat)
        fused = g * t_attended + (1.0 - g) * c_attended
        fused = self.fusion_fc(torch.cat([fused, h_trans + h_chem], dim=-1))
        
        return fused, attn_weights


class TransChemoNet(nn.Module):
    """
    End-to-End Multi-Modal Neural Network.
    Predicts:
      1. Signature Reversal Score (SRS in [-1, 1])
      2. Bioactivity / Sensitivity (pIC50 regression)
      3. Efficacy Classification (Probability of therapeutic reversal)
    """

    def __init__(
        self,
        trans_dim: int = 100,
        chem_dim: int = 2059,
        latent_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.25
    ):
        super().__init__()
        self.trans_encoder = TranscriptomicsEncoder(input_dim=trans_dim, latent_dim=latent_dim, dropout=dropout)
        self.chem_encoder = ChemoinformaticsEncoder(input_dim=chem_dim, latent_dim=latent_dim, dropout=dropout)
        self.fusion = CrossModalAttentionFusion(latent_dim=latent_dim, num_heads=num_heads)
        
        # Multi-task Prediction Heads
        self.srs_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
            nn.Tanh()  # Output in [-1.0, +1.0]
        )
        
        self.bioactivity_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1)  # pIC50 continuous regression
        )
        
        self.classifier_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Binary reversal probability
        )

    def forward(
        self, x_trans: torch.Tensor, x_chem: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        h_trans = self.trans_encoder(x_trans)
        h_chem = self.chem_encoder(x_chem)
        
        fused, attn_weights = self.fusion(h_trans, h_chem)
        
        srs_pred = self.srs_head(fused).squeeze(-1)
        pic50_pred = self.bioactivity_head(fused).squeeze(-1)
        prob_pred = self.classifier_head(fused).squeeze(-1)
        
        return {
            "srs": srs_pred,
            "pic50": pic50_pred,
            "reversal_prob": prob_pred,
            "fused_embedding": fused,
            "attn_weights": attn_weights
        }
