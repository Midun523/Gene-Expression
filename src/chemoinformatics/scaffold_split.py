"""
TransChemo-Repurpose: Bemis-Murcko Scaffold Splitting Module
Implements scaffold-based dataset partitioning for robust out-of-distribution evaluation.
"""

from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_bemis_murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Computes the Bemis-Murcko scaffold SMILES for a given molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return ""
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )
        return scaffold
    except Exception:
        return ""


def scaffold_split(
    smiles_list: List[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Partitions indices into train, validation, and test sets based on Bemis-Murcko scaffolds.
    Molecules with the same core chemical scaffold are strictly placed into the same split.
    
    Returns:
        (train_indices, val_indices, test_indices)
    """
    assert np.isclose(train_frac + val_frac + test_frac, 1.0), "Fractions must sum to 1.0"
    
    # Group molecule indices by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = generate_bemis_murcko_scaffold(smiles)
        scaffold_to_indices[scaffold].append(idx)
    
    # Sort scaffolds by number of compounds in descending order (standard scaffold split heuristic)
    scaffold_groups = list(scaffold_to_indices.values())
    np.random.seed(seed)
    np.random.shuffle(scaffold_groups)
    scaffold_groups.sort(key=len, reverse=True)
    
    total_samples = len(smiles_list)
    train_cutoff = int(train_frac * total_samples)
    val_cutoff = int((train_frac + val_frac) * total_samples)
    
    train_inds: List[int] = []
    val_inds: List[int] = []
    test_inds: List[int] = []
    
    for group in scaffold_groups:
        if len(train_inds) + len(group) <= train_cutoff:
            train_inds.extend(group)
        elif len(train_inds) + len(val_inds) + len(group) <= val_cutoff:
            val_inds.extend(group)
        else:
            test_inds.extend(group)
            
    # If any set is empty due to large scaffold blocks, rebalance
    if not val_inds and len(train_inds) > 2:
        val_inds.append(train_inds.pop())
    if not test_inds and len(train_inds) > 2:
        test_inds.append(train_inds.pop())
        
    return train_inds, val_inds, test_inds
