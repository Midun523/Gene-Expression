"""
TransChemo-Repurpose: Visualizer Module
Provides 2D SVG molecular rendering, pharmacophore highlighting, and expression plotting.
"""

from typing import List, Optional
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D


def render_molecule_svg(smiles: str, width: int = 400, height: int = 300, highlight_substructure: Optional[str] = None) -> str:
    """
    Renders a 2D depiction of a molecule into clean SVG string format.
    Optionally highlights substructure matches.
    """
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return "<svg width='400' height='300'><text x='50' y='150' fill='gray'>Invalid SMILES</text></svg>"
            
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = False
        opts.bondLineWidth = 2
        
        highlight_atoms = []
        if highlight_substructure:
            sub = Chem.MolFromSmarts(highlight_substructure)
            if sub and mol.HasSubstructMatch(sub):
                matches = mol.GetSubstructMatches(sub)
                for match in matches:
                    highlight_atoms.extend(match)
                    
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms if highlight_atoms else None)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg
    except Exception as e:
        return f"<svg width='400' height='300'><text x='20' y='150' fill='red'>Error: {str(e)}</text></svg>"
