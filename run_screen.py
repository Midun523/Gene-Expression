"""
TransChemo-Repurpose: Command-Line Virtual Screening & Repurposing Tool
Enables fast terminal-based execution of multi-modal drug repurposing screens.

Usage:
    python run_screen.py --cancer "Triple-Negative Breast Cancer (TNBC)" --top_k 10
"""

import argparse
import sys
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.transcriptomics.signature_engine import CANCER_PROFILES, TranscriptomicSignatureEngine
from src.models.reversal_scorer import DrugRepurposingEngine
from src.chemoinformatics.descriptors import MolecularFeatureExtractor


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        
    parser = argparse.ArgumentParser(
        description="TransChemo-Repurpose: Multi-Modal AI for Cancer Drug Repurposing"
    )
    parser.add_argument(
        "--cancer",
        type=str,
        default="Triple-Negative Breast Cancer (TNBC)",
        choices=list(CANCER_PROFILES.keys()),
        help="Target cancer indication profile"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top drug repurposing candidates to return"
    )
    
    args = parser.parse_args()
    console = Console(highlight=False)

    console.print("\n[bold cyan]>>> Initializing TransChemo-Repurpose AI Engine...[/bold cyan]")
    engine = DrugRepurposingEngine()
    
    console.print(f"[bold green]Target Indication:[/bold green] [yellow]{args.cancer}[/yellow]")
    console.print(f"[dim]{CANCER_PROFILES[args.cancer]['description']}[/dim]\n")
    
    with console.status("[bold blue]Running Multi-Modal Screening across Drug Library...[/bold blue]"):
        df_results = engine.screen_library(disease_profile_or_vector=args.cancer, top_k=args.top_k)

    # Format Rich Table for Terminal Output
    table = Table(title=f"Top {args.top_k} Repurposed Candidates for {args.cancer}", show_lines=True)
    table.add_column("Rank", justify="center", style="dim")
    table.add_column("Drug Name", style="bold white")
    table.add_column("FDA Status", justify="center", style="cyan")
    table.add_column("Primary Target", style="magenta")
    table.add_column("Pred SRS", justify="right", style="green")
    table.add_column("Pred pIC50", justify="right", style="yellow")
    table.add_column("Efficacy %", justify="right", style="bold blue")
    table.add_column("Composite Score", justify="right", style="bold green")

    for i, row in df_results.iterrows():
        table.add_row(
            str(i + 1),
            str(row["Drug Name"]),
            str(row["FDA Status"]),
            str(row["Primary Target"]),
            f"{row['Predicted SRS (Reversal)']:.3f}",
            f"{row['Predicted pIC50']:.2f}",
            f"{row['Efficacy Probability']:.1f}%",
            f"{row['Composite Repurposing Score']:.3f}"
        )

    console.print(table)
    console.print("\n[bold green][OK] Screening Complete.[/bold green] To launch interactive UI, run: [bold cyan]streamlit run web_app/app.py[/bold cyan]\n")


if __name__ == "__main__":
    main()
