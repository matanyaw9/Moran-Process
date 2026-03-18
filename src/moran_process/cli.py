import typer
from pathlib import Path
import os
import sys

# Ensure the package can find its own modules if run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your actual logic
from moran_process.pipeline.main import main as run_master_batch
from moran_process.pipeline.merge_batches import merge_batches as merge_logic
from moran_process.analysis.analysis_utils import process_large_batch_polars

app = typer.Typer(
    name="moran", 
    help="CLI for Evolutionary Dynamics on Graph Topologies",
    add_completion=False
)

ROOT = Path(os.getcwd())
DATA_DIR = ROOT / "simulation_data"

@app.command()
def run(
    batch_name: str = typer.Option(..., "--batch-name", "-b", help="Name of the new batch"),
):
    """
    Generate a graph zoo and submit simulation jobs to the cluster.
    """
    typer.secho(f"Starting simulation pipeline for batch: {batch_name}", fg=typer.colors.GREEN)
    run_master_batch(batch_name=batch_name)


@app.command()
def merge(
    batch1: str = typer.Argument(..., help="Name of the first batch"),
    batch2: str = typer.Argument(..., help="Name of the second batch"),
    out: str = typer.Option(..., "--out", "-o", help="Name of the new merged batch"),
):
    """
    Merge two existing simulation batches safely using Polars.
    """
    typer.secho(f"Merging {batch1} and {batch2} into {out}...", fg=typer.colors.CYAN)
    
    # Backwards compatibility trick for old pickles before executing the merge
    from moran_process.core import population_graph
    sys.modules['population_graph'] = population_graph
    
    merge_logic(batch1, batch2, out)


@app.command()
def aggregate(
    batch_name: str = typer.Argument(..., help="Name of the batch to aggregate"),
):
    """
    Process massive worker CSVs into a single lightweight statistics file.
    """
    typer.secho(f"Aggregating data for {batch_name}...", fg=typer.colors.MAGENTA)
    batch_dir = DATA_DIR / batch_name
    if not batch_dir.exists():
        typer.secho(f"Error: Directory {batch_dir} not found.", fg=typer.colors.RED)
        raise typer.Exit(1)
        
    process_large_batch_polars(batch_dir)


@app.command()
def train_ml(
    batch_name: str = typer.Argument(..., help="Batch to train on"),
    target: str = typer.Option("prob_fixation", help="Target column to predict")
):
    """
    (WIP) Train an XGBoost model on the graph statistics.
    """
    typer.secho(f"Coming soon: Training ML model for {batch_name} targeting {target}...", fg=typer.colors.YELLOW)
    # TODO: Import and call your ML pipeline here


@app.command()
def evolve():
    """
    (WIP) Run an Evolutionary Algorithm to find extreme graphs.
    """
    typer.secho("Coming soon: Evolutionary algorithm...", fg=typer.colors.YELLOW)
    # TODO: Import and call extreme_graphs.py logic here


if __name__ == "__main__":
    app()