import click


@click.group()
def main():
    """unkork — regression codec for Kokoro TTS voice creation."""
    pass


@main.command()
@click.option("--voices-dir", required=True, help="Path to Kokoro voice .pt files")
@click.option("--output", default="data/", help="Output directory for training data")
@click.option("--count", default=500, help="Number of synthetic voices to generate")
def generate(voices_dir: str, output: str, count: int):
    """Generate training dataset from existing Kokoro voices."""
    click.echo(f"Generating {count} synthetic voices from {voices_dir} -> {output}")


@main.command()
@click.option("--data", default="data/", help="Training data directory")
@click.option("--output", default="models/", help="Output directory for trained model")
@click.option("--epochs", default=200, help="Training epochs")
def train(data: str, output: str, epochs: int):
    """Train the regression codec MLP."""
    click.echo(f"Training codec on {data} for {epochs} epochs -> {output}")


@main.command()
@click.option("--model", required=True, help="Path to trained codec model")
@click.option("--audio", required=True, help="Reference audio file")
@click.option("--output", default="voice.pt", help="Output voice tensor path")
def predict(model: str, audio: str, output: str):
    """Predict a Kokoro voice tensor from reference audio."""
    click.echo(f"Predicting voice tensor from {audio} using {model} -> {output}")


@main.command()
@click.option("--start", required=True, help="Starting voice tensor (.pt)")
@click.option("--target", required=True, help="Target reference audio")
@click.option("--iterations", default=50, help="CMA-ES iterations")
@click.option("--output", default="refined.pt", help="Output refined tensor path")
def refine(start: str, target: str, iterations: int, output: str):
    """Refine a voice tensor with CMA-ES optimization."""
    click.echo(f"Refining {start} toward {target} for {iterations} iterations -> {output}")
