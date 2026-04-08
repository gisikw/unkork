"""MLP architecture for the voice regression codec."""

import torch
import torch.nn as nn

RESEMBLYZER_DIM = 256
DEFAULT_HIDDEN = 512
DEFAULT_PCA_COMPONENTS = 50


class VoiceCodec(nn.Module):
    """Maps Resemblyzer speaker embeddings to PCA-projected Kokoro voice tensors.

    Architecture: Linear → ReLU → Linear → ReLU → Linear
    """

    def __init__(
        self,
        input_dim: int = RESEMBLYZER_DIM,
        hidden_dim: int = DEFAULT_HIDDEN,
        output_dim: int = DEFAULT_PCA_COMPONENTS,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_checkpoint(model: VoiceCodec, path: str, **metadata: object) -> None:
    """Save model weights and metadata."""
    torch.save({"state_dict": model.state_dict(), **metadata}, path)


def load_checkpoint(path: str) -> tuple[VoiceCodec, dict]:
    """Load model weights and metadata. Returns (model, metadata_dict)."""
    checkpoint = torch.load(path, weights_only=False)
    state_dict = checkpoint.pop("state_dict")

    # Infer dims from weight shapes
    first_weight = state_dict["net.0.weight"]
    last_weight = state_dict["net.4.weight"]
    hidden_weight = state_dict["net.2.weight"]
    model = VoiceCodec(
        input_dim=first_weight.shape[1],
        hidden_dim=hidden_weight.shape[0],
        output_dim=last_weight.shape[0],
    )
    model.load_state_dict(state_dict)
    return model, checkpoint
