"""A small MLP policy with self-contained observation normalization."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ilgrasp.sim import ACTION_DIM, OBS_DIM


class GraspPolicy(nn.Module):
    """Map a 22-value observation to a bounded 4-value action."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        if not hidden_sizes or any(width <= 0 for width in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = tuple(int(width) for width in hidden_sizes)

        layers: list[nn.Module] = []
        input_width = obs_dim
        for width in self.hidden_sizes:
            layers.extend((nn.Linear(input_width, width), nn.ReLU()))
            input_width = width
        layers.extend((nn.Linear(input_width, action_dim), nn.Tanh()))
        self.network = nn.Sequential(*layers)

        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def set_normalization(self, observations: np.ndarray | torch.Tensor) -> None:
        values = torch.as_tensor(observations, dtype=torch.float32, device=self.obs_mean.device)
        if values.ndim != 2 or values.shape[1] != self.obs_dim:
            raise ValueError(f"Expected observations with shape (N, {self.obs_dim})")
        self.obs_mean.copy_(values.mean(dim=0))
        self.obs_std.copy_(values.std(dim=0, unbiased=False).clamp_min(1e-6))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        normalized = (observations - self.obs_mean) / self.obs_std
        return self.network(normalized)

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "hidden_sizes": self.hidden_sizes,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "GraspPolicy":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(
            obs_dim=checkpoint["obs_dim"],
            action_dim=checkpoint["action_dim"],
            hidden_sizes=checkpoint["hidden_sizes"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        return model


class NumpyPolicy:
    """Adapter that makes a PyTorch model callable from the simulator loop."""

    def __init__(self, model: GraspPolicy, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            action = self.model(tensor).squeeze(0)
        return action.cpu().numpy().astype(np.float32)

