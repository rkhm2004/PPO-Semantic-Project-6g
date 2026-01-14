import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

class RawLstmExtractor(nn.Module):
    """
    SB3-compatible extractor:
      - forward(x) -> (latent_pi, latent_vf)
      - forward_actor(x) -> latent_pi
      - forward_critic(x) -> latent_vf
    """
    def __init__(self, feature_dim: int, lstm_hidden: int = 128,
                 latent_pi: int = 64, latent_vf: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=lstm_hidden, batch_first=True)

        self.pi = nn.Sequential(nn.Linear(lstm_hidden, latent_pi), nn.ReLU())
        self.vf = nn.Sequential(nn.Linear(lstm_hidden, latent_vf), nn.ReLU())

        self.latent_dim_pi = latent_pi
        self.latent_dim_vf = latent_vf

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F] -> treat as sequence length 1
        x = x.unsqueeze(1)           # [B, 1, F]
        y, _ = self.lstm(x)          # [B, 1, H]
        return y[:, -1, :]           # [B, H]

    def forward(self, x: torch.Tensor):
        y = self._encode(x)
        return self.pi(y), self.vf(y)

    def forward_actor(self, x: torch.Tensor):
        y = self._encode(x)
        return self.pi(y)

    def forward_critic(self, x: torch.Tensor):
        y = self._encode(x)
        return self.vf(y)

class RawLstmPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.ortho_init = True

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = RawLstmExtractor(feature_dim=self.features_dim)
