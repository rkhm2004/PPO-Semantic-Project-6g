# ma_policies.py — CTDE (MAPPO) Actor-Critic for SB3 PPO
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


class CTDEExtractor(nn.Module):
    """
    CTDE feature extractor used as SB3's 'mlp_extractor':
      - Input: features (B, F) where F = num_ues * local_obs_dim
      - Actor: reshape to (B, N, L), apply shared MLP per agent, then flatten -> latent_pi (B, N*H)
      - Critic: centralized MLP over the joint input -> latent_vf (B, Hc)
    """
    def __init__(self, feature_dim: int, num_ues: int, local_obs_dim: int,
                 actor_hidden: int = 64, critic_hidden: int = 128):
        super().__init__()
        self.num_ues = num_ues
        self.local_obs_dim = local_obs_dim

        # shared actor head over local obs
        self.actor_local = nn.Sequential(
            nn.Linear(local_obs_dim, actor_hidden),
            nn.ReLU(),
            nn.Linear(actor_hidden, actor_hidden),
            nn.ReLU(),
        )
        # centralized critic over concatenated obs
        self.critic_net = nn.Sequential(
            nn.Linear(feature_dim, critic_hidden),
            nn.ReLU(),
            nn.Linear(critic_hidden, critic_hidden),
            nn.ReLU(),
        )

        # Expose latent sizes expected by SB3
        self.latent_dim_pi = actor_hidden * num_ues  # concatenated per-agent latents
        self.latent_dim_vf = critic_hidden

    def forward(self, x: torch.Tensor):
        # x: (B, F) -> actor: (B, N, L)
        B = x.shape[0]
        N = self.num_ues
        L = self.local_obs_dim

        x_agents = x.view(B, N, L)  # split per agent
        # Apply shared local head to each agent (vectorized)
        y = self.actor_local(x_agents)  # (B, N, H)
        latent_pi = y.reshape(B, -1)    # (B, N*H)

        # Critic centralized features
        latent_vf = self.critic_net(x)  # (B, Hc)

        return latent_pi, latent_vf

    # SB3 sometimes calls these explicitly:
    def forward_actor(self, x: torch.Tensor):
        B = x.shape[0]; N = self.num_ues; L = self.local_obs_dim
        y = self.actor_local(x.view(B, N, L))
        return y.reshape(B, -1)

    def forward_critic(self, x: torch.Tensor):
        return self.critic_net(x)


class CTDEPolicy(ActorCriticPolicy):
    """
    CTDE policy for MAPPO-style PPO in SB3:
    - action_space: MultiDiscrete([num_channels]*num_ues)
    - observation: concatenated local obs (num_ues * local_obs_dim)
    """
class CTDEPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, num_ues=6, local_obs_dim=4, **kwargs):
        # Set custom attributes *before* SB3 builds the network
        self.num_ues = num_ues
        self.local_obs_dim = local_obs_dim
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.ortho_init = True


    def _build_mlp_extractor(self) -> None:
        # features_dim provided by SB3 (FlattenExtractor result); should equal num_ues*local_obs_dim
        self.mlp_extractor = CTDEExtractor(
            feature_dim=self.features_dim,
            num_ues=self.num_ues,
            local_obs_dim=self.local_obs_dim,
            actor_hidden=64,
            critic_hidden=128,
        )
