import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple # <-- THIS IS THE LINE THAT WAS ADDED

class DecentralizedActorSAC(nn.Module):
    """
    The Actor network for a single agent. (This class is already correct)
    """
    def __init__(self, obs_dim, action_dim, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def get_action(self, obs, deterministic=False):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action, log_prob

class CentralizedCriticSAC(nn.Module):
    """
    The centralized Critic network, corrected to accept a single global observation.
    """
    def __init__(self, num_agents, obs_dim, action_dim):
        super().__init__()
        
        # Input is the global observation dim + all agents' action dims
        input_dim = obs_dim + (num_agents * action_dim)
        
        # Twin Q-network 1
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Twin Q-network 2
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, global_obs: torch.Tensor, all_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate the single global obs with all actions
        x = torch.cat([global_obs, all_actions.reshape(all_actions.shape[0], -1)], dim=1)
        return self.q1(x), self.q2(x)