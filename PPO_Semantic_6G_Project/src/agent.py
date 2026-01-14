import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# A single central policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_ues, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.num_ues = num_ues
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, action_dim * num_ues)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = x.unsqueeze(1)
        
        output, _ = self.lstm(x)
        output = output.squeeze(1)
        
        output = self.fc2(output)
        output = output.view(-1, self.num_ues, self.action_dim)
        return self.softmax(output)

# A single central value network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = x.unsqueeze(1)
        
        output, _ = self.lstm(x)
        output = output.squeeze(1)
        
        return self.fc2(output)
        
class PPOScheduler:
    def __init__(self, state_dim, action_dim, num_ues, lr=0.0001, gamma=0.999, clip_range=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_ues = num_ues
        self.gamma = gamma
        self.clip_range = clip_range
        self.lr = lr

        self.policy_network = PolicyNetwork(state_dim, action_dim, num_ues)
        self.value_network = ValueNetwork(state_dim)

        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.lr)
        
        self.memory = []

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        policy_output = self.policy_network(state_tensor)
        
        actions = []
        log_probs = []
        for i in range(self.num_ues):
            dist = Categorical(policy_output[0, i, :])
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            actions.append(action.item())
            log_probs.append(log_prob.item())
        
        return actions, log_probs

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        # This update method will now update a single, central network
        # (This section is simplified and would require more complex code)
        self.memory = []

    def save_models(self, path):
        torch.save(self.policy_network.state_dict(), f"{path}_policy.pt")
        torch.save(self.value_network.state_dict(), f"{path}_value.pt")
    
    def load_models(self, path):
        self.policy_network.load_state_dict(torch.load(f"{path}_policy.pt"))
        self.value_network.load_state_dict(torch.load(f"{path}_value.pt"))