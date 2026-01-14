import pytest
import torch
from src.agent import PPOScheduler
import numpy as np

def test_agent_initialization():
    agent = PPOScheduler(state_dim=50, action_dim=16, num_ues=5)
    assert agent.state_dim == 50
    assert agent.action_dim == 16
    assert agent.num_ues == 5

def test_get_action():
    agent = PPOScheduler(state_dim=50, action_dim=16, num_ues=5)
    state = np.random.rand(50)
    action = agent.get_action(state)
    assert action.shape == (5,)  # Action for each UE
    assert all(0 <= a < 16 for a in action)

def test_store_and_update():
    agent = PPOScheduler(state_dim=50, action_dim=16, num_ues=5)
    
    # Store some transitions
    for _ in range(100):
        state = np.random.rand(50)
        action = np.random.randint(0, 16, size=5)
        reward = np.random.rand()
        next_state = np.random.rand(50)
        done = np.random.rand() > 0.9
        agent.store_transition(state, action, reward, next_state, done)
    
    # Test update
    agent.update()  # Should not crash