import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment import SAMAEnvironment
import pytest
import numpy as np

def test_environment_initialization():
    env = SAMAEnvironment()
    assert env.num_channels == 16
    assert len(env.ue_states) > 0
    assert env.current_slot == 0

def test_reset():
    env = SAMAEnvironment()
    state = env.reset()
    assert isinstance(state, np.ndarray)
    assert len(state) > 0
    assert env.current_slot == 0

def test_step():
    env = SAMAEnvironment()
    env.reset()
    action = np.zeros(len(env.ue_states), dtype=int)  # All UEs choose channel 0
    state, reward, done, info = env.step(action)
    
    assert isinstance(state, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert 'urllc_success_rate' in info
    assert 'mmtc_throughput' in info
    assert 'spectral_efficiency' in info