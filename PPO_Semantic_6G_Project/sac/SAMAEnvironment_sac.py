import numpy as np
import pandas as pd
import json
from collections import defaultdict
import gymnasium as gym
from gymnasium import spaces
import os

class SAMAEnvironment_SAC(gym.Env):
    def __init__(self, is_real_time=False, num_ues=6, num_channels=3, **kwargs):
        super(SAMAEnvironment_SAC, self).__init__()
        
        self.num_ues = num_ues
        self.num_channels = num_channels
        self.config = self._load_config(kwargs.get('config_path', 'data/channel_params.json'))
        
        # (Data loading logic remains the same)
        if is_real_time:
            self.initial_states_path = kwargs.get('state_path', 'data/real/real_initial_states.csv')
            self.semantic_path = kwargs.get('semantic_path', 'data/real/real_semantic_segments.csv')
            self.traffic_path = kwargs.get('traffic_path', 'data/real/real_traffic_model.csv')
        else:
            self.initial_states_path = kwargs.get('state_path', 'data/controlled/controlled_initial_states.csv')
            self.semantic_path = kwargs.get('semantic_path', 'data/controlled/controlled_semantic_segments.csv')
            self.traffic_path = kwargs.get('traffic_path', 'data/controlled/controlled_traffic_model.csv')

        self.original_ue_states = self._load_initial_states(self.initial_states_path)
        self.original_semantic_segments = self._load_semantic_data(self.semantic_path)
        self.original_traffic_model = self._load_traffic_model(self.traffic_path)
        self.max_slots = len(self.original_traffic_model)
        
        state_dim = self.num_channels + (self.num_ues * 14) + 3
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(state_dim,), dtype=np.float32)
        
        # --- ACTION SPACE FIX: Use a continuous Box space for SAC ---
        # The agent will output a continuous value for each UE, which we will round to the nearest integer.
        self.action_space = spaces.Box(low=0, high=self.num_channels - 1, shape=(self.num_ues,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # (Reset logic remains the same)
        super().reset(seed=seed)
        self.ue_states = self.original_ue_states.copy()
        self.current_slot = 0
        self.channel_states = np.ones(self.num_channels)
        self.collisions = np.zeros(self.num_channels)
        self.spectral_efficiency = 0
        self.urllc_stats = {'success': 0, 'failures': 0}
        self.mmtc_stats = {'throughput': 0, 'packets': 0}
        
        if self.current_slot in self.original_traffic_model.index:
            traffic = self.original_traffic_model.loc[self.current_slot]
            self.ue_states['active'] = [bool(traffic[f'ue_{i}']) for i in range(len(self.ue_states))]
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        # (This method remains the same)
        state_parts = []
        state_parts.append(self.channel_states)
        ue_features = []
        total_d2lt = self.ue_states['d2lt'].sum()
        normalized_d2lt = np.zeros(len(self.ue_states))
        if total_d2lt > 0:
            normalized_d2lt = self.ue_states['d2lt'] / total_d2lt
        for idx, ue in self.ue_states.iterrows():
            ue_features.extend([
                1 if ue['active'] else 0,
                1 if ue['traffic_type'] == 'URLLC' else 0,
                ue['priority'] / 10.0,
                normalized_d2lt[idx]
            ])
            ue_segments = self.original_semantic_segments[self.original_semantic_segments['ue_id'] == ue['ue_id']]['segment_id'].tolist()
            segment_vector = [1 if seg in ue_segments else 0 for seg in range(10)]
            ue_features.extend(segment_vector)
        state_parts.append(np.array(ue_features))
        global_metrics = [
            self.spectral_efficiency,
            self.urllc_stats['success'] / (self.urllc_stats['success'] + self.urllc_stats['failures'] + 1e-6),
            self.mmtc_stats['throughput'] / (self.mmtc_stats['packets'] + 1e-6)
        ]
        state_parts.append(np.array(global_metrics))
        return np.concatenate(state_parts).astype(np.float32)

    def step(self, action: np.ndarray):
        # --- ACTION CONVERSION: Convert continuous action from SAC to discrete choices ---
        # Round to the nearest integer and clip to be within the valid channel range
        discrete_action = np.round(action).astype(int)
        discrete_action = np.clip(discrete_action, 0, self.num_channels - 1)
        
        # (The rest of the step logic uses the new discrete_action)
        reward = 0
        channel_usage = defaultdict(list)
        self.ue_states['d2lt'] += 1
        served_ue_ids = set()

        for ue_id, channel in enumerate(discrete_action):
            ue = self.ue_states.iloc[ue_id]
            if not ue['active']: continue
            channel_usage[channel].append(ue_id)
        
        # (The rest of the reward and state update logic is the same as before)
        for channel, ue_ids in channel_usage.items():
            if len(ue_ids) > 1:
                self.collisions[channel] += 1
                for ue_id in ue_ids:
                    reward -= 100 if self.ue_states.iloc[ue_id]['traffic_type'] == 'URLLC' else 50
            elif len(ue_ids) == 1:
                ue_id = ue_ids[0]
                served_ue_ids.add(ue_id)
                ue = self.ue_states.iloc[ue_id]
                d2lt_reward_factor = self.ue_states.at[ue_id, 'd2lt'] / self.max_slots if self.max_slots > 0 else 0
                self.ue_states.at[ue_id, 'd2lt'] = 0
                if ue['traffic_type'] == 'URLLC':
                    self.urllc_stats['success'] += 1
                    reward += (100 + 50 * d2lt_reward_factor)
                else:
                    self.mmtc_stats['packets'] += 1
                    self.mmtc_stats['throughput'] += 1
                    reward += (30 + 10 * d2lt_reward_factor)
        
        for idx, ue in self.ue_states.iterrows():
            if ue['active'] and ue['traffic_type'] == 'mMTC' and idx not in served_ue_ids:
                reward -= 10
        reward += (self.num_channels - len(channel_usage)) * 5
        
        self.channel_states = np.random.binomial(1, 0.8, size=self.num_channels)
        successful_ues = self.urllc_stats['success'] + self.mmtc_stats['packets']
        self.spectral_efficiency = successful_ues / (self.num_channels * (self.current_slot + 1))
        
        self.current_slot += 1
        if self.current_slot < len(self.original_traffic_model) and self.current_slot in self.original_traffic_model.index:
            traffic = self.original_traffic_model.loc[self.current_slot]
            for i in range(len(self.ue_states)):
                self.ue_states.at[i, 'active'] = bool(traffic[f'ue_{i}'])
        
        terminated = False
        truncated = self.current_slot >= self.max_slots
        
        info = {
            'urllc_success_rate': self.urllc_stats['success'] / (self.urllc_stats['success'] + self.urllc_stats['failures'] + 1e-6),
            'mmtc_throughput': self.mmtc_stats['throughput'] / (self.current_slot + 1e-6),
            'spectral_efficiency': self.spectral_efficiency,
            'collision_rate': np.sum(self.collisions) / (self.current_slot + 1e-6)
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    # (Helper methods for data loading remain the same)
    def _load_config(self, path: str):
        with open(path, 'r') as f: return json.load(f)
    def _load_initial_states(self, path: str):
        df = pd.read_csv(path); df['active'] = True; df['d2lt'] = 0; return df
    def _load_semantic_data(self, path: str):
        return pd.read_csv(path)
    def _load_traffic_model(self, path: str):
        df = pd.read_csv(path); return df.set_index('slot') if 'slot' in df.columns else df