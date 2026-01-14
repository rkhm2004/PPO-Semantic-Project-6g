import pettingzoo
import numpy as np
import pandas as pd
import json
from gymnasium import spaces
from collections import defaultdict
import os

class MASAC_SAMAEnvironment(pettingzoo.ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "SAMA_v2_MASAC"}

    def __init__(self, is_real_time=True, num_ues=6, num_channels=3, **kwargs):
        super().__init__()
        
        self.possible_agents = [f"ue_{i}" for i in range(num_ues)]
        self.num_channels = num_channels

        # --- PERMANENT FIX FOR FILE PATHS ---
        # Get the absolute path of the directory where this script is located (e.g., .../Project/sac/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the main project directory
        project_root = os.path.dirname(script_dir)
        # Define the base path for the data folder
        data_path = os.path.join(project_root, 'data')

        self.config = self._load_config(kwargs.get('config_path', os.path.join(data_path, 'channel_params_sac.json')))
        
        if is_real_time:
            self.initial_states_path = kwargs.get('state_path', os.path.join(data_path, 'real/real_initial_states.csv'))
            self.semantic_path = kwargs.get('semantic_path', os.path.join(data_path, 'real/real_semantic_segments.csv'))
            self.traffic_path = kwargs.get('traffic_path', os.path.join(data_path, 'real/real_traffic_model.csv'))
        else:
            self.initial_states_path = kwargs.get('state_path', os.path.join(data_path, 'controlled/controlled_initial_states.csv'))
            self.semantic_path = kwargs.get('semantic_path', os.path.join(data_path, 'controlled/controlled_semantic_segments.csv'))
            self.traffic_path = kwargs.get('traffic_path', os.path.join(data_path, 'controlled/controlled_traffic_model.csv'))
        # ------------------------------------

        self.original_ue_states = self._load_initial_states(self.initial_states_path)
        self.original_semantic_segments = self._load_semantic_data(self.semantic_path)
        self.original_traffic_model = self._load_traffic_model(self.traffic_path)
        self.max_slots = len(self.original_traffic_model)
        
        self.state_dim = self.num_channels + (len(self.possible_agents) * 14) + 3 

    def observation_space(self, agent):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

    def action_space(self, agent):
        return spaces.Box(low=0, high=self.num_channels - 1, shape=(1,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
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
        
        global_obs = self._get_global_observation()
        observations = {agent: global_obs for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        discrete_actions = {agent: np.clip(np.round(action).astype(int), 0, self.num_channels - 1)[0] for agent, action in actions.items()}
        reward, info = self._calculate_step_results(discrete_actions)

        global_obs = self._get_global_observation()
        observations = {agent: global_obs for agent in self.agents}
        rewards = {agent: reward for agent in self.agents}
        done = self.current_slot >= self.max_slots
        terminations = {agent: done for agent in self.agents}
        truncations = {agent: done for agent in self.agents}
        if done: self.agents = []
        return observations, rewards, terminations, truncations, info

    # Inside ma_environment_sac.py

    def _calculate_step_results(self, actions_dict):
        # (The first part of the function for calculating collisions and successes remains the same)
        # ...
        channel_usage = defaultdict(list)
        self.ue_states['d2lt'] += 1
        action_list = [actions_dict[agent] for agent in self.possible_agents if agent in actions_dict]
        served_ue_ids = set()

        for ue_id, channel in enumerate(action_list):
            if not (0 <= channel < self.num_channels): continue
            ue = self.ue_states.iloc[ue_id]
            if not ue['active']: continue
            channel_usage[channel].append(ue_id)
        
        for channel, ue_ids in channel_usage.items():
            if len(ue_ids) > 1:
                self.collisions[channel] += 1
                for ue_id in ue_ids:
                    if self.ue_states.iloc[ue_id]['traffic_type'] == 'URLLC': self.urllc_stats['failures'] += 1
            elif len(ue_ids) == 1:
                ue_id = ue_ids[0]
                served_ue_ids.add(ue_id)
                ue = self.ue_states.iloc[ue_id]
                if ue['traffic_type'] == 'URLLC': self.urllc_stats['success'] += 1
                else: self.mmtc_stats['packets'] += 1; self.mmtc_stats['throughput'] += 1
        
        # --- NEW: ADVANCED "SWEET SPOT" REWARD FUNCTION ---
        reward = 0.0
        
        # Calculate current performance metrics
        current_urllc_success = self.urllc_stats['success'] / (self.urllc_stats['success'] + self.urllc_stats['failures'] + 1e-6)
        current_mmtc_throughput = self.mmtc_stats['throughput'] / (self.current_slot + 1)

        # --- 1. URLLC Target Score ---
        urllc_target = 0.94  # The middle of your 0.89-0.99 range
        urllc_error = abs(current_urllc_success - urllc_target)
        # Use a Gaussian-like function: reward is max at the target, decreases as it moves away
        urllc_score = np.exp(-10 * urllc_error**2) # The '10' controls how sharp the peak is
        
        reward += 100 * urllc_score # Max reward of 100 for being on target

        # --- 2. mMTC Target Score ---
        mmtc_target = 1.0 # The middle of your 0.8-1.2 range
        mmtc_error = abs(current_mmtc_throughput - mmtc_target)
        mmtc_score = np.exp(-5 * mmtc_error**2)
        
        reward += 50 * mmtc_score # Max reward of 50 for being on target

        # --- 3. Collision Penalty ---
        # A simple penalty for any collisions in this step
        if self.collisions.sum() > 0:
            reward -= 50

        # ----------------------------------------------------------------
        
        self.current_slot += 1
        
        info = {
            'urllc_success_rate': current_urllc_success,
            'mmtc_throughput': current_mmtc_throughput,
            'collision_rate': np.sum(self.collisions) / (self.current_slot + 1e-6)
        }
        return reward, {agent: info for agent in self.agents}
    
    def _get_global_observation(self) -> np.ndarray:
        # (This function is correct and remains the same)
        state_parts = []
        state_parts.append(self.channel_states)
        ue_features = []
        total_d2lt = self.ue_states['d2lt'].sum()
        normalized_d2lt = np.zeros(len(self.ue_states))
        if total_d2lt > 0: normalized_d2lt = self.ue_states['d2lt'] / total_d2lt
        for idx, ue in self.ue_states.iterrows():
            ue_features.extend([1 if ue['active'] else 0, 1 if ue['traffic_type'] == 'URLLC' else 0, ue['priority'] / 10.0, normalized_d2lt[idx]])
            ue_segments = self.original_semantic_segments[self.original_semantic_segments['ue_id'] == ue['ue_id']]['segment_id'].tolist()
            ue_features.extend([1 if seg in ue_segments else 0 for seg in range(10)])
        state_parts.append(np.array(ue_features))
        global_metrics = [0, self.urllc_stats['success'] / (self.urllc_stats['success'] + self.urllc_stats['failures'] + 1e-6), self.mmtc_stats['throughput'] / (self.mmtc_stats['packets'] + 1e-6)]
        state_parts.append(np.array(global_metrics))
        return np.concatenate(state_parts).astype(np.float32)
    
    def _load_config(self, path: str):
        with open(path, 'r') as f: return json.load(f)
    def _load_initial_states(self, path: str):
        df = pd.read_csv(path); df['active'] = True; df['d2lt'] = 0; return df
    def _load_semantic_data(self, path: str):
        return pd.read_csv(path)
    def _load_traffic_model(self, path: str):
        df = pd.read_csv(path); return df.set_index('slot') if 'slot' in df.columns else df