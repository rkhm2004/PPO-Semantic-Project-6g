import numpy as np
import pandas as pd
import json
from typing import Dict, Tuple, List, Optional
import gymnasium as gym
from gymnasium import spaces

class SAMAEnvironment(gym.Env):
    """
    Single-agent, centralized scheduler for N UEs:
      - Action: MultiDiscrete([num_channels]*num_ues) -> one channel per UE each slot
      - Observation: [active, is_urllc, prio_norm] for each UE (flattened) + [free_channels]
      - Reward per step:
            +1 for each successful (no-collision) transmission
            +w_urlcc extra if UE is URLLC and success
            -pen_collision per collision event on a channel
            -pen_waste per free (unused) channel
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        state_path: str,
        traffic_path: str,
        semantic_path: Optional[str] = None,
        num_ues: int = 6,
        num_channels: int = 3,
        is_real_time: bool = False,
        reward_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        super().__init__()
        self.state_path = state_path
        self.traffic_path = traffic_path
        self.semantic_path = semantic_path
        self.num_ues = int(num_ues)
        self.num_channels = int(num_channels)
        self.is_real_time = bool(is_real_time)

        self.w = {
            "w_urlcc": 1.0,
            "pen_collision": 0.45,
            "pen_waste": 0.05,
        }
        if reward_weights:
            self.w.update(reward_weights)

        # Data
        self.states_df = self._load_initial_states(self.state_path)  # ue_id, traffic_type, priority
        self.traffic_df = self._load_traffic_model(self.traffic_path)  # slot indexed; ue_* columns
        if self.semantic_path:
            self.semantic_df = self._load_semantic_data(self.semantic_path)
        else:
            self.semantic_df = None

        # UE count consistency
        ue_cols = [c for c in self.traffic_df.columns if c.startswith("ue_")]
        if self.num_ues != len(ue_cols):
            self.num_ues = len(ue_cols)

        self.max_steps = len(self.traffic_df)

        # Precompute URLLC flags & priorities
        self.is_urllc = np.array([
            1 if self.states_df.loc[self.states_df.ue_id == i, "traffic_type"].values[0] == "URLLC" else 0
            for i in range(self.num_ues)
        ], dtype=np.int32)

        prio = np.array([
            float(self.states_df.loc[self.states_df.ue_id == i, "priority"].values[0])
            for i in range(self.num_ues)
        ], dtype=np.float32)
        prio = (prio - prio.min()) / (max(1e-9, prio.max() - prio.min()))
        self.priority_norm = prio

        # Spaces
        self.action_space = spaces.MultiDiscrete([self.num_channels] * self.num_ues)
        # obs per UE: [active, is_urlcc, prio_norm] -> 3 * num_ues + 1 global (free channels from prev step)
        self.obs_dim = 3 * self.num_ues + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # State
        self.t = 0
        self.prev_free = 1.0

    # -------- core API --------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.prev_free = 1.0
        obs = self._build_obs(self.t)
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        """
        action: array of length num_ues with channel id [0..num_channels-1]
        """
        t = self.t
        act = np.asarray(action, dtype=np.int32).reshape(-1)
        act = act[: self.num_ues]

        # Active UEs at slot t
        row = self.traffic_df.iloc[t].values[: self.num_ues]
        active = (row == 1)

        # Compute successes / collisions per channel
        reward = 0.0
        successes = 0
        urllc_succ = 0
        collisions_ch_events = 0

        for ch in range(self.num_channels):
            users = np.where((act == ch) & active)[0]
            if len(users) == 1:
                u = users[0]
                successes += 1
                if self.is_urllc[u] == 1:
                    urllc_succ += 1
            elif len(users) > 1:
                collisions_ch_events += 1

        free_channels = self._free_channels(active, act)
        free_ratio = free_channels / float(self.num_channels)

        reward += successes
        reward += self.w["w_urlcc"] * urllc_succ
        reward -= self.w["pen_collision"] * collisions_ch_events
        reward -= self.w["pen_waste"] * free_channels

        self.t += 1
        terminated = self.t >= self.max_steps
        obs = self._build_obs(self.t if not terminated else self.t - 1)

        info = dict(
            t=t,
            successes=successes,
            urllc_success=urllc_succ,
            collisions_channel_events=collisions_ch_events,
            free_channels=free_channels,
            free_ratio=free_ratio,
        )
        return obs, float(reward), terminated, False, info

    # -------- helpers --------

    def _build_obs(self, t: int) -> np.ndarray:
        if t >= self.max_steps:
            t = self.max_steps - 1
        active = (self.traffic_df.iloc[t].values[: self.num_ues] == 1).astype(np.float32)
        feats = []
        for u in range(self.num_ues):
            feats.extend([active[u], float(self.is_urllc[u]), float(self.priority_norm[u])])
        feats.append(float(self.prev_free))
        obs = np.array(feats, dtype=np.float32)
        # update prev_free from previous computed info if available
        # (this is a simple memory; exact value updated in step)
        return obs

    def _free_channels(self, active: np.ndarray, act: np.ndarray) -> int:
        free = 0
        for ch in range(self.num_channels):
            users = np.where((act == ch) & active)[0]
            if len(users) == 0:
                free += 1
        self.prev_free = free / float(self.num_channels)
        return free

    # -------- loading --------

    def _load_initial_states(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        # required cols: ue_id, traffic_type, priority
        assert {"ue_id", "traffic_type", "priority"}.issubset(df.columns), \
            "states CSV must have ue_id, traffic_type, priority"
        return df.sort_values("ue_id").reset_index(drop=True)

    def _load_semantic_data(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def _load_traffic_model(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "slot" in df.columns:
            df = df.set_index("slot")
        # expect ue_0 ... ue_{N-1}
        return df
