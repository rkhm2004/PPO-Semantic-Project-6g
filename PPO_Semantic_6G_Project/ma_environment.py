# ma_environment.py — Multi-Agent 6G Environment (CTDE-ready, Gym>=0.26 compatible)
import gym
import numpy as np
import pandas as pd
from gym import spaces


class MultiAgent6GEnv(gym.Env):
    """
    CTDE-ready multi-agent environment for 6 UEs × C channels.
    - Actor sees local obs per agent (we flatten to a single vector for SB3).
    - Centralized critic uses concatenated locals.
    - Action space: MultiDiscrete([num_channels] * num_ues).
    - Reward: scalar (centralized training).
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        state_path: str,
        traffic_path: str,
        semantic_path: str = None,
        num_channels: int = 3,
        num_ues: int = 6,
        local_obs_dim: int = 4,   # (active, is_urlcc, priority_norm, prev_free)
        collision_penalty: float = 1.0,
        success_reward: float = 1.0,
    ):
        super().__init__()
        self.state_path = state_path
        self.traffic_path = traffic_path
        self.semantic_path = semantic_path
        self.num_channels = int(num_channels)
        self.num_ues = int(num_ues)
        self.local_obs_dim = int(local_obs_dim)
        self.collision_penalty = float(collision_penalty)
        self.success_reward = float(success_reward)

        # ---------- Load static state info ----------
        self.states_df = pd.read_csv(self.state_path)
        assert {"ue_id", "traffic_type", "priority"}.issubset(self.states_df.columns)

        # ---------- Load traffic model ----------
        self.traffic_df = pd.read_csv(self.traffic_path)
        if "slot" in self.traffic_df.columns:
            self.traffic_df = self.traffic_df.set_index("slot")
        for i in range(self.num_ues):
            col = f"ue_{i}"
            if col not in self.traffic_df.columns:
                self.traffic_df[col] = 0
        self.T = len(self.traffic_df)

        # ---------- Per-UE constants ----------
        self.is_urllc = np.array([
            1.0 if self.states_df.loc[self.states_df.ue_id == i, "traffic_type"].values[0] == "URLLC" else 0.0
            for i in range(self.num_ues)
        ], dtype=np.float32)

        pr = np.array([
            float(self.states_df.loc[self.states_df.ue_id == i, "priority"].values[0])
            for i in range(self.num_ues)
        ], dtype=np.float32)
        if pr.max() > 1.0:
            pr = pr / pr.max()
        self.priority_norm = pr.astype(np.float32)

        # ---------- Runtime vars ----------
        self.prev_free = np.zeros(self.num_ues, dtype=np.float32)
        self._last_actions = np.zeros(self.num_ues, dtype=np.int64)
        self.t = 0

        # ---------- Gym spaces ----------
        obs_low = np.zeros(self.local_obs_dim * self.num_ues, dtype=np.float32)
        obs_high = np.ones(self.local_obs_dim * self.num_ues, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.num_channels] * self.num_ues)

    # =========================================================
    # Helpers
    # =========================================================
    def _get_active_vec(self, t):
        """Return binary activity vector for all UEs at slot t."""
        return self.traffic_df.iloc[t].values[: self.num_ues].astype(np.float32)

    def _build_local_obs(self, active):
        """Build concatenated local obs for all UEs."""
        parts = []
        for i in range(self.num_ues):
            parts.extend([
                active[i],
                self.is_urllc[i],
                self.priority_norm[i],
                self.prev_free[i],
            ])
        return np.array(parts, dtype=np.float32)

    # =========================================================
    # Gym API
    # =========================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.prev_free.fill(0.0)
        self._last_actions.fill(0)
        active = self._get_active_vec(self.t)
        obs = self._build_local_obs(active)
        info = {}
        return obs, info   # Gym>=0.26 expects (obs, info)

    def step(self, action):
        """
        action: array-like shape (num_ues,) with channel indices.
        Returns: obs, reward, terminated, truncated, info
        """
        action = np.asarray(action, dtype=np.int64).reshape(self.num_ues)
        active = self._get_active_vec(self.t)

        reward = 0.0
        successes = collisions = 0

        # Channel evaluation
        for cid in range(self.num_channels):
            users = np.where((action == cid) & (active == 1.0))[0]
            if len(users) == 1:
                reward += self.success_reward
                successes += 1
            elif len(users) > 1:
                reward -= self.collision_penalty
                collisions += 1

        # Update prev_free flags
        new_prev_free = np.zeros_like(self.prev_free)
        for u in range(self.num_ues):
            if active[u] == 1.0:
                cid = action[u]
                users = np.where((action == cid) & (active == 1.0))[0]
                new_prev_free[u] = 1.0 if len(users) == 1 else 0.0
        self.prev_free = new_prev_free
        self._last_actions = action.copy()

        # Advance time
        self.t += 1
        terminated = bool(self.t >= self.T)
        truncated = False

        if not terminated:
            obs = self._build_local_obs(self._get_active_vec(min(self.t, self.T - 1)))
        else:
            obs = np.zeros_like(self.observation_space.sample())

        info = {"successes": successes, "collisions": collisions}
        return obs, float(reward), terminated, truncated, info

    # =========================================================
    def render(self, mode="human"):
        return None
