# App_marl.py — CTDE (MAPPO) evaluation + plots  (UPDATED: adds channel split metrics + pie)
import os, json, argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from ma_environment import MultiAgent6GEnv
from ma_policies import CTDEPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--state_csv", required=True)
    p.add_argument("--traffic_csv", required=True)
    p.add_argument("--semantic_csv", default="")
    p.add_argument("--num_channels", type=int, default=3)
    p.add_argument("--num_ues", type=int, default=6)
    p.add_argument("--timesteps", type=int, default=1)  # minimal learn just to init
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="outputs/output_multi_agent")
    p.add_argument("--resume_from", default="", help="Path to model (omit .zip)")
    p.add_argument("--sim2real", action="store_true")
    p.add_argument("--show_plots", action="store_true")
    return p.parse_args()


def ensure_dir(d): os.makedirs(d, exist_ok=True)

def build_env(args):
    env_cfg = dict(
        state_path=args.state_csv,
        traffic_path=args.traffic_csv,
        semantic_path=args.semantic_csv or None,
        num_channels=args.num_channels,
        num_ues=args.num_ues,
        local_obs_dim=4,
    )
    vec = DummyVecEnv([lambda: MultiAgent6GEnv(**env_cfg)])
    vec = VecNormalize(vec, training=True, norm_obs=True, norm_reward=True, clip_obs=5.0)
    print("MA Env obs space:", vec.observation_space)
    return vec, env_cfg


def build_model(args, vec):
    if args.resume_from:
        model = PPO.load(args.resume_from, env=vec, device="auto", print_system_info=False)
        if args.sim2real:
            model.policy.optimizer = type(model.policy.optimizer)(
                model.policy.parameters(), lr=5e-5, weight_decay=1e-5
            )
            model.ent_coef = 0.005
        print(f"Loaded MAPPO checkpoint from: {args.resume_from}")
        return model

    # Fresh model for testing (usually you load a trained model)
    policy_kwargs = dict(num_ues=args.num_ues, local_obs_dim=4)
    model = PPO(
        CTDEPolicy, vec,
        learning_rate=3e-4, n_steps=1024, batch_size=1024, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, seed=args.seed,
        policy_kwargs=policy_kwargs
    )
    print("Initialized new CTDEPolicy (fresh).")
    return model


# ==================== metrics & utilities ====================
def reconstruct_assignments(actions, traffic_df, num_channels):
    """
    Turn per-step joint channel selections into UE×T assigned channels.
    Inactive UEs get -1.
    """
    T, num_ues = actions.shape
    assign = -1 * np.ones((num_ues, T), dtype=np.int32)
    for t in range(T):
        active = (traffic_df.iloc[t].values[:num_ues] == 1)
        for u in range(num_ues):
            if active[u] == 1:
                assign[u, t] = int(actions[t, u])
    return assign

def compute_metrics(assign, traffic_df, states_df, num_channels):
    T = assign.shape[1]
    num_ues = assign.shape[0]
    is_urllc = np.array([
        1 if states_df.loc[states_df.ue_id == i, "traffic_type"].values[0] == "URLLC" else 0
        for i in range(num_ues)
    ], dtype=np.int32)

    urllc_active = urllc_success = 0
    mmtc_active = mmtc_served = 0
    successes_total = 0
    collisions_channel_events = 0
    slots_with_collision = 0

    for t in range(T):
        active = (traffic_df.iloc[t].values[:num_ues] == 1)
        ch = assign[:, t]
        had_col = False
        for cid in range(num_channels):
            users = np.where((ch == cid) & active)[0]
            if len(users) == 1:
                u = users[0]
                successes_total += 1
                if is_urllc[u] == 1: urllc_success += 1
                else: mmtc_served += 1
            elif len(users) > 1:
                collisions_channel_events += 1
                had_col = True
        if had_col: slots_with_collision += 1
        urllc_active += int(np.sum(active * (is_urllc == 1)))
        mmtc_active += int(np.sum(active * (is_urllc == 0)))

    urllc_rel = urllc_success / max(1, urllc_active)
    mmtc_thr = mmtc_served / max(1, mmtc_active)
    se = successes_total / (T * num_channels)
    return dict(
        urllc_reliability=urllc_rel,
        mmtc_throughput=mmtc_thr,
        spectral_efficiency=se,
        collision_events=collisions_channel_events,
        collision_rate_slots=slots_with_collision / float(T),
    )

def compute_slot_stats(assign, traffic_df, num_channels):
    T = assign.shape[1]
    success_slots = np.zeros(T, dtype=int)
    collision_slots = np.zeros(T, dtype=int)
    for t in range(T):
        active = (traffic_df.iloc[t].values[:assign.shape[0]] == 1)
        ch = assign[:, t]
        for cid in range(num_channels):
            users = np.where((ch == cid) & active)[0]
            if len(users) == 1: success_slots[t] += 1
            elif len(users) > 1: collision_slots[t] += 1
    return success_slots, collision_slots

def channel_counts_from_assign(assign, num_channels):
    """
    Build counts including Inactive (-1) and active-only counts (0..K-1).
    Returns: (counts_all_dict, counts_active_dict, perc_all_dict)
    """
    flat = assign.flatten()
    # All counts (include -1)
    uniq, cnt = np.unique(flat, return_counts=True)
    counts_all = {int(k): int(c) for k, c in zip(uniq, cnt)}
    # Active-only counts (>=0 and < num_channels)
    active = flat[(flat >= 0) & (flat < num_channels)]
    if active.size == 0:
        counts_active = {c: 0 for c in range(num_channels)}
    else:
        u2, c2 = np.unique(active, return_counts=True)
        counts_active = {int(c): 0 for c in range(num_channels)}
        counts_active.update({int(k): int(v) for k, v in zip(u2, c2)})
    # Percents including -1
    total_all = float(sum(counts_all.values())) or 1.0
    perc_all = {k: (100.0 * v / total_all) for k, v in counts_all.items()}
    return counts_all, counts_active, perc_all


def _maybe_show(show):
    if show:
        plt.show(block=False); plt.pause(2.0)
    plt.close()


# ==================== plotting ====================
def plot_and_save_all(run_dir, metrics, actions, assign, traffic_df, states_df, num_channels, show=False):
    # ---- 2. Channel Utilization (pie) from ASSIGN (active-only, like your example) ----
    _, counts_active, _ = channel_counts_from_assign(assign, num_channels)
    vals = np.array([counts_active.get(c, 0) for c in range(num_channels)], dtype=float)
    total_active = float(vals.sum()) or 1.0
    fracs = 100.0 * vals / total_active

    plt.figure(figsize=(8, 8))
    plt.title("Channel Utilization (Multi-Agent)")
    plt.pie(
        fracs, startangle=90,
        labels=[f"Channel {i}" for i in range(num_channels)],
        autopct=lambda p: f"{p:.1f}%"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "2_channel_utilization.png"), bbox_inches="tight", dpi=150)
    _maybe_show(show)

    # ---- 3. Final Metrics (bars) ----
    names = ["URLLC Success","mMTC Throughput","Collision Slots","Collision Events","Spectral Efficiency"]
    vals2 = [
        metrics["urllc_reliability"],
        metrics["mmtc_throughput"],
        metrics["collision_rate_slots"],
        metrics["collision_events"]/max(1, len(traffic_df)),
        metrics["spectral_efficiency"],
    ]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, vals2, color=["#4caf50","#2196f3","#f44336","#e91e63","#ff9800"])
    for b, v in zip(bars, vals2):
        plt.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.ylim(0, 1.05)
    plt.title("Final Evaluation Metrics (Multi-Agent)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "3_final_metrics.png"), bbox_inches="tight", dpi=150)
    _maybe_show(show)

    # ---- 5. Reward components proxy ----
    s, c = compute_slot_stats(assign, traffic_df, num_channels)
    plt.figure(figsize=(18, 7))
    plt.plot(np.cumsum(s), label="Cumulative Success (+)", color="green")
    plt.plot(-np.cumsum(c), label="Cumulative Collisions (-)", color="red")
    plt.title("Reward Components (Multi-Agent)")
    plt.xlabel("Time Slot"); plt.ylabel("Cumulative")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "5_reward_components.png"), bbox_inches="tight", dpi=150)
    _maybe_show(show)

    # ---- 6. Channel Assignments Heatmap ----
    num_ues, T = assign.shape
    colors = ["#ffeb3b", "#4B0082", "#31688e", "#35b779", "#a1c9f4", "#ff9f9b"]
    cmap = ListedColormap(colors[: (num_channels + 1)])
    boundaries = np.arange(-1.5, (num_channels - 0.5) + 1e-9, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    heat = -1 * np.ones((num_ues, T), dtype=int)
    for t in range(T):
        active = np.array([traffic_df.iloc[t][f"ue_{i}"] if f"ue_{i}" in traffic_df.columns else 0 for i in range(num_ues)])
        for u in range(num_ues):
            if active[u] == 1:
                heat[u, t] = assign[u, t]

    plt.figure(figsize=(20, 7))
    im = plt.imshow(heat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm, origin="upper")
    plt.title("Channel Assignments (Multi-Agent)")
    plt.xlabel("Time Slot"); plt.ylabel("UE Index")
    cbar = plt.colorbar(im, ticks=np.arange(-1, num_channels, 1))
    cbar.ax.set_yticklabels(["Inactive"] + [f"Ch {i}" for i in range(num_channels)])
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "6_channel_assignments.png"), bbox_inches="tight", dpi=150)
    _maybe_show(show)

    # ---- 7. Traffic Load (Active UEs per slot) ----
    ue_cols = [c for c in traffic_df.columns if c.startswith("ue_")]
    if len(ue_cols) > 0:
        active = traffic_df[ue_cols].sum(axis=1).values
        plt.figure(figsize=(18, 6))
        plt.plot(active, linewidth=2)
        plt.title("Traffic Load Over Time (Active UEs per Slot)")
        plt.xlabel("Time Slot"); plt.ylabel("Active UEs")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "7_traffic_load.png"), bbox_inches="tight", dpi=150)
        _maybe_show(show)
    else:
        print("⚠️  Traffic CSV missing 'ue_' columns — skipping Traffic Load plot.")


def main():
    args = parse_args(); set_random_seed(args.seed)

    vec, env_cfg = build_env(args)
    model = build_model(args, vec)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, stamp); ensure_dir(run_dir)
    with open(os.path.join(run_dir, "env_config.json"), "w") as f:
        json.dump(env_cfg, f, indent=2)

    # Tiny learn (or bigger if you want to fine-tune here)
    model.learn(total_timesteps=args.timesteps)
    model.save(os.path.join(run_dir, "model.zip")); vec.save(os.path.join(run_dir, "vecnorm.pkl"))

    # ---------- EVAL ----------
    eval_env = DummyVecEnv([lambda: MultiAgent6GEnv(**env_cfg)])
    eval_env = VecNormalize.load(os.path.join(run_dir, "vecnorm.pkl"), eval_env)
    eval_env.training = False; eval_env.norm_reward = False

    traffic_df = pd.read_csv(env_cfg["traffic_path"])
    if "slot" in traffic_df.columns:
        traffic_df = traffic_df.set_index("slot")
    T = len(traffic_df)

    acts = []
    obs = eval_env.reset()
    for _ in range(T):
        act, _ = model.predict(obs, deterministic=True)
        acts.append(act[0])
        obs, _, dones, _ = eval_env.step(act)
        if np.all(dones): break
    actions = np.vstack(acts)  # shape: T x num_ues
    np.save(os.path.join(run_dir, "actions.npy"), actions)

    states_df = pd.read_csv(env_cfg["state_path"])
    assign = reconstruct_assignments(actions, traffic_df, env_cfg["num_channels"])

    # --- metrics + channel split ---
    metrics = compute_metrics(assign, traffic_df, states_df, env_cfg["num_channels"])
    counts_all, counts_active, perc_all = channel_counts_from_assign(assign, env_cfg["num_channels"])
    metrics["channel_split_all"] = {str(k): int(v) for k, v in counts_all.items()}
    metrics["channel_split_active"] = {str(k): int(v) for k, v in counts_active.items()}
    metrics["channel_time_share_percent"] = {str(k): float(v) for k, v in perc_all.items()}

    with open(os.path.join(run_dir, "metrics_computed.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # --- plots (includes the pie) ---
    plot_and_save_all(run_dir, metrics, actions, assign, traffic_df, states_df, env_cfg["num_channels"], show=args.show_plots)

    print("✅ Multi-Agent results saved to:", run_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
