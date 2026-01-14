# App.py — Single-Agent LSTM PPO with Automatic Visualization (with eval_only & dataset signature)
import os, json, argparse, hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from SAMAEnvironment import SAMAEnvironment
from policies import RawLstmPolicy as LstmPolicy


# ========================= ARGS =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--state_csv", required=True)
    p.add_argument("--traffic_csv", required=True)
    p.add_argument("--semantic_csv", default="")
    p.add_argument("--num_channels", type=int, default=3)
    p.add_argument("--num_ues", type=int, default=-1)
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="outputs/output_single")
    p.add_argument("--resume_from", default="", help="Checkpoint path (folder or .../model.zip)")
    p.add_argument("--sim2real", action="store_true", help="Gentle LR/entropy for fine-tuning")
    p.add_argument("--show_plots", action="store_true", help="Display graphs after saving")
    p.add_argument("--eval_only", action="store_true", help="Run evaluation only (no training)")
    return p.parse_args()


# ====================== ENV HELPERS =====================
def ensure_dir(d): os.makedirs(d, exist_ok=True)

def latest_num_ues(traffic_csv, arg_num_ues):
    df = pd.read_csv(traffic_csv)
    ue_cols = [c for c in df.columns if c.startswith("ue_")]
    return arg_num_ues if arg_num_ues > 0 else len(ue_cols)

def make_env(env_cfg): return SAMAEnvironment(**env_cfg)

def build_env(args):
    num_ues = latest_num_ues(args.traffic_csv, args.num_ues)
    env_cfg = dict(
        state_path=args.state_csv,
        traffic_path=args.traffic_csv,
        semantic_path=args.semantic_csv or None,
        num_channels=args.num_channels,
        num_ues=num_ues,
        is_real_time=False,
        reward_weights={"w_urlcc": 1.0, "pen_collision": 0.45, "pen_waste": 0.05},
    )
    vec = DummyVecEnv([lambda: make_env(env_cfg)])
    vec = VecNormalize(vec, training=True, norm_obs=True, norm_reward=True, clip_obs=5.0)
    print("Env obs space:", vec.observation_space)
    return vec, env_cfg

def resolve_ckpt(path: str) -> str:
    """Accept folder or .../model.zip; return a .zip path if exists, else ''."""
    if not path: return ""
    if os.path.isdir(path):
        z = os.path.join(path, "model.zip")
        return z if os.path.exists(z) else ""
    if path.endswith(".zip") and os.path.exists(path):
        return path
    # maybe provided without .zip
    z = f"{path}.zip"
    return z if os.path.exists(z) else ""


# ====================== MODEL BUILD =====================
def build_model(args, vec):
    ckpt_zip = resolve_ckpt(args.resume_from)
    if ckpt_zip:
        model = PPO.load(ckpt_zip, env=vec, device="auto", print_system_info=False)
        if args.sim2real:  # gentle FT
            model.policy.optimizer = type(model.policy.optimizer)(
                model.policy.parameters(), lr=5e-5, weight_decay=1e-5
            )
            model.ent_coef = 0.005
        print(f"Loaded checkpoint: {ckpt_zip}")
        return model

    model = PPO(
        LstmPolicy, vec,
        learning_rate=3e-4, n_steps=1024, batch_size=1024, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, seed=args.seed
    )
    print("Initialized new LSTM PPO model.")
    return model


# ====================== METRICS =========================
def reconstruct_assignments(actions, traffic_df, num_channels):
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

def channel_counts_from_assign(assign, num_channels):
    flat = assign.flatten()
    uniq, cnt = np.unique(flat, return_counts=True)
    counts_all = {int(k): int(c) for k, c in zip(uniq, cnt)}
    active = flat[(flat >= 0) & (flat < num_channels)]
    if active.size == 0:
        counts_active = {c: 0 for c in range(num_channels)}
    else:
        u2, c2 = np.unique(active, return_counts=True)
        counts_active = {int(c): 0 for c in range(num_channels)}
        counts_active.update({int(k): int(v) for k, v in zip(u2, c2)})
    total_all = float(sum(counts_all.values())) or 1.0
    perc_all = {k: (100.0 * v / total_all) for k, v in counts_all.items()}
    return counts_all, counts_active, perc_all


# =============== DATASET SIGNATURE HELPERS ==============
def dataset_signature(path: str, df: pd.DataFrame) -> str:
    md5 = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    ue = [c for c in df.columns if c.startswith("ue_")]
    if ue:
        k = df[ue].sum(axis=1)
        mm = f"min/mean/max={int(k.min())}/{float(k.mean()):.2f}/{int(k.max())}"
    else:
        mm = "n/a"
    return f"{os.path.basename(path)} | md5={md5[:8]} | T={len(df)} | {mm}"


# ====================== PLOTTING ========================
def _maybe_show(show):
    if show:
        plt.show(block=False); plt.pause(2.0)
    plt.close()

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

def plot_and_save_all(run_dir, metrics, actions, assign, traffic_df, states_df, num_channels, data_sig, show=False):
    # 2. Channel Utilization (pie) — active channels only
    _, counts_active, _ = channel_counts_from_assign(assign, num_channels)
    vals = np.array([counts_active.get(c, 0) for c in range(num_channels)], dtype=float)
    total_active = float(vals.sum()) or 1.0
    fracs = 100.0 * vals / total_active

    plt.figure(figsize=(8, 8))
    plt.title(f"Channel Utilization\n{data_sig}")
    plt.pie(fracs, startangle=90,
            labels=[f"Channel {i}" for i in range(num_channels)],
            autopct=lambda p: f"{p:.1f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "2_channel_utilization.png"), bbox_inches="tight", dpi=150)
    _maybe_show(show)

    # 3. Final Metrics (bar)
    names = ["URLLC Success", "mMTC Throughput", "Collision Slots", "Collision Events", "Spectral Efficiency"]
    vals_bar = [
        metrics["urllc_reliability"],
        metrics["mmtc_throughput"],
        metrics["collision_rate_slots"],
        metrics["collision_events"] / max(1, len(traffic_df)),  # normalized display
        metrics["spectral_efficiency"],
    ]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, vals_bar)
    for b, v in zip(bars, vals_bar):
        plt.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.ylim(0, 1.05)
    plt.title(f"Final Evaluation Metrics\n{data_sig}")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(os.path.join(run_dir, "3_final_metrics.png"), bbox_inches="tight", dpi=150)
    _maybe_show(show)

    # 5. Reward Components (success vs collisions)
    s, c = compute_slot_stats(assign, traffic_df, num_channels)
    plt.figure(figsize=(18, 7))
    plt.plot(np.cumsum(s), label="Cumulative Success (+)")
    plt.plot(-np.cumsum(c), label="Cumulative Collisions (-)")
    plt.title(f"Reward Components (Proxy) Over Time\n{data_sig}")
    plt.xlabel("Time Slot"); plt.ylabel("Cumulative Value"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(run_dir, "5_reward_components.png"), bbox_inches="tight", dpi=150)
    _maybe_show(show)

    # 6. Channel Assignments Heatmap
    num_ues, T = assign.shape
    base_colors = ["#ffeb3b", "#4B0082", "#31688e", "#35b779", "#a1c9f4", "#ff9f9b", "#8dd3c7", "#fb8072"]
    need = num_channels + 1
    if need > len(base_colors):
        base_colors = (base_colors * ((need // len(base_colors)) + 1))[:need]
    cmap = ListedColormap(base_colors[:need])
    boundaries = np.arange(-1.5, (num_channels - 0.5) + 1e-9, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    heat = -1 * np.ones((num_ues, T), dtype=int)
    for t in range(T):
        active = (traffic_df.iloc[t].values[:num_ues] == 1)
        for u in range(num_ues):
            if active[u] == 1:
                heat[u, t] = assign[u, t]

    plt.figure(figsize=(20, 7))
    im = plt.imshow(heat, aspect='auto', interpolation='nearest', cmap=cmap, norm=norm, origin="upper")
    plt.title(f"Channel Assignments Heatmap\n{data_sig}")
    plt.xlabel("Time Slot"); plt.ylabel("UE Index")
    cbar = plt.colorbar(im, ticks=np.arange(-1, num_channels, 1))
    cbar.ax.set_yticklabels(["Inactive"] + [f"Ch {i}" for i in range(num_channels)])

    is_urllc = np.array([
        1 if states_df.loc[states_df.ue_id == i, "traffic_type"].values[0] == "URLLC" else 0
        for i in range(num_ues)
    ])
    n_url = int(is_urllc.sum())
    if 0 < n_url < num_ues:
        plt.axhline(y=n_url - 0.5, color='red', linestyle='--', linewidth=1.5)
        plt.text(T + 1, n_url/2, "URLLC", color='red', va='center')
        plt.text(T + 1, n_url + (num_ues - n_url)/2, "mMTC", color='red', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "6_channel_assignments.png"), bbox_inches="tight", dpi=150)
    _maybe_show(show)

    # 7. Traffic Load (active UEs per slot)
    ue_cols = [c for c in traffic_df.columns if c.startswith("ue_")]
    if len(ue_cols) > 0:
        active = traffic_df[ue_cols].sum(axis=1).values
        plt.figure(figsize=(18, 6))
        plt.plot(active)
        plt.title(f"Traffic Load Over Time (Active UEs per Slot)\n{data_sig}")
        plt.xlabel("Time Slot"); plt.ylabel("Active UEs"); plt.grid(alpha=0.3)
        plt.savefig(os.path.join(run_dir, "7_traffic_load.png"), bbox_inches="tight", dpi=150)
        _maybe_show(show)
    else:
        print("⚠️  Traffic CSV missing 'ue_' columns — skipping Traffic Load plot.")


# ========================= MAIN =========================
def main():
    args = parse_args()
    set_random_seed(args.seed)

    vec, env_cfg = build_env(args)
    model = build_model(args, vec)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, stamp)
    ensure_dir(run_dir)
    with open(os.path.join(run_dir, "env_config.json"), "w") as f:
        json.dump(env_cfg, f, indent=2)

    # ---- Training or Eval-only ----
    if not args.eval_only:
        print("🧠 Training / fine-tuning ...")
        model.learn(total_timesteps=args.timesteps)
        model.save(os.path.join(run_dir, "model.zip"))
        vec.save(os.path.join(run_dir, "vecnorm.pkl"))
        eval_vec = VecNormalize.load(os.path.join(run_dir, "vecnorm.pkl"),
                                     DummyVecEnv([lambda: SAMAEnvironment(**env_cfg)]))
    else:
        print("🧪 Eval-only mode: no training.")
        vec.training = False; vec.norm_reward = False
        # try to load vecnorm from resume folder
        base_dir = os.path.dirname(resolve_ckpt(args.resume_from)) or os.path.dirname(args.resume_from)
        norm_path = os.path.join(base_dir, "vecnorm.pkl") if base_dir else ""
        if norm_path and os.path.exists(norm_path):
            print(f"Loading VecNormalize stats from {norm_path}")
            eval_vec = VecNormalize.load(norm_path, DummyVecEnv([lambda: SAMAEnvironment(**env_cfg)]))
        else:
            print("⚠️  No vecnorm.pkl found; using fresh normalization.")
            eval_vec = DummyVecEnv([lambda: SAMAEnvironment(**env_cfg)])

    # ---------- EVALUATION ----------
    eval_vec.training = False
    eval_vec.norm_reward = False

    traffic_df = pd.read_csv(env_cfg["traffic_path"])
    if "slot" in traffic_df.columns:
        traffic_df = traffic_df.set_index("slot")
    data_sig = dataset_signature(env_cfg["traffic_path"], traffic_df)
    print("[DATASET]", data_sig)

    T = len(traffic_df)
    acts = []
    obs = eval_vec.reset()
    for _ in range(T):
        act, _ = model.predict(obs, deterministic=True)
        acts.append(act[0])
        obs, _, dones, _ = eval_vec.step(act)
        if np.all(dones): break

    actions = np.vstack(acts)
    np.save(os.path.join(run_dir, "actions.npy"), actions)

    states_df = pd.read_csv(env_cfg["state_path"])
    assign = reconstruct_assignments(actions, traffic_df, env_cfg["num_channels"])

    metrics = compute_metrics(assign, traffic_df, states_df, env_cfg["num_channels"])
    counts_all, counts_active, perc_all = channel_counts_from_assign(assign, env_cfg["num_channels"])
    metrics["channel_split_all"] = {str(k): int(v) for k, v in counts_all.items()}
    metrics["channel_split_active"] = {str(k): int(v) for k, v in counts_active.items()}
    metrics["channel_time_share_percent"] = {str(k): float(v) for k, v in perc_all.items()}
    metrics["dataset_signature"] = data_sig

    with open(os.path.join(run_dir, "metrics_computed.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    plot_and_save_all(
        run_dir=run_dir,
        metrics=metrics,
        actions=actions,
        assign=assign,
        traffic_df=traffic_df,
        states_df=states_df,
        num_channels=env_cfg["num_channels"],
        data_sig=data_sig,
        show=args.show_plots
    )

    print("✅ Results saved to:", run_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
