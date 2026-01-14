# audit_two_checkpoints_single.py
# Compare two single-agent checkpoints on the SAME real dataset (apples-to-apples).
# - Accepts ckpt path as folder OR base name OR full *.zip.
# - Loads vecnorm.pkl if present; otherwise evaluates without normalization.
# - Deterministic evaluation, same horizon T for both.

import os, glob, json, argparse
import numpy as np
import pandas as pd
from math import sqrt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from SAMAEnvironment import SAMAEnvironment  # your env


# ------------------------------- Args ---------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--state_csv", required=True)
    p.add_argument("--traffic_csv", required=True)
    p.add_argument("--semantic_csv", default="")
    p.add_argument("--num_channels", type=int, default=3)
    p.add_argument("--num_ues", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)

    # Model A (e.g., Sim2Real zero-shot OR Real-only baseline)
    p.add_argument("--ckpt_a", required=True, help="Folder OR base name OR full *.zip of model A")
    p.add_argument("--vec_a", default="", help="Folder containing vecnorm.pkl for A (or file path)")

    # Model B (e.g., Fine-tuned/Hybrid)
    p.add_argument("--ckpt_b", required=True, help="Folder OR base name OR full *.zip of model B")
    p.add_argument("--vec_b", default="", help="Folder containing vecnorm.pkl for B (or file path)")
    return p.parse_args()


# --------------------------- Utilities --------------------------------
def latest_num_ues(traffic_csv, arg_num_ues):
    df = pd.read_csv(traffic_csv)
    ue_cols = [c for c in df.columns if str(c).startswith("ue_")]
    return arg_num_ues if arg_num_ues > 0 else len(ue_cols)

def make_env(cfg):
    return SAMAEnvironment(**cfg)

def resolve_ckpt(maybe_path: str) -> str:
    """
    Resolve a checkpoint to a real .zip file.
    Accepts:
      - folder (will search for model.zip / best_model.zip / *model*.zip),
      - base path without .zip (adds .zip or globs),
      - full path to a .zip.
    Returns: absolute path to *.zip
    Raises FileNotFoundError if none found.
    """
    p = os.path.normpath(maybe_path)

    # 1) exact .zip
    if p.lower().endswith(".zip") and os.path.isfile(p):
        return os.path.abspath(p)

    # 2) folder → search common names
    if os.path.isdir(p):
        candidates = []
        for name in ["model.zip", "best_model.zip", "*model*.zip"]:
            candidates.extend(glob.glob(os.path.join(p, name)))
        if candidates:
            return os.path.abspath(max(candidates, key=os.path.getmtime))

    # 3) base name without .zip
    if os.path.isfile(p + ".zip"):
        return os.path.abspath(p + ".zip")

    # 4) try glob in parent
    parent, base = os.path.dirname(p), os.path.basename(p)
    if os.path.isdir(parent):
        candidates = glob.glob(os.path.join(parent, base + "*.zip"))
        if candidates:
            return os.path.abspath(max(candidates, key=os.path.getmtime))

    raise FileNotFoundError(f"Could not resolve a checkpoint zip from '{maybe_path}'")

def load_vec(path_or_folder: str, base_env):
    """
    Try to load VecNormalize stats from:
      - a folder containing vecnorm.pkl
      - or a direct file path to vecnorm.pkl
    If not found, return base_env unchanged.
    """
    if not path_or_folder:
        return base_env

    p = os.path.normpath(path_or_folder)
    # folder
    if os.path.isdir(p):
        vp = os.path.join(p, "vecnorm.pkl")
        if os.path.isfile(vp):
            v = VecNormalize.load(vp, base_env)
            v.training = False
            v.norm_reward = False
            return v
    # direct file path
    if os.path.isfile(p) and p.lower().endswith(".pkl"):
        v = VecNormalize.load(p, base_env)
        v.training = False
        v.norm_reward = False
        return v

    # fallback: no vecnorm present
    return base_env


# --------------------------- Metrics -----------------------------------
def reconstruct_assignments(actions, traffic_df, num_channels):
    T, U = actions.shape
    assign = -1 * np.ones((U, T), dtype=np.int32)
    for t in range(T):
        active = (traffic_df.iloc[t].values[:U] == 1)
        for u in range(U):
            if active[u] == 1:
                assign[u, t] = int(actions[t, u])
    return assign

def compute_metrics(assign, traffic_df, states_df, num_channels):
    # align ue_id indexing once
    if "ue_id" in states_df.columns:
        states_df = states_df.sort_values("ue_id").reset_index(drop=True)

    U, T = assign.shape
    is_urllc = np.array([1 if states_df.iloc[i]["traffic_type"] == "URLLC" else 0 for i in range(U)], dtype=int)

    urllc_active = urllc_success = 0
    mmtc_active  = mmtc_served  = 0
    successes_total = 0
    collisions_channel_events = 0
    slots_with_collision = 0

    for t in range(T):
        active = (traffic_df.iloc[t].values[:U] == 1)
        ch = assign[:, t]
        had_col = False
        for cid in range(num_channels):
            users = np.where((ch == cid) & active)[0]
            if len(users) == 1:
                u = users[0]
                successes_total += 1
                if is_urllc[u] == 1:
                    urllc_success += 1
                else:
                    mmtc_served += 1
            elif len(users) > 1:
                collisions_channel_events += 1
                had_col = True
        if had_col:
            slots_with_collision += 1

        urllc_active += int(np.sum(active & (is_urllc == 1)))
        mmtc_active  += int(np.sum(active & (is_urllc == 0)))

    urrr = urllc_success / max(1, urllc_active)
    se   = successes_total / (T * num_channels)
    return dict(
        urllc_active=urllc_active,
        urllc_success=urllc_success,
        urllc_reliability=urrr,
        mmtc_active=mmtc_active,
        mmtc_served=mmtc_served,
        collision_slots=slots_with_collision,
        collision_rate_slots=slots_with_collision / float(T),
        collision_events=collisions_channel_events,
        spectral_efficiency=se,
        T=T,
    )

def ci95(p_hat, n):
    if n <= 0:
        return (0.0, 0.0)
    se = sqrt(p_hat * (1 - p_hat) / n)
    return (p_hat - 1.96 * se, p_hat + 1.96 * se)


# ---------------------------- Eval ------------------------------------
def eval_one(ckpt_path, vec_path, env_cfg, T_force=None):
    # base env
    base_env = DummyVecEnv([lambda: make_env(env_cfg)])
    # wrap vecnorm if present
    vec_env  = load_vec(vec_path or os.path.dirname(ckpt_path), base_env)

    # resolve and load model zip
    ckpt_zip = resolve_ckpt(ckpt_path)
    model    = PPO.load(ckpt_zip, env=vec_env, device="auto", print_system_info=False)

    # dataset to know rollout horizon
    traffic_df = pd.read_csv(env_cfg["traffic_path"])
    if "slot" in traffic_df.columns:
        traffic_df = traffic_df.set_index("slot")
    T_eval = len(traffic_df) if T_force is None else min(T_force, len(traffic_df))

    obs = vec_env.reset()
    acts = []
    for _ in range(T_eval):
        act, _ = model.predict(obs, deterministic=True)
        acts.append(act[0])
        obs, _, _, _ = vec_env.step(act)  # do not early-break; keep same horizon

    actions = np.vstack(acts)  # (T, U)
    states_df = pd.read_csv(env_cfg["state_path"])
    assign = reconstruct_assignments(actions, traffic_df.iloc[:T_eval], env_cfg["num_channels"])
    metrics = compute_metrics(assign, traffic_df.iloc[:T_eval], states_df, env_cfg["num_channels"])
    return metrics


# ----------------------------- Main -----------------------------------
def main():
    args = parse_args()

    # fixed env config for both models (same real dataset/seed)
    num_ues = latest_num_ues(args.traffic_csv, args.num_ues)
    env_cfg = dict(
        state_path=args.state_csv,
        traffic_path=args.traffic_csv,
        semantic_path=args.semantic_csv or None,
        num_channels=args.num_channels,
        num_ues=num_ues,
        is_real_time=False,
        reward_weights={"w_urlcc": 1.0, "pen_collision": 0.45, "pen_waste": 0.05},
        seed=args.seed,
    )

    # Evaluate A, then B on the SAME horizon
    A = eval_one(args.ckpt_a, args.vec_a, env_cfg, T_force=None)
    B = eval_one(args.ckpt_b, args.vec_b, env_cfg, T_force=A["T"])  # force equal T

    def pretty(name, M):
        lo, hi = ci95(M["urllc_reliability"], M["urllc_active"])
        return {
            "name": name,
            "UR_active": M["urllc_active"],
            "UR_success": M["urllc_success"],
            "URR": round(M["urllc_reliability"], 4),
            "URR_CI95": (round(lo, 4), round(hi, 4)),
            "mMTC_active": M["mmtc_active"],
            "mMTC_served": M["mmtc_served"],
            "collide_slots/T": round(M["collision_rate_slots"], 4),
            "collide_events": M["collision_events"],
            "SE": round(M["spectral_efficiency"], 4),
            "T": M["T"],
        }

    print("\n=== Apples-to-apples audit (same real dataset) ===")
    print(pretty("Model A", A))
    print(pretty("Model B", B))
    print("\nInterpretation tips:")
    print("- If URR jumps and collide_slots/T drops while SE rises, the improvement is real.")
    print("- If UR_active differs a lot between A and B, the denominator changed (check active-UE masking & labels).")
    print("- CI95 bands: non-overlap suggests a statistically solid improvement.")


if __name__ == "__main__":
    main()
