# train_marl.py — CTDE (MAPPO) training entry
import os, json, argparse
from datetime import datetime
import pandas as pd
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
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="outputs/output_multi_agent")
    p.add_argument("--resume_from", default="", help="checkpoint path (omit .zip)")
    p.add_argument("--sim2real", action="store_true")
    return p.parse_args()


def latest_num_ues(traffic_csv, arg_num_ues):
    df = pd.read_csv(traffic_csv)
    ue_cols = [c for c in df.columns if c.startswith("ue_")]
    return arg_num_ues if arg_num_ues > 0 else len(ue_cols)


def build_env(args):
    num_ues = latest_num_ues(args.traffic_csv, args.num_ues)
    env_cfg = dict(
        state_path=args.state_csv,
        traffic_path=args.traffic_csv,
        semantic_path=args.semantic_csv or None,
        num_channels=args.num_channels,
        num_ues=num_ues,
        local_obs_dim=4,
    )
    vec = DummyVecEnv([lambda: MultiAgent6GEnv(**env_cfg)])
    vec = VecNormalize(vec, training=True, norm_obs=True, norm_reward=True, clip_obs=5.0)
    return vec, env_cfg


def build_model(args, vec, num_ues):
    if args.resume_from:
        model = PPO.load(args.resume_from, env=vec, device="auto", print_system_info=False)
        if args.sim2real:
            model.policy.optimizer = type(model.policy.optimizer)(
                model.policy.parameters(), lr=5e-5, weight_decay=1e-5
            )
            model.ent_coef = 0.005
        return model

    # Pass num_ues/local_obs_dim to policy via policy_kwargs
    policy_kwargs = dict(num_ues=num_ues, local_obs_dim=4)
    model = PPO(
        CTDEPolicy, vec,
        learning_rate=3e-4, n_steps=1024, batch_size=1024, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, seed=args.seed,
        policy_kwargs=policy_kwargs
    )
    return model


def main():
    args = parse_args()
    set_random_seed(args.seed)

    vec, env_cfg = build_env(args)
    num_ues = env_cfg["num_ues"]
    model = build_model(args, vec, num_ues)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, stamp)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "env_config.json"), "w") as f:
        json.dump(env_cfg, f, indent=2)

    model.learn(total_timesteps=args.timesteps)
    model.save(os.path.join(run_dir, "model.zip"))
    vec.save(os.path.join(run_dir, "vecnorm.pkl"))

    print("✅ CTDE MAPPO training saved to:", run_dir)


if __name__ == "__main__":
    main()
