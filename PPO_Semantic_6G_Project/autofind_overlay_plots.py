import os, json, re, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== CONFIG (EDIT THIS PART) =====================

# 1) Put one or more base directories here to scan recursively.
ROOTS = [
    r"D:\PROJECTS\SEM 3\CN\CN (2)\CN\PPO_Semantic_6G_Project",
    r"D:\PROJECTS\SEM 3\CN\CN (2)\CN",  # add more if needed
]

# 2) Keyword rules to detect agent/env from folder path (case-insensitive).
AGENT_RULES = {
    "single": [r"\b(single|sa|output_single|eval_only)\b"],
    "multi":  [r"\b(multi|marl|ma|output_multi)\b"],
}

ENV_RULES = {
    "Real":      [r"\breal\b", r"\beval_real\b"],
    "Hybrid":    [r"\bhybrid\b", r"\bsyn(real)?\b", r"\bsynthetic\+?real\b", r"\baugmented\b"],
    "Sim2Real":  [r"\bsim2real\b", r"\bsim\-?to\-?real\b", r"\beval_only\b", r"\beval\_only\b"],
}

# 3) Preferred performance metric (fallbacks will be auto-detected).
PERF_METRIC = "throughput"

# 4) Channel aggregation
CHANNEL_MODE   = "count_nonzero"  # "auto" | "count_nonzero" | "mean"
CHANNEL_PREFIX = "ch_"            # for per-user channel keys like ch_1, ch_u2, ...

# 5) Reward key candidates
REWARD_KEY_CANDIDATES = ["reward_total","total_reward","reward","ep_reward","return","episode_reward"]

# 6) Optional smoothing
ROLLING_WINDOW = 1  # e.g., 20 to smooth

# 7) Output folder
OUT_DIR = "combined_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# 8) Axis labels
LABELS = {
    "performance": {"x": "Step", "y": "Performance"},
    "channel":     {"x": "Step", "y": "Active Channels (count)"},
    "reward":      {"x": "Step", "y": "Reward"},
}

# ===================== INTERNAL HELPERS =====================

def norm(s: str) -> str:
    return s.replace("\\", "/").lower()

def path_matches(path, patterns):
    p = norm(path)
    return any(re.search(rx, p) for rx in patterns)

def classify_agent(path):
    for agent, pats in AGENT_RULES.items():
        if path_matches(path, pats):
            return agent
    # gentle fallback: guess from 'multi' substring
    p = norm(path)
    return "multi" if "multi" in p or "marl" in p else "single"

def classify_env(path):
    for env, pats in ENV_RULES.items():
        if path_matches(path, pats):
            return env
    # fallback heuristics
    p = norm(path)
    if "real" in p and ("sim" not in p and "syn" not in p and "hyb" not in p):
        return "Real"
    if "sim" in p and "real" in p:
        return "Sim2Real"
    if "hyb" in p or "syn" in p:
        return "Hybrid"
    return "Real"

def safe_read_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def count_jsons(folder):
    return sum(os.path.isfile(os.path.join(folder, fn)) for fn in
               ["env_config.json", "infos.json", "metrics_computed.json"])

def discover_candidates(roots):
    """
    Recursively find folders that contain at least one of the three JSONs.
    Return list of dicts: {folder, agent, env, count, mtime}
    """
    hits = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            names = set([n.lower() for n in filenames])
            if any(n in names for n in ["env_config.json", "infos.json", "metrics_computed.json"]):
                agent = classify_agent(dirpath)
                env   = classify_env(dirpath)
                cnt   = sum(x in names for x in ["env_config.json", "infos.json", "metrics_computed.json"])
                try:
                    mtime = os.path.getmtime(dirpath)
                except Exception:
                    mtime = 0
                hits.append({
                    "folder": dirpath,
                    "agent": agent,
                    "env": env,
                    "count": cnt,
                    "mtime": mtime
                })
    return hits

def pick_best(hits):
    """
    From discovered hits, pick best folder per (agent, env): most JSONs, then newest.
    """
    best = {}
    for h in hits:
        key = (h["agent"], h["env"])
        if key not in best:
            best[key] = h
        else:
            b = best[key]
            if (h["count"], h["mtime"]) > (b["count"], b["mtime"]):
                best[key] = h
    return best

def ensure_step_len(n):
    return pd.Series(np.arange(n), name="step")

def smooth(series):
    if ROLLING_WINDOW and ROLLING_WINDOW > 1:
        return series.rolling(ROLLING_WINDOW, min_periods=1).mean()
    return series

def find_first_series(d, candidates):
    if d is None:
        return None, None
    def try_key(node, k):
        if isinstance(node, dict):
            for kk, vv in node.items():
                if kk.lower() == k.lower():
                    if isinstance(vv, list) and vv and np.isscalar(vv[0]):
                        return pd.Series(vv)
                    if isinstance(vv, list) and vv and isinstance(vv[0], dict) and k in vv[0]:
                        return pd.Series([row.get(k, np.nan) for row in vv])
        return None
    q = [d]
    while q:
        cur = q.pop(0)
        for k in candidates:
            s = try_key(cur, k)
            if s is not None:
                return k, s
        if isinstance(cur, dict):
            q.extend(cur.values())
        elif isinstance(cur, list):
            q.extend(cur)
    return None, None

def extract_all_scalars_with_prefix(d, prefix_regex):
    if d is None:
        return None
    pat = re.compile(prefix_regex, flags=re.IGNORECASE)
    rows = []
    def visit(node):
        if isinstance(node, list):
            for el in node:
                visit(el)
        elif isinstance(node, dict):
            keys = [k for k in node.keys() if pat.match(k)]
            if keys:
                rows.append({k: node.get(k, 0) for k in keys})
            for v in node.values():
                visit(v)
    visit(d)
    if not rows:
        return None
    all_keys = sorted(set().union(*[r.keys() for r in rows]))
    data = []
    for r in rows:
        data.append([r.get(k, 0) for k in all_keys])
    return pd.DataFrame(data, columns=all_keys)

def load_triplet(folder):
    env_cfg = safe_read_json(os.path.join(folder, "env_config.json"))
    infos   = safe_read_json(os.path.join(folder, "infos.json"))
    metrics = safe_read_json(os.path.join(folder, "metrics_computed.json"))
    return env_cfg, infos, metrics

def extract_performance(infos, metrics, pref="throughput"):
    alt = {
        "throughput": ["throughput","tx_rate","goodput","pkts_per_s","bits_per_s"],
        "latency":    ["latency","delay_ms","avg_latency","mean_delay","rt_latency"],
        "collision":  ["collision","collisions","collision_rate","col_rate"]
    }
    cand = [pref] + alt.get(pref.lower(), [])
    k, s = find_first_series(metrics, cand)
    if s is not None: return k, s
    k, s = find_first_series(infos, cand)
    if s is not None: return k, s
    for bag in [["performance","perf"], ["throughput","latency"], ["metric","value"]]:
        k, s = find_first_series(metrics, bag)
        if s is not None: return k, s
        k, s = find_first_series(infos, bag)
        if s is not None: return k, s
    return None, None

def extract_reward(infos, metrics):
    k, s = find_first_series(infos, REWARD_KEY_CANDIDATES)
    if s is not None: return k, s
    k, s = find_first_series(metrics, REWARD_KEY_CANDIDATES)
    if s is not None: return k, s
    return None, None

def extract_channel(infos, metrics):
    if CHANNEL_MODE == "auto":
        for bag in [["active_channels","channel_utilization","channels_active","num_active_channels"]]:
            k, s = find_first_series(infos, bag)
            if s is not None: return s
            k, s = find_first_series(metrics, bag)
            if s is not None: return s
    regex = r"^" + re.escape(CHANNEL_PREFIX) + r".*?\d+$"
    for src in (infos, metrics):
        df = extract_all_scalars_with_prefix(src, regex)
        if df is not None and len(df):
            return df.mean(axis=1) if CHANNEL_MODE == "mean" else (df != 0).sum(axis=1)
    for bag in [["channel","assigned_channel","ch"]]:
        k, s = find_first_series(infos, bag)
        if s is not None: return s
        k, s = find_first_series(metrics, bag)
        if s is not None: return s
    return None

def overlay_plot(series_map, title, x_label, y_label, outfile):
    valid = {k: v for k, v in series_map.items() if v is not None and len(v) > 0}
    if len(valid) < 1:
        print(f"[SKIP] {title}: no valid series.")
        return
    plt.figure(figsize=(8, 5))
    for label, series in valid.items():
        s = pd.Series(series).astype(float).reset_index(drop=True)
        x = ensure_step_len(len(s))
        plt.plot(x, smooth(s), label=label, linewidth=1.8)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, outfile)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[OK] Saved {path}")

def build_plots(best_map):
    # pretty print selection table
    print("\n=== Selected run folders ===")
    rows = []
    for (agent, env), h in sorted(best_map.items()):
        rows.append([agent, env, h['count'], time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(h['mtime'])), h['folder']])
    if rows:
        try:
            import textwrap
            from pprint import pprint
        except Exception:
            pass
        for r in rows:
            print(f"- {r[0]:6s} | {r[1]:9s} | jsons={r[2]} | mtime={r[3]} | {r[4]}")
    else:
        print("No candidates found. Check ROOTS in the script.")
        return

    for agent in ["single", "multi"]:
        # Performance
        perf_map = {}
        for env in ["Real","Hybrid","Sim2Real"]:
            h = best_map.get((agent, env))
            if not h: continue
            env_cfg, infos, metrics = load_triplet(h["folder"])
            k, s = extract_performance(infos, metrics, PERF_METRIC)
            if s is None:
                print(f"[WARN] {agent}/{env}: no performance series found in {h['folder']}")
            perf_map[env] = s
        overlay_plot(
            perf_map,
            f"{agent.capitalize()} Agent – Performance ({PERF_METRIC})",
            LABELS["performance"]["x"], LABELS["performance"]["y"],
            f"{agent}_performance_{PERF_METRIC}.png"
        )

        # Channel
        ch_map = {}
        for env in ["Real","Hybrid","Sim2Real"]:
            h = best_map.get((agent, env))
            if not h: continue
            env_cfg, infos, metrics = load_triplet(h["folder"])
            s = extract_channel(infos, metrics)
            if s is None:
                print(f"[WARN] {agent}/{env}: no channel series found in {h['folder']}")
            ch_map[env] = s
        overlay_plot(
            ch_map,
            f"{agent.capitalize()} Agent – Channel Assignment",
            LABELS["channel"]["x"], LABELS["channel"]["y"],
            f"{agent}_channel.png"
        )

        # Reward
        rw_map = {}
        for env in ["Real","Hybrid","Sim2Real"]:
            h = best_map.get((agent, env))
            if not h: continue
            env_cfg, infos, metrics = load_triplet(h["folder"])
            k, s = extract_reward(infos, metrics)
            if s is None:
                print(f"[WARN] {agent}/{env}: no reward series found in {h['folder']}")
            rw_map[env] = s
        overlay_plot(
            rw_map,
            f"{agent.capitalize()} Agent – Reward",
            LABELS["reward"]["x"], LABELS["reward"]["y"],
            f"{agent}_reward.png"
        )

def main():
    print("Scanning roots:")
    for r in ROOTS: print(" -", r)
    hits = discover_candidates(ROOTS)
    if not hits:
        print("\nNo JSON folders found. Add/adjust ROOTS.")
        return
    best = pick_best(hits)
    build_plots(best)
    print("\nDone. Check:", os.path.abspath(OUT_DIR))

if __name__ == "__main__":
    main()
