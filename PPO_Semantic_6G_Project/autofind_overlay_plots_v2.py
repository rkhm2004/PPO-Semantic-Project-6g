# autofind_overlay_plots_v2.py
# Recursively discovers runs, auto-picks series, and overlays:
# - Single vs Multi agent × (Performance, Channel, Reward)
# Saves figures & a TSV catalog of discovered series.

import os, json, re, time, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
ROOTS = [
    r"D:\PROJECTS\SEM 3\CN\CN (2)\CN\PPO_Semantic_6G_Project",
    r"D:\PROJECTS\SEM 3\CN\CN (2)\CN",
]

AGENT_RULES = {
    "single": [r"\b(single|sa|output_sa|output_single|eval_only)\b"],
    "multi":  [r"\b(multi|marl|ma|output_marl|output_multi)\b"],
}

ENV_RULES = {
    "Real":      [r"\breal\b", r"\beval_real\b"],
    "Hybrid":    [r"\bhybrid\b", r"\bsynthetic\+?real\b", r"\baugmented\b", r"\bsyn(real)?\b"],
    "Sim2Real":  [r"\bsim2real\b", r"\bsim\-?to\-?real\b", r"\beval_only\b", r"\beval\_only\b"],
}

# Channel aggregation if per-user keys exist:
CHANNEL_MODE   = "count_nonzero"      # "auto" | "count_nonzero" | "mean"
CHANNEL_PREFIX = "ch_"                # e.g., ch_1, ch_u2, ch_user3

ROLLING_WINDOW = 1                    # set >= 10 for smoothing
OUT_DIR = "combined_plots"
os.makedirs(OUT_DIR, exist_ok=True)

LABELS = {
    "performance": {"x": "Step", "y": "Performance"},
    "channel":     {"x": "Step", "y": "Active Channels (count)"},
    "reward":      {"x": "Step", "y": "Reward"},
}

# ===================== HELPERS =====================
def norm(s): return s.replace("\\", "/").lower()
def path_matches(path, patterns): return any(re.search(rx, norm(path)) for rx in patterns)

def classify_agent(path):
    for agent, pats in AGENT_RULES.items():
        if path_matches(path, pats): return agent
    return "multi" if ("multi" in norm(path) or "marl" in norm(path)) else "single"

def classify_env(path):
    for env, pats in ENV_RULES.items():
        if path_matches(path, pats): return env
    p = norm(path)
    if "sim" in p and "real" in p: return "Sim2Real"
    if "hyb" in p or "syn" in p:   return "Hybrid"
    return "Real"

def safe_read_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return None

def discover_candidates(roots):
    hits = []
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            names = set(n.lower() for n in filenames)
            if any(n in names for n in ["env_config.json","infos.json","metrics_computed.json"]):
                hits.append({
                    "folder": dirpath,
                    "agent": classify_agent(dirpath),
                    "env":   classify_env(dirpath),
                    "count": sum(n in names for n in ["env_config.json","infos.json","metrics_computed.json"]),
                    "mtime": os.path.getmtime(dirpath)
                })
    return hits

def pick_best(hits):
    best = {}
    for h in hits:
        key = (h["agent"], h["env"])
        if key not in best or (h["count"], h["mtime"]) > (best[key]["count"], best[key]["mtime"]):
            best[key] = h
    return best

def load_triplet(folder):
    return (
        safe_read_json(os.path.join(folder, "env_config.json")),
        safe_read_json(os.path.join(folder, "infos.json")),
        safe_read_json(os.path.join(folder, "metrics_computed.json")),
    )

# ---- series discovery ----
def is_num_list(x):
    return isinstance(x, list) and len(x) > 1 and all(
        (isinstance(v, (int, float, np.floating, np.integer)) or v is None) for v in x
    )

def flatten_series(json_obj, prefix=""):
    """Return dict: key_path -> pandas.Series for every numeric list found."""
    found = {}
    if isinstance(json_obj, dict):
        for k, v in json_obj.items():
            kp = f"{prefix}.{k}" if prefix else k
            if is_num_list(v):
                found[kp] = pd.Series(v, dtype=float)
            elif isinstance(v, dict) or isinstance(v, list):
                found.update(flatten_series(v, kp))
    elif isinstance(json_obj, list):
        # list of dicts? collect per-key arrays by position
        if json_obj and isinstance(json_obj[0], dict):
            cols = {}
            for row in json_obj:
                for k, v in row.items():
                    cols.setdefault(k, []).append(v if isinstance(v,(int,float,np.floating,np.integer)) else np.nan)
            for k, arr in cols.items():
                kp = f"{prefix}.{k}" if prefix else k
                if sum(isinstance(v,(int,float,np.floating,np.integer)) for v in arr) >= 2:
                    found[kp] = pd.Series(arr, dtype=float)
    return found

def choose_performance(series_map):
    # Prefer keys hinting perf
    perf_like = [k for k in series_map.keys()
                 if re.search(r"(throughput|goodput|latency|delay|collision|col_rate|tx_rate|pkts|bits)", k, flags=re.I)]
    if perf_like:
        cand = {k: series_map[k] for k in perf_like}
        return max(cand.items(), key=lambda kv: np.nanvar(kv[1].values))[0]
    # fallback: exclude reward-ish names, then pick highest variance
    non_reward = {k:v for k,v in series_map.items() if not re.search(r"(reward|return|ret)", k, flags=re.I)}
    if non_reward:
        return max(non_reward.items(), key=lambda kv: np.nanvar(kv[1].values))[0]
    # last resort
    return max(series_map.items(), key=lambda kv: np.nanvar(kv[1].values))[0]

def choose_reward(series_map):
    reward_like = [k for k in series_map.keys() if re.search(r"(reward|return|episode_reward|ep[_\-]?rew)", k, flags=re.I)]
    if reward_like:
        cand = {k: series_map[k] for k in reward_like}
        return max(cand.items(), key=lambda kv: np.nanvar(kv[1].values))[0]
    ep_like = [k for k in series_map.keys() if re.search(r"(episode|ep_)", k, flags=re.I)]
    if ep_like:
        cand = {k: series_map[k] for k in ep_like}
        return max(cand.items(), key=lambda kv: np.nanvar(kv[1].values))[0]
    return max(series_map.items(), key=lambda kv: np.nanvar(kv[1].values))[0]

def extract_channel(infos_map, metrics_map):
    """
    Return a tuple (key_name, series) for channel assignment.
    - Prefer any scalar time-series whose name contains 'channel' or 'utilization'
      (searches infos_map, then metrics_map). Picks the one with highest variance.
    - Otherwise aggregate per-user channel keys (e.g., ch_1, ch_u2, ...) by count-nonzero or mean.
    - If nothing is found, return (None, None).
    """
    # 1) Ready-made scalar series containing 'channel' / 'utilization'
    for m in (infos_map, metrics_map):
        if not m:
            continue
        keys = [k for k in m.keys() if re.search(r"(channel|utilization)", k, flags=re.I)]
        if keys:
            cand = {k: m[k] for k in keys if isinstance(m[k], pd.Series) and len(m[k]) > 1}
            if cand:
                sel_key, sel_series = max(cand.items(), key=lambda kv: np.nanvar(kv[1].values))
                return sel_key, sel_series

    # 2) Per-user channels: ch_1, ch_u2, ch_user3, etc. Aggregate to one series.
    pat = re.compile(rf"{re.escape(CHANNEL_PREFIX)}.*?\d+$", flags=re.I)
    all_maps = {}
    if infos_map:   all_maps.update(infos_map)
    if metrics_map: all_maps.update(metrics_map)

    per_user_keys = [k for k in all_maps.keys() if pat.search(k.split(".")[-1])]
    if per_user_keys:
        series_like = {k: v for k, v in all_maps.items() if k in per_user_keys and isinstance(v, pd.Series)}
        if series_like:
            minlen = min(len(v) for v in series_like.values())
            if minlen >= 2:
                df = pd.DataFrame({k: v.reset_index(drop=True)[:minlen] for k, v in series_like.items()})
                agg = df.mean(axis=1) if CHANNEL_MODE == "mean" else (df != 0).sum(axis=1)
                keyname = f"__AGG__{CHANNEL_MODE}_from_{CHANNEL_PREFIX}"
                return keyname, agg

    # 3) Nothing found
    return None, None

def overlay_plot(series_map, title, x_label, y_label, outfile):
    valid = {k: v for k, v in series_map.items() if v is not None and len(v) > 0}
    if len(valid) < 1:
        print(f"[SKIP] {title}: no valid series.")
        return
    plt.figure(figsize=(8, 5))
    for label, s in valid.items():
        s = pd.Series(s).astype(float).reset_index(drop=True)
        x = pd.Series(np.arange(len(s)))
        if ROLLING_WINDOW > 1:
            s = s.rolling(ROLLING_WINDOW, min_periods=1).mean()
        plt.plot(x, s, label=label, linewidth=1.8)
    plt.title(title)
    plt.xlabel(x_label); plt.ylabel(y_label)
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    path = os.path.join(OUT_DIR, outfile)
    plt.savefig(path, dpi=300); plt.close()
    print(f"[OK] Saved {path}")

# ===================== MAIN =====================
def main():
    print("Scanning roots:")
    for r in ROOTS: print(" -", r)
    hits = discover_candidates(ROOTS)
    best = pick_best(hits)

    print("\n=== Selected run folders ===")
    if not best:
        print("No candidates found. Adjust ROOTS."); return
    for (agent, env), h in sorted(best.items()):
        print(f"- {agent:6s} | {env:9s} | jsons={h['count']} | mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(h['mtime']))} | {h['folder']}")

    catalog_rows = []

    def load_maps(folder):
        env_cfg, infos, metrics = load_triplet(folder)
        infos_map   = flatten_series(infos)   if infos   else {}
        metrics_map = flatten_series(metrics) if metrics else {}
        return infos_map, metrics_map

    for agent in ["single", "multi"]:
        # ------- Performance -------
        perf_map = {}
        for env in ["Real","Hybrid","Sim2Real"]:
            h = best.get((agent, env))
            if not h: continue
            infos_map, metrics_map = load_maps(h["folder"])

            # collect catalog
            for src_name, sm in [("infos", infos_map), ("metrics", metrics_map)]:
                for k, s in sm.items():
                    try:
                        var = float(np.nanvar(s.values))
                    except Exception:
                        var = float("nan")
                    catalog_rows.append([agent, env, src_name, k, len(s), var])

            all_series = {}
            all_series.update(infos_map); all_series.update(metrics_map)
            if not all_series:
                print(f"[WARN] {agent}/{env}: no numeric series found in {h['folder']}")
                continue
            perf_key = choose_performance(all_series)
            perf_map[env] = all_series[perf_key]

        overlay_plot(perf_map,
                     f"{agent.capitalize()} Agent – Performance (auto)",
                     LABELS["performance"]["x"], LABELS["performance"]["y"],
                     f"{agent}_performance_auto.png")

        # ------- Channel -------
        ch_map = {}
        for env in ["Real","Hybrid","Sim2Real"]:
            h = best.get((agent, env))
            if not h: continue
            infos_map, metrics_map = load_maps(h["folder"])
            ck, cs = extract_channel(infos_map, metrics_map)
            if cs is None:
                print(f"[WARN] {agent}/{env}: no channel series found in {h['folder']}")
                continue
            ch_map[env] = cs
        overlay_plot(ch_map,
                     f"{agent.capitalize()} Agent – Channel Assignment",
                     LABELS["channel"]["x"], LABELS["channel"]["y"],
                     f"{agent}_channel.png")

        # ------- Reward -------
        rw_map = {}
        for env in ["Real","Hybrid","Sim2Real"]:
            h = best.get((agent, env))
            if not h: continue
            infos_map, metrics_map = load_maps(h["folder"])
            all_series = {}
            all_series.update(infos_map); all_series.update(metrics_map)
            if not all_series:
                print(f"[WARN] {agent}/{env}: no series for reward in {h['folder']}")
                continue
            rw_key = choose_reward(all_series)
            rw_map[env] = all_series[rw_key]
        overlay_plot(rw_map,
                     f"{agent.capitalize()} Agent – Reward (auto)",
                     LABELS["reward"]["x"], LABELS["reward"]["y"],
                     f"{agent}_reward_auto.png")

    # write catalog for inspection
    if catalog_rows:
        df = pd.DataFrame(catalog_rows, columns=["agent","env","source","key_path","length","variance"])
        df.sort_values(["agent","env","source","variance","length"], ascending=[True,True,True,False,False], inplace=True)
        tsv = os.path.join(OUT_DIR, "discovered_series.tsv")
        df.to_csv(tsv, sep="\t", index=False)
        print(f"\n[OK] Wrote a catalog of discovered series → {tsv}")
        print("Open it to see the exact key names I used or can use.")

if __name__ == "__main__":
    main()
