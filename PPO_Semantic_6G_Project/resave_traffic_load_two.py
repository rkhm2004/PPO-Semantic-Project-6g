# resave_traffic_load_two.py
# Robust re-render of "Traffic Load (Active UEs per Slot)" for TWO run folders.
# Saves to plots_of_cn/channel_assignment/. Handles relative paths & missing keys.

import os, json, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def _guess_project_root(run_dir):
    parts = os.path.normpath(run_dir).split(os.sep)
    if "outputs" in parts:
        i = parts.index("outputs")
        return os.sep.join(parts[:i]) or os.getcwd()
    return os.getcwd()

def _resolve_path(candidate, run_dir):
    if not candidate:
        return None
    if os.path.isabs(candidate) and os.path.exists(candidate):
        return os.path.normpath(candidate)
    p1 = os.path.normpath(os.path.join(os.getcwd(), candidate))
    if os.path.exists(p1): return p1
    p2 = os.path.normpath(os.path.join(run_dir, candidate))
    if os.path.exists(p2): return p2
    proj = _guess_project_root(run_dir)
    p3 = os.path.normpath(os.path.join(proj, candidate))
    if os.path.exists(p3): return p3
    return None

def _search_fallback(run_dir, want_name=None):
    proj = _guess_project_root(run_dir)
    data_root = os.path.join(proj, "data")
    patterns = []
    if want_name:
        patterns.append(os.path.join(data_root, "**", os.path.basename(want_name)))
    patterns += [
        os.path.join(data_root, "**", "*traffic*model*.csv"),
        os.path.join(data_root, "**", "*traffic*holdout*.csv"),
        os.path.join(data_root, "**", "*traffic*.csv"),
    ]
    for pat in patterns:
        for h in glob.glob(pat, recursive=True):
            if os.path.isfile(h):
                return os.path.normpath(h)
    return None

def load_env_config(run_dir):
    cfg_path = os.path.join(run_dir, "env_config.json")
    if not os.path.exists(cfg_path):
        candidates = glob.glob(os.path.join(run_dir, "*", "env_config.json"))
        if candidates:
            cfg_path = max(candidates, key=os.path.getmtime)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"env_config.json not found in {run_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f), os.path.dirname(cfg_path)

def find_traffic_path(env_cfg, run_dir):
    cand = env_cfg.get("traffic_path") or env_cfg.get("traffic_csv") or env_cfg.get("traffic")
    resolved = _resolve_path(cand, run_dir) if cand else None
    if resolved and os.path.isfile(resolved):
        print(f"[OK] Using traffic CSV from config: {resolved}")
        return resolved
    fallback = _search_fallback(run_dir, want_name=cand)
    if fallback:
        print(f"[OK] Using fallback traffic CSV: {fallback}")
        return fallback
    raise FileNotFoundError(
        "Could not locate a traffic CSV from env_config or by searching under data/**. "
        f"Run folder: {run_dir}; candidate: {cand!r}"
    )

def _is_ue_col(name: str) -> bool:
    s = str(name).strip().lower()
    if not s.startswith("ue_"): 
        return False
    suf = s[3:]
    return suf.isdigit()

def _ue_index(name: str) -> int:
    s = str(name).strip().lower()
    try:
        return int(s.split("ue_")[1])
    except Exception:
        return 10**9

def load_traffic_df(traffic_csv):
    df = pd.read_csv(traffic_csv)
    if "slot" in df.columns: 
        df = df.set_index("slot")
    ue_cols = [c for c in df.columns if _is_ue_col(c)]
    if not ue_cols:
        raise ValueError(f"No UE columns like ue_0, ue_1 ... found in {traffic_csv}")
    ue_cols = sorted(ue_cols, key=_ue_index)
    x = df[ue_cols].copy()
    x.columns = [f"ue_{_ue_index(c)}" for c in ue_cols]
    x = x.fillna(0).astype(int).clip(0, 1)
    return x

def plot_traffic_load(run_dir, title, outdir, font=22, T=300):
    env_cfg, resolved_run_dir = load_env_config(run_dir)
    traffic_csv = find_traffic_path(env_cfg, resolved_run_dir)
    df = load_traffic_df(traffic_csv)
    active = df.sum(axis=1).to_numpy()

    # enforce horizon 300 (trim or pad with last value)
    if T is not None and np.isfinite(T):
        T = int(T)
        if len(active) >= T:
            active = active[:T]
        else:
            pad_val = active[-1] if active.size else 0
            active = np.pad(active, (0, T - len(active)), constant_values=pad_val)

    plt.close("all")
    plt.rcParams.update({
        "font.size": font,
        "axes.titlesize": font + 6,
        "axes.labelsize": font,
        "xtick.labelsize": font,
        "ytick.labelsize": font,
    })

    fig = plt.figure(figsize=(20, 6))
    ax = plt.gca()
    ax.plot(active)
    ax.set_title(title)
    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Active UEs")
    ax.grid(alpha=0.3)

    ensure_dir(outdir)
    slug = "".join(ch if ch.isalnum() else "_" for ch in title).strip("_")
    fout = os.path.join(outdir, f"traffic_load_{slug}.png")
    plt.tight_layout()
    plt.savefig(fout, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {fout}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs=2, required=True,
        help="Two run folders (either the exact run or a parent containing a timestamp subfolder)")
    ap.add_argument("--titles", nargs=2, required=True)
    ap.add_argument("--font", type=int, default=22)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--outdir", default=os.path.join("plots_of_cn", "channel_assignment"))
    return ap.parse_args()

def main():
    args = parse_args()
    for rd, title in zip(args.run_dirs, args.titles):
        try:
            plot_traffic_load(rd, title, args.outdir, font=args.font, T=args.T)
        except Exception as e:
            print(f"[FAIL] {title} | {rd}\n  -> {e}")

if __name__ == "__main__":
    main()
