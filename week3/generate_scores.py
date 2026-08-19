"""Generate NEW Tier-1 scores from the canonical artifacts (Step 10/11).

Loads the four finalized artifacts (tier1/<mod>/model.pt + scaler.joblib +
meta.json), scores the official validation and test splits from
data/user_split_formlp/, and writes:

    tier1/scores/tier1_validation_scores.csv
    tier1/scores/tier1_test_scores.csv

Optionally also mirrors to week3/tier1_{validation,test}_scores.csv so the
existing downstream consumers (Tier-2/LangGraph/week5) keep working — use
--write-week3 to enable.

The scaler is LOADED (never refit). Feature order comes from meta.json.
Run twice (--tag run1 / --tag run2) to demonstrate reproducibility.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from week3 import tier1_lib as T


def score_split(df, bundles, modality):
    """Score one split (val or test) for one modality, returning a score Series."""
    model, scaler, feature_cols, meta = bundles[modality]
    scores = T.score_dataframe(df, model, scaler, feature_cols)
    return pd.Series(scores, index=df.index, name=f"{modality}_score")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="suffix for output filenames (run1/run2)")
    ap.add_argument("--write-week3", action="store_true",
                    help="also mirror CSVs to week3/tier1_{validation,test}_scores.csv")
    ap.add_argument("--out-dir", default=str(T.SCORES_DIR))
    args = ap.parse_args()

    tag = f"_{args.tag}" if args.tag else ""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load artifacts (verify consistency inside) ---
    print("Loading finalized artifacts...")
    bundles = T.load_all_artifacts()
    for mod, (model, scaler, feature_cols, meta) in bundles.items():
        print(f"  {mod:12s} ratio={meta['sampling_ratio']} input_dim={meta['input_dim']} "
              f"feature_cols={len(feature_cols)} scaler_mean={len(scaler.mean_)}")

    # --- load + verify split ---
    train_df, val_df, test_df = T.load_pre_split_data()
    split_info = T.verify_split(train_df, val_df, test_df)
    print(f"Split verified: {split_info}")

    # --- score validation and test ---
    results = {}
    for split_name, df in [("validation", val_df), ("test", test_df)]:
        parts = []
        for mod in T.MODALITIES:
            s = score_split(df, bundles, mod)
            parts.append(pd.DataFrame({f"{mod}_score": s}))
        out = pd.concat(parts, axis=1)
        out["user"] = df["user"].values
        out["week"] = df["week"].values
        out["day"] = df["day"].values
        out["label"] = (df["insider"].values > 0).astype(int)
        out = out[["user", "week", "day", "label"] +
                  [f"{m}_score" for m in T.MODALITIES]]
        # logon_score first to preserve the documented schema order
        out = out[["user", "week", "day", "label",
                   "logon_score", "email_score", "file_device_score", "http_score"]]
        out = out.sort_values(["user", "week", "day"]).reset_index(drop=True)
        results[split_name] = out
        path = out_dir / f"tier1_{split_name}_scores{tag}.csv"
        out.to_csv(path, index=False)
        print(f"  saved {split_name}: {out.shape[0]} rows, {out['user'].nunique()} users, "
              f"{int(out['label'].sum())} positives -> {path}")

    # --- Step 11 checks ---
    print("\n=== STEP 11 CHECKS ===")
    for split_name, out in results.items():
        # Check A: keys/dupes/missing
        assert list(out.columns[:4]) == ["user", "week", "day", "label"]
        assert out["user"].notna().all() and out["day"].notna().all()
        dupe = out.duplicated(subset=["user", "week", "day"]).sum()
        assert dupe == 0, f"{split_name}: {dupe} duplicate (user,week,day) keys"
        # Check B: score range
        for m in T.MODALITIES:
            s = out[f"{m}_score"]
            assert (s >= 0).all() and (s <= 1).all(), f"{split_name} {m}: scores out of [0,1]"
            print(f"  {split_name} {m:12s} min={s.min():.6f} max={s.max():.6f} "
                  f"mean={s.mean():.6f} std={s.std():.6f}")
        print(f"  {split_name}: rows={len(out)} users={out['user'].nunique()} "
              f"positives={int(out['label'].sum())} dupes={dupe}")

    if args.write_week3:
        week3_dir = T.PROJECT_ROOT / "week3"
        for split_name in ["validation", "test"]:
            src = out_dir / f"tier1_{split_name}_scores{tag}.csv"
            dst = week3_dir / f"tier1_{split_name}_scores.csv"
            dst.write_bytes(src.read_bytes())
            print(f"  mirrored {src.name} -> {dst}")
    print("\nSCORE GENERATION COMPLETE.")


if __name__ == "__main__":
    main()
