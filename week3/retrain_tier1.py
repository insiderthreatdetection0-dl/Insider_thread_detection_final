"""Retrain the four finalized Tier-1 MLP agents and persist canonical artifacts.

Artifacts (single source of truth, Step 15):
    tier1/http/{model.pt, scaler.joblib, meta.json}
    tier1/file_device/{...}
    tier1/email/{...}
    tier1/logon/{...}

Finalized design (DO NOT CHANGE): HTTP 6:1, File/Device 4:1, Email 4:1,
Logon 4:1 — ModalityFFNN on percentile-30 features, StandardScaler fit on the
60% train users only, 20 epochs, Adam(0.001), batch 128, SEED=42.

Run:
    uv run python -m week3.retrain_tier1
"""
import json
import time
from pathlib import Path

from week3 import tier1_lib as T

def main():
    t0 = time.time()
    print(f"Device: {T.DEVICE} | Project root: {T.PROJECT_ROOT}")

    train_df, val_df, test_df = T.load_pre_split_data()
    split_info = T.verify_split(train_df, val_df, test_df)
    print(f"Split verified: {split_info}")

    T.TIER1_DIR.mkdir(parents=True, exist_ok=True)
    T.SCORES_DIR.mkdir(parents=True, exist_ok=True)

    # --- train each finalized modality on TRAIN users only ---
    print("\n=== TRAINING FINALIZED TIER-1 AGENTS ===")
    for mod in T.MODALITIES:
        model, scaler, feature_cols, metrics = T.train_modality(mod, train_df)
        T.assert_feature_consistency(model, scaler, feature_cols, mod)
        split_info_userlists = {
            "train_users": train_df["user"].unique(),
            "val_users": val_df["user"].unique(),
            "test_users": test_df["user"].unique(),
        }
        # verify split sizes (the split_info from verify_split is shape-only;
        # user lists must come from the official split files for meta.json)
        tr_u, va_u, te_u = T.load_official_split()
        split_for_meta = {
            "train_users": [int(u) for u in tr_u],
            "val_users": [int(u) for u in va_u],
            "test_users": [int(u) for u in te_u],
            "malicious": split_info["malicious"],
        }
        d = T.save_artifacts(mod, model, scaler, feature_cols, metrics, split_for_meta)
        print(f"  saved {mod} -> {d}")

    # --- verify all four artifacts round-trip ---
    print("\n=== VERIFYING ARTIFACTS ===")
    bundles = T.load_all_artifacts()
    for mod, (model, scaler, feature_cols, meta) in bundles.items():
        assert meta["sampling_ratio"] == T.RATIO_LABEL[mod]
        print(f"  {mod:12s} input_dim={meta['input_dim']} ratio={meta['sampling_ratio']} "
              f"scaler_mean={len(scaler.mean_)} feature_cols={len(feature_cols)} OK")

    # --- MLflow (do NOT delete/overwrite historical runs) ---
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{T.PROJECT_ROOT / 'mlflow.db'}")
        mlflow.set_experiment("Week2_MLP_Percentile_Agents")
        with mlflow.start_run(run_name="finalized_tier1_retrain"):
            mlflow.log_param("seed", T.SEED)
            mlflow.log_param("purpose", "finalized reproducible Tier-1 retrain")
            for mod, (model, scaler, feature_cols, meta) in bundles.items():
                mlflow.log_param(f"{mod}_ratio", meta["sampling_ratio"])
                mlflow.log_param(f"{mod}_input_dim", meta["input_dim"])
                mlflow.log_param(f"{mod}_feature_cols", json.dumps(feature_cols))
                mlflow.log_artifact(str(T.modality_dir(mod)))
            mlflow.log_artifact(str(T.TIER1_DIR / "scores"))
        print("MLflow run logged: finalized_tier1_retrain")
    except Exception as e:
        print(f"[WARN] MLflow logging skipped: {e}")

    print(f"\nDONE in {time.time()-t0:.1f}s. Artifacts in {T.TIER1_DIR}")


if __name__ == "__main__":
    main()
