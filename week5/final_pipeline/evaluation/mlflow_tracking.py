"""MLflow tracking for the final Week-5 run (final implementation).

Creates a final Week-5 experiment/run logging parameters (seed, split, HC,
correlation threshold, fusion weights, LLM model, SHAP method/version/
background size, Python version), metrics (system metrics + explanation
comparison + degradation), and artifacts (system_metrics.csv,
system_predictions.csv, shap_attributions.csv, explanation_comparison.csv,
degradation_results.csv, final_week5_summary.json).

Idempotent: re-running updates the same experiment. No test-set tuning.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow

from ..pipeline import OUT, WEEK4_FINAL, LLM_MODEL

EXPERIMENT_NAME = "week5_final_pipeline"


def _load_json(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> dict:
    # params
    summary = _load_json("final_week5_summary.json")
    shap_meta = _load_json("shap_meta.json")
    cmp_summary = _load_json("explanation_comparison_summary.json")
    metrics_df = None
    if (OUT / "system_metrics.csv").exists():
        import pandas as pd
        metrics_df = pd.read_csv(OUT / "system_metrics.csv")

    params = {
        "seed": 42,
        "split": "60/20/20",
        "HC_threshold": 0.60,
        "correlation_threshold": 0.30,
        "fusion_weights": "http=0.50,file_device=0.20,logon=0.15,email=0.15",
        "llm_model": LLM_MODEL,
        "shap_method": shap_meta.get("shap_method", "shap.DeepExplainer"),
        "shap_version": shap_meta.get("shap_version", ""),
        "shap_background_n": shap_meta.get("background_n", ""),
        "shap_background_source": shap_meta.get("background_source", ""),
        "python_version": sys.version.split()[0],
        "tier2_input": "week4/frozen_inputs/improved_tier2_test_predictions.csv",
        "tier3_outputs": "week4/final_llm_reasoning/outputs/llm_triage_outputs_final.csv",
    }

    metrics = {}
    if metrics_df is not None:
        for _, r in metrics_df.iterrows():
            metrics[str(r["metric"]).replace("tier2_", "")] = float(r["value"])
    # explanation comparison metrics
    for k in ("channel_jaccard_mean", "feature_jaccard_mean", "unsupported_claim_rate",
              "supported_claim_rate", "channel_f1", "feature_f1"):
        if k in cmp_summary:
            metrics[k] = float(cmp_summary[k])
    # degradation metrics
    if (OUT / "degradation_results.csv").exists():
        import pandas as pd
        deg = pd.read_csv(OUT / "degradation_results.csv")
        for _, r in deg.iterrows():
            if r["missing"] != "none":
                metrics[f"degraded_{r['missing']}_recall"] = float(r["recall"])

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="final_week5") as run:
        mlflow.log_params(params)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        for artifact in ("system_metrics.csv", "system_predictions.csv",
                         "shap_attributions.csv", "explanation_comparison.csv",
                         "degradation_results.csv", "final_week5_summary.json",
                         "shap_meta.json", "shap_sanity.csv"):
            p = OUT / artifact
            if p.exists():
                mlflow.log_artifact(str(p))
        mlflow.log_param("experiment", EXPERIMENT_NAME)
        run_id = run.info.run_id

    print(f"\nMLflow run logged: experiment='{EXPERIMENT_NAME}' run_id={run_id}")
    print(f"  params: {len(params)} | metrics: {len(metrics)} | artifacts: 8")
    return {"experiment": EXPERIMENT_NAME, "run_id": run_id,
            "n_params": len(params), "n_metrics": len(metrics)}


if __name__ == "__main__":
    main()
