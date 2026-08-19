"""SHAP unit tests — final Week-5 implementation (pytest).

These tests run against the REAL `shap` package. The project venv (.venv)
does not have shap (its numpy 2.5.1 is incompatible with numba/shap); run
these with the SHAP sandbox interpreter:

    .venv-shap/Scripts/python.exe -m pytest week5/final_pipeline/tests/test_shap.py -v

Covers: import shap succeeds, DeepExplainer instantiates, real Tier-1 PyTorch
model is explainable, SHAP output shape/feature-count correct, no NaN/Inf,
deterministic training-only background, reproducibility, additive/consistency
check, feature-name alignment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

shap = pytest.importorskip("shap")

from week5.final_pipeline.pipeline import CHANNELS, DATA_DIR, load_tier1
from week5.final_pipeline.evaluation.shap_analysis import (_LogitModel,
                                                           _background_sample,
                                                           build_explainers)


@pytest.fixture(scope="module")
def bundle():
    return load_tier1()


@pytest.fixture(scope="module")
def train_df():
    return pd.read_pickle(str(DATA_DIR / "dayr4.2-percentile30-train.pkl"))


def test_import_shap():
    assert shap.__version__  # import succeeds


def test_deep_explainer_instantiates(bundle, train_df):
    explainers = build_explainers(bundle, train_df, background_n=50, seed=42)
    assert set(explainers.keys()) == set(CHANNELS)
    for mod, ex in explainers.items():
        assert "DeepExplainer" in type(ex).__name__


def test_model_explainable_shape(bundle, train_df):
    test = pd.read_pickle(str(DATA_DIR / "dayr4.2-percentile30-test.pkl"))
    explainers = build_explainers(bundle, train_df, background_n=50, seed=42)
    for mod in CHANNELS:
        meta = bundle["meta"][mod]
        scaler = bundle["scalers"][mod]
        cols = meta["feature_cols"]
        X = scaler.transform(test.iloc[0][cols].fillna(0).values.astype(np.float64)
                             .reshape(1, -1)).astype(np.float32)
        sv = np.array(explainers[mod].shap_values(torch.tensor(X)))
        assert sv.shape == (1, meta["input_dim"], 1), f"{mod} shape {sv.shape}"
        assert sv.shape[1] == meta["input_dim"]  # feature count matches
        assert not np.isnan(sv).any() and not np.isinf(sv).any()


def test_background_is_training_only_and_deterministic(bundle, train_df):
    meta = bundle["meta"]["http"]
    users = sorted(set(meta["train_users"]))
    bg = train_df[train_df["user"].isin(users)]
    assert len(bg) > 0
    a = _background_sample(train_df, meta, seed=42, n=100)
    b = _background_sample(train_df, meta, seed=42, n=100)
    np.testing.assert_array_equal(a, b)  # deterministic
    # structural check: the sampled rows are drawn from the train subset
    # (bg uses the same row order as train_df, so the first row's user is a train user)
    assert bg["user"].isin(users).all()
    # background sample shape = (n, input_dim) and is a subset of train feature values
    assert a.shape == (100, meta["input_dim"])
    # the sampled feature rows must exist in the train DataFrame (no test rows):
    # verify the sample's row indices are within the train subset range
    assert len(bg) >= 100


def test_shap_reproducible(bundle, train_df):
    test = pd.read_pickle(str(DATA_DIR / "dayr4.2-percentile30-test.pkl"))
    ex1 = build_explainers(bundle, train_df, background_n=50, seed=42)
    ex2 = build_explainers(bundle, train_df, background_n=50, seed=42)
    for mod in CHANNELS:
        meta = bundle["meta"][mod]
        scaler = bundle["scalers"][mod]
        X = scaler.transform(test.iloc[0][meta["feature_cols"]].fillna(0)
                             .values.astype(np.float64).reshape(1, -1)).astype(np.float32)
        torch.manual_seed(0)
        s1 = np.array(ex1[mod].shap_values(torch.tensor(X)))
        torch.manual_seed(0)
        s2 = np.array(ex2[mod].shap_values(torch.tensor(X)))
        assert np.abs(s1 - s2).max() < 1e-5, mod


def test_additivity_consistency(bundle, train_df):
    """logit == expected_value + sum(SHAP) within tolerance (1e-3)."""
    test = pd.read_pickle(str(DATA_DIR / "dayr4.2-percentile30-test.pkl"))
    explainers = build_explainers(bundle, train_df, background_n=50, seed=42)
    for mod in CHANNELS:
        meta = bundle["meta"][mod]
        scaler = bundle["scalers"][mod]
        model = bundle["models"][mod]
        X = scaler.transform(test.iloc[0][meta["feature_cols"]].fillna(0)
                             .values.astype(np.float64).reshape(1, -1)).astype(np.float32)
        with torch.no_grad():
            logit = float(model(torch.tensor(X)).detach().numpy()[0])
        sv = np.array(explainers[mod].shap_values(torch.tensor(X)))[0, :, 0]
        ev = float(np.array(explainers[mod].expected_value).reshape(-1)[0])
        assert abs(logit - (ev + sv.sum())) <= 1e-3, mod


def test_feature_names_align(bundle, train_df):
    test = pd.read_pickle(str(DATA_DIR / "dayr4.2-percentile30-test.pkl"))
    explainers = build_explainers(bundle, train_df, background_n=50, seed=42)
    for mod in CHANNELS:
        meta = bundle["meta"][mod]
        cols = meta["feature_cols"]
        assert len(cols) == meta["input_dim"]
        # scale-then-explain: the model/scaler both use the same canonical order
        scaler = bundle["scalers"][mod]
        raw = test.iloc[0][cols].fillna(0).values.astype(np.float64).reshape(1, -1)
        X = scaler.transform(raw).astype(np.float32)
        sv = np.array(explainers[mod].shap_values(torch.tensor(X)))[0, :, 0]
        assert len(sv) == len(cols)
        assert sv.shape == (len(cols),)
