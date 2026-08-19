"""Shared helpers for Week 2 experiment notebooks.


Provides:
  - load_official_user_split(): load the canonical 60/20/20 user split
    from ``data/user_splits/`` (train_users.npy / val_users.npy / test_users.npy).
  - split_df_by_users(): partition a dataframe by user membership.
  - start_overwritable_run(): MLflow run wrapper that OVERWRITES the
    previous run recorded under the same experiment name + run name, so
    re-running the same experiment updates (rather than appends) the record.

This file does NOT change any model architecture, hyperparameters, or
feature engineering — it only provides split + MLflow plumbing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import mlflow
except Exception:  # pragma: no cover - mlflow may not be installed in some envs
    mlflow = None


def _project_root() -> Path:
    """Walk up to the directory containing pyproject.toml (portability convention)."""
    curr = Path.cwd()
    while not (curr / "pyproject.toml").exists() and curr != curr.parent:
        curr = curr.parent
    return curr


def _tracking_uri(root: Path | None = None) -> str:
    """Absolute sqlite tracking URI anchored to the project root, so MLflow
    always writes to <project_root>/mlflow.db regardless of the notebook's
    current working directory."""
    root = root or _project_root()
    return f"sqlite:///{(root / 'mlflow.db').as_posix()}"


def _fix_artifact_location_sql(experiment_id, root: Path):
    """Repair artifact locations in the SQLite DB: point the experiment and
    its runs at <project_root>/mlruns/<exp_id>/... instead of a stale path
    under the notebook CWD (which may not exist, breaking the UI runs list)."""
    import sqlite3

    db_path = _tracking_uri().removeprefix("sqlite:///").replace("\\", "/")
    try:
        con = sqlite3.connect(db_path, timeout=10)
        cur = con.cursor()
        root_s = root.as_posix()
        cur.execute(
            "UPDATE experiments SET artifact_location=? WHERE experiment_id=?",
            (f"file:{root_s}/mlruns/{experiment_id}", str(experiment_id)),
        )
        rows = cur.execute(
            "SELECT run_uuid FROM runs WHERE experiment_id=?", (str(experiment_id),)
        ).fetchall()
        for (ru,) in rows:
            cur.execute(
                "UPDATE runs SET artifact_uri=? WHERE run_uuid=?",
                (f"file:{root_s}/mlruns/{experiment_id}/{ru}/artifacts", ru),
            )
        con.commit()
        con.close()
    except Exception:
        pass


def _purge_runs_by_tag(client, experiment_id, run_name: str) -> int:
    """PERMANENTLY delete every run in ``experiment_id`` whose ``exp_run_name``
    tag equals ``run_name`` — including soft-deleted and RUNNING runs.

    This works against the SQLite backend (project root ``mlflow.db``): it
    resolves the runs table (the "runs" row key is ``run_uuid``), finds the
    matching run uuids via the ``tags`` table, and removes them plus all
    associated rows (metrics, params, tags, latest_metrics) so the run is
    truly gone from the UI instead of lingering as a deleted entry.

    Returns the number of runs purged. Runs of other names are untouched.
    """
    import sqlite3

    uri = _tracking_uri()
    db_path = uri.removeprefix("sqlite:///")
    # Use the client's tracking store to stay consistent.
    db_path = db_path.replace("\\", "/")

    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        # mlflow's runs table: run_uuid is the primary key; experiments in the
        # sqlite store may use either 'runs' (legacy) with run_uuid column.
        # Also handle the case where runs live in a table keyed by run_id.
        cur.execute(
            "SELECT r.run_uuid FROM runs r "
            "JOIN tags t ON t.run_uuid = r.run_uuid "
            "WHERE r.experiment_id = ? AND t.key = 'exp_run_name' AND t.value = ?",
            (str(experiment_id), run_name),
        )
        uuids = [row[0] for row in cur.fetchall()]

        if not uuids:
            return 0

        placeholders = ",".join("?" * len(uuids))
        # Related tables that reference run_uuid.
        for table in (
            "tags", "metrics", "params", "latest_metrics",
            "inputs", "input_tags", "logged_models",
        ):
            try:
                cur.execute(f"DELETE FROM {table} WHERE run_uuid IN ({placeholders})", uuids)
            except sqlite3.Error:
                pass
        try:
            cur.execute(f"DELETE FROM runs WHERE run_uuid IN ({placeholders})", uuids)
        except sqlite3.Error:
            pass
        con.commit()
        return len(uuids)
    finally:
        con.close()


def load_official_user_split(split_dir=None) -> dict:
    """Load the canonical user-level 60/20/20 split.

    Returns dict with keys 'train', 'val', 'test' -> np.ndarray of user ids.
    """
    root = _project_root()
    split_dir = Path(split_dir) if split_dir else root / "data" / "user_splits"
    files = {
        "train": split_dir / "train_users.npy",
        "val": split_dir / "val_users.npy",
        "test": split_dir / "test_users.npy",
    }
    out = {}
    for key, path in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Official split file not found: {path}. "
                f"Run week1/week1_official_split.ipynb first."
            )
        out[key] = np.load(path)
    return out


def split_df_by_users(df: pd.DataFrame, split: dict):
    """Partition ``df`` into train/val/test by the official user split.

    Returns (train_df, val_df, test_df, train_users, val_users, test_users).
    """
    train_df = df[df["user"].isin(split["train"])].copy()
    val_df = df[df["user"].isin(split["val"])].copy()
    test_df = df[df["user"].isin(split["test"])].copy()
    return train_df, val_df, test_df, split["train"], split["val"], split["test"]


# ---------------------------------------------------------------------------
# MLflow overwrite helper
# ---------------------------------------------------------------------------
def start_overwritable_run(
    experiment_name: str,
    run_name: str,
    tracking_uri: str | None = None,
    overwrite: bool = True,
):
    """Start an MLflow run that overwrites a previous run of the same name.

    ``overwrite=True`` (default) deletes any existing ACTIVE run in the
    experiment with the matching tag ``exp_run_name`` before starting a new
    one, so re-running the same experiment records new results in place of
    the old rather than appending a duplicate.

    ``tracking_uri`` defaults to the project-root ``mlflow.db`` (``sqlite:///``
    anchored to the directory containing pyproject.toml), so recording is
    consistent regardless of the notebook's working directory.

    Returns an ``mlflow.ActiveRun`` (caller should use ``with``/``end_run``).
    """
    if mlflow is None:
        raise ImportError("mlflow is not installed")

    if tracking_uri is None:
        tracking_uri = _tracking_uri()

    mlflow.set_tracking_uri(tracking_uri)

    # Reuse the existing experiment if present, else create it.
    # A soft-deleted experiment is restored so it can be re-used (the
    # default is to treat a previously-deleted experiment as new).
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)

    # Artifact root anchored to the project root (NOT the notebook CWD), so
    # the UI can always find run artifacts regardless of where the notebook
    # is executed from.
    root = _project_root()
    artifact_loc = f"file:{(root / 'mlruns').as_posix()}/{{experiment_id}}"

    if exp is None:
        exp_id = client.create_experiment(
            experiment_name,
            artifact_location=f"file:{(root / 'mlruns').as_posix()}",
        )
    else:
        exp_id = exp.experiment_id
        if exp.lifecycle_stage != "active":
            # Soft-deleted (via delete_experiment) -> restore so re-runs work.
            try:
                client.restore_experiment(exp_id)
            except Exception:
                pass
        # Repair a stale artifact_location (e.g. pointing at a missing dir
        # under the notebook CWD) so the UI renders runs reliably.
        if exp.artifact_location and "week2" in exp.artifact_location.replace("\\", "/"):
            try:
                client.update_experiment(
                    exp_id,
                    new_name=experiment_name,
                )
                _fix_artifact_location_sql(exp_id, root)
            except Exception:
                pass

    # Overwrite: permanently purge ANY prior run (active OR soft-deleted,
    # any status incl. stuck RUNNING) that carries this exact run-name tag,
    # so re-running the same experiment replaces its previous instance
    # rather than appending a duplicate. Runs of OTHER names are untouched.
    if overwrite:
        try:
            # 1) Hard-delete matching runs (soft-deleted ones included).
            _purge_runs_by_tag(client, exp_id, run_name)
        except Exception:
            # 2) Fallback: best-effort soft delete via the public API.
            try:
                runs = client.search_runs(
                    experiment_ids=[exp_id],
                    filter_string=f"tags.exp_run_name = '{run_name}'",
                )
                for r in runs:
                    try:
                        client.delete_run(r.info.run_id)
                    except Exception:
                        pass
            except Exception:
                pass

    # Idempotency: start the run in the (re)created experiment. We pass
    # experiment_id explicitly, so there is no need for mlflow.set_experiment.
    run = mlflow.start_run(run_name=run_name, experiment_id=exp_id)
    # Tag with the stable run name so a future overwrite can find it.
    mlflow.set_tag("exp_run_name", run_name)
    return run


def mlflow_available() -> bool:
    """True if mlflow is importable and settable."""
    try:
        import mlflow  # noqa: F401
        return True
    except Exception:
        return False
