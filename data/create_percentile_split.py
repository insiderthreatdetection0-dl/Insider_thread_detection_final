"""Create a deterministic 60/20/20 user-level split of the percentile pickle.

Replicates EXACTLY the split logic used by week3/tier1_score_generation.ipynb
(stratified user-level 60/20/20, random_state=42, stratify on per-user max
insider label) so the MLP Tier-1 agents see identical user assignments.

Source : data/dayr4.2-percentile30.pkl  (NOT modified)
Outputs: data/dayr4.2-percentile30-train.pkl
         data/dayr4.2-percentile30-validation.pkl
         data/dayr4.2-percentile30-test.pkl
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent
SRC_PKL = DATA_DIR / "dayr4.2-percentile30.pkl"
SEED = 42  # same seed as week3/tier1_score_generation.ipynb (SEED=42 / random_state=42)

OUT_TRAIN = DATA_DIR / "dayr4.2-percentile30-train.pkl"
OUT_VAL = DATA_DIR / "dayr4.2-percentile30-validation.pkl"
OUT_TEST = DATA_DIR / "dayr4.2-percentile30-test.pkl"


def load_source():
    """Load the original percentile pickle (read-only)."""
    if not SRC_PKL.exists():
        raise FileNotFoundError(f"Source percentile pickle not found: {SRC_PKL}")
    df = pd.read_pickle(str(SRC_PKL))
    return df


def split_users(df):
    """Deterministic stratified 60/20/20 user-level split (tier-1 logic).

    Returns (train_users, val_users, test_users) as numpy arrays.
    """
    user_labels = df.groupby("user")["insider"].max().astype(int)
    unique_users = user_labels.index.values

    # 1) 60% Train vs 40% Temp (stratified by per-user insider label)
    train_users, temp_users = train_test_split(
        unique_users, test_size=0.4, random_state=SEED, stratify=user_labels.values)

    # 2) Temp split into two equal halves -> 20% Validation, 20% Test
    val_users, test_users = train_test_split(
        temp_users, test_size=0.5, random_state=SEED,
        stratify=user_labels.loc[temp_users].values)

    return train_users, val_users, test_users


def verify_no_overlap(train_users, val_users, test_users):
    t, v, te = set(train_users), set(val_users), set(test_users)
    assert not (t & v), "Train ∩ Validation must be empty"
    assert not (t & te), "Train ∩ Test must be empty"
    assert not (v & te), "Validation ∩ Test must be empty"
    return len(t), len(v), len(te)


def summarize(name, users, df_part):
    print(f"  {name:<12} users={len(users):>5}  rows={len(df_part):>7}")


def main():
    print("=" * 64)
    print("PERCENTILE PICKLE 60/20/20 USER-LEVEL SPLIT")
    print("=" * 64)
    print(f"Source : {SRC_PKL}")
    print(f"Seed   : {SEED} (from week3/tier1_score_generation.ipynb)")

    df = load_source()
    total_users = df["user"].nunique()
    total_rows = len(df)
    print(f"Source shape: {df.shape} | unique users: {total_users}")

    # Run the split twice to prove determinism.
    split_a = split_users(df)
    split_b = split_users(df)
    assert all(np.array_equal(a, b) for a, b in zip(split_a, split_b)), \
        "Split is NOT deterministic across runs!"
    train_users, val_users, test_users = split_a
    print("Determinism check: two identical split runs -> identical user assignments (OK)")

    # Partition the dataframe by user membership.
    train_df = df[df["user"].isin(train_users)].copy()
    val_df = df[df["user"].isin(val_users)].copy()
    test_df = df[df["user"].isin(test_users)].copy()

    # Verify each split contains exactly its users' full record sets.
    for name, users, part in (("Train", train_users, train_df),
                              ("Validation", val_users, val_df),
                              ("Test", test_users, test_df)):
        part_users = set(part["user"].unique())
        assert part_users == set(users), f"{name} partition user mismatch"
    print("User integrity: every user's full record set is in exactly one split (OK)")

    # Save (do NOT overwrite the source).
    train_df.to_pickle(str(OUT_TRAIN))
    val_df.to_pickle(str(OUT_VAL))
    test_df.to_pickle(str(OUT_TEST))

    # Summary.
    n = total_users
    print("\n" + "=" * 64)
    print("SPLIT SUMMARY")
    print("=" * 64)
    print(f"{'Split':<12}{'Users':>7}{'Rows':>10}{'Users %':>9}")
    print("-" * 42)
    print(f"{'Train':<12}{len(train_users):>7}{len(train_df):>10}{100*len(train_users)/n:>8.1f}%")
    print(f"{'Validation':<12}{len(val_users):>7}{len(val_df):>10}{100*len(val_users)/n:>8.1f}%")
    print(f"{'Test':<12}{len(test_users):>7}{len(test_df):>10}{100*len(test_users)/n:>8.1f}%")
    print("-" * 42)
    print(f"{'Total':<12}{total_users:>7}{total_rows:>10}")

    # Row integrity: all rows accounted for.
    assert len(train_df) + len(val_df) + len(test_df) == total_rows, \
        "Row counts do not sum to the original total!"
    print(f"\nRow integrity: {len(train_df)} + {len(val_df)} + {len(test_df)} = {total_rows} (OK)")

    # Overlap verification.
    t_n, v_n, te_n = verify_no_overlap(train_users, val_users, test_users)
    print(f"User overlap: Train&Validation=0, Train&Test=0, Validation&Test=0 "
          f"({t_n}/{v_n}/{te_n} users) (OK)")

    print("\nSaved output files (source untouched):")
    for p in (OUT_TRAIN, OUT_VAL, OUT_TEST):
        print(f"  {p}  ({p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
