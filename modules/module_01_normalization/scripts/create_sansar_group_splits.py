"""Create leakage-aware splits from the bounded Sansar sample."""

from pathlib import Path
import sys
import random

import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
if str(MODULE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "src"))
SAMPLE_PATH = PROJECT_ROOT / "datasets" / "processed" / "sansar_source_aware_sample.parquet"
OUTPUT_DIRECTORY = PROJECT_ROOT / "datasets" / "splits"
SPLIT_PATH = OUTPUT_DIRECTORY / "sansar_source_aware_group_splits.parquet"
DUPLICATE_PATH = OUTPUT_DIRECTORY / "sansar_duplicate_text_exclusions.csv"

SPLIT_SEED = 42
SPLIT_RATIOS = {
    "train": 0.80,
    "validation": 0.10,
    "test": 0.10,
}


def assign_groups(candidate_df):
    """Assign whole sampling groups to splits with approximate row ratios."""
    group_sizes = candidate_df.groupby("sampling_group", sort=False).size()
    groups = list(group_sizes.items())

    random_generator = random.Random(SPLIT_SEED)
    random_generator.shuffle(groups)
    groups.sort(key=lambda item: item[1], reverse=True)

    total_rows = len(candidate_df)
    target_rows = {
        split_name: total_rows * ratio
        for split_name, ratio in SPLIT_RATIOS.items()
    }
    assigned_rows = {split_name: 0 for split_name in SPLIT_RATIOS}
    group_to_split = {}

    for group_name, group_size in groups:
        split_name = max(
            SPLIT_RATIOS,
            key=lambda name: (target_rows[name] - assigned_rows[name]) / target_rows[name],
        )
        group_to_split[group_name] = split_name
        assigned_rows[split_name] += group_size

    return group_to_split, assigned_rows


def main():
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Source-aware sample not found: {SAMPLE_PATH}")
    if SPLIT_PATH.exists() or DUPLICATE_PATH.exists():
        raise FileExistsError(
            "Split outputs already exist. Move or rename them before creating a new run."
        )

    sample_df = pd.read_parquet(SAMPLE_PATH)
    if "normalized_text" not in sample_df.columns:
        raise KeyError("The sample must contain a normalized_text column.")
    if "sampling_group" not in sample_df.columns:
        raise KeyError("The sample must contain a sampling_group column.")

    duplicate_mask = sample_df["normalized_text"].duplicated(keep="first")
    duplicate_rows = sample_df.loc[duplicate_mask].copy()
    candidate_df = sample_df.loc[~duplicate_mask].copy()

    first_rows = candidate_df.set_index("normalized_text")
    duplicate_ledger = duplicate_rows[["id", "sampling_group", "licence"]].copy()
    duplicate_ledger = duplicate_ledger.rename(columns={"id": "omitted_id"})
    duplicate_ledger["representative_id"] = duplicate_rows["normalized_text"].map(
        first_rows["id"]
    )
    duplicate_ledger["representative_group"] = duplicate_rows["normalized_text"].map(
        first_rows["sampling_group"]
    )

    group_to_split, assigned_rows = assign_groups(candidate_df)
    candidate_df["split"] = candidate_df["sampling_group"].map(group_to_split)

    split_count_by_group = candidate_df.groupby("sampling_group")["split"].nunique()
    if not split_count_by_group.eq(1).all():
        raise ValueError("A sampling group was assigned to more than one split.")

    split_count_by_text = candidate_df.groupby("normalized_text")["split"].nunique()
    if not split_count_by_text.eq(1).all():
        raise ValueError("A normalized text value crossed split boundaries.")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    candidate_df.to_parquet(SPLIT_PATH, index=False)
    duplicate_ledger.to_csv(DUPLICATE_PATH, index=False, encoding="utf-8")

    print(f"Rows in source-aware sample: {len(sample_df):,}")
    print(f"Unique text rows assigned: {len(candidate_df):,}")
    print(f"Duplicate rows excluded: {len(duplicate_rows):,}")
    print(f"Source/work groups assigned: {len(group_to_split):,}")
    print("Rows by split:")
    for split_name in SPLIT_RATIOS:
        rows = int(assigned_rows[split_name])
        print(f"  {split_name}: {rows:,} ({rows / len(candidate_df):.1%})")
    print(f"Saved split data: {SPLIT_PATH}")
    print(f"Saved duplicate ID ledger: {DUPLICATE_PATH}")
    print("Split unit: source/work group; this is not a source-held-out evaluation.")


if __name__ == "__main__":
    main()
