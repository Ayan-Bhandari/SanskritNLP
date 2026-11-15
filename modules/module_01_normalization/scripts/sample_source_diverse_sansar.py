"""Create a bounded source/work-aware sample from the Sansar corpus."""

from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
import sys

import pandas as pd
from datasets import load_dataset


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
if str(MODULE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "src"))

from normalization import normalize_text


DATASET_ID = "MuseMesh/sansar-sanskrit-corpus"
DATASET_REVISION = "a268a87223c38575883c7ee6af25b1d2e65fd0d7"
CONFIG_NAME = "default"
SPLIT_NAME = "train"

MAX_SCAN_ROWS = 50_000
MAX_SAMPLE_ROWS = 5_000
MAX_ROWS_PER_GROUP = 300
SHUFFLE_BUFFER_SIZE = 10_000
SHUFFLE_SEED = 42


def make_group_key(record):
    """Group by source/work, falling back to source when work is missing."""
    source = record.get("source")
    if source is None or str(source).strip() == "":
        source = "<missing source>"
    else:
        source = str(source).strip()

    work = record.get("work")
    if work is None or str(work).strip() == "":
        return (source, "<source-level>")

    return (source, str(work).strip())


def main():
    print(f"Dataset: {DATASET_ID}")
    print(f"Pinned revision: {DATASET_REVISION}")
    print(f"Maximum yielded records scanned: {MAX_SCAN_ROWS:,}")
    print(f"Maximum records saved: {MAX_SAMPLE_ROWS:,}")

    stream = load_dataset(
        DATASET_ID,
        name=CONFIG_NAME,
        split=SPLIT_NAME,
        revision=DATASET_REVISION,
        streaming=True,
    )
    shuffled_stream = stream.shuffle(
        seed=SHUFFLE_SEED,
        buffer_size=SHUFFLE_BUFFER_SIZE,
    )

    selected_by_group = defaultdict(list)
    scanned_rows = 0

    for record in islice(shuffled_stream, MAX_SCAN_ROWS):
        scanned_rows += 1
        group_key = make_group_key(record)

        if len(selected_by_group[group_key]) >= MAX_ROWS_PER_GROUP:
            continue

        selected_record = dict(record)
        selected_record["sampling_group"] = " | ".join(group_key)
        selected_by_group[group_key].append(selected_record)

        selected_rows = sum(len(rows) for rows in selected_by_group.values())
        if selected_rows >= MAX_SAMPLE_ROWS:
            break

    selected_records = []
    for group_rows in selected_by_group.values():
        selected_records.extend(group_rows)

    if not selected_records:
        raise RuntimeError("The bounded scan did not return any records.")

    sample_df = pd.DataFrame.from_records(selected_records)
    if "text" not in sample_df.columns:
        raise KeyError(f"Expected a 'text' field; got {list(sample_df.columns)}")

    sample_df = sample_df.rename(columns={"text": "raw_text"})
    sample_df["normalized_text"] = sample_df["raw_text"].map(normalize_text)

    print(f"Records scanned: {scanned_rows:,}")
    print(f"Records selected: {len(sample_df):,}")
    print(f"Source/work groups selected: {len(selected_by_group):,}")
    print("Rows per sampling group:")
    group_counts = Counter(sample_df["sampling_group"])
    for group_name, count in group_counts.most_common():
        print(f"  {group_name}: {count}")

    if "licence" in sample_df.columns:
        print("Rows by licence:")
        for licence, count in sample_df["licence"].value_counts(dropna=False).items():
            print(f"  {licence}: {count}")

    processed_directory = PROJECT_ROOT / "datasets" / "processed"
    processed_directory.mkdir(parents=True, exist_ok=True)
    output_path = processed_directory / "sansar_source_aware_sample.parquet"
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing sample: {output_path}"
        )

    sample_df.to_parquet(output_path, index=False)
    print(f"Saved sample to: {output_path}")

    if len(selected_by_group) < 10:
        print(
            "Warning: fewer than 10 source/work groups were found. "
            "Do not use this sample for a source-held-out 80/10/10 split."
        )


if __name__ == "__main__":
    main()
