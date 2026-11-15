"""Stream and normalize the pinned Sansar corpus into Parquet shards."""

from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import sys


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
if str(MODULE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "src"))

import datasets
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features, Value, load_dataset

from normalization import normalize_text


DATASET_ID = "MuseMesh/sansar-sanskrit-corpus"
DATASET_REVISION = "a268a87223c38575883c7ee6af25b1d2e65fd0d7"
CONFIG_NAME = "default"
SPLIT_NAME = "train"
BATCH_SIZE = 10_000

OUTPUT_DIRECTORY = PROJECT_ROOT / "datasets" / "processed" / "sansar_normalized"
MANIFEST_PATH = OUTPUT_DIRECTORY / "manifest.json"


def write_manifest(manifest):
    """Write progress atomically so an interrupted run can resume."""
    temporary_path = OUTPUT_DIRECTORY / "manifest.json.tmp"
    temporary_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(MANIFEST_PATH)


def make_output_features(source_features):
    """Keep the source schema while retaining raw and normalized text."""
    feature_fields = {}
    for field_name, field_type in source_features.items():
        if field_name == "text":
            feature_fields["raw_text"] = field_type
        else:
            feature_fields[field_name] = field_type
    feature_fields["normalized_text"] = Value("string")
    output_features = Features(feature_fields)
    return output_features.arrow_schema


def convert_record(record, source_field_names):
    """Copy metadata and add raw_text and normalized_text fields."""
    output_record = {}
    for field_name in source_field_names:
        value = record.get(field_name)
        if field_name == "text":
            output_record["raw_text"] = value
            output_record["normalized_text"] = normalize_text(value)
        else:
            output_record[field_name] = value
    return output_record


def write_shard(rows, schema, shard_number):
    """Write one batch to a Parquet shard, then rename it into place."""
    shard_name = f"part-{shard_number:05d}.parquet"
    final_path = OUTPUT_DIRECTORY / shard_name
    temporary_path = OUTPUT_DIRECTORY / f"{shard_name}.tmp"

    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, temporary_path, compression="zstd")
    temporary_path.replace(final_path)
    return shard_name


def start_or_resume():
    """Create a manifest or validate progress from an interrupted run."""
    if not OUTPUT_DIRECTORY.exists():
        OUTPUT_DIRECTORY.mkdir(parents=True)
        manifest = {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "config": CONFIG_NAME,
            "split": SPLIT_NAME,
            "batch_size": BATCH_SIZE,
            "records_written": 0,
            "shards_written": 0,
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "datasets_version": datasets.__version__,
            "pandas_version": pd.__version__,
            "pyarrow_version": pa.__version__,
        }
        write_manifest(manifest)
        return manifest

    if not MANIFEST_PATH.exists():
        raise FileExistsError(
            f"Output directory already exists without a manifest: {OUTPUT_DIRECTORY}"
        )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_settings = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "config": CONFIG_NAME,
        "split": SPLIT_NAME,
        "batch_size": BATCH_SIZE,
    }
    for setting_name, expected_value in expected_settings.items():
        if manifest.get(setting_name) != expected_value:
            raise ValueError(
                f"Manifest setting {setting_name!r} does not match this run."
            )

    if manifest.get("status") == "complete":
        print("The full-corpus normalization is already complete.")
        print(f"Records written: {manifest['records_written']:,}")
        return None

    for shard_number in range(manifest["shards_written"]):
        shard_path = OUTPUT_DIRECTORY / f"part-{shard_number:05d}.parquet"
        if not shard_path.exists():
            raise FileNotFoundError(f"Expected completed shard is missing: {shard_path}")

    print(f"Resuming after {manifest['records_written']:,} records.")
    return manifest


def main():
    manifest = start_or_resume()
    if manifest is None:
        return

    dataset_stream = load_dataset(
        DATASET_ID,
        name=CONFIG_NAME,
        split=SPLIT_NAME,
        revision=DATASET_REVISION,
        streaming=True,
    )
    source_field_names = list(dataset_stream.features.keys())
    if "text" not in source_field_names:
        raise KeyError(f"Expected a 'text' field; found {source_field_names}")

    output_schema = make_output_features(dataset_stream.features)
    stream_iterator = iter(dataset_stream)

    rows_to_skip = manifest["records_written"]
    for _ in range(rows_to_skip):
        try:
            next(stream_iterator)
        except StopIteration as error:
            raise ValueError("The source stream ended before the saved checkpoint.") from error

    batch = []
    records_written = rows_to_skip
    shard_number = manifest["shards_written"]

    for source_record in stream_iterator:
        batch.append(convert_record(source_record, source_field_names))
        if len(batch) >= BATCH_SIZE:
            shard_name = write_shard(batch, output_schema, shard_number)
            records_written += len(batch)
            shard_number += 1
            manifest["records_written"] = records_written
            manifest["shards_written"] = shard_number
            manifest["last_shard"] = shard_name
            write_manifest(manifest)
            print(f"Wrote {records_written:,} records ({shard_name}).")
            batch = []

    if batch:
        shard_name = write_shard(batch, output_schema, shard_number)
        records_written += len(batch)
        shard_number += 1
        manifest["last_shard"] = shard_name

    manifest["records_written"] = records_written
    manifest["shards_written"] = shard_number
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["output_columns"] = list(output_schema.names)
    write_manifest(manifest)

    print(f"Normalization complete: {records_written:,} records.")
    print(f"Parquet shards: {shard_number:,}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
