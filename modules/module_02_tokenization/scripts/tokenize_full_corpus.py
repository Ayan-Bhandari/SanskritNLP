"""Create resumable BPE token-ID shards for the complete Sansar train split."""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "processed" / "sansar_normalized"
MANIFEST_PATH = DATASET_ROOT / "manifest.json"
MODEL_PATH = PROJECT_ROOT / "models" / "module_02_tokenization" / "full_corpus" / "bpe_vocab8192.json"
MODEL_META_PATH = MODEL_PATH.with_name("bpe_vocab8192.training.json")
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "tokenized" / "sansar_default_train_bpe8192"
REPORT_ROOT = PROJECT_ROOT / "experiments" / "tokenization" / "full_corpus"
EXPECTED_DATASET = "MuseMesh/sansar-sanskrit-corpus"
EXPECTED_CONFIG = "default"
EXPECTED_SPLIT = "train"
MODEL_TYPE = "bpe"
VOCAB_SIZE = 8192
ARROW_BATCH_ROWS = 256
ENCODE_BATCH_ROWS = 64
ENCODE_BATCH_CHARACTERS = 250_000
PROGRESS_EVERY_ROWS = 10_000


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_path, path)


def load_source():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing Module 1 manifest: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = {
        "dataset_id": EXPECTED_DATASET,
        "config": EXPECTED_CONFIG,
        "split": EXPECTED_SPLIT,
        "status": "complete",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Manifest {key} is {manifest.get(key)!r}; expected {value!r}"
            )
    paths = sorted(DATASET_ROOT.glob("part-*.parquet"))
    if len(paths) != manifest.get("shards_written"):
        raise ValueError(
            f"Found {len(paths)} source shards; manifest expects "
            f"{manifest.get('shards_written')}"
        )
    rows_found = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
    if rows_found != manifest["records_written"]:
        raise ValueError(
            f"Source shards contain {rows_found:,} rows; manifest reports "
            f"{manifest['records_written']:,}"
        )
    return manifest, paths


def load_tokenizer(manifest):
    if not MODEL_PATH.is_file() or not MODEL_META_PATH.is_file():
        raise FileNotFoundError(
            "The full-corpus BPE model is missing. Run "
            "train_full_corpus_tokenizers.py --vocab-size 8192 --models bpe first."
        )
    training_metadata = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    if (
        training_metadata.get("training_rows") != manifest["records_written"]
        or training_metadata.get("dataset_revision") != manifest["dataset_revision"]
    ):
        raise ValueError("The local BPE model was not fitted on this exact dataset revision")
    tokenizer = Tokenizer.from_file(str(MODEL_PATH))
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise ValueError(
            f"BPE vocabulary has {tokenizer.get_vocab_size()} entries; expected {VOCAB_SIZE}"
        )
    return tokenizer, sha256_file(MODEL_PATH)


def uniform_sample_positions(total_rows, sample_rows):
    if sample_rows < 2:
        return {0}
    return {
        round(index * (total_rows - 1) / (sample_rows - 1))
        for index in range(sample_rows)
    }


def encode_batches(tokenizer, examples, batch_rows=ENCODE_BATCH_ROWS):
    """Encode complete records in bounded batches; never split a record."""
    texts = []
    owners = []
    total_characters = 0

    def emit():
        encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
        yield from zip(owners, encodings)

    for owner, text in examples:
        if not isinstance(text, str):
            raise ValueError(f"normalized_text for row {owner} is not a string")
        if texts and (
            len(texts) >= batch_rows
            or total_characters + len(text) > ENCODE_BATCH_CHARACTERS
        ):
            yield from emit()
            texts = []
            owners = []
            total_characters = 0
        texts.append(text)
        owners.append(owner)
        total_characters += len(text)
    if texts:
        yield from emit()


def run_pilot(tokenizer, paths, manifest, sample_rows):
    selected = uniform_sample_positions(manifest["records_written"], sample_rows)
    processed = 0
    token_total = 0
    encoded_rows = []
    started = time.perf_counter()
    started_at = utc_now()
    encoding_seconds = 0.0
    next_report = 1_000
    print(
        f"Timing pilot: encoding {len(selected):,} evenly spaced rows "
        "from the complete train split...",
        flush=True,
    )
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(
            columns=["normalized_text"], batch_size=ARROW_BATCH_ROWS
        ):
            batch_texts = batch.column(0).to_pylist()
            examples = [
                (processed + offset, text)
                for offset, text in enumerate(batch_texts)
                if processed + offset in selected
            ]
            processed += len(batch_texts)
            encoding_started = time.perf_counter()
            encoded_batch = list(encode_batches(tokenizer, examples))
            encoding_seconds += time.perf_counter() - encoding_started
            for row_index, encoding in encoded_batch:
                encoded_rows.append((row_index, len(encoding.ids)))
                token_total += len(encoding.ids)
            if len(encoded_rows) >= next_report:
                print(f"  pilot encoded {len(encoded_rows):,}/{len(selected):,} rows", flush=True)
                next_report += 1_000

    duration = time.perf_counter() - started
    if processed != manifest["records_written"] or len(encoded_rows) != len(selected):
        raise ValueError(
            f"Pilot scanned {processed:,} rows and encoded {len(encoded_rows):,}/"
            f"{len(selected):,} selected rows"
        )
    estimated_encode_seconds = (
        encoding_seconds / len(encoded_rows) * manifest["records_written"]
    )
    estimated_end_to_end_seconds = duration + max(
        0.0, estimated_encode_seconds - encoding_seconds
    )
    report = {
        "mode": "timing_pilot",
        "dataset_id": manifest["dataset_id"],
        "dataset_revision": manifest["dataset_revision"],
        "split": manifest["split"],
        "source_rows_scanned": processed,
        "sample_rows_encoded": len(encoded_rows),
        "sample_token_instances": token_total,
        "sample_mean_tokens_per_row": token_total / len(encoded_rows),
        "pilot_seconds": duration,
        "pilot_encoding_seconds": encoding_seconds,
        "estimated_encoding_hours": estimated_encode_seconds / 3600,
        "estimated_end_to_end_hours_before_output_write_overhead": (
            estimated_end_to_end_seconds / 3600
        ),
        "pilot_rows_per_second": len(encoded_rows) / duration,
        "model_sha256": sha256_file(MODEL_PATH),
        "started_utc": started_at,
        "completed_utc": utc_now(),
        "note": "Uniform-row estimate separates one complete source scan from sample encoding; Parquet output writing adds overhead and row lengths vary.",
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    pilot_path = REPORT_ROOT / "bpe_full_corpus_tokenization_pilot.json"
    atomic_json_write(pilot_path, report)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Pilot report: {pilot_path}", flush=True)


def make_output_schema(id_type, metadata):
    schema_metadata = {
        key.encode("utf-8"): str(value).encode("utf-8")
        for key, value in metadata.items()
    }
    return pa.schema(
        [
            pa.field("id", id_type),
            pa.field("source_row_index", pa.uint32()),
            pa.field("token_ids", pa.list_(pa.uint16())),
            pa.field("token_count", pa.uint32()),
        ],
        metadata=schema_metadata,
    )


def check_output_metadata(path, source_name, source_size, expected_rows, revision, model_sha):
    if not path.is_file():
        return False
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.schema_arrow.metadata or {}
    expected = {
        "source_shard": source_name,
        "source_size_bytes": str(source_size),
        "source_rows": str(expected_rows),
        "dataset_revision": revision,
        "model_sha256": model_sha,
    }
    actual = {key.decode("utf-8"): value.decode("utf-8") for key, value in metadata.items()}
    return (
        parquet_file.metadata.num_rows == expected_rows
        and all(actual.get(key) == value for key, value in expected.items())
        and parquet_file.schema_arrow.names
        == ["id", "source_row_index", "token_ids", "token_count"]
    )


def load_progress(progress_path, manifest, model_sha, paths):
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        expected_identity = {
            "dataset_id": manifest["dataset_id"],
            "dataset_revision": manifest["dataset_revision"],
            "config": manifest["config"],
            "split": manifest["split"],
            "model_sha256": model_sha,
            "vocabulary_size": VOCAB_SIZE,
            "tokenizers_version": __import__("tokenizers").__version__,
            "source_shards": [path.name for path in paths],
        }
        for key, value in expected_identity.items():
            if progress.get(key) != value:
                raise ValueError(
                    f"Resume state {key} does not match current input/model; "
                    "use a fresh output directory for a different run."
                )
        return progress
    return {
        "dataset_id": manifest["dataset_id"],
        "dataset_revision": manifest["dataset_revision"],
        "config": manifest["config"],
        "split": manifest["split"],
        "model_type": MODEL_TYPE,
        "model_sha256": model_sha,
        "vocabulary_size": VOCAB_SIZE,
        "tokenizers_version": __import__("tokenizers").__version__,
        "source_shards": [path.name for path in paths],
        "rows_expected": manifest["records_written"],
        "rows_completed": 0,
        "token_ids_completed": 0,
        "completed_shards": {},
        "started_utc": utc_now(),
        "updated_utc": utc_now(),
        "status": "in_progress",
    }


def tokenize_shard(tokenizer, source_path, output_path, manifest, model_sha):
    source_file = pq.ParquetFile(source_path)
    source_rows = source_file.metadata.num_rows
    source_size = source_path.stat().st_size
    if "id" not in source_file.schema_arrow.names or "normalized_text" not in source_file.schema_arrow.names:
        raise ValueError(f"Source shard lacks id or normalized_text: {source_path.name}")
    id_type = source_file.schema_arrow.field("id").type
    schema = make_output_schema(
        id_type,
        {
            "source_shard": source_path.name,
            "source_size_bytes": source_size,
            "source_rows": source_rows,
            "dataset_revision": manifest["dataset_revision"],
            "model_sha256": model_sha,
            "tokenizer_type": MODEL_TYPE,
            "vocabulary_size": VOCAB_SIZE,
            "tokenizers_version": __import__("tokenizers").__version__,
            "id_storage_type": "uint16; cast to int64 when creating model tensors",
        },
    )
    temporary_path = output_path.with_name(output_path.name + ".inprogress")
    if temporary_path.exists():
        temporary_path.unlink()
    writer = pq.ParquetWriter(
        temporary_path,
        schema,
        compression="zstd",
        compression_level=3,
        use_dictionary=["id"],
        write_statistics=True,
    )
    rows_written = 0
    token_total = 0
    max_row_tokens = 0
    min_token_id = None
    max_token_id = None
    started = time.perf_counter()
    try:
        for batch in source_file.iter_batches(
            columns=["id", "normalized_text"], batch_size=ARROW_BATCH_ROWS
        ):
            ids = batch.column(0).to_pylist()
            texts = batch.column(1).to_pylist()
            encoded = list(
                encode_batches(
                    tokenizer,
                    enumerate(texts, start=rows_written),
                    batch_rows=ENCODE_BATCH_ROWS,
                )
            )
            expected_indices = list(range(rows_written, rows_written + len(texts)))
            if [row_index for row_index, _ in encoded] != expected_indices:
                raise ValueError("Adaptive encoding changed source row order")
            token_lists = [encoding.ids for _, encoding in encoded]
            token_counts = [len(token_ids) for token_ids in token_lists]
            for token_ids in token_lists:
                if any(token_id < 0 or token_id > 65535 for token_id in token_ids):
                    raise ValueError("Token ID cannot be represented by uint16")
                if token_ids:
                    token_min = min(token_ids)
                    token_max = max(token_ids)
                    min_token_id = token_min if min_token_id is None else min(min_token_id, token_min)
                    max_token_id = token_max if max_token_id is None else max(max_token_id, token_max)
            table = pa.Table.from_arrays(
                [
                    pa.array(ids, type=id_type),
                    pa.array(range(rows_written, rows_written + len(ids)), type=pa.uint32()),
                    pa.array(token_lists, type=pa.list_(pa.uint16())),
                    pa.array(token_counts, type=pa.uint32()),
                ],
                schema=schema,
            )
            writer.write_table(table, row_group_size=ARROW_BATCH_ROWS)
            rows_written += len(ids)
            token_total += sum(token_counts)
            if token_counts:
                max_row_tokens = max(max_row_tokens, max(token_counts))
            if rows_written % PROGRESS_EVERY_ROWS == 0:
                print(
                    f"  {source_path.name}: {rows_written:,}/{source_rows:,} rows; "
                    f"{token_total:,} token IDs",
                    flush=True,
                )
        writer.close()
        if rows_written != source_rows:
            raise ValueError(
                f"Wrote {rows_written:,} rows from {source_path.name}; expected {source_rows:,}"
            )
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            writer.close()
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        raise
    return {
        "source_shard": source_path.name,
        "output_shard": output_path.name,
        "source_rows": source_rows,
        "token_ids": token_total,
        "max_tokens_in_row": max_row_tokens,
        "min_token_id": min_token_id,
        "max_token_id": max_token_id,
        "output_size_bytes": output_path.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_utc": utc_now(),
    }


def run_full(tokenizer, paths, manifest, model_sha, max_shards=None):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    progress_path = OUTPUT_ROOT / "progress.json"
    progress = load_progress(progress_path, manifest, model_sha, paths)
    shards_this_run = 0
    for source_path in paths:
        output_path = OUTPUT_ROOT / source_path.name
        source_rows = pq.ParquetFile(source_path).metadata.num_rows
        source_size = source_path.stat().st_size
        completed = progress["completed_shards"].get(source_path.name)
        if completed:
            if not check_output_metadata(
                output_path,
                source_path.name,
                source_size,
                source_rows,
                manifest["dataset_revision"],
                model_sha,
            ):
                raise ValueError(f"Completed output failed validation: {output_path}")
            continue
        # Recover safely if the output was atomically renamed just before an
        # interruption prevented its checkpoint update.
        if output_path.exists():
            if not check_output_metadata(
                output_path,
                source_path.name,
                source_size,
                source_rows,
                manifest["dataset_revision"],
                model_sha,
            ):
                raise ValueError(
                    f"Untracked output already exists and does not match this run: {output_path}"
                )
            recovered = {
                "source_shard": source_path.name,
                "output_shard": output_path.name,
                "source_rows": source_rows,
                "token_ids": sum(
                    pq.ParquetFile(output_path).read(columns=["token_count"])
                    .column("token_count").to_pylist()
                ),
                "output_size_bytes": output_path.stat().st_size,
                "recovered_from_output_metadata": True,
                "completed_utc": utc_now(),
            }
            progress["completed_shards"][source_path.name] = recovered
            progress["rows_completed"] += source_rows
            progress["token_ids_completed"] += recovered["token_ids"]
            progress["updated_utc"] = utc_now()
            atomic_json_write(progress_path, progress)
            continue
        if max_shards is not None and shards_this_run >= max_shards:
            break
        print(f"Tokenizing {source_path.name} ({source_rows:,} rows)...", flush=True)
        result = tokenize_shard(
            tokenizer, source_path, output_path, manifest, model_sha
        )
        progress["completed_shards"][source_path.name] = result
        progress["rows_completed"] += result["source_rows"]
        progress["token_ids_completed"] += result["token_ids"]
        progress["updated_utc"] = utc_now()
        atomic_json_write(progress_path, progress)
        shards_this_run += 1
        print(
            f"Completed {source_path.name}: {result['source_rows']:,} rows, "
            f"{result['token_ids']:,} IDs, {result['elapsed_seconds'] / 60:.1f} min; "
            f"overall {progress['rows_completed']:,}/{manifest['records_written']:,} rows",
            flush=True,
        )

    all_complete = len(progress["completed_shards"]) == len(paths)
    progress["status"] = "complete" if all_complete else "in_progress"
    progress["updated_utc"] = utc_now()
    atomic_json_write(progress_path, progress)
    if all_complete:
        if progress["rows_completed"] != manifest["records_written"]:
            raise ValueError("Completed shard row totals do not match the source manifest")
        print(
            f"Full tokenization complete: {progress['rows_completed']:,} rows, "
            f"{progress['token_ids_completed']:,} token IDs. Output: {OUTPUT_ROOT}",
            flush=True,
        )
    else:
        print(
            f"Run stopped at a shard boundary: {progress['rows_completed']:,}/"
            f"{manifest['records_written']:,} rows complete. Rerun the same command "
            "to resume at the next shard.",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-rows",
        type=int,
        default=0,
        help="Encode an evenly spaced timing sample only, then exit (for example 10000).",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Process at most this many new source shards, then save progress and exit.",
    )
    args = parser.parse_args()
    if args.pilot_rows < 0 or args.max_shards is not None and args.max_shards < 1:
        parser.error("--pilot-rows must be nonnegative and --max-shards must be positive")
    manifest, paths = load_source()
    tokenizer, model_sha = load_tokenizer(manifest)
    print(
        f"Dataset {manifest['dataset_id']} {manifest['config']}/{manifest['split']}\n"
        f"Revision {manifest['dataset_revision']}\n"
        f"Rows {manifest['records_written']:,}; shards {len(paths)}\n"
        f"BPE vocabulary {tokenizer.get_vocab_size():,}; model SHA-256 {model_sha}",
        flush=True,
    )
    if args.pilot_rows:
        run_pilot(tokenizer, paths, manifest, min(args.pilot_rows, manifest["records_written"]))
        return
    run_full(tokenizer, paths, manifest, model_sha, args.max_shards)


if __name__ == "__main__":
    main()
