"""Train and evaluate byte-level subword tokenizers on all Sansar train rows.

The source Parquet shards are streamed in batches. No decoded corpus copy or
per-row tokenized dataset is written. Only tokenizer JSON files and aggregate
metrics are saved under the git-ignored models/ and experiments/ directories.
"""

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "processed" / "sansar_normalized"
MANIFEST_PATH = DATASET_ROOT / "manifest.json"
MODEL_ROOT = PROJECT_ROOT / "models" / "module_02_tokenization" / "full_corpus"
REPORT_ROOT = PROJECT_ROOT / "experiments" / "tokenization" / "full_corpus"
EXPECTED_DATASET = "MuseMesh/sansar-sanskrit-corpus"
EXPECTED_CONFIG = "default"
EXPECTED_SPLIT = "train"
SPECIAL_TOKENS = ["<unk>"]
BYTE_ALPHABET = pre_tokenizers.ByteLevel.alphabet()
REPORT_EVERY_ROWS = 100_000
EVALUATION_CHUNK_CHARACTERS = 1_024
EVALUATION_SAMPLE_ROWS = 5_000


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_manifest():
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
    if not manifest.get("records_written") or not manifest.get("shards_written"):
        raise ValueError("Manifest does not report a complete, non-empty dataset")
    return manifest


def parquet_paths(manifest):
    paths = sorted(DATASET_ROOT.glob("part-*.parquet"))
    if len(paths) != manifest["shards_written"]:
        raise ValueError(
            f"Found {len(paths)} Parquet shards; manifest expects "
            f"{manifest['shards_written']}"
        )
    return paths


def iter_text_rows(paths, report_progress=False):
    """Yield normalized rows without materializing any shard or corpus."""
    rows_seen = 0
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        if "normalized_text" not in parquet_file.schema_arrow.names:
            raise ValueError(f"Missing normalized_text in {path.name}")
        for batch in parquet_file.iter_batches(
            columns=["normalized_text"], batch_size=64
        ):
            for text in batch.column(0).to_pylist():
                if not isinstance(text, str):
                    raise ValueError(
                        f"Expected normalized_text string at row {rows_seen} "
                        f"in {path.name}"
                    )
                rows_seen += 1
                if report_progress and rows_seen % REPORT_EVERY_ROWS == 0:
                    print(f"  streamed {rows_seen:,} rows", flush=True)
                yield text


def build_tokenizer(model_type, vocab_size):
    if model_type == "bpe":
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            initial_alphabet=BYTE_ALPHABET,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )
    else:
        tokenizer = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            initial_alphabet=BYTE_ALPHABET,
            special_tokens=SPECIAL_TOKENS,
            unk_token="<unk>",
            show_progress=True,
        )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer, trainer


def percentile(sorted_values, fraction):
    if not sorted_values:
        return 0
    index = max(0, math.ceil(fraction * len(sorted_values)) - 1)
    return sorted_values[index]


def train_one(model_type, vocab_size, paths, manifest):
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_ROOT / f"{model_type}_vocab{vocab_size}.json"
    training_metadata_path = MODEL_ROOT / f"{model_type}_vocab{vocab_size}.training.json"
    tokenizer, trainer = build_tokenizer(model_type, vocab_size)
    started_at = utc_now()
    started = time.perf_counter()
    training_rows = 0

    def training_rows_iterator():
        nonlocal training_rows
        for text in iter_text_rows(paths, report_progress=True):
            training_rows += 1
            yield text

    model_reused = model_path.exists()
    if model_reused:
        tokenizer = Tokenizer.from_file(str(model_path))
        training_rows = manifest["records_written"]
        training_metadata = (
            json.loads(training_metadata_path.read_text(encoding="utf-8"))
            if training_metadata_path.is_file()
            else {}
        )
        if (
            training_metadata.get("training_rows") != manifest["records_written"]
            or training_metadata.get("dataset_revision") != manifest["dataset_revision"]
        ):
            raise ValueError(
                f"Cannot confirm that {model_path.name} was fitted on this full "
                "dataset revision. Remove the local model file and its training "
                "metadata to retrain it, or restore matching metadata."
            )
        training_minutes = training_metadata.get("training_minutes")
        print(f"Reusing saved {model_type.upper()} model; verifying full-corpus sample...", flush=True)
    else:
        print(f"Training {model_type.upper()} on the full corpus...", flush=True)
        tokenizer.train_from_iterator(
            training_rows_iterator(), trainer=trainer, length=manifest["records_written"]
        )
        if training_rows != manifest["records_written"]:
            raise ValueError(
                f"Trainer consumed {training_rows:,} rows; manifest reports "
                f"{manifest['records_written']:,}"
            )
        tokenizer.save(str(model_path), pretty=True)
        train_seconds = time.perf_counter() - started
        training_minutes = train_seconds / 60
        training_metadata_path.write_text(
            json.dumps(
                {
                    "training_rows": training_rows,
                    "training_minutes": training_minutes,
                    "dataset_revision": manifest["dataset_revision"],
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(
            f"Training complete ({training_rows:,} rows, "
            f"{train_seconds / 60:.1f} min); verifying full-corpus sample...",
            flush=True,
        )

    row_token_counts = []
    total_tokens = 0
    total_characters = 0
    total_utf8_bytes = 0
    unknown_tokens = 0
    exact_rows = 0
    max_row_tokens = 0
    evaluation_started = time.perf_counter()
    unknown_id = tokenizer.token_to_id("<unk>")
    rows_seen = 0
    sample_positions = {
        round(index * (manifest["records_written"] - 1) / (EVALUATION_SAMPLE_ROWS - 1))
        for index in range(EVALUATION_SAMPLE_ROWS)
    }
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(
            columns=["normalized_text"], batch_size=64
        ):
            batch_texts = batch.column(0).to_pylist()
            selected_offsets = [
                offset for offset in range(len(batch_texts))
                if rows_seen + offset in sample_positions
            ]
            texts = [batch_texts[offset] for offset in selected_offsets]
            rows_seen += len(batch_texts)
            if not texts:
                continue
            encoded_chunks = []
            chunk_owners = []
            for row_index, text in enumerate(texts):
                if not isinstance(text, str):
                    raise ValueError(
                        f"Expected normalized_text string at evaluation row "
                        f"{rows_seen + row_index} in {path.name}"
                    )
                starts = range(0, len(text), EVALUATION_CHUNK_CHARACTERS)
                chunks = (
                    [text[start : start + EVALUATION_CHUNK_CHARACTERS] for start in starts]
                    if text
                    else [""]
                )
                encoded_chunks.extend(chunks)
                chunk_owners.extend([row_index] * len(chunks))

            encodings = tokenizer.encode_batch(
                encoded_chunks, add_special_tokens=False
            )
            decoded_chunks = tokenizer.decode_batch(
                [encoding.ids for encoding in encodings], skip_special_tokens=False
            )
            row_piece_counts = [0] * len(texts)
            row_unknown_counts = [0] * len(texts)
            row_decoded_chunks = [[] for _ in texts]
            for owner, encoding, decoded in zip(
                chunk_owners, encodings, decoded_chunks
            ):
                row_piece_counts[owner] += len(encoding.ids)
                row_unknown_counts[owner] += encoding.ids.count(unknown_id)
                row_decoded_chunks[owner].append(decoded)

            for row_index, text in enumerate(texts):
                token_count = row_piece_counts[row_index]
                if "".join(row_decoded_chunks[row_index]) == text:
                    exact_rows += 1
                row_token_counts.append(token_count)
                total_tokens += token_count
                total_characters += len(text)
                total_utf8_bytes += len(text.encode("utf-8"))
                unknown_tokens += row_unknown_counts[row_index]
                max_row_tokens = max(max_row_tokens, token_count)
            if len(row_token_counts) % 1_000 == 0:
                print(
                    f"  verified {len(row_token_counts):,}/"
                    f"{len(sample_positions):,} evenly spaced sample rows",
                    flush=True,
                )

    if rows_seen != manifest["records_written"]:
        raise ValueError(f"Evaluation scan read {rows_seen:,} rows from source shards")
    if len(row_token_counts) != len(sample_positions):
        raise ValueError(
            f"Evaluation covered {len(row_token_counts):,} sample rows; expected "
            f"{len(sample_positions):,}"
        )
    row_token_counts.sort()
    eval_seconds = time.perf_counter() - evaluation_started
    result = {
        "model_type": model_type,
        "tokenizers_version": __import__("tokenizers").__version__,
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "dataset_id": manifest["dataset_id"],
        "dataset_revision": manifest["dataset_revision"],
        "config": manifest["config"],
        "split": manifest["split"],
        "training_rows": training_rows,
        "evaluation_rows": len(row_token_counts),
        "evaluation_sample_method": (
            "5,000 deterministic evenly spaced row positions across the full train split"
        ),
        "evaluation_sample_utf8_bytes": total_utf8_bytes,
        "evaluation_sample_characters": total_characters,
        "total_token_instances": total_tokens,
        "mean_tokens_per_row": total_tokens / len(row_token_counts),
        "median_tokens_per_row": statistics.median(row_token_counts),
        "p90_tokens_per_row": percentile(row_token_counts, 0.90),
        "p95_tokens_per_row": percentile(row_token_counts, 0.95),
        "max_tokens_per_row": max_row_tokens,
        "tokens_per_codepoint": total_tokens / total_characters if total_characters else 0,
        "unknown_token_instances": unknown_tokens,
        "exact_round_trip_rows": exact_rows,
        "exact_round_trip_rate": exact_rows / len(row_token_counts),
        "training_minutes": training_minutes,
        "model_reused_during_report_run": model_reused,
        "evaluation_minutes": eval_seconds / 60,
        "started_utc": started_at,
        "completed_utc": utc_now(),
        "evaluation_scope": (
            "in-corpus sample; tokenizer was fitted on the complete train split, "
            "so these measurements are not held-out generalization results"
        ),
        "tokenization_pipeline": {
            "pre_tokenizer": "ByteLevel(add_prefix_space=False)",
            "decoder": "ByteLevel",
            "special_tokens": SPECIAL_TOKENS,
            "unknown_handling": (
                "BPE and Unigram trainers include the complete ByteLevel alphabet"
            ),
            "evaluation_chunk_characters": EVALUATION_CHUNK_CHARACTERS,
            "evaluation_chunking_note": (
                "Sampled very long rows are encoded in codepoint-safe chunks to bound memory; "
                "chunk boundaries may slightly increase piece counts. Chunks are "
                "decoded and rejoined to verify exact row reconstruction."
            ),
        },
    }
    report_path = REPORT_ROOT / f"{model_type}_vocab{vocab_size}.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Finished {model_type.upper()}: vocab={result['actual_vocab_size']:,}, "
        f"mean={result['mean_tokens_per_row']:.2f} tokens/row, "
        f"round-trip={exact_rows:,}/{len(row_token_counts):,}, "
        f"report={report_path}",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument(
        "--models", nargs="+", choices=("bpe", "unigram"), default=("bpe", "unigram")
    )
    args = parser.parse_args()
    if args.vocab_size < 512:
        parser.error("--vocab-size must be at least 512")

    manifest = load_manifest()
    paths = parquet_paths(manifest)
    print(
        f"Dataset: {manifest['dataset_id']} {manifest['config']}/{manifest['split']}\n"
        f"Revision: {manifest['dataset_revision']}\n"
        f"Rows: {manifest['records_written']:,}; shards: {len(paths)}\n"
        f"Vocabulary target: {args.vocab_size:,}; models: {', '.join(args.models)}",
        flush=True,
    )
    results = [
        train_one(model_type, args.vocab_size, paths, manifest)
        for model_type in args.models
    ]
    summary_path = REPORT_ROOT / "summary.json"
    summary = {
        "dataset_id": manifest["dataset_id"],
        "dataset_revision": manifest["dataset_revision"],
        "config": manifest["config"],
        "split": manifest["split"],
        "rows": manifest["records_written"],
        "models": results,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"All models complete. Aggregate summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
