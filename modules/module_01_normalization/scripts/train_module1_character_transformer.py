"""Train a small character-level Transformer for Sanskrit OCR correction."""

import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import unicodedata

import pyarrow.parquet as pq
import torch
from rapidfuzz.distance import Levenshtein
from torch import nn


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
if str(MODULE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "src"))

from normalization import normalize_text


SPLIT_DIRECTORY = PROJECT_ROOT / "datasets" / "splits" / "ocr_target_disjoint"
OUTPUT_DIRECTORY = PROJECT_ROOT / "experiments" / "normalization" / "real_ocr" / "character_transformer_10k"
MODEL_DIRECTORY = PROJECT_ROOT / "models" / "module1_character_transformer_10k"

SEED = 42
MAX_CHAR_LENGTH = 256
BATCH_SIZE = 128
MAX_TRAIN_EXAMPLES = 10_000
MODEL_DIMENSION = 64
ATTENTION_HEADS = 4
ENCODER_LAYERS = 2
DECODER_LAYERS = 2
FEED_FORWARD_DIMENSION = 128
DROPOUT = 0.1
MAX_EPOCHS = 10
LEARNING_RATE = 0.0005
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


def read_split(split_name):
    file_path = SPLIT_DIRECTORY / f"{split_name}.parquet"
    parquet_file = pq.ParquetFile(file_path)
    rows = []
    for batch in parquet_file.iter_batches(
        columns=["row_number", "raw_input_text", "input_text", "target_text"],
        batch_size=2_000,
    ):
        for row in batch.to_pylist():
            row["split"] = split_name
            rows.append(row)
    return rows


def keep_supported_lengths(rows):
    kept = []
    excluded = 0
    for row in rows:
        if max(len(row["input_text"]), len(row["target_text"])) <= MAX_CHAR_LENGTH:
            kept.append(row)
        else:
            excluded += 1
    return kept, excluded


def build_vocabulary(training_rows):
    characters = set()
    for row in training_rows:
        characters.update(row["input_text"])
        characters.update(row["target_text"])

    tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    tokens.extend(sorted(characters))
    token_to_id = {token: index for index, token in enumerate(tokens)}
    return tokens, token_to_id


def encode_text(text, token_to_id):
    unknown_id = token_to_id[UNK_TOKEN]
    return [token_to_id.get(character, unknown_id) for character in text]


def make_batches(rows, token_to_id, batch_size, shuffle_batches, random_generator):
    ordered_indices = sorted(
        range(len(rows)),
        key=lambda index: max(
            len(rows[index]["input_text"]),
            len(rows[index]["target_text"]),
        ),
    )
    batches = []
    for start in range(0, len(ordered_indices), batch_size):
        batch_indices = ordered_indices[start:start + batch_size]
        if shuffle_batches:
            random_generator.shuffle(batch_indices)
        batches.append(batch_indices)
    if shuffle_batches:
        random_generator.shuffle(batches)

    for batch_indices in batches:
        selected_rows = [rows[index] for index in batch_indices]
        source_sequences = [
            encode_text(row["input_text"], token_to_id) + [token_to_id[EOS_TOKEN]]
            for row in selected_rows
        ]
        target_sequences = [
            encode_text(row["target_text"], token_to_id)
            for row in selected_rows
        ]

        source_length = max(len(sequence) for sequence in source_sequences)
        target_length = max(len(sequence) for sequence in target_sequences)
        source = torch.full(
            (len(selected_rows), source_length),
            token_to_id[PAD_TOKEN],
            dtype=torch.long,
        )
        decoder_input = torch.full(
            (len(selected_rows), target_length + 1),
            token_to_id[PAD_TOKEN],
            dtype=torch.long,
        )
        labels = torch.full(
            (len(selected_rows), target_length + 1),
            token_to_id[PAD_TOKEN],
            dtype=torch.long,
        )

        for row_index, (source_ids, target_ids) in enumerate(
            zip(source_sequences, target_sequences)
        ):
            source[row_index, :len(source_ids)] = torch.tensor(source_ids)
            decoder_ids = [token_to_id[BOS_TOKEN]] + target_ids
            label_ids = target_ids + [token_to_id[EOS_TOKEN]]
            decoder_input[row_index, :len(decoder_ids)] = torch.tensor(decoder_ids)
            labels[row_index, :len(label_ids)] = torch.tensor(label_ids)

        yield source, decoder_input, labels, selected_rows


class CharacterTransformer(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.dimension = MODEL_DIMENSION
        self.embedding = nn.Embedding(vocabulary_size, MODEL_DIMENSION)
        self.position_embedding = nn.Embedding(MAX_CHAR_LENGTH + 2, MODEL_DIMENSION)
        self.dropout = nn.Dropout(DROPOUT)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIMENSION,
            nhead=ATTENTION_HEADS,
            dim_feedforward=FEED_FORWARD_DIMENSION,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=MODEL_DIMENSION,
            nhead=ATTENTION_HEADS,
            dim_feedforward=FEED_FORWARD_DIMENSION,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=ENCODER_LAYERS,
            norm=nn.LayerNorm(MODEL_DIMENSION),
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=DECODER_LAYERS,
            norm=nn.LayerNorm(MODEL_DIMENSION),
        )
        self.output_layer = nn.Linear(MODEL_DIMENSION, vocabulary_size)

    def embed(self, token_ids):
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        positions = positions.unsqueeze(0).expand(token_ids.size(0), -1)
        embeddings = self.embedding(token_ids) * math.sqrt(self.dimension)
        embeddings = embeddings + self.position_embedding(positions)
        return self.dropout(embeddings)

    def encode(self, source, pad_id):
        source_padding_mask = source.eq(pad_id)
        memory = self.encoder(
            self.embed(source),
            src_key_padding_mask=source_padding_mask,
        )
        return memory, source_padding_mask

    def decode(self, decoder_input, memory, source_padding_mask, pad_id):
        target_padding_mask = decoder_input.eq(pad_id)
        length = decoder_input.size(1)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=decoder_input.device),
            diagonal=1,
        )
        decoded = self.decoder(
            self.embed(decoder_input),
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=source_padding_mask,
        )
        return self.output_layer(decoded)

    def forward(self, source, decoder_input, pad_id):
        memory, source_padding_mask = self.encode(source, pad_id)
        return self.decode(decoder_input, memory, source_padding_mask, pad_id)


def loss_for_rows(model, rows, token_to_id, device, random_generator):
    model.eval()
    loss_function = nn.CrossEntropyLoss(ignore_index=token_to_id[PAD_TOKEN])
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        batches = make_batches(
            rows,
            token_to_id,
            BATCH_SIZE,
            False,
            random_generator,
        )
        for source, decoder_input, labels, _ in batches:
            source = source.to(device)
            decoder_input = decoder_input.to(device)
            labels = labels.to(device)
            logits = model(source, decoder_input, token_to_id[PAD_TOKEN])
            token_count = int(labels.ne(token_to_id[PAD_TOKEN]).sum().item())
            loss = loss_function(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            total_loss += float(loss.item()) * token_count
            total_tokens += token_count

    return total_loss / max(total_tokens, 1)


def greedy_decode(model, source, token_to_id, id_to_token, device):
    model.eval()
    source = source.to(device)
    pad_id = token_to_id[PAD_TOKEN]
    bos_id = token_to_id[BOS_TOKEN]
    eos_id = token_to_id[EOS_TOKEN]
    memory, source_padding_mask = model.encode(source, pad_id)
    generated = torch.full((source.size(0), 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(source.size(0), dtype=torch.bool, device=device)

    with torch.no_grad():
        for _ in range(MAX_CHAR_LENGTH + 1):
            logits = model.decode(generated, memory, source_padding_mask, pad_id)
            next_logits = logits[:, -1, :]
            next_logits[:, pad_id] = -1e9
            next_logits[:, bos_id] = -1e9
            next_token = next_logits.argmax(dim=-1)
            next_token = torch.where(finished, eos_id, next_token)
            generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)
            finished = finished | next_token.eq(eos_id)
            if bool(finished.all()):
                break

    predictions = []
    for token_ids in generated[:, 1:].tolist():
        characters = []
        for token_id in token_ids:
            if token_id == eos_id:
                break
            token = id_to_token[token_id]
            if token == UNK_TOKEN:
                characters.append("\ufffd")
            elif token not in {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}:
                characters.append(token)
        predictions.append("".join(characters))
    return predictions


def character_error_rate(prediction, reference):
    prediction = unicodedata.normalize("NFC", prediction)
    reference = unicodedata.normalize("NFC", reference)
    distance = Levenshtein.distance(prediction, reference)
    if not reference:
        return distance, 0.0 if not prediction else 1.0
    return distance, distance / len(reference)


def score_predictions(rows, model_predictions):
    measures = {
        "identity": {"distances": 0, "reference_chars": 0, "row_cers": [], "exact": 0},
        "normalizer": {"distances": 0, "reference_chars": 0, "row_cers": [], "exact": 0},
        "model": {"distances": 0, "reference_chars": 0, "row_cers": [], "exact": 0},
    }
    scored_rows = []

    for row, model_prediction in zip(rows, model_predictions):
        reference = unicodedata.normalize("NFC", row["target_text"])
        predictions = {
            "identity": unicodedata.normalize("NFC", row["raw_input_text"]),
            "normalizer": row["input_text"],
            "model": model_prediction,
        }
        scores = {}

        for baseline_name, prediction in predictions.items():
            distance, cer = character_error_rate(prediction, reference)
            scores[baseline_name] = {"distance": distance, "cer": cer}
            measures[baseline_name]["distances"] += distance
            measures[baseline_name]["reference_chars"] += len(reference)
            measures[baseline_name]["row_cers"].append(cer)
            measures[baseline_name]["exact"] += prediction == reference

        scored_rows.append({
            "split": row["split"],
            "source_row_number": row["row_number"],
            "target_text": reference,
            "identity_prediction": predictions["identity"],
            "normalizer_prediction": predictions["normalizer"],
            "model_prediction": predictions["model"],
            "identity_cer": scores["identity"]["cer"],
            "normalizer_cer": scores["normalizer"]["cer"],
            "model_cer": scores["model"]["cer"],
            "model_exact_match": predictions["model"] == reference,
        })

    summary = {}
    row_count = len(rows)
    for name, values in measures.items():
        summary[name] = {
            "examples": row_count,
            "micro_cer": values["distances"] / max(values["reference_chars"], 1),
            "mean_row_cer": sum(values["row_cers"]) / max(row_count, 1),
            "exact_match_rate": values["exact"] / max(row_count, 1),
        }

    improved_vs_identity = 0
    worsened_vs_identity = 0
    improved_vs_normalizer = 0
    worsened_vs_normalizer = 0
    for row in scored_rows:
        improved_vs_identity += row["model_cer"] < row["identity_cer"]
        worsened_vs_identity += row["model_cer"] > row["identity_cer"]
        improved_vs_normalizer += row["model_cer"] < row["normalizer_cer"]
        worsened_vs_normalizer += row["model_cer"] > row["normalizer_cer"]

    summary["model_row_comparisons"] = {
        "improved_vs_identity": improved_vs_identity,
        "worsened_vs_identity": worsened_vs_identity,
        "improved_vs_normalizer": improved_vs_normalizer,
        "worsened_vs_normalizer": worsened_vs_normalizer,
    }
    return summary, scored_rows


def train_one_epoch(model, training_rows, token_to_id, optimizer, device, random_generator, epoch):
    model.train()
    loss_function = nn.CrossEntropyLoss(ignore_index=token_to_id[PAD_TOKEN])
    total_loss = 0.0
    total_tokens = 0
    total_batches = math.ceil(len(training_rows) / BATCH_SIZE)
    started = time.time()

    batches = make_batches(
        training_rows,
        token_to_id,
        BATCH_SIZE,
        True,
        random_generator,
    )
    for batch_number, (source, decoder_input, labels, _) in enumerate(batches, start=1):
        source = source.to(device)
        decoder_input = decoder_input.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(source, decoder_input, token_to_id[PAD_TOKEN])
        token_count = int(labels.ne(token_to_id[PAD_TOKEN]).sum().item())
        loss = loss_function(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count

        if batch_number % 100 == 0 or batch_number == total_batches:
            elapsed = time.time() - started
            print(
                f"Epoch {epoch}: batch {batch_number:,}/{total_batches:,}; "
                f"token_loss={total_loss / max(total_tokens, 1):.4f}; "
                f"elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    return total_loss / max(total_tokens, 1)


def evaluate_split(model, rows, token_to_id, id_to_token, device, random_generator):
    model.eval()
    predictions_by_row_number = {}
    batches = make_batches(
        rows,
        token_to_id,
        BATCH_SIZE,
        False,
        random_generator,
    )
    for source, _, _, selected_rows in batches:
        decoded = greedy_decode(model, source, token_to_id, id_to_token, device)
        for row, prediction in zip(selected_rows, decoded):
            predictions_by_row_number[row["row_number"]] = prediction
        if len(predictions_by_row_number) % 1024 < BATCH_SIZE:
            print(
                f"Decoded {len(predictions_by_row_number):,}/{len(rows):,} rows.",
                flush=True,
            )
    predictions = [
        predictions_by_row_number[row["row_number"]]
        for row in rows
    ]
    return score_predictions(rows, predictions)


def write_pair_metrics(path, scored_rows):
    if not scored_rows:
        raise ValueError(f"No scored rows to save to {path}.")
    field_names = list(scored_rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(scored_rows)


def main():
    if OUTPUT_DIRECTORY.exists() or MODEL_DIRECTORY.exists():
        raise FileExistsError("Model output already exists; refusing to overwrite.")

    torch.manual_seed(SEED)
    random.seed(SEED)
    random_generator = random.Random(SEED)
    thread_count = min(8, max(1, (os.cpu_count() or 2) // 2))
    torch.set_num_threads(thread_count)
    device = torch.device("cpu")

    split_rows = {}
    length_exclusions = {}
    for split_name in ["train", "validation", "test", "ood_test"]:
        rows = read_split(split_name)
        rows, excluded = keep_supported_lengths(rows)
        if not rows:
            raise ValueError(f"No examples within the {MAX_CHAR_LENGTH}-character limit in {split_name}.")
        split_rows[split_name] = rows
        length_exclusions[split_name] = excluded
        print(
            f"{split_name}: {len(rows):,} rows kept; {excluded:,} rows over "
            f"the {MAX_CHAR_LENGTH}-character limit.",
            flush=True,
        )

    available_training_rows = split_rows["train"]
    training_rows = available_training_rows
    if len(training_rows) > MAX_TRAIN_EXAMPLES:
        training_rows = random_generator.sample(training_rows, MAX_TRAIN_EXAMPLES)
    validation_rows = split_rows["validation"]
    tokens, token_to_id = build_vocabulary(training_rows)
    id_to_token = tokens
    model = CharacterTransformer(len(tokens)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    sample_path = OUTPUT_DIRECTORY / "training_sample_row_numbers.csv"
    with sample_path.open("w", encoding="utf-8-sig", newline="") as sample_file:
        writer = csv.writer(sample_file)
        writer.writerow(["prepared_train_row_number"])
        for row in training_rows:
            writer.writerow([row["row_number"]])
    history = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print(
        f"Training {len(training_rows):,} of {len(available_training_rows):,} "
        f"eligible train rows on {device}; vocabulary={len(tokens)}; "
        f"threads={thread_count}; parameters={sum(parameter.numel() for parameter in model.parameters()):,}.",
        flush=True,
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        training_loss = train_one_epoch(
            model,
            training_rows,
            token_to_id,
            optimizer,
            device,
            random_generator,
            epoch,
        )
        validation_loss = loss_for_rows(
            model,
            validation_rows,
            token_to_id,
            device,
            random_generator,
        )
        history.append({
            "epoch": epoch,
            "training_token_loss": training_loss,
            "validation_token_loss": validation_loss,
        })
        print(
            f"Epoch {epoch} complete: train_loss={training_loss:.4f}, "
            f"validation_loss={validation_loss:.4f}.",
            flush=True,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tokens": tokens,
                    "config": {
                        "max_char_length": MAX_CHAR_LENGTH,
                        "max_train_examples": MAX_TRAIN_EXAMPLES,
                        "model_dimension": MODEL_DIMENSION,
                        "attention_heads": ATTENTION_HEADS,
                        "encoder_layers": ENCODER_LAYERS,
                        "decoder_layers": DECODER_LAYERS,
                        "feed_forward_dimension": FEED_FORWARD_DIMENSION,
                        "dropout": DROPOUT,
                        "seed": SEED,
                        "torch_version": torch.__version__,
                    },
                },
                MODEL_DIRECTORY / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 1:
                print("Stopping early because validation loss did not improve.", flush=True)
                break

    with (OUTPUT_DIRECTORY / "training_history.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as history_file:
        writer = csv.DictWriter(history_file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(
        MODEL_DIRECTORY / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluation_summaries = {}
    for split_name in ["validation", "test", "ood_test"]:
        rows = split_rows[split_name]
        print(f"Generating {split_name} predictions...", flush=True)
        summary, scored_rows = evaluate_split(
            model,
            rows,
            token_to_id,
            id_to_token,
            device,
            random_generator,
        )
        evaluation_summaries[split_name] = summary
        write_pair_metrics(
            OUTPUT_DIRECTORY / f"{split_name}_predictions.csv",
            scored_rows,
        )
        print(
            f"{split_name}: identity CER={summary['identity']['micro_cer']:.4f}, "
            f"normalizer CER={summary['normalizer']['micro_cer']:.4f}, "
            f"model CER={summary['model']['micro_cer']:.4f}, "
            f"model exact={summary['model']['exact_match_rate']:.3%}",
            flush=True,
        )

    report = {
        "dataset_id": "acomquest/sanskrit-ocr-post-correction",
        "dataset_revision": "35de91d045dfd8c9bd875b44f2bfda52f09f72ce",
        "prepared_split_manifest": str(
            (SPLIT_DIRECTORY / "manifest.json").relative_to(PROJECT_ROOT)
        ),
        "model": "character-level Transformer encoder-decoder",
        "device": str(device),
        "torch_version": torch.__version__,
        "seed": SEED,
        "max_char_length": MAX_CHAR_LENGTH,
        "max_train_examples": MAX_TRAIN_EXAMPLES,
        "max_epochs": MAX_EPOCHS,
        "best_epoch": best_epoch,
        "best_validation_token_loss": best_validation_loss,
        "eligible_training_rows": len(available_training_rows),
        "training_rows": len(training_rows),
        "training_sample_strategy": (
            "Seeded random sample without replacement from target-disjoint "
            "training rows in file order."
        ),
        "training_sample_row_numbers_file": str(
            sample_path.relative_to(PROJECT_ROOT)
        ),
        "validation_rows": len(validation_rows),
        "length_exclusions": length_exclusions,
        "vocabulary_size": len(tokens),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_history": history,
        "evaluation": evaluation_summaries,
        "limits": [
            "The model uses Unicode code points, not linguistic akshara units.",
            "Rows longer than max_char_length were excluded from the model run.",
            "The benchmark mirror's file counts differ from its current card and paper; see split_audit.json.",
            "The OCR data mirror declares MIT, but this result does not independently verify rights in each source book.",
        ],
    }
    report_path = OUTPUT_DIRECTORY / "model_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Best validation epoch: {best_epoch}", flush=True)
    print(f"Saved model: {MODEL_DIRECTORY / 'best_model.pt'}", flush=True)
    print(f"Saved report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
