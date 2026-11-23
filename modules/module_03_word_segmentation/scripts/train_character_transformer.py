"""Train and evaluate a character-level Sanskrit segmentation Transformer.

Input and target are both character sequences so the model can learn sandhi
repairs as well as word boundaries. The source CSV is read-only; model files,
metrics, and predictions go to ignored local experiment/model directories.
"""

import argparse
import csv
import json
import math
import random
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from rapidfuzz.distance import Levenshtein


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "external" / "Hackathon_dcs.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "module_03_word_segmentation"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "module_03_segmentation" / "character_transformer"
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]
SEED = 42
MAX_OUTPUT_EXTRA_CHARS = 32
MAX_CHARACTER_EDIT_RATIO = 0.30
ROUTE_TO_CHUNKS_ABOVE_LENGTH = 55


def read_dataset(path, max_length):
    frame = pd.read_csv(path, dtype={"input": "string", "output": "string", "split": "string", "DCS-ID": "string"})
    required = {"input", "output", "split", "DCS-ID"}
    if set(frame.columns) != required:
        raise ValueError(f"Expected columns {sorted(required)}, found {list(frame.columns)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Dataset contains missing required values; refusing to train.")
    rows = frame.to_dict("records")
    splits = {name: [] for name in ("train", "dev", "test")}
    excluded_by_length = Counter()
    for row in rows:
        split = str(row["split"])
        if split not in splits:
            raise ValueError(f"Unexpected split value: {split!r}")
        source = unicodedata.normalize("NFC", str(row["input"]))
        target = unicodedata.normalize("NFC", str(row["output"]))
        if max(len(source), len(target)) > max_length:
            excluded_by_length[split] += 1
            continue
        splits[split].append({"input": source, "output": target, "split": split, "DCS-ID": str(row["DCS-ID"])})
    # Exact duplicate pairs do not add distinct supervision. Conflicting
    # outputs for the same input remain separate training targets.
    deduplicated = {}
    duplicate_counts = {}
    for split, split_rows in splits.items():
        seen = set()
        unique = []
        for row in split_rows:
            key = (row["input"], row["output"])
            if key not in seen:
                seen.add(key)
                unique.append(row)
        deduplicated[split] = unique
        duplicate_counts[split] = len(split_rows) - len(unique)
    return deduplicated, {"excluded_by_length": dict(excluded_by_length), "exact_duplicate_rows_removed": duplicate_counts}


def build_vocab(train_rows):
    chars = set()
    for row in train_rows:
        chars.update(row["input"])
        chars.update(row["output"])
    tokens = SPECIALS + sorted(chars - set(SPECIALS))
    return tokens, {token: index for index, token in enumerate(tokens)}


def make_batches(rows, token_to_id, batch_size, shuffle, rng):
    indices = sorted(range(len(rows)), key=lambda i: max(len(rows[i]["input"]), len(rows[i]["output"])))
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    if shuffle:
        rng.shuffle(batches)
        for batch in batches:
            rng.shuffle(batch)
    pad_id, bos_id, eos_id = (token_to_id[x] for x in (PAD, BOS, EOS))
    for batch_indices in batches:
        selected = [rows[i] for i in batch_indices]
        source_ids = [[token_to_id.get(c, token_to_id[UNK]) for c in r["input"]] + [eos_id] for r in selected]
        target_ids = [[token_to_id.get(c, token_to_id[UNK]) for c in r["output"]] for r in selected]
        src_len = max(map(len, source_ids))
        tgt_len = max(map(len, target_ids)) + 1
        source = torch.full((len(selected), src_len), pad_id, dtype=torch.long)
        decoder = torch.full((len(selected), tgt_len), pad_id, dtype=torch.long)
        labels = torch.full((len(selected), tgt_len), pad_id, dtype=torch.long)
        for i, (src, tgt) in enumerate(zip(source_ids, target_ids)):
            source[i, :len(src)] = torch.tensor(src)
            decoder_ids = [bos_id] + tgt
            label_ids = tgt + [eos_id]
            decoder[i, :len(decoder_ids)] = torch.tensor(decoder_ids)
            labels[i, :len(label_ids)] = torch.tensor(label_ids)
        yield source, decoder, labels, selected


class CharacterTransformer(nn.Module):
    def __init__(self, vocab_size, max_length, dimension=128, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.dimension = dimension
        self.embedding = nn.Embedding(vocab_size, dimension)
        self.position = nn.Embedding(max_length + 2, dimension)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dimension, heads, dimension * 2, dropout, batch_first=True, norm_first=True),
            layers, norm=nn.LayerNorm(dimension), enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dimension, heads, dimension * 2, dropout, batch_first=True, norm_first=True),
            layers, norm=nn.LayerNorm(dimension),
        )
        self.output = nn.Linear(dimension, vocab_size)

    def embed(self, ids):
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        return self.dropout(self.embedding(ids) * math.sqrt(self.dimension) + self.position(pos))

    def encode(self, source, pad_id):
        pad_mask = source.eq(pad_id)
        return self.encoder(self.embed(source), src_key_padding_mask=pad_mask), pad_mask

    def decode(self, decoder, memory, source_pad, pad_id):
        length = decoder.size(1)
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=decoder.device), diagonal=1)
        out = self.decoder(self.embed(decoder), memory, tgt_mask=mask,
                           tgt_key_padding_mask=decoder.eq(pad_id), memory_key_padding_mask=source_pad)
        return self.output(out)

    def forward(self, source, decoder, pad_id):
        memory, source_pad = self.encode(source, pad_id)
        return self.decode(decoder, memory, source_pad, pad_id)


def run_loss(model, rows, vocab, batch_size, device, rng):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab[PAD], reduction="sum")
    total, count = 0.0, 0
    with torch.no_grad():
        for source, decoder, labels, _ in make_batches(rows, vocab, batch_size, False, rng):
            source, decoder, labels = source.to(device), decoder.to(device), labels.to(device)
            logits = model(source, decoder, vocab[PAD])
            total += loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)).item()
            count += int(labels.ne(vocab[PAD]).sum())
    return total / max(count, 1)


def train_epoch(model, rows, vocab, batch_size, device, optimizer, rng, epoch, scaler):
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab[PAD])
    total_loss, total_tokens = 0.0, 0
    batches = make_batches(rows, vocab, batch_size, True, rng)
    total_batches = math.ceil(len(rows) / batch_size)
    started = time.time()
    for number, (source, decoder, labels, _) in enumerate(batches, 1):
        source, decoder, labels = source.to(device), decoder.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(source, decoder, vocab[PAD])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        token_count = int(labels.ne(vocab[PAD]).sum())
        total_loss += float(loss.detach()) * token_count
        total_tokens += token_count
        if number % 100 == 0 or number == total_batches:
            print(f"epoch={epoch} batch={number}/{total_batches} loss={total_loss/max(total_tokens,1):.4f} elapsed_min={(time.time()-started)/60:.1f}", flush=True)
    return total_loss / max(total_tokens, 1)


def greedy_decode(model, source, vocab, tokens, max_length):
    model.eval()
    pad_id, bos_id, eos_id = (vocab[x] for x in (PAD, BOS, EOS))
    memory, source_pad = model.encode(source, pad_id)
    generated = torch.full((source.size(0), 1), bos_id, device=source.device, dtype=torch.long)
    finished = torch.zeros(source.size(0), device=source.device, dtype=torch.bool)
    source_lengths = source.ne(pad_id).sum(dim=1) - 1  # Do not count the source EOS.
    output_limits = (source_lengths + MAX_OUTPUT_EXTRA_CHARS).clamp(min=1, max=max_length)
    with torch.no_grad():
        for step in range(max_length):
            logits = model.decode(generated, memory, source_pad, pad_id)[:, -1, :]
            logits[:, pad_id] = -1e4
            logits[:, bos_id] = -1e4
            next_id = logits.argmax(-1)
            next_id = torch.where(finished, torch.full_like(next_id, eos_id), next_id)
            generated = torch.cat([generated, next_id[:, None]], dim=1)
            finished |= next_id.eq(eos_id) | ((step + 1) >= output_limits)
            if bool(finished.all()):
                break
    results = []
    for sequence in generated[:, 1:].tolist():
        chars = []
        for index in sequence:
            if index == eos_id:
                break
            if index >= len(SPECIALS):
                chars.append(tokens[index])
        results.append("".join(chars))
    return results


def token_scores(prediction, references):
    pred = prediction.split()
    best = (0.0, 0.0, 0.0, False, references[0])
    for reference in references:
        gold = reference.split()
        overlap = sum((Counter(pred) & Counter(gold)).values())
        precision = overlap / len(pred) if pred else float(not gold)
        recall = overlap / len(gold) if gold else float(not pred)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        best = max(best, (precision, recall, f1, prediction == reference, reference), key=lambda v: (v[2], v[3]))
    return best


def evaluate(model, rows, vocab, tokens, batch_size, device, max_length):
    grouped = defaultdict(list)
    row_counts = Counter()
    for row in rows:
        if row["output"] not in grouped[row["input"]]:
            grouped[row["input"]].append(row["output"])
        row_counts[row["input"]] += 1
    short_inputs = [
        {"input": key, "output": values[0]}
        for key, values in grouped.items()
        if len(key) <= ROUTE_TO_CHUNKS_ABOVE_LENGTH
    ]
    long_inputs = [key for key in grouped if len(key) > ROUTE_TO_CHUNKS_ABOVE_LENGTH]
    predictions = {}
    chunk_fallbacks = {}
    for source, _, _, selected in make_batches(short_inputs, vocab, batch_size, False, random.Random(SEED)):
        source = source.to(device)
        decoded = greedy_decode(model, source, vocab, tokens, max_length)
        predictions.update({row["input"]: prediction for row, prediction in zip(selected, decoded)})

    chunk_rows = []
    for sentence in long_inputs:
        for chunk_index, chunk in enumerate(sentence.split()):
            chunk_rows.append({"input": chunk, "output": "", "sentence": sentence,
                               "chunk_index": chunk_index})
    decoded_chunks = defaultdict(list)
    for source, _, _, selected in make_batches(chunk_rows, vocab, batch_size, False, random.Random(SEED)):
        decoded = greedy_decode(model, source.to(device), vocab, tokens, max_length)
        for row, prediction in zip(selected, decoded):
            source_chars = row["input"]
            predicted_chars = "".join(prediction.split())
            ratio = Levenshtein.distance(source_chars, predicted_chars) / max(len(source_chars), 1)
            fallback = ratio > MAX_CHARACTER_EDIT_RATIO
            decoded_chunks[row["sentence"]].append(
                (row["chunk_index"], source_chars if fallback else prediction, fallback)
            )
    for sentence, chunks in decoded_chunks.items():
        chunks.sort(key=lambda item: item[0])
        predictions[sentence] = " ".join(item[1] for item in chunks)
        chunk_fallbacks[sentence] = any(item[2] for item in chunks)
    exact = 0
    p_sum = r_sum = f_sum = 0.0
    total_gold_words = total_pred_words = total_overlap = 0
    fallback_count = 0
    prediction_rows = []
    for source, references in grouped.items():
        raw_prediction = predictions[source]
        source_chars = "".join(source.split())
        predicted_chars = "".join(raw_prediction.split())
        char_error_ratio = Levenshtein.distance(source_chars, predicted_chars) / max(len(source_chars), 1)
        if source in chunk_fallbacks:
            fallback_applied = chunk_fallbacks[source]
            prediction = raw_prediction
        else:
            fallback_applied = char_error_ratio > MAX_CHARACTER_EDIT_RATIO
            prediction = source if fallback_applied else raw_prediction
        fallback_count += int(fallback_applied)
        precision, recall, f1, is_exact, selected_reference = token_scores(prediction, references)
        exact += int(is_exact)
        p_sum += precision
        r_sum += recall
        f_sum += f1
        pred_tokens = Counter(prediction.split())
        ref_tokens = Counter(selected_reference.split())
        total_pred_words += sum(pred_tokens.values())
        total_gold_words += sum(ref_tokens.values())
        total_overlap += sum((pred_tokens & ref_tokens).values())
        prediction_rows.append({"input": source, "prediction": prediction,
                                "references_json": json.dumps(references, ensure_ascii=False),
                                "source_rows": row_counts[source], "exact_match_any_reference": is_exact,
                                "fallback_applied": fallback_applied,
                                "char_edit_ratio": char_error_ratio})
    n = max(len(grouped), 1)
    micro_p = total_overlap / total_pred_words if total_pred_words else 0.0
    micro_r = total_overlap / total_gold_words if total_gold_words else 0.0
    return {
        "rows": len(rows), "unique_inputs": len(grouped),
        "macro_word_precision": p_sum / n, "macro_word_recall": r_sum / n,
        "macro_word_f1": f_sum / n, "micro_word_precision": micro_p,
        "micro_word_recall": micro_r,
        "micro_word_f1": 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0,
        "perfect_sentence_match_any_reference": exact / n,
        "multiple_reference_inputs": sum(len(v) > 1 for v in grouped.values()),
        "fallback_count": fallback_count,
    }, prediction_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.input}")
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("epochs and batch-size must be positive")

    torch.manual_seed(SEED)
    random.seed(SEED)
    rng = random.Random(SEED)
    torch.set_num_threads(min(8, max(1, (torch.get_num_threads()))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()

    splits, preparation = read_dataset(args.input, args.max_length)
    if any(not splits[name] for name in splits):
        raise ValueError(f"Empty split after length filtering: {preparation}")
    tokens, vocab = build_vocab(splits["train"])
    model = CharacterTransformer(len(tokens), args.max_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / "best_model.pt"
    start_epoch, best_loss, bad_epochs, history = 1, float("inf"), 0, []
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot resume; missing {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["tokens"] != tokens:
            raise ValueError("Dataset vocabulary differs from checkpoint; refusing to resume.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_validation_loss"]
        bad_epochs = checkpoint.get("bad_epochs", 0)
        history = checkpoint.get("history", [])
        print(f"Resuming at epoch {start_epoch}.", flush=True)

    json_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}

    print(json.dumps({"device": str(device), "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                      "rows": {k: len(v) for k, v in splits.items()},
                      "vocabulary": len(tokens), "parameters": sum(p.numel() for p in model.parameters()),
                      "preparation": preparation}, ensure_ascii=False), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(model, splits["train"], vocab, args.batch_size, device, optimizer, rng, epoch, scaler)
        validation_loss = run_loss(model, splits["dev"], vocab, args.batch_size, device, rng)
        history.append({"epoch": epoch, "train_token_loss": train_loss, "dev_token_loss": validation_loss})
        print(f"epoch={epoch} train_loss={train_loss:.5f} dev_loss={validation_loss:.5f}", flush=True)
        if validation_loss < best_loss:
            best_loss, bad_epochs = validation_loss, 0
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "tokens": tokens, "epoch": epoch, "best_validation_loss": best_loss,
                        "bad_epochs": bad_epochs, "history": history,
                        "config": json_config | {"device": str(device), "torch_version": torch.__version__,
                                                 "seed": SEED, "dimension": 128, "heads": 4, "layers": 2}},
                       checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping: dev loss did not improve.", flush=True)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    summaries = {}
    for split_name in ("dev", "test"):
        summaries[split_name], predictions = evaluate(model, splits[split_name], vocab, tokens,
                                                       args.batch_size, device, args.max_length)
        with (OUTPUT_DIR / f"{split_name}_predictions.csv").open("w", encoding="utf-8-sig", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        print(f"{split_name}: {json.dumps(summaries[split_name])}", flush=True)

    # Report benchmark results and a leakage-aware test subset separately.
    train_inputs = {r["input"] for r in splits["train"]}
    leak_rows = [r for r in splits["test"] if r["input"] in train_inputs]
    leakage_free_rows = [r for r in splits["test"] if r["input"] not in train_inputs]
    leakage_free, _ = evaluate(model, leakage_free_rows, vocab, tokens, args.batch_size, device, args.max_length)
    report = {"dataset": str(args.input), "dataset_sha256": __import__("hashlib").sha256(args.input.read_bytes()).hexdigest(),
              "model": "character-level Transformer encoder-decoder", "device": str(device),
              "torch_version": torch.__version__, "seed": SEED, "config": json_config,
              "vocabulary_size": len(tokens), "parameter_count": sum(p.numel() for p in model.parameters()),
              "best_epoch": checkpoint["epoch"], "best_dev_token_loss": checkpoint["best_validation_loss"],
              "history": history, "preparation": preparation,
              "evaluation": summaries,
              "leakage_audit": {"test_rows_with_train_input_overlap": len(leak_rows),
                                "test_unique_inputs_with_train_overlap": len({r['input'] for r in leak_rows}),
                                "leakage_free_test": leakage_free},
              "metric_definition": "Per unique input, maximize token multiset precision/recall/F1 over available gold references; report macro and micro word scores and exact sentence match.",
              "decoding_safety": {"route_to_chunk_decoding_above_input_characters": ROUTE_TO_CHUNKS_ABOVE_LENGTH,
                                  "max_output_extra_characters": MAX_OUTPUT_EXTRA_CHARS,
                                  "identity_fallback_char_edit_ratio": MAX_CHARACTER_EDIT_RATIO,
                                  "fallbacks": "If generated text differs from input by more than the threshold after removing whitespace, return the original input."},
              "limits": ["Original test split is retained for benchmark compatibility.",
                         "The CSV has known train/test input overlaps; leakage-free test metrics are also reported.",
                         "This is a character Transformer baseline, not the full TransLIST lattice/PRCP architecture."]}
    report_path = OUTPUT_DIR / "training_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_DIR / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {checkpoint_path}", flush=True)
    print(f"Saved metrics: {report_path}", flush=True)


if __name__ == "__main__":
    main()
