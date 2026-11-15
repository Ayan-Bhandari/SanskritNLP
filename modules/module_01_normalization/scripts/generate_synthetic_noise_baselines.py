"""Generate controlled normalization pairs and compare simple baselines."""

import hashlib
from pathlib import Path
import random
import sys
import unicodedata


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
if str(MODULE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT / "src"))

import pandas as pd
from rapidfuzz.distance import Levenshtein

from normalization import normalize_text


SPLIT_PATH = PROJECT_ROOT / "datasets" / "splits" / "sansar_source_aware_group_splits.parquet"
OUTPUT_DIRECTORY = PROJECT_ROOT / "experiments" / "normalization" / "generated"
PAIRS_PATH = OUTPUT_DIRECTORY / "synthetic_noise_pairs.parquet"
METRICS_PATH = OUTPUT_DIRECTORY / "synthetic_baseline_metrics.csv"

BASE_SEED = 43
CHARACTER_EDIT_RATES = (0.01, 0.03, 0.05)
VIRAMA = "\u094d"
JOIN_CONTROLS = {"\u200c", "\u200d"}
DEVANAGARI_RANGES = ((0x0900, 0x097F), (0x1CD0, 0x1CFF), (0x11B00, 0x11B5F))


def stable_seed(record_id, split_name, noise_type, severity):
    """Create a reproducible seed without Python's process-randomized hash."""
    seed_text = f"{BASE_SEED}|{record_id}|{split_name}|{noise_type}|{severity}"
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def contains_devanagari_letter(text):
    for character in text:
        code_point = ord(character)
        is_devanagari = any(start <= code_point <= end for start, end in DEVANAGARI_RANGES)
        if is_devanagari and unicodedata.category(character).startswith("L"):
            return True
    return False


def split_into_akshara_like_units(text):
    """Keep vowel marks and virama conjuncts attached during synthetic edits."""
    units = []
    current_unit = ""

    for character in text:
        category = unicodedata.category(character)
        is_boundary = character.isspace() or category.startswith("P")

        if is_boundary:
            if current_unit:
                units.append(current_unit)
                current_unit = ""
            units.append(character)
            continue

        if not current_unit:
            current_unit = character
            continue

        previous_core = current_unit.rstrip("\u200c\u200d")
        follows_virama = previous_core.endswith(VIRAMA)
        is_mark_or_joiner = category.startswith("M") or character in JOIN_CONTROLS

        if is_mark_or_joiner or follows_virama:
            current_unit += character
        else:
            units.append(current_unit)
            current_unit = character

    if current_unit:
        units.append(current_unit)

    return units


def make_formatting_noise(clean_text, random_generator):
    """Introduce reversible whitespace and line-ending variations."""
    noisy_text = f" {clean_text} "

    if " " in clean_text:
        first_space = noisy_text.find(" ", 1)
        if first_space >= 0:
            noisy_text = (
                noisy_text[:first_space]
                + "\t  "
                + noisy_text[first_space + 1:]
            )

    if "\n" in clean_text and random_generator.random() < 0.5:
        noisy_text = noisy_text.replace("\n", "\r\n", 1)

    decomposed_text = unicodedata.normalize("NFD", noisy_text)
    if normalize_text(decomposed_text) == clean_text:
        noisy_text = decomposed_text

    return noisy_text


def editable_positions(units):
    positions = []
    for position, unit in enumerate(units):
        if contains_devanagari_letter(unit):
            positions.append(position)
    return positions


def make_character_edit_noise(clean_text, rate, alphabet, random_generator):
    """Apply generic random akshara-like edits; this is not an OCR simulator."""
    units = split_into_akshara_like_units(clean_text)
    positions = editable_positions(units)
    if not positions or not alphabet:
        return clean_text

    edit_count = max(1, round(len(positions) * rate))
    for _ in range(edit_count):
        positions = editable_positions(units)
        if not positions:
            break

        position = random_generator.choice(positions)
        operation = random_generator.choice(("substitute", "delete", "insert"))

        if operation == "delete" and len(positions) > 1:
            del units[position]
        elif operation == "insert":
            inserted_unit = random_generator.choice(alphabet)
            units.insert(position + 1, inserted_unit)
        else:
            replacements = [unit for unit in alphabet if unit != units[position]]
            if replacements:
                units[position] = random_generator.choice(replacements)
            else:
                units.insert(position + 1, random_generator.choice(alphabet))

    return "".join(units)


def character_error_rate(prediction, reference):
    prediction = unicodedata.normalize("NFC", prediction)
    reference = unicodedata.normalize("NFC", reference)
    if not reference:
        return 0.0 if not prediction else 1.0
    distance = Levenshtein.distance(prediction, reference)
    return distance / len(reference)


def make_pair(source_row, noise_type, severity, target_edit_rate, noisy_text, seed):
    clean_text = source_row["normalized_text"]
    normalized_prediction = normalize_text(noisy_text)

    pair = source_row.to_dict()
    pair["clean_text"] = pair.pop("normalized_text")
    pair["noisy_text"] = noisy_text
    pair["noise_type"] = noise_type
    pair["severity"] = severity
    pair["target_edit_rate"] = target_edit_rate
    pair["seed"] = seed
    pair["identity_cer"] = character_error_rate(noisy_text, clean_text)
    pair["normalizer_cer"] = character_error_rate(normalized_prediction, clean_text)
    pair["identity_exact_match"] = noisy_text == clean_text
    pair["normalizer_exact_match"] = normalized_prediction == clean_text
    return pair


def main():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Group split data not found: {SPLIT_PATH}")
    if OUTPUT_DIRECTORY.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT_DIRECTORY}")

    split_df = pd.read_parquet(SPLIT_PATH)
    training_texts = split_df.loc[split_df["split"] == "train", "normalized_text"]

    alphabet = set()
    for clean_text in training_texts:
        for unit in split_into_akshara_like_units(clean_text):
            if contains_devanagari_letter(unit):
                alphabet.add(unit)
    alphabet = sorted(alphabet)
    if not alphabet:
        raise ValueError("No Devanagari units were available in the training split.")

    pairs = []
    for _, source_row in split_df.iterrows():
        clean_text = source_row["normalized_text"]
        record_id = source_row["id"]
        split_name = source_row["split"]

        noise_type = "formatting"
        severity = "mixed"
        seed = stable_seed(record_id, split_name, noise_type, severity)
        random_generator = random.Random(seed)
        noisy_text = make_formatting_noise(clean_text, random_generator)
        pairs.append(make_pair(source_row, noise_type, severity, None, noisy_text, seed))

        for rate in CHARACTER_EDIT_RATES:
            noise_type = "generic_akshara_edits"
            severity = f"{rate:.0%} target unit edits"
            seed = stable_seed(record_id, split_name, noise_type, severity)
            random_generator = random.Random(seed)
            noisy_text = make_character_edit_noise(
                clean_text,
                rate,
                alphabet,
                random_generator,
            )
            pairs.append(
                make_pair(source_row, noise_type, severity, rate, noisy_text, seed)
            )

    pairs_df = pd.DataFrame.from_records(pairs)
    formatting_rows = pairs_df[pairs_df["noise_type"] == "formatting"]
    formatting_failures = int((~formatting_rows["normalizer_exact_match"]).sum())
    if formatting_failures:
        raise ValueError(
            f"The deterministic normalizer failed on {formatting_failures} "
            "formatting-noise examples."
        )

    metrics_df = (
        pairs_df.groupby(["split", "noise_type", "severity"], dropna=False)
        .agg(
            pairs=("id", "count"),
            identity_mean_cer=("identity_cer", "mean"),
            normalizer_mean_cer=("normalizer_cer", "mean"),
            identity_exact_match_rate=("identity_exact_match", "mean"),
            normalizer_exact_match_rate=("normalizer_exact_match", "mean"),
        )
        .reset_index()
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    pairs_df.to_parquet(PAIRS_PATH, index=False)
    metrics_df.to_csv(METRICS_PATH, index=False, encoding="utf-8")

    print(f"Clean split rows: {len(split_df):,}")
    print(f"Synthetic pairs: {len(pairs_df):,}")
    print(f"Formatting pairs recovered exactly by the normalizer: {len(formatting_rows):,}")
    print("Metrics by split, noise type, and severity:")
    print(metrics_df.to_string(index=False))
    print(f"Saved synthetic pairs: {PAIRS_PATH}")
    print(f"Saved baseline metrics: {METRICS_PATH}")
    print("Generic character edits are synthetic robustness checks, not OCR evidence.")


if __name__ == "__main__":
    main()
