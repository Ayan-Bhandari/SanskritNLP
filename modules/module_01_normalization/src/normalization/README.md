# Module 1 normalization baseline

`normalize_text` is a deterministic first pass for Sanskrit text. It makes
Unicode and whitespace representation more consistent while keeping original
verse line breaks available for later processing.

## Rules

The function applies these operations in order:

1. Apply Unicode NFC normalization.
2. Convert Windows CRLF and old Mac CR line endings to LF (`\n`).
3. Replace each run of horizontal whitespace within a line with one ordinary
   space. This includes tabs and repeated spaces.
4. Remove whitespace at the start and end of each line. Preserve line breaks
   and internal blank lines.
5. Leave punctuation unchanged. Return non-string values unchanged so missing
   records remain visible to the caller.

The initial 5,000-row pilot found 12 rows with line feeds. A later source-aware
5,000-row sample found line breaks in 143 rows. They can mark verse structure,
so the baseline preserves them instead of flattening them into spaces. The
full normalized corpus also retains each original string in `raw_text`.

The full pinned Sansar `default/train` split contains 1,829,785 normalized
records stored as 183 Parquet shards under
`datasets/processed/sansar_normalized/`. See
`../../docs/research/03_module1_completion.md` for the completion
summary and manifest details.

## Examples

| Case | Input | Output |
|---|---|---|
| NFC composition (general Unicode example) | `e` followed by combining acute accent | `é` |
| Repeated spaces | `रामः   वनम्` | `रामः वनम्` |
| Tab between words | `रामः\tवनम्` | `रामः वनम्` |
| CRLF line ending | `रामः\r\nवनम्` | `रामः\nवनम्` |
| Existing line break | `रामः\nवनम्` | `रामः\nवनम्` |
| Blank line | `रामः\n\nवनम्` | `रामः\n\nवनम्` |
| Edge spaces | `  रामः  ` | `रामः` |
| Punctuation | `रामः॥` | `रामः॥` |
| Missing value | `None` | `None` |

Run the examples from the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe modules\module_01_normalization\scripts\normalization_examples.py
```

## Limits

This baseline does not correct OCR errors, alter danda punctuation, validate
Sanskrit grammar, identify word boundaries, split Sandhi, or decide whether an
unusual character is corrupt. Those checks and corrections need separate
evidence and evaluation. The original raw text must remain available for every
normalization experiment.
