# Dataset card

## Fields

| Field | Description |
|---|---|
| `sentence` | Bodo sentence in Devanagari script |
| `tonal_word` | Target word whose tone is labeled |
| `label` | `H` (High) or `L` (Low) |

## File format

`example_with_tone.csv` is a UTF-8 CSV with three fields per row and no header row. Read it with `header=None`.

## Important before publication

- Confirm the dataset's original source and citation.
- Confirm that redistribution is permitted.
- Document who assigned the tone labels and how disagreements were handled.
- Avoid publishing any sensitive or restricted data.
