# Data Access

The full raw image dataset used in experiments is not redistributed in this public repository.

## What is available here

- Aggregated and sample experiment outputs in `data/results_sample/`.

## What must be provided by the user

- A local dataset folder compatible with the expected class layout used by the scripts.
- If required by licensing, data acquisition must follow the original source terms.

## Expected local path convention

Set your local dataset root and output folder through environment variables used by the scripts:

- `DATASET_PATH`
- `OUTPUT_PATH`

## Reproducibility recommendation

For publication-grade reproducibility:

1. Add dataset provenance and license details.
2. Add exact retrieval/preparation steps.
3. Add checksums or version identifiers when possible.
