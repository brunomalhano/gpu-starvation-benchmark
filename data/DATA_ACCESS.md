# Data Access

The full raw image dataset used in experiments is not redistributed in this public repository.

## What is available here

- Aggregated and sample experiment outputs in `data/results_sample/`.

### Sample results by environment

Each subfolder under `data/results_sample/` holds one run environment and maps to
the environments reported in the manuscript:

| Folder | Environment | Storage | Accelerator |
|--------|-------------|---------|-------------|
| `colab_fuse/` | Public cloud / Colab | FUSE network mount | CPU baseline + GPU |
| `local/` | Edge / local workstation | Local SSD | CPU + GPU (RTX 3050) |
| `azure_aca/` | Enterprise cloud (Azure Container Apps) | NVMe / Azure Files | CPU + GPU (Tesla T4) |
| `azure_aca_sweep/` | Azure Container Apps | NVMe | GPU worker-count sweep (consolidated) |

Each environment provides `training_results_cpu.{csv,txt}` and
`training_results_gpu.{csv,txt}` (the sweep folder provides a single
`consolidated_sweep.csv`). Values are single-epoch runs per scenario.

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
