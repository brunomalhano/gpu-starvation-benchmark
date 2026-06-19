# GPU Starvation Benchmark

A reproducible benchmark that measures **GPU starvation** caused by data-loading
I/O bottlenecks during deep-learning training. It compares three data-access
strategies for a ResNet-18 image-classification workload and reports throughput,
GPU compute time, data-wait time, and the resulting GPU starvation rate across
CPU, GPU, local, and Azure Container Apps environments.

## Motivation

In modern training pipelines the GPU is often idle while it waits for the next
batch to be read, decoded, and transferred from storage. This idle time —
**GPU starvation** — wastes expensive accelerator capacity. This project
quantifies that effect and shows how the data-loading strategy, not the model,
can dominate end-to-end training time.

## Scenarios

| Scenario | Strategy | Description |
|----------|----------|-------------|
| **A** | Naive POSIX read | Reads loose image files directly from remote storage with `ImageFolder`. |
| **B** | WebDataset (sequential) | Streams a sharded `.tar` archive sequentially with `webdataset`. |
| **C** | Burst Buffer (staging) | Stages the dataset to a local SSD in parallel, then trains from local disk. |

Each scenario trains the same model and is measured with the shared
instrumentation in `src/measurement_utils.py`.

## Metrics

For every epoch the benchmark records:

- `total_time_s` — wall-clock epoch time
- `throughput_mb_s` — data throughput (MB/s)
- `gpu_compute_s` — time spent in forward/backward/optimizer
- `data_wait_s` — time the loop spent waiting for the next batch
- `gpu_starvation_percent` — `data_wait_s / total_time_s * 100`

Jitter experiments additionally report per-batch latency distributions
(p50 / p95 / p99) via `src/check_jitter.py` and `src/experiment_jitter.py`.

## Repository structure

```
src/                     Experiment and training scripts
  measurement_utils.py   Shared training + measurement instrumentation
  scenario_a.py          Scenario A — naive POSIX read
  scenario_b.py          Scenario B — WebDataset sequential read
  scenario_c.py          Scenario C — burst-buffer staging to local SSD
  experiment_cpu.py      CPU run orchestration
  experiment_gpu.py      GPU run orchestration
  experiment_azure.py    Azure Container Apps run orchestration
  experiment_jitter.py   Per-batch latency / jitter experiment
  check_jitter.py        Jitter analysis helper
env/                     Runtime assets (Dockerfile, requirements.txt)
infra/main.bicep         Azure infrastructure (ACR, Storage, Container Apps Jobs)
scripts/deploy.sh        Build, deploy, and dataset-upload automation
data/results_sample/     Lightweight sample results used in the paper
  colab_fuse/            Public cloud / Colab run over FUSE network storage (CPU and GPU)
  local/                 Local single-run results (CPU and GPU)
  azure_aca/             Azure Container Apps single-run results (CPU and GPU)
  azure_aca_sweep/       Azure worker-count sweep (consolidated)
data/DATA_ACCESS.md      How to obtain the raw dataset (not redistributed here)
docs/                    Publication scope and content boundaries
```

## Requirements

- Python 3.10+
- PyTorch and torchvision
- `webdataset` (Scenario B)
- A CUDA-capable GPU for GPU runs (CPU runs work without one)

Install dependencies:

```bash
pip install -r env/requirements.txt
```

## Running locally

Adjust the dataset paths at the top of each scenario script, then run the
desired experiment from the `src/` directory:

```bash
cd src

# Single scenario
python scenario_a.py
python scenario_b.py
python scenario_c.py

# Full CPU / GPU experiment runners
python experiment_cpu.py
python experiment_gpu.py

# Jitter / tail-latency experiment
python experiment_jitter.py
```

Results are written as `training_results_*.csv` / `.txt` and
`jitter_results_*` / `batch_latencies_*` files.

## Running on Azure Container Apps

`infra/main.bicep` provisions an Azure Container Registry, a Storage Account with
Azure Files shares for the dataset and outputs, Log Analytics, a user-assigned
managed identity (with `AcrPull`), and a Container Apps Environment with CPU and
GPU workload profiles exposed as two Jobs (`job-experiment-cpu`,
`job-experiment-gpu`).

```bash
# Build the image, deploy infrastructure, and upload the dataset
./scripts/deploy.sh
```

See `scripts/deploy.sh` for the required parameters (subscription, resource
group, location, registry name).

## Sample results

Sample results are grouped by environment under `data/results_sample/`:
`colab_fuse/`, `local/`, `azure_aca/`, and `azure_aca_sweep/`.

Public cloud / Colab GPU run over FUSE network storage
(`data/results_sample/colab_fuse/training_results_gpu.csv`) — the headline
I/O-bound case, where naive POSIX reads leave the GPU idle ~99% of the time:

| Scenario | Total time (s) | Throughput (MB/s) | Data wait (s) | GPU starvation (%) |
|----------|---------------:|------------------:|--------------:|-------------------:|
| A | 1014.49 |  1.58 | 1002.54 | 98.8 |
| B |   86.77 | 18.53 |   77.76 | 89.6 |
| C |   39.04 | 41.18 |   29.97 | 76.8 |

Local CPU run (`data/results_sample/local/training_results_cpu.csv`):

| Scenario | Total time (s) | Throughput (MB/s) | Data wait (s) | GPU starvation (%) |
|----------|---------------:|------------------:|--------------:|-------------------:|
| A | 134.75 | 11.93 | 38.81 | 28.8 |
| B | 171.67 |  9.37 | 76.66 | 44.7 |
| C | 265.07 | 12.13 | 61.06 | 23.0 |

Azure Container Apps GPU run (`data/results_sample/azure_aca/training_results_gpu.csv`):

| Scenario | Total time (s) | Throughput (MB/s) | Data wait (s) | GPU starvation (%) |
|----------|---------------:|------------------:|--------------:|-------------------:|
| A | 15.10 | 39.82 | 11.95 | 79.2 |
| B | 49.31 | 32.61 | 41.77 | 84.7 |
| C | 14.57 | 41.27 | 11.81 | 81.1 |

Numbers are illustrative samples from one environment; absolute values depend on
storage, network, and accelerator. The benchmark is intended for **relative**
comparison of data-loading strategies.

## Data

The raw image dataset is **not** redistributed in this repository. See
`data/DATA_ACCESS.md` for the approved source and access constraints. Only
small sample result files are included under `data/results_sample/`.

## License

Released under the [MIT License](LICENSE).
