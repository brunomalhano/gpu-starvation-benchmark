# Content Matrix (Public vs Private)

| Source path | Public GitHub | Destination | Reason |
|---|---|---|---|
| `01_code/src/*.py` | Yes | `src/` | Core experiment logic and reproducibility scripts |
| `01_code/env/*` | Yes | `env/` | Runtime setup and dependency lock |
| `01_code/infra/main.bicep` | Yes | `infra/` | Infrastructure definition for cloud experiments |
| `01_code/automation/deploy.sh` | Yes | `scripts/` | Deployment automation |
| `01_code/configs/config_host_local.txt` | No | N/A | Contains machine/network-specific details |
| `01_code/configs/bkp_config_host_local_carmo.txt` | No | N/A | Contains machine/network-specific details |
| `02_data/raw/**` | No | N/A | Large raw dataset and redistribution constraints |
| `02_data/results/output/*.csv` and `*.txt` (selected) | Yes | `data/results_sample/` | Lightweight published evidence |
| `02_data/results/output/_dataset_local/**` | No | N/A | Data mirror and large local artifacts |
| `03_manuscript/source/main.tex` | No | N/A | Manuscript stays outside public code repository |
| `03_manuscript/source/figures/*.png` | No | N/A | Manuscript assets stay outside public code repository |
| `03_manuscript/build/**` | No | N/A | Generated build artifacts |
| `03_manuscript/submission/**` | Optional | Usually no | Journal submission package can remain separate |
| `06_workspace/**` | No | N/A | Local cache/system artifacts |

## Recommended policy

- Keep the repository focused on reproducibility assets.
- Keep the manuscript submission package and heavy data archives outside the public root.
- If you need complete reproducibility with full data, publish an external archive and link it from README.
