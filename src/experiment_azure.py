#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment_azure.py
Version unificada para Azure Container Apps — CPU e GPU via workload profiles.

Environment variables:
  DATASET_PATH  — raiz do dataset com subpastas de classe + dataset_solto.tar
  OUTPUT_PATH   — output directory (resultados .txt/.csv + config_host)
  FORCE_CPU     — "1" para force modo CPU (mesmo com GPU available)
  NUM_WORKERS   — workers do DataLoader (default: 0)
  NUM_EPOCHS    — epochs per scenario (default: 1)
"""

import os
import sys
import time
import csv
import socket
import subprocess
import shutil
import tarfile
from pathlib import Path
from datetime import datetime

import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import webdataset as wds

def _silenciar_keyboard_interrupt(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        print("\nInterrupted by user.")
        return
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _silenciar_keyboard_interrupt


DATASET_PATH = Path(os.environ.get("DATASET_PATH", "/workspace/dataset"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/workspace/output"))
FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "1"))

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

CAMINHO_TAR_DATASET = DATASET_PATH / "dataset_solto.tar"
STAGING_PATH = Path("/tmp/dataset_local")


if FORCE_CPU:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE_TAG = "gpu" if device.type == "cuda" else "cpu"
HOSTNAME = socket.gethostname()

print(f"Host: {HOSTNAME}")
print(f"Device: {device} ({'forced' if FORCE_CPU else 'auto-detected'})")
print(f"Dataset: {DATASET_PATH}")
print(f"Output: {OUTPUT_PATH}")
print(f"Torch: {torch.__version__}")
if device.type == "cuda":
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")


ARQUIVO_TXT = OUTPUT_PATH / f"training_results_{DEVICE_TAG}.txt"
ARQUIVO_CSV = OUTPUT_PATH / f"training_results_{DEVICE_TAG}.csv"


def _sh(cmd):
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True,
            stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except Exception:
        return "N/A"

def coletar_config_host():
    linhas = [
        f"**Configuracao Azure Container Apps — {DEVICE_TAG.upper()}**",
        "",
        f"Hostname: `{HOSTNAME}`",
        f"Data/Hora: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
    ]

    cpu_model = _sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2").strip()
    cpu_cores = _sh("nproc")
    cpu_threads = _sh("lscpu | grep '^CPU(s):' | head -1 | awk '{print $2}'")
    linhas += [
        f"CPU: `{cpu_model}`",
        f"Cores: `{cpu_cores}`",
        f"Threads: `{cpu_threads}`",
        "",
    ]

    mem_total = _sh("free -h | grep Mem | awk '{print $2}'")
    linhas += [f"Memory: `{mem_total}`", ""]

    gpu_name = _sh("nvidia-smi --query-gpu=name --format=csv,noheader")
    gpu_mem = _sh("nvidia-smi --query-gpu=memory.total --format=csv,noheader")
    gpu_driver = _sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    linhas += [
        f"GPU: `{gpu_name}`",
        f"VRAM: `{gpu_mem}`",
        f"Driver NVIDIA: `{gpu_driver}`",
        "",
        f"PyTorch: `{torch.__version__}`",
        f"CUDA available: `{torch.cuda.is_available()}`",
    ]
    if torch.cuda.is_available():
        linhas.append(f"CUDA version: `{torch.version.cuda}`")

    config_file = OUTPUT_PATH / f"config_host_{DEVICE_TAG}.txt"
    config_file.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"Config do host salva em {config_file}")

coletar_config_host()


model = models.resnet18().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def salvar_resultados(scenario, epoch, tempo_total, throughput,
                      tempo_computacao, tempo_espera_dados, taxa_metrica):
    if device.type == "cuda":
        rotulo_device, rotulo_espera = "GPU", "CPU"
        rotulo_taxa = f"GPU Starvation Rate: {taxa_metrica:.1f}%"
    else:
        rotulo_device, rotulo_espera = "CPU", "I/O"
        rotulo_taxa = f"I/O wait rate: {taxa_metrica:.1f}%"

    linhas = [
        f"--- Results da Epoch {epoch} ({scenario}) ---",
        f"Total Epoch Time: {tempo_total:.2f} s",
        f"Throughput: {throughput:.2f} MB/s",
        f"Compute time on {rotulo_device}: {tempo_computacao:.2f} s",
        f"Time waiting for data ({rotulo_espera}): {tempo_espera_dados:.2f} s",
        rotulo_taxa,
        "-" * 30,
    ]

    with ARQUIVO_TXT.open("a", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")

    csv_existe = ARQUIVO_CSV.exists()
    with ARQUIVO_CSV.open("a", newline="", encoding="utf-8") as f:
        colunas = [
            "timestamp",
            "host",
            "scenario",
            "epoch",
            "total_time_s",
            "throughput_mb_s",
            "gpu_compute_s",
            "data_wait_s",
            "gpu_starvation_percent",
        ]
        writer = csv.DictWriter(f, fieldnames=colunas)
        if not csv_existe:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "host": HOSTNAME,
            "scenario": scenario,
            "epoch": epoch,
            "total_time_s": f"{tempo_total:.2f}",
            "throughput_mb_s": f"{throughput:.2f}",
            "gpu_compute_s": f"{tempo_computacao:.2f}",
            "data_wait_s": f"{tempo_espera_dados:.2f}",
            "gpu_starvation_percent": f"{taxa_metrica:.1f}",
        })

    for linha in linhas:
        print(linha)

def treinar_e_medir(dataloader, num_epochs=1, scenario="Sem nome"):
    model.train()
    for epoch in range(num_epochs):
        inicio_epoch = time.time()
        tempo_espera_dados = 0.0
        tempo_computacao = 0.0
        bytes_processados = 0

        t0 = time.time()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            tempo_espera_dados += (time.time() - t0)
            bytes_processados += inputs.element_size() * inputs.nelement()
            inputs, targets = inputs.to(device), targets.to(device)

            t1 = time.time()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()

            tempo_computacao += (time.time() - t1)
            t0 = time.time()

        tempo_total = time.time() - inicio_epoch
        mb_processados = bytes_processados / (1024 * 1024)
        throughput = mb_processados / tempo_total
        taxa = (tempo_espera_dados / tempo_total) * 100

        salvar_resultados(scenario, epoch + 1, tempo_total, throughput,
                          tempo_computacao, tempo_espera_dados, taxa)


def preparar_dataset():
    if any(DATASET_PATH.rglob("*.jpg")) or any(DATASET_PATH.rglob("*.png")):
        return

    if not CAMINHO_TAR_DATASET.is_file():
        raise FileNotFoundError(
            f"Nenhum dataset encontrado em {DATASET_PATH}. "
            "Upload das images e/ou dataset_solto.tar para o Azure Files share 'dataset'."
        )

    print(f"Extraindo dataset de {CAMINHO_TAR_DATASET}...")
    DATASET_PATH.mkdir(parents=True, exist_ok=True)
    with tarfile.open(CAMINHO_TAR_DATASET, "r") as tf:
        for membro in tf.getmembers():
            partes = Path(membro.name).parts
            if len(partes) <= 1:
                continue
            membro.name = str(Path(*partes[1:]))
            tf.extract(membro, DATASET_PATH)

preparar_dataset()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


print("\n" + "=" * 60)
print("Scenario A: Read POSIX direto do Azure Files")
print("=" * 60)
try:
    if not any(p.is_dir() for p in DATASET_PATH.iterdir()):
        raise FileNotFoundError(f"Nenhuma subpasta de classe em {DATASET_PATH}")

    dataset_a = datasets.ImageFolder(str(DATASET_PATH), transform=transform)
    dataloader_a = DataLoader(dataset_a, batch_size=32, shuffle=True,
                              num_workers=NUM_WORKERS)
    treinar_e_medir(dataloader_a, num_epochs=NUM_EPOCHS, scenario="Scenario A")
except Exception as e:
    print(f"Error Scenario A: {e}")


print("\n" + "=" * 60)
print("Scenario B: WebDataset (sequential read)")
print("=" * 60)
try:
    if not CAMINHO_TAR_DATASET.is_file():
        raise FileNotFoundError(f".tar not found: {CAMINHO_TAR_DATASET}")

    caminho_tar_url = "file:" + str(CAMINHO_TAR_DATASET)
    dataset_b = (
        wds.WebDataset(caminho_tar_url, empty_check=False, shardshuffle=False)
        .shuffle(1000)
        .decode("torchrgb")
        .to_tuple("jpg;png")
        .map_tuple(transforms.Resize((224, 224)))
        .map(lambda x: (x[0], 0))
    )
    dataloader_b = DataLoader(dataset_b, batch_size=32, num_workers=NUM_WORKERS)
    treinar_e_medir(dataloader_b, num_epochs=NUM_EPOCHS, scenario="Scenario B")
except Exception as e:
    print(f"Error Scenario B: {e}")


class DatasetImagensSoltas(Dataset):
    def __init__(self, diretorio, transform=None):
        self.transform = transform
        self.arquivos = []
        for root, _, files in os.walk(diretorio):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.arquivos.append(os.path.join(root, f))

    def __len__(self):
        return len(self.arquivos)

    def __getitem__(self, idx):
        img = Image.open(self.arquivos[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0

def copiar_arquivo(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

def stage_data_parallel(src_dir, dst_dir, max_workers=16):
    from concurrent.futures import ThreadPoolExecutor
    print("Staging (parallel copy) for local ephemeral storage...")
    t0 = time.time()
    tarefas = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, _, files in os.walk(src_dir):
            for f in files:
                src_file = os.path.join(root, f)
                rel = os.path.relpath(src_file, src_dir)
                dst_file = os.path.join(dst_dir, rel)
                tarefas.append(executor.submit(copiar_arquivo, src_file, dst_file))
    for t in tarefas:
        t.result()
    print(f"Staging completed in {time.time() - t0:.2f}s")

print("\n" + "=" * 60)
print("Scenario C: Burst Buffer (local ephemeral storage)")
print("=" * 60)
try:
    if not any(DATASET_PATH.rglob("*.jpg")) and not any(DATASET_PATH.rglob("*.png")):
        raise FileNotFoundError(f"Nenhuma imagem em {DATASET_PATH}")

    stage_data_parallel(str(DATASET_PATH), str(STAGING_PATH))
    dataset_c = DatasetImagensSoltas(str(STAGING_PATH), transform=transform)
    dataloader_c = DataLoader(dataset_c, batch_size=32, shuffle=True,
                              num_workers=NUM_WORKERS)
    print(f"Imagens no staging: {len(dataset_c)}")
    treinar_e_medir(dataloader_c, num_epochs=NUM_EPOCHS, scenario="Scenario C")
except Exception as e:
    print(f"Error Scenario C: {e}")


print(f"\nExperiment completed. Results at {OUTPUT_PATH}")


for arq in sorted(OUTPUT_PATH.glob("*")):
    print(f"\n{'='*60}")
    print(f">>> FILE: {arq.name}")
    print(f"{'='*60}")
    print(arq.read_text(encoding="utf-8"))
