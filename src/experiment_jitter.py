
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import os
import csv
import socket
import tarfile
from pathlib import Path
from datetime import datetime
import sys
import numpy as np
import shutil
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import webdataset as wds


def _silenciar_keyboard_interrupt(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        print("\nInterrupted by user.")
        return
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _silenciar_keyboard_interrupt


NUM_RODADAS = 10       # Independent runs per scenario
NUM_EPOCHS = 3         # Epochs per run
NUM_WORKERS = 0        # Baseline: sem prefetch paralelo
BATCH_SIZE = 32


BASE_DIR = Path(__file__).resolve().parent / "output"
BASE_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR_DATASET = Path(r"G:\Meu Drive\VSC\PPD\dataset")
CAMINHO_DATASET_LOCAL = BASE_DIR_DATASET
CAMINHO_TAR_DATASET = BASE_DIR_DATASET / "dataset_solto.tar"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
if device.type == "cpu":
    print(f"  Threads torch: {torch.get_num_threads()}")


HOSTNAME = socket.gethostname()
NOME_DISPOSITIVO_ARQUIVO = "gpu" if device.type == "cuda" else "cpu"

ARQUIVO_RESULTADOS_TXT = BASE_DIR / f"jitter_results_{NOME_DISPOSITIVO_ARQUIVO}.txt"
ARQUIVO_RESULTADOS_CSV = BASE_DIR / f"jitter_results_{NOME_DISPOSITIVO_ARQUIVO}.csv"
ARQUIVO_LATENCIAS_CSV = BASE_DIR / f"batch_latencies_{NOME_DISPOSITIVO_ARQUIVO}.csv"

print(f"Results will be saved at:")
print(f"  Epoch CSV:  {ARQUIVO_RESULTADOS_CSV}")
print(f"  Batch CSV:  {ARQUIVO_LATENCIAS_CSV}")
print(f"  TXT:        {ARQUIVO_RESULTADOS_TXT}")


def preparar_dataset_local():
    if any(CAMINHO_DATASET_LOCAL.rglob("*.jpg")) or any(CAMINHO_DATASET_LOCAL.rglob("*.png")):
        return

    if not CAMINHO_TAR_DATASET.is_file():
        raise FileNotFoundError(f"File .tar not found: {CAMINHO_TAR_DATASET}")

    print(f"Extraindo dataset para {CAMINHO_DATASET_LOCAL}...")
    CAMINHO_DATASET_LOCAL.mkdir(parents=True, exist_ok=True)
    with tarfile.open(CAMINHO_TAR_DATASET, "r") as arquivo_tar:
        for membro in arquivo_tar.getmembers():
            partes = Path(membro.name).parts
            if len(partes) <= 1:
                continue
            membro.name = str(Path(*partes[1:]))
            arquivo_tar.extract(membro, CAMINHO_DATASET_LOCAL)

preparar_dataset_local()


criterion = torch.nn.CrossEntropyLoss()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class DatasetImagensSoltas(Dataset):
    def __init__(self, diretorio, transform=None):
        self.diretorio = diretorio
        self.transform = transform
        self.arquivos = []

        # os.walk entra em todas as subpastas recursivamente buscando imagens
        for root, _, files in os.walk(diretorio):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.arquivos.append(os.path.join(root, f))

    def __len__(self):
        return len(self.arquivos)

    def __getitem__(self, idx):
        caminho_img = self.arquivos[idx]
        imagem = Image.open(caminho_img).convert('RGB')

        if self.transform:
            imagem = self.transform(imagem)

        return imagem, 0  # Return dummy class 0


def copiar_arquivo(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

def stage_data_parallel(src_dir, dst_dir, max_workers=16):
    print("Starting staging (parallel copy) to local SSD...")
    t0 = time.time()
    tarefas = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, _, files in os.walk(src_dir):
            for file in files:
                src_file = os.path.join(root, file)
                rel_path = os.path.relpath(src_file, src_dir)
                dst_file = os.path.join(dst_dir, rel_path)
                tarefas.append(executor.submit(copiar_arquivo, src_file, dst_file))

    for tarefa in tarefas:
        tarefa.result()

    print(f"Staging completed in {time.time() - t0:.2f} seconds!")


caminho_scenario_a = str(CAMINHO_DATASET_LOCAL)
caminho_scenario_b = "file:" + str(CAMINHO_TAR_DATASET).replace("\\", "/")
caminho_origem_gdrive = str(CAMINHO_DATASET_LOCAL)
caminho_destino_ssd_local = str(BASE_DIR / "_dataset_local")


COLUNAS_EPOCA = [
    "timestamp",
    "host",
    "scenario",
    "run",
    "epoch",
    "total_time_s",
    "throughput_mb_s",
    "gpu_compute_s",
    "data_wait_s",
    "gpu_starvation_percent",
    "jitter_espera_s",
    "jitter_compute_s",
    "cv_espera",
    "p50_espera_s",
    "p95_espera_s",
    "p99_espera_s",
    "num_batches",
]

COLUNAS_BATCH = [
    "scenario",
    "run",
    "epoch",
    "batch_idx",
    "latencia_espera_s",
    "latencia_compute_s",
]

def _inicializar_csv(caminho_csv, colunas):
    """Unified jitter experiment script."""
    if not caminho_csv.exists():
        with caminho_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=colunas)
            writer.writeheader()

def salvar_resultados_epoch(
    scenario, run, epoch, tempo_total, throughput,
    tempo_computacao, tempo_espera_dados, taxa_inanicao,
    jitter_espera, jitter_compute, cv_espera,
    p50_espera, p95_espera, p99_espera, num_batches,
):
    """Salva uma linha no CSV de epochs e imprime/escreve no TXT."""
    # --- Readable TXT ---
    if device.type == "cuda":
        nome_dispositivo = "GPU"
        nome_espera = "CPU"
        linha_taxa = f"GPU Starvation Rate: {taxa_inanicao:.1f}%"
    else:
        nome_dispositivo = "CPU"
        nome_espera = "I/O"
        linha_taxa = f"I/O wait rate: {taxa_inanicao:.1f}%"

    linhas = [
        f"--- Epoch Results {epoch} ({scenario}) - Run {run}/{NUM_RODADAS} ---",
        f"Total Epoch Time: {tempo_total:.2f} s",
        f"Throughput: {throughput:.2f} MB/s",
        f"Tempo processando na {nome_dispositivo}: {tempo_computacao:.2f} s",
        f"Time waiting for data ({nome_espera}): {tempo_espera_dados:.2f} s",
        linha_taxa,
        f"Jitter (σ espera dados): {jitter_espera:.4f} s",
        f"Jitter (σ compute): {jitter_compute:.4f} s",
        f"Coefficient of Variation: {cv_espera:.4f}",
        f"Latency P50 / P95 / P99: {p50_espera:.4f} / {p95_espera:.4f} / {p99_espera:.4f} s",
        f"Nº de batches: {num_batches}",
        "-" * 30,
    ]

    with ARQUIVO_RESULTADOS_TXT.open("a", encoding="utf-8") as arquivo_txt:
        arquivo_txt.write("\n".join(linhas) + "\n")

    for linha in linhas:
        print(linha)

    # --- CSV de epochs ---
    with ARQUIVO_RESULTADOS_CSV.open("a", newline="", encoding="utf-8") as arquivo_csv:
        writer = csv.DictWriter(arquivo_csv, fieldnames=COLUNAS_EPOCA)
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "host": HOSTNAME,
            "scenario": scenario,
            "run": run,
            "epoch": epoch,
            "total_time_s": f"{tempo_total:.4f}",
            "throughput_mb_s": f"{throughput:.4f}",
            "gpu_compute_s": f"{tempo_computacao:.4f}",
            "data_wait_s": f"{tempo_espera_dados:.4f}",
            "gpu_starvation_percent": f"{taxa_inanicao:.2f}",
            "jitter_espera_s": f"{jitter_espera:.6f}",
            "jitter_compute_s": f"{jitter_compute:.6f}",
            "cv_espera": f"{cv_espera:.6f}",
            "p50_espera_s": f"{p50_espera:.6f}",
            "p95_espera_s": f"{p95_espera:.6f}",
            "p99_espera_s": f"{p99_espera:.6f}",
            "num_batches": num_batches,
        })

def salvar_batch_latencies(scenario, run, epoch, latencias_espera, latencias_compute):
    """Grava as latencys individuais de cada batch no CSV granular."""
    with ARQUIVO_LATENCIAS_CSV.open("a", newline="", encoding="utf-8") as arquivo_csv:
        writer = csv.DictWriter(arquivo_csv, fieldnames=COLUNAS_BATCH)
        for idx, (lat_espera, lat_compute) in enumerate(zip(latencias_espera, latencias_compute)):
            writer.writerow({
                "scenario": scenario,
                "run": run,
                "epoch": epoch,
                "batch_idx": idx,
                "latencia_espera_s": f"{lat_espera:.6f}",
                "latencia_compute_s": f"{lat_compute:.6f}",
            })


def treinar_e_medir(model, optimizer, dataloader, num_epochs, scenario, run):
    """Treina o modelo e coleta metrics por epoch e por batch."""
    model.train()

    for epoch in range(num_epochs):
        inicio_epoch = time.time()
        tempo_espera_dados = 0.0
        tempo_computacao = 0.0
        bytes_processados = 0

        latencias_espera_batch = []   # latency de espera por batch (seconds)
        latencias_compute_batch = []  # latency de compute por batch (seconds)

        t0 = time.time()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Medir o tempo de espera para os dados
            latencia_espera = time.time() - t0
            latencias_espera_batch.append(latencia_espera)
            tempo_espera_dados += latencia_espera

            # Approximate MB processed (tensor size)
            bytes_processados += inputs.element_size() * inputs.nelement()

            # Transferir para o dispositivo
            inputs, targets = inputs.to(device), targets.to(device)

            # Start compute timing
            t1 = time.time()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Sincronizar GPU para garantir que terminou o trabalho
            if device.type == "cuda":
                torch.cuda.synchronize()

            latencia_compute = time.time() - t1
            latencias_compute_batch.append(latencia_compute)
            tempo_computacao += latencia_compute

            t0 = time.time()  # Reset timer for next batch

        tempo_total = time.time() - inicio_epoch
        mb_processados = bytes_processados / (1024 * 1024)
        throughput = mb_processados / tempo_total if tempo_total > 0 else 0.0
        taxa_inanicao = (tempo_espera_dados / tempo_total) * 100 if tempo_total > 0 else 0.0

        # --- Jitter metrics ---
        arr_espera = np.array(latencias_espera_batch)
        arr_compute = np.array(latencias_compute_batch)

        # Jitter = standard deviation of per-batch wait latencies
        jitter_espera = float(np.std(arr_espera))
        jitter_compute = float(np.std(arr_compute))

        # Coefficient of Variation
        media_espera = float(np.mean(arr_espera))
        cv_espera = jitter_espera / media_espera if media_espera > 0 else 0.0

        # Percentis (em seconds)
        p50_espera = float(np.percentile(arr_espera, 50))
        p95_espera = float(np.percentile(arr_espera, 95))
        p99_espera = float(np.percentile(arr_espera, 99))

        num_batches = len(latencias_espera_batch)

        # Salvar resultados da epoch
        salvar_resultados_epoch(
            scenario, run, epoch + 1, tempo_total, throughput,
            tempo_computacao, tempo_espera_dados, taxa_inanicao,
            jitter_espera, jitter_compute, cv_espera,
            p50_espera, p95_espera, p99_espera, num_batches,
        )

        # Salvar latencys individuais por batch
        salvar_batch_latencies(
            scenario, run, epoch + 1,
            latencias_espera_batch, latencias_compute_batch,
        )


_inicializar_csv(ARQUIVO_RESULTADOS_CSV, COLUNAS_EPOCA)
_inicializar_csv(ARQUIVO_LATENCIAS_CSV, COLUNAS_BATCH)


print(f"\nExperiment configuration:")
print(f"  Dispositivo:   {device}")
print(f"  Runs:       {NUM_RODADAS}")
print(f"  Epochs/run: {NUM_EPOCHS}")
print(f"  Batch size:    {BATCH_SIZE}")
print(f"  Num workers:   {NUM_WORKERS}")
print(f"  Host:          {HOSTNAME}")

for run in range(1, NUM_RODADAS + 1):
    print(f"\n{'='*60}")
    print(f"RODADA {run}/{NUM_RODADAS}")
    print(f"{'='*60}")

    # Re-instantiate model and optimizer for statistical independence
    model = models.resnet18().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # ---------------------------------------------------------------
    # Scenario A: naive POSIX read (ImageFolder)
    # ---------------------------------------------------------------
    print(f"\nStarting Scenario A: naive POSIX read — Run {run}/{NUM_RODADAS}")
    try:
        if not any(p.is_dir() for p in CAMINHO_DATASET_LOCAL.iterdir()):
            raise FileNotFoundError(
                f"Nenhuma subpasta de classe encontrada em {CAMINHO_DATASET_LOCAL}"
            )

        dataset_a = datasets.ImageFolder(caminho_scenario_a, transform=transform)
        dataloader_a = DataLoader(
            dataset_a, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        )

        treinar_e_medir(
            model, optimizer, dataloader_a,
            num_epochs=NUM_EPOCHS, scenario="Scenario A", run=run,
        )
    except Exception as e:
        print(f"Error in Scenario A: {e}")

    # Re-instanciar modelo para isolar scenarios dentro da mesma run
    model = models.resnet18().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # ---------------------------------------------------------------
    # Scenario B: WebDataset (leitura sequencial de .tar)
    # ---------------------------------------------------------------
    print(f"\nStarting Scenario B: WebDataset (sequential read) — Run {run}/{NUM_RODADAS}")
    try:
        if not CAMINHO_TAR_DATASET.is_file():
            raise FileNotFoundError(f"File .tar not found: {CAMINHO_TAR_DATASET}")

        # Construindo pipeline de dados
        dataset_b = (
            wds.WebDataset(caminho_scenario_b, empty_check=False, shardshuffle=False)
            .shuffle(1000)
            .decode("torchrgb")
            .to_tuple("jpg;png")
            .map_tuple(transforms.Resize((224, 224)))
            .map(lambda x: (x[0], 0))
        )

        dataloader_b = DataLoader(
            dataset_b, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        )

        treinar_e_medir(
            model, optimizer, dataloader_b,
            num_epochs=NUM_EPOCHS, scenario="Scenario B", run=run,
        )
    except Exception as e:
        print(f"Error in Scenario B: {e}")

    # Re-instanciar modelo para isolar scenarios dentro da mesma run
    model = models.resnet18().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # ---------------------------------------------------------------
    # Scenario C: Burst Buffer (staging para SSD local)
    # ---------------------------------------------------------------
    print(f"\nStarting Scenario C: Burst Buffer (SSD Local) — Run {run}/{NUM_RODADAS}")
    try:
        if not any(p.is_file() for p in CAMINHO_DATASET_LOCAL.rglob("*")):
            raise FileNotFoundError(f"Nenhuma imagem encontrada em {CAMINHO_DATASET_LOCAL}")

        # Run staging (fast copy)
        stage_data_parallel(caminho_origem_gdrive, caminho_destino_ssd_local)

        # Carregar do SSD local usando o Dataset customizado
        dataset_c = DatasetImagensSoltas(caminho_destino_ssd_local, transform=transform)

        dataloader_c = DataLoader(
            dataset_c, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        )

        print(f"Sucesso! Encontradas {len(dataset_c)} imagens no SSD local.")
        treinar_e_medir(
            model, optimizer, dataloader_c,
            num_epochs=NUM_EPOCHS, scenario="Scenario C", run=run,
        )
    except Exception as e:
        print(f"Error in Scenario C: {e}")


print(f"\n{'='*60}")
print("EXPERIMENT COMPLETED")
print(f"{'='*60}")
print(f"Dispositivo: {device}")
print(f"Runs completed: {NUM_RODADAS}")
print(f"Epochs per run:  {NUM_EPOCHS}")
print(f"Scenarios:           A (POSIX), B (WebDataset), C (Burst Buffer)")
print(f"\nArquivos gerados:")
print(f"  {ARQUIVO_RESULTADOS_CSV}")
print(f"  {ARQUIVO_LATENCIAS_CSV}")
print(f"  {ARQUIVO_RESULTADOS_TXT}")
print(f"{'='*60}")
