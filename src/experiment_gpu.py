import time
import torch
import torchvision.models as models
import os
import csv
import socket
import tarfile
from pathlib import Path
from datetime import datetime
import sys

def _silenciar_keyboard_interrupt(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        print("\nInterrupted by user.")
        return
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _silenciar_keyboard_interrupt

BASE_DIR = Path(__file__).resolve().parent / "output"
BASE_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR_DATASET = Path(r"G:\Meu Drive\VSC\PPD\dataset")
CAMINHO_DATASET_LOCAL = BASE_DIR_DATASET
CAMINHO_TAR_DATASET = BASE_DIR_DATASET / "dataset_solto.tar"
NUM_WORKERS = 0

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

HOSTNAME = socket.gethostname()
NOME_DISPOSITIVO_ARQUIVO = "gpu" if device.type == "cuda" else "cpu"
TIMESTAMP_EXECUCAO = datetime.now().strftime("%Y%m%d_%H%M%S")
NOME_BASE_RESULTADOS = f"training_results_{NOME_DISPOSITIVO_ARQUIVO}"
ARQUIVO_RESULTADOS_TXT = BASE_DIR / f"{NOME_BASE_RESULTADOS}.txt"
ARQUIVO_RESULTADOS_CSV = BASE_DIR / f"{NOME_BASE_RESULTADOS}.csv"
print(f"Results will be saved at: {NOME_BASE_RESULTADOS}.[txt|csv]")

preparar_dataset_local()

model = models.resnet18().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def salvar_resultados(scenario, epoch, tempo_total, throughput, tempo_computacao, tempo_espera_dados, taxa_inanicao_gpu):
    if device.type == "cuda":
        nome_dispositivo = "GPU"
        nome_espera = "CPU"
        linha_taxa = f"GPU Starvation Rate: {taxa_inanicao_gpu:.1f}%"
    else:
        nome_dispositivo = "CPU"
        nome_espera = "I/O"
        linha_taxa = f"I/O wait rate: {taxa_inanicao_gpu:.1f}%"
    linhas = [
        f"--- Results da Epoch {epoch} ({scenario}) ---",
        f"Total Epoch Time: {tempo_total:.2f} s",
        f"Throughput: {throughput:.2f} MB/s",
        f"Compute time on {nome_dispositivo}: {tempo_computacao:.2f} s",
        f"Time waiting for data ({nome_espera}): {tempo_espera_dados:.2f} s",
        linha_taxa,
        "-" * 30,
    ]

    with ARQUIVO_RESULTADOS_TXT.open("a", encoding="utf-8") as arquivo_txt:
        arquivo_txt.write("\n".join(linhas) + "\n")

    csv_existe = ARQUIVO_RESULTADOS_CSV.exists()
    with ARQUIVO_RESULTADOS_CSV.open("a", newline="", encoding="utf-8") as arquivo_csv:
        colunas = [
            "timestamp",
            "scenario",
            "epoch",
            "total_time_s",
            "throughput_mb_s",
            "gpu_compute_s",
            "data_wait_s",
            "gpu_starvation_percent",
        ]
        writer = csv.DictWriter(arquivo_csv, fieldnames=colunas)
        if not csv_existe:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "scenario": scenario,
            "epoch": epoch,
            "total_time_s": f"{tempo_total:.2f}",
            "throughput_mb_s": f"{throughput:.2f}",
            "gpu_compute_s": f"{tempo_computacao:.2f}",
            "data_wait_s": f"{tempo_espera_dados:.2f}",
            "gpu_starvation_percent": f"{taxa_inanicao_gpu:.1f}",
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

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            tempo_computacao += (time.time() - t1)
            t0 = time.time() # Reset timer for the next iteration

        tempo_total = time.time() - inicio_epoch
        mb_processados = bytes_processados / (1024 * 1024)
        throughput = mb_processados / tempo_total
        taxa_inanicao_gpu = (tempo_espera_dados / tempo_total) * 100

        salvar_resultados(
            scenario,
            epoch + 1,
            tempo_total,
            throughput,
            tempo_computacao,
            tempo_espera_dados,
            taxa_inanicao_gpu,
        )


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
caminho_scenario_a = str(CAMINHO_DATASET_LOCAL)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

print("Starting Scenario A: naive POSIX read from GDrive")
try:
    if not any(p.is_dir() for p in CAMINHO_DATASET_LOCAL.iterdir()):
        raise FileNotFoundError(
            f"Nenhuma subpasta de classe encontrada em {CAMINHO_DATASET_LOCAL}"
        )

    dataset_a = datasets.ImageFolder(caminho_scenario_a, transform=transform)
    dataloader_a = DataLoader(dataset_a, batch_size=32, shuffle=True, num_workers=NUM_WORKERS)

    treinar_e_medir(dataloader_a, num_epochs=1, scenario="Scenario A")
except Exception as e:
    print(f"Errorr: Check the path. Details: {e}")



from torchvision import transforms
from torch.utils.data import DataLoader
import webdataset as wds

caminho_scenario_b = "file:" + str(CAMINHO_TAR_DATASET).replace("\\", "/")
print("Starting Scenario B: WebDataset (sequential read)")
try:
    if not CAMINHO_TAR_DATASET.is_file():
        raise FileNotFoundError(f"File .tar not found: {CAMINHO_TAR_DATASET}")

    dataset_b = (
        wds.WebDataset(caminho_scenario_b, empty_check=False, shardshuffle=False)
        .shuffle(1000)
        .decode("torchrgb")
        .to_tuple("jpg;png")
        .map_tuple(transforms.Resize((224, 224)))
        .map(lambda x: (x[0], 0))
    )

    dataloader_b = DataLoader(dataset_b, batch_size=32, num_workers=NUM_WORKERS)
    
    treinar_e_medir(dataloader_b, num_epochs=1, scenario="Scenario B")
except Exception as e:
    print(f"Errorr: Check the .tar path. Details: {e}")


import shutil
import time
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader # FIX HERE

caminho_origem_gdrive = str(CAMINHO_DATASET_LOCAL)
caminho_destino_ssd_local = str(BASE_DIR / "_dataset_local")

class DatasetImagensSoltas(Dataset):
    def __init__(self, diretorio, transform=None):
        self.diretorio = diretorio
        self.transform = transform
        self.arquivos = []
        
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
            
        return imagem, 0 # Returns dummy class 0

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
meu_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

print("Starting Scenario C: Burst Buffer (SSD Local)")
try:
    if not any(p.is_file() for p in CAMINHO_DATASET_LOCAL.rglob("*")):
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {CAMINHO_DATASET_LOCAL}")

    stage_data_parallel(caminho_origem_gdrive, caminho_destino_ssd_local)
    
    dataset_c = DatasetImagensSoltas(caminho_destino_ssd_local, transform=meu_transform)
    
    dataloader_c = DataLoader(dataset_c, batch_size=32, shuffle=True, num_workers=NUM_WORKERS)
    
    print(f"Sucesso! Encontradas {len(dataset_c)} images no SSD local.")
    treinar_e_medir(dataloader_c, num_epochs=1, scenario="Scenario C")

except Exception as e:
    print(f"Error no Scenario C: {e}")
