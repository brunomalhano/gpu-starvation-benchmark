#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:39:19 2026

@author: fabiolicht
"""

import shutil
from concurrent.futures import ThreadPoolExecutor
from torchvision import datasets, transform
from torch.utils.data import DataLoader
import measurement_utils as fm
import os
import time

caminho_origem_gdrive = '/content/drive/MeuDrive/dataset_solto'
caminho_destino_ssd_local = '/content/dataset_local'

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

print("Starting Scenario C: Burst Buffer (SSD Local)")
try:
    stage_data_parallel(caminho_origem_gdrive, caminho_destino_ssd_local)
    
    dataset_c = datasets.ImageFolder(caminho_destino_ssd_local, transform=transform)
    dataloader_c = DataLoader(dataset_c, batch_size=32, shuffle=True, num_workers=4) # No SSD local, testar com mais workers
    
    fm.treinar_e_medir(dataloader_c, num_epochs=1)
except Exception as e:
    print(f"Error no Scenario C: {e}")