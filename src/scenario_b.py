#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:35:35 2026

@author: fabiolicht
"""

from torchvision import transforms
from torch.utils.data import DataLoader
import measurement_utils as fm
import webdataset as wds

caminho_scenario_b = '/content/drive/MeuDrive/dataset.tar'

print("Starting Scenario B: WebDataset (sequential read)")
try:
    dataset_b = (
        wds.WebDataset(caminho_scenario_b)
        .shuffle(1000)
        .decode("torchrgb") # Decodifica a imagem para tensor
        .to_tuple("jpg;png", "cls") # Pega a imagem e a classe
        .map_tuple(transforms.Resize((224, 224)), lambda x: x)
    )
    
    dataloader_b = DataLoader(dataset_b, batch_size=32, num_workers=2)
    
    fm.treinar_e_medir(dataloader_b, num_epochs=1)
except Exception as e:
    print(f"Errorr: Check the .tar path. Details: {e}")