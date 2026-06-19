#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:45:20 2026

@author: fabiolicht
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import measurement_utils as fm
caminho_scenario_a = '/content/drive/MeuDrive/dataset_solto'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

print("Starting Scenario A: naive POSIX read from GDrive")
try:
    dataset_a = datasets.ImageFolder(caminho_scenario_a, transform=transform)
    dataloader_a = DataLoader(dataset_a, batch_size=32, shuffle=True, num_workers=2)
    
    fm.treinar_e_medir(dataloader_a, num_epochs=1)
except Exception as e:
    print(f"Errorr: Check the path. Details: {e}")