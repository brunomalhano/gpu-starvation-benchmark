# -*- coding: utf-8 -*-
"""
Editor Spyder

This is a temporary script file.
"""

import time
import torch
import torchvision.models as models

from google.colab import drive
drive.mount('/content/drive')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

model = models.resnet18().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def treinar_e_medir(dataloader, num_epochs=1):
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

        print(f"--- Results da Epoch {epoch+1} ---")
        print(f"Total Epoch Time: {tempo_total:.2f} s")
        print(f"Throughput: {throughput:.2f} MB/s")
        print(f"Compute time on GPU: {tempo_computacao:.2f} s")
        print(f"Time waiting for data (CPU): {tempo_espera_dados:.2f} s")
        print(f"GPU Starvation Rate: {taxa_inanicao_gpu:.1f}%")
        print("-" * 30)