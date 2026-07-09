"""
treino.py (versao multimodal)

Funcoes que executam uma epoca de treino e uma avaliacao (sem treinar)
sobre um DataLoader multimodal - a unica mudanca real em relacao a versao
so-dermoscopica e que cada batch agora traz DUAS imagens (imgs_derm,
imgs_clin) em vez de uma, e o forward do modelo recebe as duas.
"""

import torch
from sklearn.metrics import f1_score


def train_one_epoch(model, loader, criterion, optimizer, device):  # treina o modelo por 1 epoca completa
    model.train()                                                    # ativa modo de treino (dropout/batchnorm ativos)
    running_loss = 0.0                                                # acumulador do erro total da epoca
    for imgs_derm, imgs_clin, labels in loader:                       # percorre todos os batches (agora com 2 imagens cada)
        imgs_derm = imgs_derm.to(device)                              # move a dermoscopica pra GPU/CPU
        imgs_clin = imgs_clin.to(device)                              # move a clinica pra GPU/CPU
        labels = labels.to(device)                                    # move os rotulos pra GPU/CPU

        optimizer.zero_grad()                                         # zera os gradientes acumulados do batch anterior
        outputs = model(imgs_derm, imgs_clin)                         # forward pass - modelo recebe as DUAS imagens
        loss = criterion(outputs, labels)                             # calcula o erro entre previsao e rotulo real
        loss.backward()                                               # calcula os gradientes (backpropagation)
        optimizer.step()                                              # atualiza os pesos da rede (backbone compartilhado + classifier)
        running_loss += loss.item() * imgs_derm.size(0)                # soma o erro do batch (ponderado pelo tamanho dele)
    return running_loss / len(loader.dataset)                         # devolve a media do erro na epoca inteira


def evaluate(model, loader, criterion, device):                       # avalia o modelo (sem treinar)
    model.eval()                                                       # ativa modo de avaliacao (dropout/batchnorm desativados)
    running_loss = 0.0                                                 # acumulador do erro total
    all_preds, all_labels = [], []                                     # listas pra guardar todas as previsoes e rotulos reais
    with torch.no_grad():                                              # desliga calculo de gradiente (economiza memoria/tempo)
        for imgs_derm, imgs_clin, labels in loader:                    # percorre todos os batches
            imgs_derm = imgs_derm.to(device)
            imgs_clin = imgs_clin.to(device)
            labels = labels.to(device)

            outputs = model(imgs_derm, imgs_clin)                      # forward pass - modelo recebe as DUAS imagens
            loss = criterion(outputs, labels)                          # calcula o erro entre previsao e rotulo real
            running_loss += loss.item() * imgs_derm.size(0)             # soma o erro do batch (ponderado pelo tamanho dele)
            preds = outputs.argmax(dim=1)                               # pega a classe com maior "pontuacao" prevista, por lesao
            all_preds.extend(preds.cpu().numpy())                       # guarda as previsoes desse batch na lista geral
            all_labels.extend(labels.cpu().numpy())                     # guarda os rotulos reais desse batch na lista geral
    avg_loss = running_loss / len(loader.dataset)                                # erro medio do conjunto
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) # F1 medio entre as classes
    return avg_loss, macro_f1, all_labels, all_preds                             # devolve os resultados
