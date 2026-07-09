"""
treino.py (versao multimodal)

Funcoes que executam uma epoca de treino e uma avaliacao (sem treinar)
sobre um DataLoader multimodal - a unica mudanca real em relacao a versao
so-dermoscopica e que cada batch agora traz DUAS imagens (imgs_derm,
imgs_clin) em vez de uma, e o forward do modelo recebe as duas.
"""

import torch
from sklearn.metrics import f1_score

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs_derm, imgs_clin, metadata, labels in loader:   # MUDANÇA: agora vem tb o vetor de metadados
        imgs_derm = imgs_derm.to(device)
        imgs_clin = imgs_clin.to(device)
        metadata = metadata.to(device)   # NOVO
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs_derm, imgs_clin, metadata)   # MUDANÇA: passa os metadados tb
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs_derm.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs_derm, imgs_clin, metadata, labels in loader:   # MUDANÇA
            imgs_derm = imgs_derm.to(device)
            imgs_clin = imgs_clin.to(device)
            metadata = metadata.to(device)   # NOVO
            labels = labels.to(device)

            outputs = model(imgs_derm, imgs_clin, metadata)   # MUDANÇA
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs_derm.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = running_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_labels, all_preds