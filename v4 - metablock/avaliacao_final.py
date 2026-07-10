"""
avaliacao_final.py

Recarrega o melhor checkpoint salvo durante o treino (nao o modelo da
ultima epoca, que pode estar mais overfitado) e avalia no conjunto de
TESTE - nunca visto nem no treino, nem na escolha da melhor epoca.
Gera o relatorio por classe (precision/recall/f1), que e o numero que
entra no relatorio final da atividade.

Identico ao avaliacao_final.py do PAD-UFES-20 - nenhuma mudanca
necessaria, ja que a funcao so depende de model/loader/criterion/device
e da lista de classes, nunca do dataset especifico.
"""

import torch
from sklearn.metrics import classification_report, confusion_matrix

from treino import evaluate


def avaliar_no_teste(model, test_loader, criterion, device, caminho_melhor_modelo, classes):
    """
    Recarrega os pesos do melhor checkpoint no modelo (sobrescrevendo os
    pesos atuais, que sao da ultima epoca de treino), avalia no
    test_loader, e imprime o relatorio de classificacao por classe.
    """
    model.load_state_dict(torch.load(caminho_melhor_modelo, map_location=device))  # recarrega os pesos da melhor epoca (nao a ultima)
    model.to(device)

    test_metrics, test_labels, test_preds = evaluate(model, test_loader, criterion, device)

    print(f"loss val: {test_metrics['avg_loss']:.4f} | "
        f"Macro F1: {test_metrics['macro_f1']:.4f} | Acc: {test_metrics['accuracy']:.4f} | BAcc : {test_metrics['balanced_accuracy']:.4f} |"
        f"Precision: {test_metrics['macro_precision']:.4f} | Recall: {test_metrics['macro_recall']:.4f}")

    print("Relatorio por classe (teste):")
    cr = classification_report(test_labels, test_preds, target_names=classes, zero_division=0)
    print(cr)
    cm = confusion_matrix(test_labels, test_preds)

    return test_metrics, cr, cm, test_labels, test_preds
