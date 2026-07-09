"""
loop_treino.py

Loop principal de treino: roda N_EPOCHS epocas, chamando train_one_epoch
e evaluate a cada uma, e salva em disco os pesos do modelo sempre que o
Macro F1 na validacao melhora - esse arquivo salvo e o modelo final que a
avaliacao final vai carregar depois.

Identico ao loop_treino.py do PAD-UFES-20 - nenhuma mudanca necessaria,
ja que a funcao so depende de model/loaders/criterion/optimizer/device,
nunca do dataset especifico.
"""

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

from treino import train_one_epoch, evaluate


def treinar(model, train_loader, val_loader, criterion, optimizer, device, n_epochs, caminho_melhor_modelo):
    """
    Roda o loop de treino por n_epochs epocas. A cada epoca, treina e
    avalia na validacao; se o Macro F1 da validacao melhorar em relacao
    a melhor epoca anterior, salva os pesos do modelo em
    caminho_melhor_modelo. Devolve o melhor Macro F1 alcancado.
    """
    best_macro_f1 = 0.0  # melhor Macro F1 visto ate agora

    for epoch in range(1, n_epochs + 1):  # repete uma vez por epoca
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)  # treina 1 epoca, pega o erro medio
        val_loss, val_macro_f1, val_labels, val_preds = evaluate(model, val_loader, criterion, device)  # avalia na validacao

        val_accuracy = accuracy_score(val_labels, val_preds)                                      # acuracia - % de acertos totais
        val_precision = precision_score(val_labels, val_preds, average="macro", zero_division=0)  # precision media entre as classes
        val_recall = recall_score(val_labels, val_preds, average="macro", zero_division=0)         # recall medio entre as classes

        print(f"Epoca {epoch:02d} | loss treino: {train_loss:.4f} | loss val: {val_loss:.4f} | "
              f"Macro F1: {val_macro_f1:.4f} | Acuracia: {val_accuracy:.4f} | "
              f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")

        if val_macro_f1 > best_macro_f1:                            # melhorou em relacao a melhor epoca ate agora
            best_macro_f1 = val_macro_f1                            # atualiza o recorde
            torch.save(model.state_dict(), caminho_melhor_modelo)   # salva os pesos desse momento
            print(f"  -> melhor epoca ate agora (Macro F1 val: {val_macro_f1:.4f}) - modelo salvo em {caminho_melhor_modelo}")

    print(f"\nTreino finalizado - melhor Macro F1 (val): {best_macro_f1:.4f}")
    return best_macro_f1
