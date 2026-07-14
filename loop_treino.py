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

from treino import train_one_epoch, evaluate


def treinar(model, train_loader, val_loader, criterion, optimizer, device, tolerance, n_epochs, caminho_melhor_modelo, early_stopping_metric='macro_f1'):
    """
    Roda o loop de treino por n_epochs epocas. A cada epoca, treina e
    avalia na validacao; se o Macro F1 da validacao melhorar em relacao
    a melhor epoca anterior, salva os pesos do modelo em
    caminho_melhor_modelo. Devolve o melhor Macro F1 alcancado.
    """
    cont_tolerance = 0
    best_metrics = {}
    best_metrics[early_stopping_metric] = None
    for epoch in range(1, n_epochs + 1):  # repete uma vez por epoca
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)  # treina 1 epoca, pega o erro medio
        val_metrics, _, _= evaluate(model, val_loader, criterion, device)  # avalia na validacao

        print(f"Epoca {epoch:02d} | loss treino: {train_loss:.4f} | loss val: {val_metrics['avg_loss']:.4f} | "
              f"Macro F1: {val_metrics['macro_f1']:.4f} | Acc: {val_metrics['accuracy']:.4f} | BAcc : {val_metrics['balanced_accuracy']:.4f} |"
              f"Precision: {val_metrics['macro_precision']:.4f} | Recall: {val_metrics['macro_recall']:.4f}")

        if best_metrics[early_stopping_metric] == None:
            best_metrics = val_metrics
            torch.save(model.state_dict(), caminho_melhor_modelo)   # salva os pesos desse momento
            print(f"  -> melhor epoca ate agora ({early_stopping_metric} val: {val_metrics[early_stopping_metric]:.4f}) - modelo salvo em {caminho_melhor_modelo}")
        elif val_metrics[early_stopping_metric] > best_metrics[early_stopping_metric]:
            best_metrics = val_metrics
            cont_tolerance = 0                            
            torch.save(model.state_dict(), caminho_melhor_modelo)   # salva os pesos desse momento
            print(f"  -> melhor epoca ate agora ({early_stopping_metric} val: {val_metrics[early_stopping_metric]:.4f}) - modelo salvo em {caminho_melhor_modelo}")
        else:
            cont_tolerance += 1
            if cont_tolerance >= tolerance:
                print(f'Early stopping - Epoca{epoch:02d}')
                break 
    
    print(f"\nTreino finalizado - melhor {early_stopping_metric} (val): {best_metrics[early_stopping_metric]:.4f}")
    return best_metrics
