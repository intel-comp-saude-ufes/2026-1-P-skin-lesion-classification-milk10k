"""
loss_otimizador.py

Cria a funcao de perda (loss) e o otimizador usados no treino.

Nenhuma mudanca de logica em relacao ao loss_otimizador.py do PAD-UFES-20
- a funcao ja era generica (recebe model e class_weights como parametro).
"""

import torch.nn as nn
import torch.optim as optim


def criar_loss_otimizador(model, class_weights, learning_rate=1e-4):
    """
    criterion: CrossEntropyLoss ponderada pelo peso de cada classe
    (class_weights), pra compensar o desbalanceamento entre as 11 classes
    do MILK10k.

    optimizer: Adam, atualizando todos os parametros do modelo (incluindo
    a nova camada final e o restante da ResNet, ja que nenhuma camada foi
    congelada).
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return criterion, optimizer
