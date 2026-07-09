"""
pesos_classe.py

Calcula o peso de cada classe, inversamente proporcional a frequencia
dela no conjunto de treino - classes raras do MILK10k (como MAL_OTH, DF,
INF, VASC, BEN_OTH) recebem peso mais alto, pra nao serem "ignoradas"
pelo modelo durante o treino.

Nenhuma mudanca de logica em relacao ao pesos_classe.py do PAD-UFES-20 -
a funcao ja era generica o suficiente (recebe train_df e classes como
parametro), so muda o numero de classes (11 em vez de 7) e a lista em si,
que vem de preparar_rotulos.py.
"""

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


def calcular_class_weight(train_df, classes):
    """
    Calcula o peso de cada classe (na mesma ordem da lista 'classes'),
    usando o metodo 'balanced' do sklearn: peso = n_amostras / (n_classes * contagem_da_classe).
    Devolve um tensor do PyTorch, pronto pra passar em nn.CrossEntropyLoss(weight=...).
    """
    pesos = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(classes)),   # os indices numericos (0 a 10), na mesma ordem de CLASSES
        y=train_df["label"],               # os rotulos numericos de cada imagem do treino
    )

    class_weights = torch.tensor(pesos, dtype=torch.float)

    print("Peso de cada classe (na ordem de CLASSES):")
    for classe, peso in zip(classes, pesos):
        print(f"  {classe}: {peso:.3f}")

    return class_weights
