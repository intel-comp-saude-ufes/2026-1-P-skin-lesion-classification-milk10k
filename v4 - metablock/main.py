"""
main.py (versao multimodal)

Treino do classificador MILK10k usando as DUAS imagens de cada lesao
(dermoscopica + clinica) numa unica previsao, com backbone ResNet18
COMPARTILHADO (mesmos pesos processam as duas imagens) e fusao por
concatenacao das features antes da camada final.

Adaptado da versao so-dermoscopica: mudam carregar_dados.py (1 linha por
lesao, com as 2 colunas de imagem), dataset.py (devolve as 2 imagens),
modelo.py (arquitetura de 2 entradas) e treino.py (forward com 2
imagens). pesos_classe/loss_otimizador/loop_treino/avaliacao_final nao
precisaram mudar - so dependem de train_df/model/device, nunca da
arquitetura interna do modelo.
"""

import random

import numpy as np
import torch

from carregar_dados import carregar_dados
from preparar_rotulos import preparar_rotulos, CLASSES
from transformacoes import train_transform, eval_transform
from dataloaders import criar_dataloaders
from pesos_classe import calcular_class_weight
from modelo import criar_modelo
from loss_otimizador import criar_loss_otimizador
from loop_treino import treinar
from avaliacao_final import avaliar_no_teste

# ----------------------------- SEMENTES / REPRODUTIBILIDADE -----------------------------

SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ----------------------------- 1. CARREGAR OS DADOS (1 linha por lesao, 2 imagens cada) -----------------------------

df, metadata_dim = carregar_dados()

# ----------------------------- 2. MAPEAR AS CLASSES PRA INDICES NUMERICOS -----------------------------

df = preparar_rotulos(df)

# ----------------------------- 3. DATALOADERS DE TREINO, VALIDACAO E TESTE -----------------------------

BATCH_SIZE = 32

train_loader, val_loader, test_loader, train_df, val_df, test_df = criar_dataloaders(
    df, train_transform, eval_transform, batch_size=BATCH_SIZE
)

# ----------------------------- 4. PESO DE CADA CLASSE (DESBALANCEAMENTO) -----------------------------

class_weights = calcular_class_weight(train_df, CLASSES)
class_weights = class_weights.to(device)

# ----------------------------- 5. MODELO (RESNET18 COMPARTILHADO + FUSAO POR CONCATENACAO + METABLOCK) -----------------------------

# re-fixa a semente logo antes de criar o modelo, pra garantir que a camada
# final (classifier) nasca sempre com os mesmos pesos, independente de
# quantos numeros aleatorios ja foram consumidos ate aqui (shuffle do
# DataLoader, augmentation, etc.) - deixa o script reprodutivel de
# execucao pra execucao.
torch.manual_seed(SEED)
model = criar_modelo(num_classes=len(CLASSES), metadata_dim=metadata_dim, device=device)
print(model)

# ----------------------------- 6. LOSS E OTIMIZADOR -----------------------------

LEARNING_RATE = 1e-4

criterion, optimizer = criar_loss_otimizador(model, class_weights, learning_rate=LEARNING_RATE)
print(criterion)
print(optimizer)

# ----------------------------- 7. LOOP DE TREINO -----------------------------

N_EPOCHS = 10
MELHOR_MODELO_PATH = "melhor_modelo_milk10k_multimodal.pt"

best_macro_f1 = treinar(
    model, train_loader, val_loader, criterion, optimizer, device,
    n_epochs=N_EPOCHS, caminho_melhor_modelo=MELHOR_MODELO_PATH
)

# ----------------------------- 8. AVALIACAO FINAL NO TESTE -----------------------------

test_macro_f1, test_labels, test_preds = avaliar_no_teste(
    model, test_loader, criterion, device, MELHOR_MODELO_PATH, CLASSES
)