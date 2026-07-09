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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from carregar_dados import carregar_dados
from preparar_rotulos import preparar_rotulos, separar_rotulos, CLASSES
from transformacoes import train_transform, eval_transform
from pesos_classe import calcular_class_weight
from modelo import criar_modelo
from loss_otimizador import criar_loss_otimizador
from loop_treino import treinar
from avaliacao_final import avaliar_no_teste
from dataset import MILK10kMultimodalDataset
from transformacoes import eval_transform, train_transform

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
train_df, test_df = separar_rotulos(df)

BATCH_SIZE = 32
K_FOLDS = 2
TOLERANCE = 10
EPOCHS = 1

y = train_df['label'].values.astype(np.int64)

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):

    print(f"\n--- Fold {fold+1}/{K_FOLDS} ---")
    fold_path = Path(f'fold{fold+1}')
    fold_path.mkdir(exist_ok=True)

    train_folder_df = train_df.iloc[train_idx]   
    val_folder_df   = train_df.iloc[val_idx]

    train_dataset = MILK10kMultimodalDataset(train_folder_df, transform=train_transform)
    val_dataset = MILK10kMultimodalDataset(val_folder_df, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True)

    y_train_fold = y[train_idx]
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = criar_modelo(num_classes=len(CLASSES), metadata_dim=metadata_dim, device=device)
    criterion, optimizer = criar_loss_otimizador(model, class_weights)
    best_metrics = treinar(model, train_loader, val_loader, criterion, optimizer, device, TOLERANCE, EPOCHS, f'{fold_path}/melhor_modelo_f{fold+1}.pt')

    pd.DataFrame(best_metrics, index=[0]).to_csv(f"{fold_path}/best_metrics.csv", index=False)
