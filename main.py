"""
main.py (versao multimodal com ResNet50 + selecao de teste)

Treino do classificador MILK10k usando as DUAS imagens de cada lesao
(dermoscopica + clinica) numa unica previsao, com backbone ResNet50
COMPARTILHADO (mesmos pesos processam as duas imagens).

MUDANÇA em relacao a versao anterior: em vez de rodar sempre o
TripleBlock, agora da pra escolher qual dos 3 testes rodar passando um
argumento na linha de comando, o numero de folds, e o backbone:

    python main.py sem_metadado            -> TESTE 1, K_FOLDS=7, resnet50 (padrao)
    python main.py metablock 7             -> TESTE 2, K_FOLDS=7, resnet50
    python main.py tripleblock 7 resnet18  -> TESTE 3, K_FOLDS=7, resnet18

Se nenhum argumento for passado, roda 'tripleblock' com K_FOLDS=7 e
'resnet50' por padrao - ISSO NAO MUDOU, de proposito: a rodada que ja
estava rodando em segundo plano (sem passar backbone) continua usando
resnet50 e salvando nas mesmas pastas de sempre (sem_metadado/,
metablock/, tripleblock/), sem nenhuma quebra.

NOVO: quando backbone != "resnet50", a pasta de saida ganha um prefixo
com o nome do backbone (ex.: resnet18_sem_metadado/), pra nunca
sobrescrever os resultados da rodada de resnet50.
"""

import random
import sys   # NOVO
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

# ----------------------------- 0. QUAL TESTE RODAR -----------------------------
# NOVO: le tipo de modelo, numero de folds e backbone da linha de comando
TIPO_MODELO = sys.argv[1] if len(sys.argv) > 1 else "tripleblock"
K_FOLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 7   # default 7, ver docstring do arquivo
BACKBONE = sys.argv[3] if len(sys.argv) > 3 else "resnet50"   # NOVO: default resnet50, p/ nao mudar a rodada ja em andamento
if TIPO_MODELO not in ("sem_metadado", "metablock", "tripleblock"):
    raise ValueError(f"Tipo de modelo invalido: {TIPO_MODELO!r}. Use 'sem_metadado', 'metablock' ou 'tripleblock'.")
if BACKBONE not in ("resnet18", "resnet50"):
    raise ValueError(f"Backbone invalido: {BACKBONE!r}. Use 'resnet18' ou 'resnet50'.")
print(f"Rodando teste: {TIPO_MODELO} | backbone: {BACKBONE} | K_FOLDS={K_FOLDS}")

# ----------------------------- 1. CARREGAR OS DADOS (1 linha por lesao, 2 imagens cada) -----------------------------

df, metadata_dim = carregar_dados()

# ----------------------------- 2. MAPEAR AS CLASSES PRA INDICES NUMERICOS -----------------------------

df = preparar_rotulos(df)

# ----------------------------- 3. DATALOADERS DE TREINO, VALIDACAO E TESTE -----------------------------
train_df, test_df = separar_rotulos(df)

BATCH_SIZE = 32   # ATENÇÃO: se der CUDA out of memory com a ResNet50, reduza pra 16 ou 8
TOLERANCE = 10
EPOCHS = 50

test_dataset = MILK10kMultimodalDataset(test_df, eval_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

y = train_df['label'].values.astype(np.int64)

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

test_metrics = []

prefixo = "" if BACKBONE == "resnet50" else f"{BACKBONE}_"   # NOVO: so prefixa quando NAO for resnet50 (compatibilidade com a rodada ja rodando)
path = Path(f'{prefixo}{TIPO_MODELO}')   # pasta de saida com o nome do teste (e do backbone, se != resnet50), pra nao sobrescrever resultados
path.mkdir(exist_ok=True)

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):

    print(f"\n--- Fold {fold+1}/{K_FOLDS} ---")
    fold_path = Path(f'{path}/fold{fold+1}')
    fold_path.mkdir(exist_ok=True)

    train_folder_df = train_df.iloc[train_idx]
    val_folder_df   = train_df.iloc[val_idx]

    train_dataset = MILK10kMultimodalDataset(train_folder_df, transform=train_transform)
    val_dataset   = MILK10kMultimodalDataset(val_folder_df, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True)

    y_train_fold = y[train_idx]
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_fold), y=y_train_fold)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = criar_modelo(num_classes=len(CLASSES), metadata_dim=metadata_dim, device=device, tipo=TIPO_MODELO, backbone=BACKBONE)   # NOVO: passa o backbone escolhido
    criterion, optimizer = criar_loss_otimizador(model, class_weights)
    treinar(model, train_loader, val_loader, criterion, optimizer, device, TOLERANCE, EPOCHS, f'{fold_path}/melhor_modelo_f{fold+1}.pt')

    fold_test_metrics, *_ = avaliar_no_teste(model, test_loader, criterion, device,
                                            f'{fold_path}/melhor_modelo_f{fold+1}.pt',
                                            CLASSES)
    test_metrics.append(fold_test_metrics)

df_metrics = pd.DataFrame(test_metrics)
df_metrics.insert(0, 'fold', range(1, len(test_metrics)+1))
df_metrics.to_csv(f'{path}/test_metrics_per_fold.csv', index=False)
