import glob
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from avaliacao_final import avaliar_no_teste
from carregar_dados import carregar_dados
from preparar_rotulos import preparar_rotulos, separar_rotulos, CLASSES
from dataset import MILK10kMultimodalDataset
from transformacoes import eval_transform
from modelo import criar_modelo

def carregar_e_avaliar(modelo, test_loader, device, caminho_checkpoint):
    """
    Carrega os pesos de um checkpoint e roda inferência multimodal no test_loader,
    sem depender de criterion/pesos de classe. Retorna y_true, y_pred.
    """
    modelo.load_state_dict(torch.load(caminho_checkpoint, map_location=device))
    modelo.to(device)
    modelo.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs_derm, imgs_clin, metadata, labels in test_loader:
            imgs_derm = imgs_derm.to(device)
            imgs_clin = imgs_clin.to(device)
            metadata = metadata.to(device)

            outputs = modelo(imgs_derm, imgs_clin, metadata)
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    return np.array(y_true), np.array(y_pred)

def gerar_matriz_confusao(y_true, y_pred, classes, titulo="Matriz de Confusão",
                            normalizar=True, cmap='Oranges', salvar_em=None):
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalizar else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=cmap, values_format='.2f' if normalizar else 'd', xticks_rotation=45)
    plt.title(titulo)
    plt.tight_layout()
    if salvar_em:
        fig.savefig(salvar_em, dpi=300, bbox_inches='tight')
    return fig, cm


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    df, metadata_dim = carregar_dados()
    df = preparar_rotulos(df)
    train_df, test_df = separar_rotulos(df)

    test_dataset = MILK10kMultimodalDataset(test_df, eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # ==== 5. Recarrega os modelos salvos de uma pasta ====
    #MUDAR: pasta modelos
    PASTA_MODELOS = "tripleblock"  # <- pasta raiz onde estão as subpastas fold1, fold2...
    padrao_arquivo = os.path.join(PASTA_MODELOS, "fold*", "melhor_modelo_f*.pt")

    checkpoints = sorted(glob.glob(padrao_arquivo))
    print(f"Encontrados {len(checkpoints)} modelos: {checkpoints}")

    resultados = {}

    for caminho in checkpoints:
        nome_modelo = os.path.splitext(os.path.basename(caminho))[0]

        #MUDAR: modelo
        modelo = criar_modelo(num_classes=len(CLASSES), metadata_dim=metadata_dim, device=device)
        y_true, y_pred = carregar_e_avaliar(modelo, test_loader, device, caminho)
        resultados[nome_modelo] = (y_true, y_pred)
        
        y_true_agregado = []
        y_pred_agregado = []

        for nome_modelo, (y_true, y_pred) in resultados.items():
            y_true_agregado.extend(y_true)
            y_pred_agregado.extend(y_pred)

    y_true_agregado = np.array(y_true_agregado)
    y_pred_agregado = np.array(y_pred_agregado)

    fig, cm = gerar_matriz_confusao(
        y_true_agregado, y_pred_agregado, CLASSES,
        titulo="Matriz de Confusão Agregada — Todos os Folds (Teste)",
        salvar_em=os.path.join(PASTA_MODELOS, "matriz_agregada.png")
    )
    plt.show()

    print(f"Matriz agregada gerada com {len(y_true_agregado)} predições no total "
        f"({len(resultados)} folds × {len(test_df)} amostras de teste).")

    print("Matrizes geradas para todos os modelos.")

