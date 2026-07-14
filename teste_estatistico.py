"""
teste_estatistico.py

Compara os 3 testes do main.py (sem_metadado / metablock / tripleblock)
usando o mesmo procedimento estatistico que esta dentro do pacote raug
(raug/utils/common.py, funcao statistical_test) - o mesmo raug do Andre
Pacheco, autor do MetaBlock que voce ja usa no projeto.

O QUE O TESTE FAZ:
  1. Teste de Friedman entre os 3 modelos (usa os 5 valores de macro_f1
     por fold de cada um). E o teste "global": existe alguma diferenca
     entre os 3 modelos?
  2. SO SE o p-valor do Friedman for menor que PV_REF, roda o teste de
     Wilcoxon (pareado) entre cada par de modelos, pra ver ESPECIFICA-
     MENTE quais pares diferem.

Por que nao instalar o pacote raug inteiro: pra so importar essa funcao,
o common.py dele carrega opencv, torchvision, matplotlib, tqdm etc (usados
em outras funcoes do mesmo arquivo) - dependencias pesadas desnecessarias.
A logica real do teste sao so ~15 linhas, usando so scipy.stats - entao
reescrevi essa parte aqui como script standalone (funcao
statistical_test() abaixo e a mesma logica, so reescrita sem as
dependencias que o raug carrega mas que a funcao em si nao usa).

ATENCAO - LEIA ANTES DE POR NO RELATORIO (ponto IMPORTANTE):
Com K_FOLDS=5, o Wilcoxon pareado NUNCA consegue dar p < 0.05, mesmo que
um modelo vença o outro nos 5 folds - o p-valor minimo possivel do teste
exato com 5 pares e 0.0625 (verificado rodando scipy.stats.wilcoxon com o
caso mais extremo possivel: 5 diferencas, todas com o mesmo sinal). Com
K_FOLDS=7 (valor atual do main.py) o piso cai pra 0.0078 - ver aviso
automatico no final, calculado pro numero de folds que voce realmente
usou.

NOVO: aceita o nome do backbone como argumento de linha de comando, pra
comparar a rodada certa:
    python teste_estatistico.py            -> compara resnet50 (pastas sem prefixo, igual sempre foi)
    python teste_estatistico.py resnet18   -> compara resnet18 (pastas resnet18_sem_metadado/ etc.)
"""

import sys   # NOVO
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

# ----------------------------- CONFIGURACAO — AJUSTE conforme sua maquina -----------------------------
# NOVO: le o backbone da linha de comando - "resnet50" (padrao, SEM prefixo
# na pasta, igual a rodada original) ou "resnet18" (pastas com prefixo,
# criadas pelo main.py quando voce passa "resnet18" como 3o argumento)
BACKBONE = sys.argv[1] if len(sys.argv) > 1 else "resnet50"
PREFIXO = "" if BACKBONE == "resnet50" else f"{BACKBONE}_"

# Caminho do test_metrics_per_fold.csv gerado por CADA chamada do main.py
ARQUIVOS = {
    "sem_metadado": f"{PREFIXO}sem_metadado/test_metrics_per_fold.csv",
    "metablock":    f"{PREFIXO}metablock/test_metrics_per_fold.csv",
    "tripleblock":  f"{PREFIXO}tripleblock/test_metrics_per_fold.csv",
}
METRICA = "macro_f1"   # coluna a comparar (pode trocar por 'balanced_accuracy', 'accuracy', etc.)
PV_REF = 0.05           # limiar de referencia pro Friedman decidir se faz o Wilcoxon


def carregar_metrica(arquivos, metrica):
    """
    Le os CSVs (um por teste/modelo) e monta a matriz 'modelos x folds'
    que o teste de Friedman/Wilcoxon espera: uma linha por modelo, uma
    coluna por fold - o formato que test_metrics_per_fold.csv ja tem,
    so precisando pegar 1 coluna de cada arquivo.

    IMPORTANTE pra validade do teste pareado: como o main.py usa
    StratifiedKFold(..., random_state=42) fixo nos 3 testes, o fold i de
    um teste usa o MESMO split de treino/validacao que o fold i dos
    outros 2 - por isso da pra tratar os 3 valores do fold i como um
    "trio pareado" (mesma particao de dados, arquitetura diferente).
    """
    nomes = list(arquivos.keys())
    linhas = []
    for nome in nomes:
        df = pd.read_csv(arquivos[nome])
        linhas.append(df[metrica].values)

    n_folds = {nome: len(v) for nome, v in zip(nomes, linhas)}
    if len(set(n_folds.values())) > 1:
        raise ValueError(
            f"Os testes tem numeros diferentes de folds: {n_folds} - "
            f"o teste pareado exige o mesmo numero de folds em todos."
        )

    return np.array(linhas), nomes


def statistical_test(data, names, pv_ref=0.05, verbose=True):
    """
    Mesma logica de raug/utils/common.py::statistical_test (Andre
    Pacheco), reescrita sem as dependencias pesadas que o arquivo
    original carrega (cv2, torchvision, matplotlib, tqdm) mas que essa
    funcao especifica nao usa - so scipy.stats mesmo.

    data (np.array): matriz (n_modelos x n_folds).
    names (list[str]): nome de cada modelo, na mesma ordem das linhas de data.
    pv_ref (float): limiar de referencia do p-valor do Friedman.
    """
    data = np.asarray(data)
    if data.shape[0] != len(names):
        raise ValueError("data.shape[0] tem que ser igual a len(names)")

    linhas_saida = []

    # ---- 1. Teste de Friedman (global, entre todos os modelos de uma vez) ----
    stat_fri, pv_fri = friedmanchisquare(*[data[i, :] for i in range(data.shape[0])])
    linhas_saida.append(f"Friedman: estatistica={stat_fri:.4f} | p-valor={pv_fri:.4f}")

    if pv_fri > pv_ref:
        linhas_saida.append(
            f"  -> p-valor > {pv_ref}: NAO ha evidencia de diferenca significativa entre "
            f"os modelos; nao faz sentido comparar par a par."
        )
    else:
        linhas_saida.append(
            f"  -> p-valor <= {pv_ref}: ha evidencia de diferenca significativa global; "
            f"comparando par a par com Wilcoxon:"
        )
        # ---- 2. So roda o Wilcoxon par a par se o Friedman deu significativo ----
        for i, j in combinations(range(data.shape[0]), 2):
            stat_w, pv_w = wilcoxon(data[i, :], data[j, :])
            marca = "***" if pv_w < pv_ref else "   "
            linhas_saida.append(
                f"  {marca} {names[i]:>13} vs {names[j]:<13}: p-valor={pv_w:.4f}"
            )

    texto = "\n".join(linhas_saida)
    if verbose:
        print(texto)
    return texto


if __name__ == "__main__":
    data, nomes = carregar_metrica(ARQUIVOS, METRICA)
    n_folds = data.shape[1]

    print(f"Comparando '{METRICA}' entre {len(nomes)} modelos, {n_folds} folds cada:\n")
    for nome, valores in zip(nomes, data):
        print(f"  {nome:>13}: media={valores.mean():.4f} | desvio={valores.std(ddof=1):.4f} "
              f"| valores por fold={np.round(valores, 4).tolist()}")
    print()

    statistical_test(data, nomes, pv_ref=PV_REF)

    # ---- aviso automatico sobre o piso de p-valor do Wilcoxon, dado o numero de folds usado ----
    p_minimo = 2 / (2 ** n_folds)
    if p_minimo > PV_REF:
        print(
            f"\nATENCAO: com {n_folds} folds, o p-valor minimo POSSIVEL do Wilcoxon exato "
            f"e {p_minimo:.4f} - ou seja, NENHUMA comparacao par a par pode dar "
            f"p < {PV_REF} aqui, mesmo que um modelo vença o outro em todos os "
            f"{n_folds} folds. Pra essa comparacao par a par ter alguma chance de "
            f"significancia, seria preciso rodar com pelo menos "
            f"{int(np.ceil(np.log2(2 / PV_REF)))} folds."
        )
