"""
metadados.py

Prepara os 4 metadados clinicos (age_approx, sex, skin_tone_class, site)
pra virar um vetor numerico fixo, pronto pro MetaBlock.

IMPORTANTE (mesmo principio de vazamento de dados ja usado no resto do
projeto): a media de idade e as CATEGORIAS de sex/site/skin_tone_class
sao calculadas SO com o conjunto de TREINO, e depois aplicadas do mesmo
jeito em validacao e teste - senao, estatisticas do teste vazariam pro
pre-processamento (mesmo tipo de cuidado que ja tomamos ao dividir por
lesion_id, so que agora pra estatisticas, nao pra imagens).
"""

import pandas as pd


def preparar_metadados(df):
    """
    Recebe o df JA COM A COLUNA 'split' preenchida (train/val/test).
    Ajusta idade media e categorias one-hot usando SO df[split == 'train'],
    aplica em todo mundo, e devolve (df com coluna 'metadata', metadata_dim).
    """
    df = df.copy()
    train_mask = df["split"] == "train"

    # ---- idade: preenche NaN com a media do TREINO, depois normaliza pra [0,1] ----
    idade_media_treino = df.loc[train_mask, "age_approx"].mean()
    df["age_norm"] = df["age_approx"].fillna(idade_media_treino) / 100.0

    # ---- categoricas: preenche NaN com "unknown" / -1 antes do one-hot ----
    df["sex"] = df["sex"].fillna("unknown")
    df["site"] = df["site"].fillna("unknown")
    df["skin_tone_class"] = df["skin_tone_class"].fillna(-1).astype(int).astype(str)

    # categorias fixadas pelo TREINO - se val/teste tiver uma categoria nova
    # que o treino nunca viu, ela e descartada (vira tudo 0 nas colunas
    # existentes), pra nao criar uma coluna que o modelo nunca aprendeu a usar
    categorias_sex = sorted(df.loc[train_mask, "sex"].unique())
    categorias_site = sorted(df.loc[train_mask, "site"].unique())
    categorias_tone = sorted(df.loc[train_mask, "skin_tone_class"].unique())

    dummies_sex = pd.get_dummies(df["sex"]).reindex(columns=categorias_sex, fill_value=0).add_prefix("sex_")
    dummies_site = pd.get_dummies(df["site"]).reindex(columns=categorias_site, fill_value=0).add_prefix("site_")
    dummies_tone = pd.get_dummies(df["skin_tone_class"]).reindex(columns=categorias_tone, fill_value=0).add_prefix("tone_")

    metadata_df = pd.concat([df[["age_norm"]], dummies_sex, dummies_site, dummies_tone], axis=1)
    metadata_cols = metadata_df.columns.tolist()
    metadata_dim = len(metadata_cols)

    print(f"Vetor de metadados: {metadata_dim} posicoes -> {metadata_cols}")

    df["metadata"] = metadata_df.astype(float).values.tolist()  # cada lesao guarda seu vetor como lista de floats

    return df, metadata_dim