"""
preparar_rotulos.py

Define a lista fixa das 11 classes do MILK10k e mapeia a coluna 'classe'
(string, ja derivada do one-hot em carregar_dados.py) pra uma coluna
'label' numerica (0 a 10), formato que o PyTorch espera pra treinar.

Equivalente ao preparar_rotulos.py do projeto PAD-UFES-20, so com as 11
classes do MILK10k no lugar das 6 do outro dataset (e sem a questao da
classe SEM, que era especifica de la).
"""

CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
           "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
class_to_idx = {c: i for i, c in enumerate(CLASSES)}  # cria um dicionario com um numero pra cada rotulo


def preparar_rotulos(df):
    """
    Adiciona a coluna 'label' (inteiro, 0 a 10) ao DataFrame, a partir da
    coluna 'classe' (texto). Devolve o mesmo DataFrame, ja com a coluna
    nova.
    """
    df["label"] = df["classe"].map(class_to_idx)  # add o numero do rotulo no DF

    print(class_to_idx)
    print(df["classe"].value_counts())  # conta quantas vezes cada classe aparece, em ordem decrescente

    return df

def separar_rotulos(df):
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"Treino:     {len(train_df)} lesoes")
    print(f"Teste:      {len(test_df)} lesoes")

    return train_df, test_df

if __name__ == "__main__":
    # permite rodar so esse arquivo isoladamente pra testar
    from carregar_dados import carregar_dados

    df = carregar_dados()
    preparar_rotulos(df)
