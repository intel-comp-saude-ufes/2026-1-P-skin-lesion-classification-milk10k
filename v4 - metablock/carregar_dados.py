"""
carregar_dados.py (versao multimodal)

Le o milk10k_merged.csv e monta UMA LINHA POR LESAO (nao mais uma por
imagem) - cada linha traz as duas modalidades da mesma lesao, em colunas
separadas ('img_path_derm' e 'img_path_clin'), pra alimentar um modelo
que usa as duas imagens numa unica previsao.

Diferenca em relacao a versao so-dermoscopica (que voce ja tinha): la, o
DataFrame tinha uma linha por IMAGEM (so dermoscopica). Aqui, filtramos as
duas modalidades separadamente e juntamos (merge) por lesion_id - cada
lesao ja aparece nos dois subconjuntos (conferido: os 5240 lesion_id do
MILK10k tem sempre exatamente 1 imagem dermoscopica + 1 clinica, nenhuma
faltando), entao o merge e 1-pra-1 e nao perde nenhuma lesao.

O rotulo ('classe') e identico nas duas modalidades da mesma lesao (o
diagnostico e da LESAO, nao da imagem) - por isso pegamos ele so da
metade dermoscopica.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from metadados import preparar_metadados  

# ----------------------------- CONFIGURACAO — AJUSTE conforme sua maquina -----------------------------
CSV_PATH = Path("milk10k_merged.csv")
IMAGES_DIR = Path("imagens")   # raiz onde ficam as pastas IMAGES_DIR/{lesion_id}/{isic_id}.jpg

CLASSES_ONE_HOT = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
                    "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]

SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15


def carregar_dados():
    df = pd.read_csv(CSV_PATH)

    derm = df[df["image_type"] == "dermoscopic"].copy()
    clin = df[df["image_type"] == "clinical: close-up"].copy()

    derm["classe"] = derm[CLASSES_ONE_HOT].idxmax(axis=1)

    derm["img_path_derm"] = derm.apply(
        lambda row: str(IMAGES_DIR / row["lesion_id"] / f"{row['isic_id']}.jpg"), axis=1
    )
    clin["img_path_clin"] = clin.apply(
        lambda row: str(IMAGES_DIR / row["lesion_id"] / f"{row['isic_id']}.jpg"), axis=1
    )

    # MUDANÇA: adicionadas as 4 colunas de metadados clinicos (vem so da
    # metade dermoscopica, ja que sao do PACIENTE/LESAO, iguais nas 2 modalidades)
    lesoes = derm[["lesion_id", "classe", "img_path_derm",
                   "age_approx", "sex", "skin_tone_class", "site"]].merge(
        clin[["lesion_id", "img_path_clin"]], on="lesion_id", how="inner"
    )
    print(f"Total de lesoes com as 2 modalidades: {len(lesoes)}")
    print(lesoes["classe"].value_counts())

    existe_derm = lesoes["img_path_derm"].apply(lambda p: Path(p).exists())
    existe_clin = lesoes["img_path_clin"].apply(lambda p: Path(p).exists())
    print(f"\nDermoscopicas encontradas: {existe_derm.sum()} / {len(lesoes)}")
    print(f"Clinicas encontradas:      {existe_clin.sum()} / {len(lesoes)}")
    faltando = ~(existe_derm & existe_clin)
    if faltando.any():
        print("Atencao: algumas lesoes tem pelo menos uma imagem faltando. Exemplos:")
        print(lesoes.loc[faltando, ["lesion_id", "classe", "img_path_derm", "img_path_clin"]].head())

    lesoes = _dividir_treino_val_teste(lesoes)

    # NOVO: so depois do split (pra usar as estatisticas SO do treino)
    lesoes, metadata_dim = preparar_metadados(lesoes)

    return lesoes, metadata_dim   # MUDANÇA: agora devolve tb a dimensao do vetor de metadados


def _dividir_treino_val_teste(df):
    """
    Split 70/15/15 em duas chamadas encadeadas de train_test_split,
    estratificado por 'classe'. Como o DataFrame ja tem uma linha por
    LESAO (nao por imagem), esse split e automaticamente seguro contra
    vazamento - nao ha como a dermoscopica e a clinica da mesma lesao
    caírem em particoes diferentes, porque elas agora sao a MESMA linha.
    """
    rest_size = VAL_SIZE + TEST_SIZE
    train_df, temp_df = train_test_split(
        df, test_size=rest_size, random_state=SEED, stratify=df["classe"]
    )
    relative_test_size = TEST_SIZE / rest_size
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_size, random_state=SEED, stratify=temp_df["classe"]
    )

    df = df.copy()
    df["split"] = None
    df.loc[train_df.index, "split"] = "train"
    df.loc[val_df.index, "split"] = "val"
    df.loc[test_df.index, "split"] = "test"

    print(f"\nTreino:     {(df['split'] == 'train').sum()} lesoes")
    print(f"Validacao:  {(df['split'] == 'val').sum()} lesoes")
    print(f"Teste:      {(df['split'] == 'test').sum()} lesoes")

    return df