"""
dataloaders.py (versao multimodal)

Monta os DataLoaders de treino, validacao e teste a partir da coluna
'split' que ja vem pronta no DataFrame devolvido por carregar_dados.py
(split 70/15/15 estratificado por classe, por LESAO).

Unica mudanca em relacao a versao so-dermoscopica: importa
MILK10kMultimodalDataset em vez de MILK10kDataset. O resto da logica e
identico.
"""

from torch.utils.data import DataLoader

from dataset import MILK10kMultimodalDataset


def criar_dataloaders(df, train_transform, eval_transform, batch_size=32):
    """
    Filtra o df pelas 3 particoes (train/val/test), monta um
    MILK10kMultimodalDataset pra cada uma (treino COM augmentation,
    validacao e teste SEM augmentation) e devolve os tres DataLoaders
    correspondentes, junto com os DataFrames de cada particao.
    """
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"Treino:     {len(train_df)} lesoes")
    print(f"Validacao:  {len(val_df)} lesoes")
    print(f"Teste:      {len(test_df)} lesoes")

    train_dataset = MILK10kMultimodalDataset(train_df, transform=train_transform)  # com augmentation
    val_dataset = MILK10kMultimodalDataset(val_df, transform=eval_transform)       # sem augmentation
    test_dataset = MILK10kMultimodalDataset(test_df, transform=eval_transform)     # sem augmentation

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, train_df, val_df, test_df
