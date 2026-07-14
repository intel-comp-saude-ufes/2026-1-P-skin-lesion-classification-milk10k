import torch   # NOVO
from PIL import Image
from torch.utils.data import Dataset


class MILK10kMultimodalDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_derm = Image.open(row["img_path_derm"]).convert("RGB")
        img_clin = Image.open(row["img_path_clin"]).convert("RGB")

        if self.transform:
            img_derm = self.transform(img_derm)
            img_clin = self.transform(img_clin)

        metadata = torch.tensor(row["metadata"], dtype=torch.float)  # NOVO: vetor de metadados da lesao
        label = row["label"]
        return img_derm, img_clin, metadata, label   # MUDANÇA: agora devolve 4 itens (antes eram 3)