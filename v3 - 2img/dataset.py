"""
dataset.py (versao multimodal)

Dataset customizado do PyTorch para o MILK10k multimodal. Abre as DUAS
imagens da mesma lesao (dermoscopica e clinica), aplica as
transformacoes em cada uma (com sorteios de augmentation
INDEPENDENTES entre elas - nao faz sentido forcar o mesmo flip/rotacao
nas duas, ja que sao fotos diferentes da mesma lesao) e devolve a tripla
(imagem dermoscopica, imagem clinica, rotulo).
"""

from PIL import Image
from torch.utils.data import Dataset


class MILK10kMultimodalDataset(Dataset):  # herda de Dataset (o "molde" que o PyTorch exige)
    def __init__(self, df, transform=None):  # construtor: roda uma vez, quando o objeto e criado
        self.df = df.reset_index(drop=True)  # guarda o dataframe recebido, com indice reorganizado do zero
        self.transform = transform            # guarda a transformacao recebida (ou None)

    def __len__(self):              # conecta essa classe a funcao nativa len() do Python
        return len(self.df)         # total de exemplos do dataset (agora, lesoes - nao imagens)

    def __getitem__(self, idx):                                       # chamado pelo DataLoader pra buscar UMA lesao
        row = self.df.iloc[idx]                                        # pega a linha do dataframe na posicao idx
        img_derm = Image.open(row["img_path_derm"]).convert("RGB")      # abre a imagem dermoscopica (garante 3 canais RGB)
        img_clin = Image.open(row["img_path_clin"]).convert("RGB")      # abre a imagem clinica (garante 3 canais RGB)

        if self.transform:                     # se uma transformacao foi passada...
            img_derm = self.transform(img_derm)  # ...aplica ela na dermoscopica (sorteio proprio de augmentation)
            img_clin = self.transform(img_clin)  # ...e aplica de novo na clinica (sorteio INDEPENDENTE - nao e o mesmo flip/rotacao)

        label = row["label"]                    # rotulo numerico (0 a 10), o mesmo pras duas imagens (e da lesao, nao da imagem)
        return img_derm, img_clin, label         # tripla que o DataLoader empacota em batches
