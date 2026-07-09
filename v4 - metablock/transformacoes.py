"""
transformacoes.py

Define as transformacoes (pre-processamento + data augmentation)
aplicadas as imagens antes de entrarem na rede.

Mesma logica usada no projeto PAD-UFES-20 - a nota sobre o Gaussian blur
da classe SEM nao se aplica aqui, ja que o MILK10k nao tem esse problema
de mistura de fontes de imagem (so estamos usando a modalidade
dermoscopica, todas vindas do mesmo tipo de captura). Augmentation leve
no treino, so resize+normalizacao na avaliacao; normalizacao com
media/desvio do ImageNet, porque a ResNet18 foi pre-treinada nela.
"""

import torchvision.transforms as transforms

IMG_SIZE = 224                                    # tamanho fixo (224x224) que a ResNet espera como entrada

imagenet_mean = [0.485, 0.456, 0.406]             # media de cada canal (R, G, B), calculada sobre o dataset ImageNet
imagenet_std = [0.229, 0.224, 0.225]              # desvio-padrao de cada canal, tambem da ImageNet

train_transform = transforms.Compose([            # agrupa varias transformacoes numa unica "receita", aplicadas em sequencia
    transforms.Resize((IMG_SIZE, IMG_SIZE)),      # redimensiona a imagem pra 224x224, independente do tamanho original
    transforms.RandomHorizontalFlip(),            # espelha a imagem horizontalmente, com 50% de chance (augmentation)
    transforms.RandomVerticalFlip(),              # espelha a imagem verticalmente, com 50% de chance (augmentation)
    transforms.RandomRotation(20),                # rotaciona a imagem aleatoriamente entre -20 e +20 graus (augmentation)
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # varia brilho e contraste levemente e aleatoriamente (augmentation)
    transforms.ToTensor(),                        # converte a imagem PIL em tensor do PyTorch, com pixels reescalados de 0-255 para 0-1
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),  # normaliza cada canal usando media/desvio da ImageNet (centraliza os valores em torno de 0)
])

eval_transform = transforms.Compose([             # outra "receita" de transformacoes, so que sem nenhum augmentation
    transforms.Resize((IMG_SIZE, IMG_SIZE)),      # mesmo resize pra 224x224, pra manter compatibilidade com a rede
    transforms.ToTensor(),                        # mesma conversao pra tensor
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),  # mesma normalizacao - precisa ser identica a do treino, senao a rede recebe dados fora do padrao que aprendeu
])
