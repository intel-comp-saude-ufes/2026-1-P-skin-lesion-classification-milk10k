"""
modelo.py (versao multimodal, backbone COMPARTILHADO)

Modelo multimodal: uma UNICA ResNet18 pre-treinada (mesmos pesos, mesmos
filtros) processa a imagem dermoscopica e a clinica, cada uma gerando um
vetor de 512 "features" (o resumo que a rede faz da imagem, logo antes da
antiga camada de classificacao). Os dois vetores sao concatenados
(512 + 512 = 1024) e passam por UMA UNICA camada linear nova, que decide
entre as 11 classes da lesao - uma decisao so, baseada nas duas imagens
juntas.

Bridging p/ C/Java: pense no backbone como uma FUNCAO reaproveitada -
extrair_features(imagem) -> vetor de 512 numeros - chamada duas vezes,
com dois argumentos diferentes (a dermoscopica e a clinica), sempre com a
MESMA implementacao (mesmos pesos). E diferente de ter duas funcoes
separadas (dois backbones, com pesos proprios cada) - aqui e uma so,
reusada.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class MILK10kMultimodalModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # ResNet18 com pesos pre-treinados na ImageNet
        self.num_features = backbone.fc.in_features                  # 512 - tamanho do vetor de features da ResNet18
        backbone.fc = nn.Identity()  # remove a camada de classificacao original - queremos so o vetor de features, nao uma decisao ainda

        self.backbone = backbone  # UMA SO instancia - reaproveitada pras duas imagens (pesos COMPARTILHADOS)

        # camada nova: recebe os 2 vetores de 512 concatenados (1024) e decide entre as 11 classes
        self.classifier = nn.Linear(self.num_features * 2, num_classes)

    def forward(self, img_derm, img_clin):
        feat_derm = self.backbone(img_derm)  # (batch, 512) - passa a dermoscopica pela ResNet18
        feat_clin = self.backbone(img_clin)  # (batch, 512) - passa a clinica pela MESMA ResNet18 (pesos compartilhados)

        combined = torch.cat([feat_derm, feat_clin], dim=1)  # (batch, 1024) - concatena os dois vetores de features

        return self.classifier(combined)  # (batch, 11) - decisao final, baseada nas 2 imagens juntas


def criar_modelo(num_classes: int, device) -> nn.Module:
    """Cria o modelo multimodal e move pro device (GPU/CPU)."""
    model = MILK10kMultimodalModel(num_classes)
    return model.to(device)
