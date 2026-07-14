"""
modelo.py (versao com backbone selecionavel: ResNet18 ou ResNet50 + 3 arquiteturas)

Define 3 modelos diferentes, todos com o MESMO backbone COMPARTILHADO
(mesmos pesos processam a imagem dermoscopica e a clinica) e a mesma
assinatura de forward (img_derm, img_clin, metadata) - assim treino.py
nao precisa saber qual dos 3 esta sendo usado.

  1. MILK10kMultimodalNoMeta -> TESTE 1: so concatena derm+clin, SEM
     usar metadado nenhum (metadado e recebido no forward mas ignorado).
  2. MILK10kMultimodalModel  -> TESTE 2: concatena derm+clin e aplica UM
     MetaBlock sobre o vetor combinado, usando o metadado.
  3. TripleBlock             -> TESTE 3: aplica um MetaBlock em CADA
     ramo (derm filtrado pelo clin e vice-versa) e depois mais um
     MetaBlock final sobre o vetor combinado, usando o metadado.

NOVO: 'backbone' agora e parametro ("resnet18" ou "resnet50"), com
"resnet50" como padrao - pra manter identico o comportamento da rodada
que ja estava rodando em segundo plano antes dessa mudanca. Como
self.num_features e lido dinamicamente de backbone.fc.in_features
(512 na resnet18, 2048 na resnet50), o MetaBlock e o classifier se
ajustam sozinhos - nenhum numero fixo precisa mudar entre os dois.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights   # NOVO: resnet18 tambem


def _criar_backbone(nome: str):
    """
    Devolve (backbone_sem_camada_final, num_features) pro nome pedido.
    "resnet18" -> num_features=512 | "resnet50" -> num_features=2048
    """
    if nome == "resnet18":
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif nome == "resnet50":
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"backbone desconhecido: {nome!r} (use 'resnet18' ou 'resnet50')")

    num_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, num_features



class MetaBlock(nn.Module):
    """
    MetaBlock (Pacheco & Krohling): usa o vetor de metadados (M) pra gerar
    um "portao" que realça ou suprime canais do vetor de features de
    imagem (V), em vez de so concatenar os dois.

    Bridging p/ C/Java: e como uma mascara de floats (0 a 1-ish), calculada
    a partir dos metadados e da propria imagem, multiplicada elemento-a-
    elemento pelo vetor de imagem - um "AND fuzzy" que pesa cada posicao.
    """
    def __init__(self, feature_dim, metadata_dim):
        super().__init__()
        self.fb = nn.Linear(feature_dim, feature_dim)    # transforma o proprio vetor de imagem
        self.gb = nn.Linear(metadata_dim, feature_dim)   # projeta os metadados pro mesmo tamanho do vetor de imagem

    def forward(self, v, m):
        t1 = torch.sigmoid(self.fb(v))   # "portao" (0 a 1)
        t2 = torch.tanh(self.gb(m))      # influencia dos metadados (-1 a 1)
        return v * (t1 * t2)             # aplica o portao sobre o vetor de imagem original


class MILK10kMultimodalNoMeta(nn.Module):   # TESTE 1 — 2 imagens, SEM metadado
    """
    So concatena as features de derm+clin e manda direto pro classificador.
    'metadata' fica na assinatura do forward so pra bater com a chamada
    padrao model(imgs_derm, imgs_clin, metadata) que treino.py sempre faz,
    mas e completamente ignorado aqui dentro.
    """
    def __init__(self, num_classes: int, metadata_dim: int = None, backbone: str = "resnet50"):   # metadata_dim aceito e ignorado, so p/ manter a mesma assinatura de criar_modelo
        super().__init__()

        self.backbone, self.num_features = _criar_backbone(backbone)   # NOVO: backbone escolhido dinamicamente

        combined_dim = self.num_features * 2   # 4096 (resnet50) ou 1024 (resnet18)
        self.classifier = nn.Linear(combined_dim, num_classes)

    def forward(self, img_derm, img_clin, metadata):   # 'metadata' recebido, NAO USADO
        feat_derm = self.backbone(img_derm)
        feat_clin = self.backbone(img_clin)

        combined = torch.cat([feat_derm, feat_clin], dim=1)
        return self.classifier(combined)


class MILK10kMultimodalModel(nn.Module):   # TESTE 2 — 2 imagens + metadado, MetaBlock unico
    def __init__(self, num_classes: int, metadata_dim: int, backbone: str = "resnet50"):
        super().__init__()

        self.backbone, self.num_features = _criar_backbone(backbone)   # NOVO

        combined_dim = self.num_features * 2   # 4096 (resnet50) ou 1024 (resnet18)

        self.metablock = MetaBlock(feature_dim=combined_dim, metadata_dim=metadata_dim)
        self.classifier = nn.Linear(combined_dim, num_classes)

    def forward(self, img_derm, img_clin, metadata):
        feat_derm = self.backbone(img_derm)
        feat_clin = self.backbone(img_clin)

        combined = torch.cat([feat_derm, feat_clin], dim=1)
        combined = self.metablock(combined, metadata)   # metadados "filtram" as features de imagem

        return self.classifier(combined)


class TripleBlock(nn.Module):   # TESTE 3 — 2 imagens + metadado, MetaBlock em cada ramo + MetaBlock final
    def __init__(self, num_classes: int, metadata_dim: int, backbone: str = "resnet50"):
        super().__init__()

        self.backbone, self.num_features = _criar_backbone(backbone)   # NOVO

        self.metablock_derm = MetaBlock(feature_dim=self.num_features, metadata_dim=self.num_features)
        self.metablock_clin = MetaBlock(feature_dim=self.num_features, metadata_dim=self.num_features)

        combined_dim = self.num_features * 2   # 4096 (resnet50) ou 1024 (resnet18)
        self.metablock_metadata = MetaBlock(feature_dim=combined_dim, metadata_dim=metadata_dim)
        self.classifier = nn.Linear(combined_dim, num_classes)

    def forward(self, img_derm, img_clin, metadata):
        feat_derm = self.backbone(img_derm)
        feat_clin = self.backbone(img_clin)

        feat_metablock_derm = self.metablock_derm(feat_derm, feat_clin)
        feat_metablock_clin = self.metablock_clin(feat_clin, feat_derm)

        combined = torch.cat([feat_metablock_derm, feat_metablock_clin], dim=1)
        combined = self.metablock_metadata(combined, metadata)   # metadados "filtram" as features de imagem

        return self.classifier(combined)


def criar_modelo(num_classes: int, metadata_dim: int, device, tipo: str = "tripleblock", backbone: str = "resnet50") -> nn.Module:   # NOVO: parametro 'backbone'
    """
    tipo:
      "sem_metadado" -> Teste 1 (so concatenacao derm+clin, sem metadado)
      "metablock"    -> Teste 2 (MetaBlock unico sobre derm+clin combinados)
      "tripleblock"  -> Teste 3 (MetaBlock em cada ramo + MetaBlock final com metadado)
    backbone:
      "resnet50" (padrao) ou "resnet18"
    """
    if tipo == "sem_metadado":
        model = MILK10kMultimodalNoMeta(num_classes, metadata_dim, backbone=backbone)
    elif tipo == "metablock":
        model = MILK10kMultimodalModel(num_classes, metadata_dim, backbone=backbone)
    elif tipo == "tripleblock":
        model = TripleBlock(num_classes, metadata_dim, backbone=backbone)
    else:
        raise ValueError(f"tipo de modelo desconhecido: {tipo!r} (use 'sem_metadado', 'metablock' ou 'tripleblock')")

    return model.to(device)
