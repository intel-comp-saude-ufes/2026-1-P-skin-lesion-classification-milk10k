import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class MetaBlock(nn.Module):   # NOVO
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


class MILK10kMultimodalModel(nn.Module):
    def __init__(self, num_classes: int, metadata_dim: int):   # MUDANÇA: recebe tb metadata_dim
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        combined_dim = self.num_features * 2   # 1024 (derm + clin concatenados)

        self.metablock = MetaBlock(feature_dim=combined_dim, metadata_dim=metadata_dim)  # NOVO
        self.classifier = nn.Linear(combined_dim, num_classes)

    def forward(self, img_derm, img_clin, metadata):   # MUDANÇA: agora recebe tb metadata
        feat_derm = self.backbone(img_derm)
        feat_clin = self.backbone(img_clin)

        combined = torch.cat([feat_derm, feat_clin], dim=1)
        combined = self.metablock(combined, metadata)   # NOVO: metadados "filtram" as features de imagem

        return self.classifier(combined)

class TripleBlock(nn.Module):
    def __init__(self, num_classes: int, metadata_dim: int):   # MUDANÇA: recebe tb metadata_dim
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.metablock_derm = MetaBlock(feature_dim=self.num_features, metadata_dim=self.num_features)
        self.metablock_clin = MetaBlock(feature_dim=self.num_features, metadata_dim=self.num_features)

        combined_dim = self.num_features * 2   # 1024 (derm + clin concatenados)
        self.metablock_metadata = MetaBlock(feature_dim=combined_dim, metadata_dim=metadata_dim)  # NOVO
        self.classifier = nn.Linear(combined_dim, num_classes)
    
    def forward(self, img_derm, img_clin, metadata):   # MUDANÇA: agora recebe tb metadata
        feat_derm = self.backbone(img_derm)
        feat_clin = self.backbone(img_clin)

        feat_metablock_derm = self.metablock_derm(feat_derm, feat_clin)
        feat_metablock_clin = self.metablock_clin(feat_clin, feat_derm)

        combined = torch.cat([feat_metablock_derm, feat_metablock_clin], dim=1)
        combined = self.metablock_metadata(combined, metadata)   # NOVO: metadados "filtram" as features de imagem

        return self.classifier(combined)

def criar_modelo(num_classes: int, metadata_dim: int, device) -> nn.Module:   # MUDANÇA
    #model = MILK10kMultimodalModel(num_classes, metadata_dim)
    model = TripleBlock(num_classes, metadata_dim)
    return model.to(device)