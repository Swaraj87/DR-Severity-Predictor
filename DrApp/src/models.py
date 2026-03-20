# models.py — final correct version

import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss    = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt         = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DRModelManager:
    """Manages loading and inference for a single CNN backbone."""

    def __init__(self, config, model_name, tuning_params=None):
        self.config          = config
        self.model_name      = model_name
        self.device          = config.device
        self.params          = tuning_params if tuning_params else config.default_head_params
        self.model           = None
        self.feature_extractor = None
        self._initialize_model_finetune()

    def _build_dynamic_head(self, in_features):
        layers = []
        layers.append(nn.Dropout(self.params['dropout']))
        layers.append(nn.Linear(in_features, self.params['fc_dim']))
        layers.append(nn.BatchNorm1d(self.params['fc_dim']))
        layers.append(nn.LeakyReLU(inplace=True))

        if self.params.get('fc_layers', 2) == 2:
            layers.append(nn.Dropout(self.params['dropout'] * 0.5))
            layers.append(nn.Linear(self.params['fc_dim'], self.params['fc_dim'] // 2))
            layers.append(nn.BatchNorm1d(self.params['fc_dim'] // 2))
            layers.append(nn.LeakyReLU(inplace=True))
            last_dim = self.params['fc_dim'] // 2
        else:
            last_dim = self.params['fc_dim']

        layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(last_dim, self.config.num_classes))
        return nn.Sequential(*layers)

    def _initialize_model_finetune(self):
        if self.model_name == 'resnet50':
            self._initialize_resnet50_finetune()
        elif self.model_name == 'inceptionV3':
            self._initialize_inception_v3_finetune()
        elif self.model_name == 'densenet121':
            self._initialize_densenet121_finetune()
        else:
            raise ValueError(f"Model {self.model_name} not supported")
        self.model.to(self.device) # type: ignore
        self._feature_extractor()

    def _initialize_resnet50_finetune(self):
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        for module in self.model.layer4.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()
                for param in module.parameters():
                    param.requires_grad = True
        num_feature  = self.model.fc.in_features
        self.model.fc = self._build_dynamic_head(num_feature) # type: ignore
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def _initialize_inception_v3_finetune(self):
        self.model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True
        )
        for param in self.model.parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if 'Mixed_7' in name or 'Mixed_6e' in name or 'Mixed_6d' in name:
                param.requires_grad = True
            if ('bn' in name or 'BatchNorm' in name) and ('Mixed_7' in name or 'Mixed_6' in name):
                param.requires_grad = True
        if self.model.AuxLogits is not None:
            num_aux = self.model.AuxLogits.fc.in_features # type: ignore
            self.model.AuxLogits.fc = nn.Linear(num_aux, self.config.num_classes) # type: ignore
            for param in self.model.AuxLogits.fc.parameters():
                param.requires_grad = True
        num_features  = self.model.fc.in_features
        self.model.fc = self._build_dynamic_head(num_features) # type: ignore
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def _initialize_densenet121_finetune(self):
        self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if 'denseblock4' in name or 'norm5' in name or 'transition3' in name:
                param.requires_grad = True
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if 'denseblock4' in name or 'norm5' in name:
                    module.train()
                    for param in module.parameters():
                        param.requires_grad = True
        num_features         = self.model.classifier.in_features
        self.model.classifier = self._build_dynamic_head(num_features) # type: ignore
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def _feature_extractor(self):
        if self.model is None:
            raise ValueError(f"Model is None for {self.model_name}")

        if self.model_name == 'resnet50':
            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])

        elif self.model_name == 'inceptionV3':
            self.feature_extractor            = copy.deepcopy(self.model)
            self.feature_extractor.aux_logits = False # type: ignore
            self.feature_extractor.fc         = nn.Identity() # type: ignore

        elif self.model_name == 'densenet121':
            self.feature_extractor = nn.Sequential(
                self.model.features, # type: ignore
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        self.feature_extractor.to(self.device) # type: ignore
        self.feature_extractor.eval() # type: ignore

    def get_model(self):
        return self.model

    def get_feature_extractor(self):
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not initialized.")
        return self.feature_extractor

    def load_model(self, path):
        """Load saved weights into the already-initialised architecture."""
        if self.model is None:
            self._initialize_model_finetune()

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict']) # pyright: ignore[reportOptionalMemberAccess]

        if 'trainable_layers' in checkpoint:
            for param in self.model.parameters(): # type: ignore
                param.requires_grad = False
            for name, param in self.model.named_parameters(): # type: ignore
                if name in checkpoint['trainable_layers']:
                    param.requires_grad = True

        self._feature_extractor()
        self.model.eval() # type: ignore
        print(f"✓ {self.model_name} loaded from {path}")
        return self.model

        