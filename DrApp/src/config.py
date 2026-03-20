# config.py — final correct version

from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image

# ── Module-level constants — accessible as config.BEST_HEAD_PARAMS etc. ──

BEST_HEAD_PARAMS = {
    'resnet50':    {'fc_dim': 128, 'dropout': 0.436595217760556, 'fc_layers': 3},
    'densenet121': {'fc_dim': 128, 'dropout': 0.34287, 'fc_layers': 3},
    'inceptionV3': {'fc_dim': 128, 'dropout': 0.34287, 'fc_layers': 3},
}

GRADE_INFO = {
    0: {'name': 'No DR',            'colour': '#27ae60',
        'urgency': 'Routine',           'action': 'Annual screening recommended'},
    1: {'name': 'Mild DR',          'colour': '#f1c40f',
        'urgency': 'Monitor',           'action': 'Repeat screening in 12 months'},
    2: {'name': 'Moderate DR',      'colour': '#e67e22',
        'urgency': 'Monitor Closely',   'action': 'Ophthalmologist in 3-6 months'},
    3: {'name': 'Severe DR',        'colour': '#e74c3c',
        'urgency': '⚠ Urgent Referral', 'action': 'Ophthalmologist within weeks'},
    4: {'name': 'Proliferative DR', 'colour': '#8e44ad',
        'urgency': '🚨 Immediate',       'action': 'Emergency specialist review'},
}

MODEL_DISPLAY_NAMES = {
    'resnet50_cnn':    'ResNet50 (CNN only)',
    'densenet121_cnn': 'DenseNet121 (CNN only)',
    'inceptionV3_cnn': 'InceptionV3 (CNN only)',
    'resnet50_xgb':    'ResNet50 + XGBoost',
    'densenet121_xgb': 'DenseNet121 + XGBoost',
    'inceptionV3_xgb': 'InceptionV3 + XGBoost',
    'ensemble':        'SMOTE Ensemble (Best)',
}


class Config:
    def __init__(self):
        self.model_dir = Path('savemodels')
        self.num_classes = 5
        self.img_size = (512, 512)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.default_head_params = {'fc_dim': 512, 'dropout': 0.5, 'fc_layers': 2}

    @staticmethod
    def preprocess_image(image_path):
        """Preprocess a single image for inference. Returns tensor of shape (1, 3, 512, 512)."""
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)  # type: ignore # (1, 3, 512, 512)