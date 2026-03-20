""" 
src/inference.py 

Loads all 7 models once and exposes a single predict(pil_image) 
mthod that returns predictions from every model.

"""

import pickle
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F 
from torchvision import transforms
from PIL import Image

from src.config import Config, BEST_HEAD_PARAMS
from src.models import DRModelManager

#Preporcessing transfrom:

def get_inference_transfrom(img_size=(512, 512)):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])

class Inference:
    """
    Loads all 7 trained models and provides single-image prediction.
 
    Models loaded:
      CNN-only  (3): resnet50, densenet121, inceptionV3 — direct FC head output
      CNN+XGB   (3): each CNN's features fed to its trained XGBoost
      Ensemble (1): all 3 CNN features -> PCA -> SMOTE XGBoost
    """

    def __init__(self, model_dir: str='savemodels'):
        self.config = Config()
        self.device = self.config.device
        self.model_dir = model_dir
        self.transfrom = get_inference_transfrom(self.config.img_size)

        print(f"Loading models on: {self.device}")

        # Loading the 3 CNN manages
        self.manager = {}
        for model_name in ['resnet50', 'inceptionV3', "densenet121"]:
            ckpt = self.model_dir / f"{model_name}_finetune_best.pth" # type: ignore

            if not ckpt.exists():
                print(f" Checkpoint not found :{ckpt} - skipping {model_name}")
                continue

            manager = DRModelManager(self.config,
                                     model_name,
                                     tuning_params= BEST_HEAD_PARAMS)
            manager.load_model(ckpt)
            manager.model.eval() # type: ignore
            self.manager[model_name] = manager
            print(f" {model_name} laoded")

        #Loading the individual XGBoost with CNNs
        self.xgb = {}
        for model_name in self.manager:
            xgb_path = self.model_dir / f"{model_name}_xgb.pkl" # type: ignore
            if xgb_path.exists():
                with open(xgb_path, 'rb') as f:
                    self.xgb_models[model_name] = pickle.load(f) # type: ignore
                print(f"{model_name}_Xgb loaded")

        #Loading the SOMETE Ensemble model + its PCA:
        self.ensemble_model = None
        self.ensemble_pca = None
        ens_path = self.model_dir/"ensemble_xgb.pkl" # type: ignore
        pca_path = self.model_dir/"ensemble_pca.pkl" # pyright: ignore[reportOperatorIssue]

        if ens_path.exists() and pca_path.eixsts():
            with open(ens_path, 'rb') as f:
                self.ensemble_model = pickle.load(f)
            
            with open(pca_path, 'rb') as f:
                self.ensemble_pca = pickle.load(f)
            print("Ensemble SMOTE + PCA LOADED.")
        print("ALL MODELS LOADED.")

    def _preprocess(self, pil_images: Image.Image) -> torch.Tensor:
        "Converts a PIL images to a (1,3,512,512) tensor."
        img = pil_images.convert("RGB")
        tensor = self.transfrom(img)
        return tensor.unsqueeze(0).to(self.device)
    
    def _cnn_predict(self, model_name:str, tensor: torch.Tensor) -> dict:
        """
        Runs the CNN's FC head directly on the input tensor.
        Returns grade, confidence, and full probability array.
        InceptionV3 returns (logits, aux) during training but only logits in eval.
        """
        model = self.manager[model_name].get_model()
        model.eval()
        with torch.no_grad():
            output = model(tensor)
            #for InceptionV3 in eval model returns a plain tensor, not a tuple
            if isinstance(output, tuple):
                output = output[0]
            probs = F.softmax(output, dim=1).cpu().numpy()[0]

        grade = int(probs.argmax())
        confidence = float(probs[grade])
        return {"grade": grade, "confidence": confidence, "probs": probs.tolist()}
        
    def _extract_features(self, model_name: str, tensor: torch.Tensor) -> np.ndarray:

        """
        Passes the image through the backbone (no FC head).
        Returns a (feature_dim,) numpy array — 2048 for ResNet/Inception, 1024 for DenseNet.

        """
        extractor = self.manager[model_name].get_feature_extractor()
        extractor.eval()
        with torch.no_grad():
            feats = extractor(tensor)
            feats = feats.view(feats.size(0), -1)
        return feats.cpu().numpy()
    
    def _xgb_predict(self, model_name: str, features: np.ndarray) -> dict:
        "Runs a individual XGBoost model on CNN features."
        xgb_model = self.xgb[model_name]
        probs = self.xgb.predict_proba(features)[0]
        grade = int(probs.argmax())
        return{'grade': grade, 'confidence': float(probs[grade]), 'probs': probs.tolist()}
    
    def predict(self, pil_image: Image.Image) ->dict:
        "Runs the full 7-model pipeline on a single PIL image"
        tensor = self._preprocess(pil_image)
        results = {}
        all_grades = []

        #CNN ONLY PREDICTION
        cnn_features = {}
        for model_name in self.manager:
            pred = self._cnn_predict(model_name, tensor)
            results[f"{model_name}_CNN"] = pred
            all_grades.append(pred['grade'])

            #Extracting features once will reuse for xgb
            cnn_features[model_name] = self._extract_features(model_name, tensor)

        #Individual XGBOOST + CNN predictions
        for model_name, xgb_model in self.xgb.items():
            if model_name in cnn_features:
                pred = self._xgb_predict(model_name, cnn_features[model_name])
                results[f"{model_name}_xgb"] = pred
                all_grades.append(pred["grade"])
        
        #Ensemble prediction
        if self.ensemble_model is not None and self.ensemble_pca is not None:
            available = [m for m in ['resnet50', 'densenet121', 'inceptionV3']
                         if m in cnn_features]
            X_combined = np.hstack([cnn_features[m] for m in available])
            X_pca = self.ensemble_pca.transform(X_combined)            
            probs = self.ensemble_model.predict_proba(X_pca)[0]
            grade = int(probs.argmax())
            results["ensemble"] = {
                "grade": grade,
                "confidence": float(probs[grade]),
                "probs": probs.tolist()
            }
            all_grades.append(grade)
 
        # ── Consensus grade (majority vote) ────────────────────────────────
        if all_grades:
            results["consensus_grade"] = int(
                np.bincount(all_grades).argmax()
            )
 
        return results