"""
src/gradcam.py
────────────────────────────────────────────────────────────
GradCAMEngine: generates Grad-CAM heatmap overlays for all
three CNN backbones using the pytorch-grad-cam library.

Install:  pip install grad-cam

Target layers (the last convolutional layer before the classifier):
ResNet50    → model.layer4[-1]               last residual block
DenseNet121 → model.features.denseblock4     last dense block
InceptionV3 → model.Mixed_7c                 last inception module

These are the layers whose activation maps are most semantically
rich — they encode high-level features like microaneurysms,
haemorrhages, and neovascularisation patterns.
────────────────────────────────────────────────────────────
"""

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAIL = True

except ImportError:
    GRADCAM_AVAIL = False

def get_target_layer(model, model_name: str):
    """
    Returns the correct target layer for Grad-CAM for each architecture.
    Must be the last convolutional layer before global average pooling.
    """
    if model_name == 'resnet50':
        #layer4 is a last Sequential layer of 3 bottleneck block, while we want the last one
        return [model.layer4[-1]]

    elif model_name == "densenet121":
        #here also we want the last block which is denseblock4 befor norm5
        return [model.features.denseblock4]
    
    elif model_name == "inceptionV3":
        #Require the Mixed_7c last inception module befor adaptive avg pool
        return [model.Mixed_7c]
    
    else:
        raise ValueError(f"No target layer defined for: {model_name}")
    
def preprocess_for_gradcam(pil_image: Image.Image, img_size: tuple = (512, 512)) -> tuple:
    """
    Returns:
    input_tensor : (1, 3, H, W) torch.Tensor — normalised, for model forward pass
    rgb_array    : (H, W, 3)    np.float32   — [0,1] range, for overlay blending

    """
    resize = transforms.Resize(img_size)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_resized = resize(pil_image.convert("RGB"))
    input_tensor = to_tensor(img_resized).unsqueeze(0)

    rgb_array = np.array(img_resized).astype(np.float32) / 255.0

    return input_tensor, rgb_array

class GradCAMEngine:
    """
    Generates Grad-CAM visualisations for a single image
    across all three CNN backbones.

    """

    def __init__(self, managers: dict, device):
        """
        managers : dict  model_name → DRModelManager (from InferenceEngine)
        device   : torch.device
        """
        self.managers = managers
        self.device   = device
 
        if not GRADCAM_AVAIL:
            print("pytorch-grad-cam not installed.")
            print("  Run: pip install grad-cam")
 
    def generate(self, pil_image: Image.Image,
                 target_grade: int = None) -> dict:

        """
        Generates Grad-CAM overlays for all loaded CNN models.
 
        Parameters
        ----------
        pil_image    : PIL.Image   — the uploaded retinal image
        target_grade : int or None — grade to explain (None = use predicted grade)
 
        Returns
        -------
        dict  model_name → PIL.Image  (overlay heatmap, same size as input)

        """
        if not GRADCAM_AVAIL:
            return {}
        
        input_tensor, rgb_array = preprocess_for_gradcam(pil_image)
        input_tensor = input_tensor.to(self.device)

        overlays = {}

        for model_name, self.managers in self.managers.items():
            model = self.managers.get_model()
            model = self.managers.get_model()
            model.eval()
            
            try:
                target_layers = get_target_layer(model, model_name)

                #Special for Case for InceptionV3 as this outputs multiple outputs with auxlogits
                # we disable aux_logits during Grad-CAM.
                if model_name == 'inceptionV3':
                    original_aux = model.aux_logits
                    model.aux_logits = False

                with GradCAM(model=model, target_layers=target_layers) as cam:

                    targets = None
                    if target_grade is not None:
                         from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                         targets = [ClassifierOutputTarget(target_grade)] 
                    
                    grayscale_cam = cam(
                        input_tensor=input_tensor,
                        targets = targets
                    )[0]

                    overlays_array = show_cam_on_image(
                    rgb_array,
                    grayscale_cam,
                    use_rgb=True,
                    colormap=4,
                    image_weight=0.5
                    )

                    overlays[model_name] = Image.fromarray(overlays_array)

                    if model_name == "inceptionV3":
                        model.aux_logits = original_aux
                    
            except Exception as e:
                print(f"Grad-Cam failed for {model_name}: {e}")
                overlays[model_name] = None
            
        return overlays