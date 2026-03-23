# Diabetic Retinopathy Detection: A Hybrid Deep Ensemble Approach:

## Overcoming the "Accuracy Paradox" in Medical Imaging-
This project uses a multi-stage Hybrid Stacking Ensemble to detect and grade Diabetic Retinopathy (DR) from fundus images. Our main goal was to address the Accuracy Paradox, where standard deep learning models achieve high overall accuracy by overlooking rare, high-risk "Severe DR" cases in favor of the "No DR" majority. The Problem: The Accuracy Paradox In medical screening, standard CNNs often achieve 80-90% accuracy simply by predicting "Healthy" for every patient. The Clinical Risk: Misclassifying a patient with Severe DR as "Healthy" can have serious consequences. The Data Reality: Medical datasets like APTOS 2019 are very imbalanced, making traditional training unreliable for minority classes.

### Classwise dataset sample
<img width="1490" height="330" alt="image" src="https://github.com/user-attachments/assets/2f306496-699e-4b8c-b9b9-4e9039fecaf5" />

## The Architecture
Our pipeline uses a "Model Surgery" approach to decouple spatial feature learning from final classification.

* **Triple-Backbone Feature Extraction:** We utilized three distinct CNN architectures, each capturing different spatial hierarchies:
  * **ResNet50:** Residual skip connections for structural integrity.
  * **DenseNet121:** Dense connectivity for fine-grained textural reuse.
  * **InceptionV3:** Factorized convolutions for multi-scale lesion detection.
* **Advanced Optimization:** * **Selective Partial Unfreezing:** Only the deepest blocks (e.g., `layer4`, `Mixed_7`) were fine-tuned to prevent "Catastrophic Forgetting."
  * **Focal Loss:** Applied during training to penalize the model more heavily for misclassifying difficult minority samples.
  * **Optuna Tuning:** Bayesian optimization was used to find the perfect hyperparameters for our custom classification heads.
* **Feature Fusion & SMOTE:** * Extracted 5,120 high-dimensional features per image.
  * Applied SMOTE (Synthetic Minority Over-sampling Technique) in the 1D feature space to mathematically balance the classes without distorting raw pixel data.
* **Meta-Classification:** * An XGBoost meta-learner was trained on the balanced feature space to learn the final non-linear decision boundaries.

### Pipline Architecture
<img width="2816" height="1536" alt="image" src="https://github.com/user-attachments/assets/e9b96391-05b3-43f5-934f-65991b66cc30" />

##  Key Results (APTOS 2019)
By implementing the SMOTE Ensemble, we achieved a significant boost in clinical sensitivity compared to standalone models.

| Metric | Standalone ResNet50 | Proposed SMOTE Ensemble |
| :--- | :---: | :---: |
| **Macro F1-Score** | 0.4120 | **0.6886** |
| **ROC AUC** | 0.8850 | **0.9662** |
| **Recall (Severe DR)** | ~23.0% | **46.1%** |
| **Accuracy** | 76.41% | **82.81%** |

### Overall pipeline performance
<img width="3499" height="1976" alt="image" src="https://github.com/user-attachments/assets/caacb2c4-1417-43c2-a0b7-80de78c4c85b" />

### Resolving the accurancy paradox
<img width="2898" height="1702" alt="image" src="https://github.com/user-attachments/assets/4f2da18a-b043-4ea2-af1a-99f6e63ff31a" />

## Clinical Transparency (XAI)
To ensure "Black Box" models are trustworthy, we integrated Grad-CAM (Gradient-weighted Class Activation Mapping). This visually confirms that the model is looking at true biological markers—like hemorrhages and exudates—rather than image artifacts or lighting noise.

### Grad-CAM Image
<img width="3470" height="1828" alt="image" src="https://github.com/user-attachments/assets/44697e8e-396f-4c25-bd8e-59c0e497946c" />

## Tech Stack

* **Frameworks:** PyTorch, XGBoost
* **Optimization:** Optuna, SMOTE (imbalanced-learn)
* **Visualization:** Matplotlib, Grad-CAM
* **Validation:** IDRiD (Cross-Dataset Testing)




