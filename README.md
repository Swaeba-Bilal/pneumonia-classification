# Chest X-Ray Pneumonia Classification using Transfer Learning

A deep learning project comparing four different transfer-learning strategies for detecting pneumonia from chest X-ray images, built using PyTorch and a pretrained ResNet-18 backbone. Developed as part of preparation for the Mitacs Globalink Research Internship application.

## Overview

Pneumonia diagnosis from chest X-rays is a well-studied medical imaging task, commonly used as a benchmark for evaluating transfer-learning approaches in low-data medical settings. This project explores how much of a pretrained ImageNet network needs to be fine-tuned to get reliable, generalizable performance on a relatively small, imbalanced medical dataset — and finds a counter-intuitive result: less fine-tuning generalized better than more.

## Dataset

**Source**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kermany et al.), downloaded via the Kaggle API.

- **5,856 total images**, labeled NORMAL or PNEUMONIA, sourced from pediatric patients
- Provided as three splits: train (5,216), test (624), val (16)
- **Class imbalance**: training data has a NORMAL:PNEUMONIA ratio of roughly 1:2.89 (1,341 NORMAL vs 3,875 PNEUMONIA)

**Validation split issue**: the dataset's provided validation folder contains only 16 images — too small to give a statistically meaningful signal during training. To address this, a new validation set was carved out of the training data using a stratified 85/15 split (`sklearn.model_selection.train_test_split` with `stratify`), preserving the original class ratio in both the new training and validation sets. The original test set (624 images) was left completely untouched throughout training and only used for final evaluation.

## Preprocessing

- All images resized to 224×224 (matching ResNet-18's expected ImageNet input size)
- Normalized using ImageNet's channel-wise mean/std (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`), since the pretrained backbone expects inputs in this distribution
- **Training-only augmentation**: random rotation (±15°) and brightness/contrast jitter, simulating natural variation in patient positioning and X-ray exposure
- **Deliberately excluded**: horizontal flipping, since some images contain laterality markers ("R" for right side) that could be scrambled by flipping; vertical flipping, which is anatomically meaningless for a chest X-ray

## Methodology: Four Transfer-Learning Strategies

All four experiments use a ResNet-18 backbone pretrained on ImageNet, with its final classification layer replaced to output 2 classes instead of 1,000. They differ in how much of the pretrained network is allowed to keep learning:

### Experiment 1: Full Fine-Tune + Class-Weighted Loss
Every layer of the network remains trainable. To address the class imbalance, `CrossEntropyLoss` was weighted inversely proportional to class frequency, so mistakes on the minority class (NORMAL) are penalized more heavily during training.

### Experiment 2: Full Fine-Tune + Regularization
Same as Experiment 1, but with `Dropout(0.5)` added before the final layer and L2 weight decay (`1e-4`) applied via the optimizer, both intended to reduce overfitting by discouraging the model from relying too heavily on narrow, training-specific patterns.

### Experiment 3: Frozen Backbone (Linear Probe)
All pretrained layers are frozen (`requires_grad=False`); only the new final layer is trained. This is the lightest-touch approach — essentially using ResNet-18 purely as a fixed feature extractor. No class weighting was applied here.

### Experiment 4: Partial Unfreeze with Differential Learning Rates
Only the last residual block (`layer4`) is unfrozen alongside the new final layer, using two different learning rates: a very small one (`1e-5`) for the pretrained `layer4` (to adjust gently without destroying existing knowledge) and a larger one (`1e-3`) for the new final layer (which starts from scratch and needs to learn faster). Dropout and weight decay were also applied.

All four experiments used the same optimizer (Adam), the same early stopping criterion (stop after 2 epochs without validation improvement), and a fixed random seed (42) for a fair, controlled comparison.

## Results

Evaluated on the held-out, never-touched test set (624 images):

| Experiment | Val Accuracy | Test Accuracy | NORMAL Recall | PNEUMONIA Recall |
|---|---|---|---|---|
| 1: Full Fine-Tune + Class Weights | 99.11% | 84.13% | 58% | 99% |
| 2: Full Fine-Tune + Regularization | 98.60% | 86.70% | 65% | 99% |
| 4: Partial Unfreeze (layer4) | 98.08% | 86.70% | 66% | 99% |
| **3: Frozen Backbone (final model)** | 94.25% | **87.66%** | **70%** | 98% |

## Key Finding

Validation accuracy and test accuracy moved in **opposite directions** as more of the network was allowed to fine-tune: the model with the *highest* validation accuracy (Experiment 1, 99.11%) had the *lowest* test accuracy (84.13%), while the model with the *lowest* validation accuracy (Experiment 3, 94.25%) had the *highest* test accuracy (87.66%). This pattern held even under a controlled comparison with a fixed random seed across all four experiments.

This suggests a genuine distribution shift between the validation and test portions of this dataset (a characteristic noted in other work using this dataset, likely due to differing image sources or acquisition equipment). Freezing the pretrained backbone — training only a lightweight linear classifier on top of frozen ImageNet features — proved the most robust to this shift, likely because it has far less capacity to overfit to validation-specific quirks.

Class-weighted loss (Experiments 1, 2) was also found to bias predictions toward PNEUMONIA, achieving near-perfect PNEUMONIA recall (99%) at a real cost to NORMAL recall (58-65%). Removing class weighting (Experiments 3, 4) produced more balanced, and ultimately better-generalizing, classifiers.

## Final Model

**Experiment 3 (Frozen Backbone)** was selected as the project's final model:
- Test Accuracy: 87.66%
- NORMAL: 96% precision, 70% recall
- PNEUMONIA: 84% precision, 98% recall

## Project Structure
pneumonia-classification/
├── Notebooks/
│ └── My_Project.ipynb # Full pipeline: data loading through evaluation
├── models/
│ ├── best_model.pth # Experiment 1
│ ├── best_model_regularized.pth # Experiment 2
│ ├── best_model_frozen.pth # Experiment 3 (final)
│ └── best_model_partial_unfreeze.pth
├── results/
│ ├── figures/ # Confusion matrices, training curves
│ └── logs/ # Per-epoch training history (JSON)
└── README.md


## Tools & Libraries

PyTorch, torchvision (ResNet-18, ImageNet weights), scikit-learn (stratified splitting, evaluation metrics), Google Colab (T4 GPU)

## Limitations

- Single run per experiment (with a fixed seed for fair comparison); a fully rigorous comparison would average multiple seeded runs per strategy
- Trained and evaluated on pediatric chest X-rays only; generalization to adult patients is untested
- Binary classification only (NORMAL vs. PNEUMONIA); does not distinguish between pneumonia subtypes (viral vs. bacterial)
- The validation-to-test distribution shift observed suggests any deployment would need evaluation on external, independently-sourced data before real-world use
