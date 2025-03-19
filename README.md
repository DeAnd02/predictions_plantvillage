# 🌿 PlantVillage Disease Detection with CNN and Transfer Learning

## Overview

This project focuses on the automatic detection of plant leaf diseases using Convolutional Neural Networks (CNN) and Transfer Learning (MobileNet V2). The goal is to create a lightweight, efficient system suitable for deployment on mobile or embedded devices (e.g., Raspberry Pi), capable of classifying 38 different categories of healthy and infected leaves across 14 plant species.

## Dataset

- **Source**: PlantVillage dataset
- **Size**: 50,000+ images
- **Classes**: 38 (includes healthy and diseased classes for 14 species)
- **Split**:
  - 70% Training
  - 15% Validation
  - 15% Testing
- **Image Size**: Resized to 224x224 pixels (required input size for MobileNet V2)

## Models

### 1. **MobileNet V2 (Transfer Learning)**
- Pre-trained on ImageNet
- Fine-tuned on PlantVillage dataset
- Final Dense layer with SoftMax for 38 classes
- Achieved **94% accuracy** before fine-tuning
- **98% accuracy** after fine-tuning last 20 layers
- Low computational cost, ideal for resource-constrained environments

### 2. **Custom CNN**
- 5 Convolutional Blocks (filters 64 → 256)
- MaxPooling after each block
- Dense layers with Dropout (rate: 0.5)
- Achieved **94% accuracy**
- Trained from scratch on PlantVillage dataset

## Training Details
- **Epochs**: 15
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam with dynamic learning rate
- **Regularization**: Dropout and Early Stopping
- **Data Augmentation**: Rotation, flipping, scaling to reduce overfitting

## Files

| File Name                        | Description                                                               |
|---------------------------------|---------------------------------------------------------------------------|
| `plant_village.ipynb`           | Main Colab notebook for MobileNet V2 training and evaluation              |
| `plantvillage_customCNN.ipynb`  | Notebook for training and evaluating the Custom CNN model                 |
| `cofusion_matrix.ipynb`         | Notebook to compute and visualize confusion matrices                      |
| `predict_index.ipynb`           | Prediction notebook using index-based approach                            |
| `predict_mobilenet.ipynb`       | Prediction notebook using the trained MobileNet V2 model                  |
| `ML_plantvillage_CNN_presentation.pdf` | Presentation detailing dataset, models, training results       |

## Usage

1. Open the desired notebook in [Google Colab](https://colab.research.google.com/)
2. Upload or mount the PlantVillage dataset
3. Run:
   - `plant_village.ipynb` for MobileNet V2
   - `plantvillage_customCNN.ipynb` for Custom CNN
4. Evaluate model performance with confusion matrices and accuracy metrics

## Results Summary

| Model           | Accuracy | Parameters (Trainable) | Notes                               |
|-----------------|----------|------------------------|-------------------------------------|
| MobileNet V2    | 94%      | 168,870 / 2.4M         | Transfer Learning                   |
| MobileNet V2 FT | 98%      | Fine-tuned last 20     | Fine-tuned, higher training time    |
| Custom CNN      | 94%      | 1.26M                  | From scratch, more computationally intense |

## Future Work
- Test with other pre-trained models (e.g., EfficientNet, ResNet)
- Deploy trained models into a real-world mobile application
- Expand dataset with real-world images for robustness

## Author
Andrea De Biasi  
Ca’ Foscari University of Venice  
Department of Environmental Sciences, Informatics and Statistics  
Email: 890488@stud.unive.it

---

> *“Reducing crop disease through machine learning for a sustainable future.”*
