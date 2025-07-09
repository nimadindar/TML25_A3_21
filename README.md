# Trustworthy Machine Learning Course Assignment 3

This repository contains the implementation and supporting materials for Assignment 3 of the Trustworthy Machine Learning course, completed by Nima Dindarsafa (7072844) and Samira Abedini (7072848) during the Summer Semester 2025. The objective of this assignment is to develop neural network models with enhanced robustness against adversarial examples, utilizing ResNet18, ResNet34, and ResNet50 architectures. Three distinct methods were implemented: Geometric Aware Instance Reweighted Adversarial Training (GAIRAT), Pretraining, and Dataset-Specific Adversarial Training. The project leverages PyTorch for model training and evaluation, with results submitted to a leaderboard. A comprehensive report detailing the methodology and findings is provided in `Report.pdf`.

## Project Description
The assignment focuses on training models to maintain high clean accuracy while resisting adversarial perturbations generated using the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD). The dataset, stored in `Train.pt`, consists of 32x32x3 RGB images, split into 80% training and 20% testing sets. The implementations address challenges related to model input compatibility and computational resource constraints, as documented in the report.

## Folder Structure

- **`data/`**: Contains the dataset file `Train.pt`.
- **`GAIRAT/`**: Implements the GAIRAT method, which assigns weights to adversarial examples based on their geometric proximity to decision boundaries to improve robustness. Key files include:
  - `GAIRAT.py`: Primary script for training and evaluating GAIRAT models, recording metrics such as natural and PGD20 accuracies.
  - `gairat_config.py`: Configuration file specifying hyperparameters.
  - `attack_generator.py`: Module for generating adversarial examples using FGSM and PGD.
- **`Pretrain/`**: Contains the Pretraining method, which involves adversarial pre-training on Downsampled ImageNet followed by fine-tuning. Key file:
  - `train_imagenet.py`: Implements a pre-training scheme from Hendrycks et al. (2019), using PyTorch to train a ResNet model on downsampled ImageNet with PGD-generated adversarial examples, SGD or Adam optimization with cosine annealing, and multi-GPU support with checkpoint resumption.
- **`DatasetSpecific/`**: Implements the Dataset-Specific Adversarial Training method, optimized for ResNet34 with dataset-specific normalization. Key file:
  - `robust_classifier.py`: Script for training ResNet34 over 100 epochs, employing FGSM and PGD attacks, and managing leaderboard submission.
- **`dataset/`**: Houses data handling utilities.
- **`models/`**: Includes utilities for model management. Key file:
  - `load_model.py`: Loads and adapts pre-trained ResNet models.
- **`results/`**: Directory for storing output files.
- **`README.md`**: This document, providing an overview of the repository structure and file functionalities.
- **`Report.pdf`**: Contains the project report.
- **`requirements.txt`**: Lists required Python packages, including `torch`, `torchvision`, `numpy`, and `requests`.
## Results
The following table presents the best clean and adversarial accuracies achieved on the test set for each method. Values reflect experimental results from training ResNet18, ResNet34, and ResNet50 models on the `Train.pt` dataset, with an 80/20 train-test split. Due to resource constraints, some results (e.g., Pretraining) are incomplete, while others (e.g., GAIRAT, Dataset-Specific) are based on partial or full runs as noted.

| Method                     | Best Model | Best Clean Accuracy (%) | Best Adversarial Accuracy (%) |
|----------------------------|------------|--------------------------|--------------------------------|
| GAIRAT [1]                 | ResNet50   | 52.7950                 | 42.0450                       |
| Pretraining [2]            | -          | -                       | -                             |
| Dataset-Specific Adversarial | ResNet34   | 42.1400                 | 37.2840                       |

- **GAIRAT**: Achieves balanced performance with 52.79% clean accuracy and 42.05% adversarial accuracy on CIFAR-like datasets, leveraging geometry-aware weighting to focus on vulnerable data points. Results are indicative due to resource constraints preventing full optimization.
- **Pretraining**: Potentially outperforms GAIRAT due to adversarial pre-training on Downsampled ImageNet, enhancing generalization. However, results are unavailable due to high computational demands (incomplete after 2.5 days).
- **Dataset-Specific Adversarial**: Reaches 42.14% clean accuracy and 37.28% adversarial accuracy with ResNet34 after 150 epochs. Its clean-only focus (`beta=1.0`) and dataset-specific normalization address the dataset’s specialized distribution, though clean accuracy did not exceed 50% despite extended training at a lower learning rate (0.0001). Future improvements may involve reintroducing adversarial training (`beta=0.8`, `epsilon=4/255`) once clean accuracy surpasses 50%.


## Implementation Notes
The project encountered several technical challenges:
- **Input Dimension Mismatch**: Pre-trained ResNet models require 224x224x3 inputs, necessitating adjustments for 32x32x3 data, which initially caused shape errors.
- **Computational Resources**: Training, especially for the Pretraining method, required significant GPU resources, with runs exceeding 2.5 days on available hardware.
- **Submission Format**: A shift from ONNX (Assignment 2) to PT files required adapting submission code, highlighting a need for consistent formats across assignments.

These issues impacted the ability to fully optimize models and compete on the leaderboard, as detailed in the `Report.pdf`.

## Usage
1. Install dependencies by executing: `pip install -r requirements.txt`.
2. Run the respective scripts within their directories, e.g., `python GAIRAT/GAIRAT.py` or `python DatasetSpecific/robust_classifier.py`, on a GPU-enabled environment.
3. Verify output files, including model weights, in `out/models/`.

For further details on the methodology, experimental setup, and results, consult the `Report.pdf` or access the associated [GitHub repository](https://github.com/nimadindar/TML25_A3_21).
