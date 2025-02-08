
# Multi-Modal Classifier for Research

This repository contains the implementation of a multi-modal classifier designed for research purposes. The model fuses features from various branches (text, image, and vision transformer) to perform classification tasks on multi-modal data. The code is developed for academic research and has been used in our paper.



## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Database](#database)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Key Arguments](#Key-Arguments)
  - [Example Command](#example-command)
- [Training Configuration](#training-configuration)
- [File Structure](#file-structure)
- [Results and Logging](#results-and-logging)
- [Contact](#contact)


## Overview

The repository implements a multi-modal classifier that integrates multiple data modalities:
- **Text Branch**: Uses Sentence Transformers (`all-MiniLM-L6-v2`) for textual feature extraction.
- **Mamba Branch**: Utilizes NVIDIA's MambaVision for image classification.
- **ViT Branch**: Employs OpenAI's CLIP ViT for visual feature extraction.

The classifier fuses these features using a configurable combined classifier, which can be extended with additional hidden layers. Training procedures include options for early stopping, learning rate scheduling, and detailed logging of per-epoch and final results.



## Features

- **Multi-Modal Fusion**: Combines text, image, and vision transformer features.
- **Configurable Architecture**: Customize the number of hidden layers and neurons for the combined classifier.
- **Flexible Training**: Options to enable/disable a learning rate scheduler.
- **Detailed Logging**: Logs training progress to `training.log` and final results (including per-class accuracies) to `result.log`.
- **Reproducible Research**: Designed to support reproducible experiments for academic research.

## Database

The dataset used for this project is publicly available and has been carefully curated to support multi-modal book cover classification. The database includes:

You can download the complete dataset from the following link:

[Download Dataset](https://drive.google.com/file/d/14Vx5WhkFe4LZM4x1p8kfybbYkAiQ52oS/view?usp=sharing)

*Note: Please ensure that the dataset is extracted to the repository root or update the file paths in the code accordingly.*

## Requirements

- Python 3.7 or later
- [PyTorch](https://pytorch.org/) (tested with version 1.7+)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [Transformers](https://github.com/huggingface/transformers)
- [Pandas](https://pandas.pydata.org/)
- [TQDM](https://github.com/tqdm/tqdm)
- [Pillow](https://python-pillow.org/)

Run the following command to install the required libraries:
```sh
pip install -r requirements.txt
```



## Installation
1. Clone repository:
   ```bash
   git clone https://github.com/RezaToosii/BookCover-Classifier
   cd BookCover-Classifier
   ```


Ensure that your dataset CSV files (`train_dataset.csv`, `test_dataset.csv`, and `class_map.csv`) are in the repository root or adjust the hard-coded paths in the code as needed.



## Usage

The training code is contained in `train.py`. The dataset file paths and batch size are hard-coded in the script for reproducibility. All model and training hyperparameters are passed via command-line arguments.

### Key Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--model_type` | Modality combination (mamba, vit, text, or combinations) | Required |
| `--learning_rate` | Initial learning rate | 2e-5 |
| `--epochs` | Maximum training epochs | 7 |
| `--using_scheduler` | Boolean flag to enable or disable the learning rate scheduler |True |
| `--step` | Step size for the learning rate scheduler | 5 |
| `--gamma` | Gamma value for the learning rate scheduler | 0.1 |
| `--early_stop` | Early stopping patience | 3 |
| `--num_hidden_layers` | Combined classifier hidden layers | 0 |
| `--hidden_neurons` | Neurons per hidden layer | 512 |

### Example Command

Run the training script with a configuration that uses the ViT, text, and Mamba branches, with a scheduler enabled:

```bash
python train.py --model_type vit+text+mamba --learning_rate 3e-5 --using_scheduler True --step 5 --gamma 0.1 --epochs 100 --early_stop 3 --num_hidden_layers 2 --hidden_neurons 256 
```

This command will:
- Train the model for up to 100 epochs with early stopping.
- Use 2 hidden layers with 256 neurons each in the combined classifier.
- Log training details to `training.log` and final results to `result.log`.
- Save the best model weights in the `weight/` folder with a filename reflecting key hyperparameters.



## Training Configuration
### Optimization
- **Optimizer**: AdamW
- **Scheduler**: StepLR (optional)
- **Loss Function**: Cross-Entropy

### Image Processing
- Resize: 224x224
- Normalization: ImageNet statistics
- Augmentations: None (add via `transforms`)


## File Structure

```
.
├── weight/               # Directory where the best model weights are saved.
├── Covers/               # Directory where train & test dataset are exist.
├── dataloader.py         # Custom dataset loader for the book cover dataset.
├── model.py              # Contains the model definition, training, evaluation, and logging functions.
├── train.py              # Main training script that parses arguments and initiates training.
├── result.log            # (Generated after training) Contains final training results and model summary.
├── training.log          # (Generated during training) Contains per-epoch training and validation logs.
├── train_dataset.csv     # Training dataset CSV file (hard-coded in train.py).
├── test_dataset.csv      # Validation dataset CSV file (hard-coded in train.py).
└── class_map.csv         # CSV mapping class numbers to class names.
```



### Output Files
---
- **training.log**: Full training process with epoch-wise metrics
- **result.log**: Final results including:
  - Best validation accuracies (Top-1/Top-3)
  - Per-class accuracy table
  - Used hyperparameters
- **weight/**: Saved model checkpoints

### Example Result
```
Parsed Hyperparameters:


Final Training Loss:
Final Training Top1:

Best Epoch:
Best Val Top1:
Best Val Top3:

Per-Class Validation Accuracies (Best Epoch):
```

## Contact

For any questions or inquiries, please contact [Reza Toosi](rtoosi81@gmail.com).

---