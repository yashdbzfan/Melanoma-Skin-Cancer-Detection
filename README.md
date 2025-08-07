# Melanoma Skin Cancer Detection (SIIM-ISIC 2020)

## Overview

This project builds a deep learning classifier for melanoma detection using the SIIM-ISIC 2020 dataset, which contains 33,126 dermoscopic training images with per-image metadata and patient identifiers for patient-level splitting. The training pipeline uses PyTorch with a ResNet50 backbone, patient-wise GroupShuffleSplit to prevent leakage, WeightedRandomSampler for class imbalance, and OneCycleLR scheduling for stable convergence. A Streamlit app is provided for inference from uploaded skin lesion images, loading a saved model checkpoint (.pth).[^1][^2][^3][^4]

## Features

- Dataset: SIIM-ISIC 2020 Challenge dataset with 33,126 dermoscopic images and metadata including patient_id and target(malignant/benign).[^2][^4][^1]
- Patient-wise split: GroupShuffleSplit on patient_id to avoid information leakage across train/validation sets.[^3]
- Class imbalance handling: WeightedRandomSampler to oversample minority class (malignant).[^5]
- Model: ResNet50 with a custom classifier head and BCEWithLogitsLoss; ROC-AUC used for evaluation.[^2]
- Training: AdamW optimizer, OneCycleLR scheduler, standard ImageNet normalization and strong augmentations for training.[^2]
- Inference app: Streamlit interface to upload an image and get benign/malignant prediction with confidence; loads saved PyTorch .pth weights.[^2]
- Checkpointing: Saves model weights to melanoma_resnet50.pth for later use with the web app.[^2]


## Dataset Download

- Official challenge site: ISIC 2020 Challenge Dataset (details and license).[^1]
- Alternative index pages for ISIC challenge datasets (download entry points may vary over time).[^6]
- Kaggle competition page (requires login to access data terms).[^7][^8]

Notes:

- The dataset includes 33,126 training images from 2,000+ patients, designed for patient-centric evaluation and licensed under CC BY-NC 4.0.[^1][^2]
- Malignant prevalence is low (~1.8%), so addressing class imbalance is important.[^5][^2]


## Project Structure

```
melanoma-detection/
├── app.py                 # Training pipeline: data loading, model, training loop, saving .pth
├── online.py              # Streamlit inference app (loads melanoma_resnet50.pth)
├── train-image/           # Root folder for images (DATA_DIR)
├── train-metadata.csv     # Metadata with isic_id, patient_id, target (CSV_PATH)
├── melanoma_resnet50.pth  # Saved model weights after training
├── README.md              # This documentation
└── requirements.txt       # Python dependencies (torch, torchvision, pandas, streamlit, etc.)
```


## Requirements

- Python 3.8+
- Recommended GPU with CUDA for training
- Key Python packages:
    - torch, torchvision
    - pandas, scikit-learn
    - Pillow, tqdm
    - streamlit


## Setup Instructions

1) Clone the repository
```bash
git clone <repo-url>
cd melanoma-detection
```

2) Prepare the dataset

- Download SIIM-ISIC 2020 images and metadata CSV from ISIC or Kaggle.[^8][^7][^1]
- Place image folders into train-image/ and the CSV file as train-metadata.csv, or update DATA_DIR and CSV_PATH in app.py accordingly.[^1][^2]
- Dataset description confirms 33,126 training images and patient identifiers for grouped splits.[^4][^1][^2]

3) Install dependencies
```bash
pip install -r requirements.txt
```

4) Configure paths

- In app.py, set:

```
DATA_DIR = r"C:\path\to\train-image"
CSV_PATH = r"C:\path\to\train-metadata.csv"
```

- Adjust batch size, epochs, and workers as needed.


## Training

Run:

```bash
python app.py
```

What happens:

- Reads train-metadata.csv (columns: isic_id, patient_id, target).[^2]
- Recursively indexes images by isic_id under DATA_DIR and fails fast if any metadata IDs are missing.[^2]
- Splits data using GroupShuffleSplit with patient_id to prevent leakage.[^3]
- Applies training augmentations (RandomResizedCrop, flips, ColorJitter) and ImageNet normalization; validation uses resize+normalize only.[^2]
- Uses WeightedRandomSampler for class imbalance and computes loss with BCEWithLogitsLoss and AUC for monitoring.[^5][^2]
- Optimizes with AdamW and OneCycleLR for stable learning rate scheduling.[^2]
- Saves model weights to melanoma_resnet50.pth upon completion.[^2]

Why GroupShuffleSplit:

- Ensures all samples from a given patient_id appear in either train or validation, not both—important for clinical datasets to avoid overly optimistic evaluation.[^3]


## Inference (Streamlit App)

Run:

```bash
streamlit run online.py
```

Usage:

- Upload a dermoscopic image (.jpg/.jpeg/.png).
- The app loads melanoma_resnet50.pth, applies the same normalization, and outputs:
    - Prediction: Benign or Malignant
    - Confidence score
- The app uses a ResNet50 with a matched final layer configuration and loads state_dict with map_location for CPU/GPU portability.[^2]

Note on model architecture consistency:

- The model head in online.py must match the head defined during training, otherwise state_dict loading will fail; error handling is included to surface mismatches.[^2]


## Model Details

- Backbone: torchvision ResNet50 (ImageNet weights for transfer learning).[^2]
- Head: Fully connected layers with BatchNorm/ReLU/Dropout ending in a single logit for binary classification; evaluated with ROC-AUC.[^2]
- Loss: BCEWithLogitsLoss; Sigmoid applied for probability during metric computation and inference.[^2]
- Scheduler: OneCycleLR; Optimizer: AdamW with weight decay.[^2]
- Metrics: AUC printed per epoch for train and validation to track discrimination performance.[^2]


## Example Paths and Splits

- The official dataset page confirms 33,126 dermoscopic training images and patient-centric design, which aligns with the patient-wise GroupShuffleSplit used here.[^4][^1][^2]
- Severely imbalanced positive class motivates the use of WeightedRandomSampler during training.[^5]


## Troubleshooting

- Missing images for metadata IDs: The dataset class raises with a sample of missing IDs if path mapping fails; verify image folder structure and filenames match isic_id.[^2]
- State dict loading errors: Ensure the classifier head in online.py matches app.py; version differences may cause key mismatches—adjust architecture accordingly.[^2]
- Low validation AUC: Verify patient-wise splitting, normalization, and sampler settings; the class imbalance and prevalence can impact performance.[^5][^2]
- Dataset access: If Kaggle access is limited, use the ISIC challenge portal; confirm license and citation requirements (CC BY-NC 4.0).[^7][^8][^1]


## References and Access

- SIIM-ISIC 2020 dataset official page with description, sources, and license details, confirming 33,126 training images and patient-centric design.[^1]
- Peer-reviewed dataset description with counts, composition, and prevalence statistics.[^2]
- Kaggle competition pages for metadata, download, and context of the challenge.[^8][^7]
- GroupShuffleSplit documentation for patient-level splitting strategy.[^3]
- Notes on class imbalance magnitude from community analyses of the competition dataset.[^5]

<div style="text-align: center">⁂</div>

[^1]: https://challenge2020.isic-archive.com

[^2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7843971/

[^3]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html

[^4]: https://paperswithcode.com/dataset/isic-2020-challenge-dataset

[^5]: https://www.kaggle.com/code/neelgajare/siim-isic-melanoma-classification

[^6]: https://challenge.isic-archive.com/data/

[^7]: https://www.kaggle.com/competitions/siim-isic-melanoma-classification

[^8]: https://www.kaggle.com/c/siim-isic-melanoma-classification/data

[^9]: https://www.kaggle.com/code/ladanova/siim-isic-melanoma-classification

[^10]: https://www.kaggle.com/code/muhakabartay/keras-siim-isic-melanoma-classification-resnet

[^11]: https://scikit-learn.org/0.20/modules/generated/sklearn.model_selection.GroupShuffleSplit.html

[^12]: https://github.com/kabartay/kaggle-siim-isic-melanoma-classification/blob/master/siim-isic-melanoma-classification-efficientnet.ipynb

[^13]: https://www.sciencedirect.com/science/article/pii/S0010482524015774

[^14]: https://www.geeksforgeeks.org/machine-learning/how-to-generate-a-train-test-split-based-on-a-group-id/

[^15]: https://www.kaggle.com/c/siim-isic-melanoma-classification/leaderboard

[^16]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12192895/

[^17]: https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic

[^18]: https://stackoverflow.com/questions/68321527/why-i-am-getting-the-error-for-groupshufflesplit-train-test-split

[^19]: https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/154271

[^20]: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/164092

