import os
import gc
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# 1. CONFIG — point these to your folders/files:
DATA_DIR     = r"train-image"
CSV_PATH     = r"train-metadata.csv"

BATCH_SIZE   = 10
IMG_SIZE     = 224
NUM_WORKERS  = 4   # set to 0 if you still see spawn issues
EPOCHS       = 10
LR           = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ISICDataset(Dataset):
    """
    Recursively scans DATA_DIR and builds a map from isic_id → full image path.
    Raises if metadata IDs are not found.
    """
    def __init__(self, df, img_dir, transforms=None):
        self.df= df.reset_index(drop=True)
        self.transforms = transforms

        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        id2path = {}
        for root, _, files in os.walk(img_dir):
            for fname in files:
                base, ext = os.path.splitext(fname)
                if ext.lower() in exts:
                    id2path[base] = os.path.join(root, fname)
        self.id2path = id2path

        found = len(self.id2path)
        expected = len(df)
        print(f"[Dataset INIT] scanned {found:,} image files; metadata has {expected:,} entries.")

        missing = set(df.isic_id) - set(self.id2path.keys())
        if missing:
            sample = list(missing)[:10]
            raise RuntimeError(
                f"ERROR: {len(missing):,} metadata IDs not found on disk. "
                f"Examples: {', '.join(sample)}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.id2path[row.isic_id]
        img  = Image.open(path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        label = torch.tensor(row.target, dtype=torch.float32)
        return img, label


def get_data_loaders():
    df = pd.read_csv(CSV_PATH)[['isic_id','patient_id','target']]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, valid_idx = next(gss.split(df, groups=df.patient_id))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    valid_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = ISICDataset(train_df, DATA_DIR, transforms=train_tfms)
    valid_ds = ISICDataset(valid_df, DATA_DIR, transforms=valid_tfms)

    counts = train_df.target.value_counts().sort_index().tolist()
    class_weights = [1.0/c for c in counts]
    sample_weights = [class_weights[int(t)] for t in train_df.target]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    return train_loader, valid_loader


def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 1)

    )
    return model.to(DEVICE)


def train_loop(loader, model, criterion, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    all_preds, all_targs = [], []

    for imgs, labels in tqdm(loader, desc="TRAIN"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * imgs.size(0)

        # detach before storing
        all_preds.append(torch.sigmoid(logits).detach().cpu())
        all_targs.append(labels.detach().cpu())

    preds = torch.cat(all_preds).numpy()
    targs = torch.cat(all_targs).numpy()
    return total_loss/len(loader.dataset), roc_auc_score(targs, preds)


def valid_loop(loader, model, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_targs = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="VALID"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)

            all_preds.append(torch.sigmoid(logits).detach().cpu())
            all_targs.append(labels.detach().cpu())

    preds = torch.cat(all_preds).numpy()
    targs = torch.cat(all_targs).numpy()
    return total_loss/len(loader.dataset), roc_auc_score(targs, preds)


def train():
    train_loader, valid_loader = get_data_loaders()
    model     = build_model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(optimizer, max_lr=LR,
                          steps_per_epoch=len(train_loader), epochs=EPOCHS)

    for epoch in range(1, EPOCHS+1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        tr_loss, tr_auc = train_loop(train_loader, model, criterion, optimizer, scheduler)
        va_loss, va_auc = valid_loop(valid_loader, model, criterion)
        print(
            f"Train Loss: {tr_loss:.4f} | AUC: {tr_auc:.4f}  ||  "
            f"Val Loss: {va_loss:.4f} | AUC: {va_auc:.4f}"
        )

    torch.save(model.state_dict(), "melanoma_resnet50.pth")
    print("\nTraining complete. Model saved to melanoma_resnet50.pth")

    # --- FREE UP MEMORY ---
    print("Clearing Python and GPU caches...")
    # delete big objects
    del model, optimizer, scheduler
    del train_loader, valid_loader
    # run garbage collection
    gc.collect()
    # free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    # On Windows executables (.exe), uncomment next line:
    # from torch.multiprocessing import freeze_support; freeze_support()
    train()
