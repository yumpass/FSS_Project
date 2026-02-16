# vit_speckle_regressor_torchvision.py
import os
import re
import argparse
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from transformers import ViTModel   # Eğer bu import hata verirse transformers'ı güncelle
from tqdm import tqdm

# ---------------------------
# Dataset (torchvision transforms ile)
# ---------------------------
class SpeckleDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".tiff", ".tif", ".png", ".jpg"))])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        filepath = os.path.join(self.img_dir, filename)

        # label parse: örn "002.2°C" veya "-12.3°C"
        match = re.search(r"([-+]?\d*\.?\d+|[-+]?\d+)C", filename)
        if match:
            label = float(match.group(1))
        else:
            raise ValueError(f"Filename {filename} did not contain a temperature in expected format!")

        # image load
        image = Image.open(filepath).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)  # => tensor (C,H,W) normalized

        return {"pixel_values": image, "labels": torch.tensor(label, dtype=torch.float)}

# ---------------------------
# Model
# ---------------------------
class ViTRegressor(nn.Module):
    def __init__(self, pretrained_model="google/vit-base-patch16-224", hidden_dim=256):
        super(ViTRegressor, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model)
        self.regressor = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, pixel_values):
        # pixel_values: (B, C, H, W)
        outputs = self.vit(pixel_values=pixel_values)  # return dict-like
        # CLS token embedding
        cls_emb = outputs.last_hidden_state[:, 0, :]   # shape (B, hidden_size)
        out = self.regressor(cls_emb).squeeze(-1)     # (B,)
        return out

# ---------------------------
# Training loop
# ---------------------------
def train_one_epoch(model, loader, optim, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train batch"):
        x = batch["pixel_values"].to(device)
        y = batch["labels"].to(device)

        preds = model(x)
        loss = criterion(preds, y)

        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val batch"):
            x = batch["pixel_values"].to(device)
            y = batch["labels"].to(device)
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item()
    return total_loss / len(loader)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT for FSS Temperature Regression - Training")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing specklegram images (e.g., './data/Original dataset/')")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="vit_speckle_regressor.pt", help="Path to save the trained model")
    args = parser.parse_args()

    # 2. Argümanları kullan
    IMG_DIR = args.data_dir
    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = args.lr
    

    # Transform: ViT pretrained ImageNet normalizasyonu
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = SpeckleDataset(IMG_DIR, transform=transform)

    # Train/val split
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2  , random_state=42)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTRegressor().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} — train_loss: {train_loss:.4f} — val_loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "vit_speckle_regressor_combined_Feb9.pt")
    print("Model saved.")
    
    
    

