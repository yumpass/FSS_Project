# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 17:05:27 2025

@author: SADIK
"""

# vit_speckle_regressor_test.py
import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # sessiz mod
import re
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from transformers import ViTModel
from tqdm import tqdm
import numpy as np

# ------------------------------------------------------------
# Dataset class (aynı olmalı)
# ------------------------------------------------------------
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
        match = re.search(r"([-+]?\d*\.?\d+|[-+]?\d+)C", filename)
        if match:
            label = float(match.group(1))
        else:
            raise ValueError(f"Filename {filename} did not contain a temperature in expected format!")
        image = Image.open(filepath).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"pixel_values": image, "labels": torch.tensor(label, dtype=torch.float)}

# ------------------------------------------------------------
# Model (aynı yapı olmalı)
# ------------------------------------------------------------
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
        outputs = self.vit(pixel_values=pixel_values)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        out = self.regressor(cls_emb).squeeze(-1)
        return out

# ------------------------------------------------------------
# Test script
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Esto SOLO se ejecuta si corro el archivo directamente")
    parser = argparse.ArgumentParser(description="ViT for FSS Temperature Regression - Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model .pt file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the (test) data directory")
    
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    IMG_DIR = args.data_dir
    batch_size = args.batch_size
    dataset = SpeckleDataset(IMG_DIR, transform=transform)

    # Train/val split aynı şekilde olmalı
    indices = list(range(len(dataset)))
    _, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_ds = Subset(dataset, val_idx)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTRegressor().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    preds_list, labels_list = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Testing"):
            x = batch["pixel_values"].to(device)
            y = batch["labels"].cpu().numpy()
            preds = model(x).cpu().numpy()
            preds_list.extend(preds)
            labels_list.extend(y)

    preds_array = np.array(preds_list)
    labels_array = np.array(labels_list)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import pearsonr
    
    # =======================
    #  Temel metrikler
    # =======================
    mae = mean_absolute_error(labels_array, preds_array)
    rmse = np.sqrt(mean_squared_error(labels_array, preds_array))
    r2 = r2_score(labels_array, preds_array)
    corr, _ = pearsonr(labels_array, preds_array)
    
    print(f"\n✅ MAE: {mae:.3f} °C")
    print(f"✅ RMSE: {rmse:.3f} °C")
    print(f"✅ R² Score: {r2:.3f}")
    print(f"✅ Pearson Correlation: {corr:.3f}")
    
    # =======================
    #  Grafik 1: Gerçek vs Tahmin (scatter)
    # =======================
    plt.figure(figsize=(6,6))
    plt.scatter(labels_array, preds_array, alpha=0.6, edgecolor='k')
    lims = [min(labels_array.min(), preds_array.min()), max(labels_array.max(), preds_array.max())]
    plt.plot(lims, lims, 'r--', label="Ideal (y = x)")
    plt.xlabel("Real Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("regression_line.svg")
    
    # =======================
    #  Grafik 2: Residual (hata) dağılımı
    # =======================
    residuals = preds_array - labels_array
    plt.figure(figsize=(6,4))
    plt.hist(residuals, bins=30, alpha=0.7, color='gray', edgecolor='black')
    plt.axvline(0, color='r', linestyle='--')
    plt.title("Residual (Hata) Dağılımı")
    plt.xlabel("Tahmin - Gerçek (°C)")
    plt.ylabel("Frekans")
    plt.tight_layout()
    plt.show()
    
    # # =======================
    mean_vals = (labels_array + preds_array) / 2
    diffs = preds_array - labels_array
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    
    plt.figure(figsize=(6,4))
    plt.scatter(mean_vals, diffs, alpha=0.6)
    plt.axhline(mean_diff, color='red', linestyle='--', label=f"Mean diff = {mean_diff:.2f}")
    plt.axhline(mean_diff + 1.96*std_diff, color='gray', linestyle='--', label='±1.96 SD')
    plt.axhline(mean_diff - 1.96*std_diff, color='gray', linestyle='--')
    plt.xlabel('Mean of True & Predicted [°C]')
    plt.ylabel('Difference (Pred - True) [°C]')
    plt.title('Bland–Altman Agreement Plot')
    plt.legend()
    plt.tight_layout()
    plt.savefig("bland_altman.svg")
    
    # =======================
    #  Grafik 4: Residuals vs Gerçek değerler
    # =======================
    plt.figure(figsize=(6,4))
    plt.scatter(labels_array, residuals, alpha=0.6, edgecolor='k')
    plt.axhline(0, color='r', linestyle='--')
    plt.title("Residuals vs Gerçek Değerler")
    plt.xlabel("Gerçek sıcaklık (°C)")
    plt.ylabel("Hata (°C)")
    plt.tight_layout()
    plt.show()
    
    
    # =======================
    #  Grafik 5: Hataların dağılım aralıkları
    # =======================
    import pandas as pd
    df_results = pd.DataFrame({"true": labels_array, "pred": preds_array})
    df_results["abs_err"] = np.abs(df_results["pred"] - df_results["true"])
    
    # --- İYİLEŞTİRME BAŞLANGICI ---
    
    # 1. Temiz, manuel aralıklar tanımlayın (0'dan 120'ye 15'er derece)
    bins = [0, 15, 30, 45, 60, 75, 90, 105, 120]
    
    # 2. Bu aralıklar için temiz metin etiketleri oluşturun
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    # Bu, ["0-15", "15-30", "30-45", ...] şeklinde bir liste oluşturur
    
    # 3. pd.cut içinde bu aralıkları ve etiketleri kullanın
    # 'right=True' varsayılanıdır, yani (0, 15] aralığını temsil eder
    df_results["temp_range"] = pd.cut(df_results["true"], bins=bins, labels=labels, right=True, include_lowest=True)
    
    # 4. Temiz etiketlere göre gruplayın
    bin_means = df_results.groupby("temp_range")["abs_err"].mean()
    
    # --- İYİLEŞTİRME SONU ---
    
    # --- Orijinal plot kodunuz ---
    plt.figure(figsize=(6,4))
    bin_means.plot(kind='bar', figsize=(8,6))
    plt.ylabel("Ortalama Mutlak Hata [°C]")
    plt.xlabel("Sıcaklık Aralığı [°C]")
    
    # Bonus: Uzun etiketlerin üst üste binmemesi için döndürün
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig("error_dist_clean.svg")
    
    
    # =======================
#  Hata (Residual) Dağılım Analizi
# =======================

    import seaborn as sns
    from scipy.stats import norm, shapiro
    residuals = preds_array - labels_array
    
    # Normal dağılım parametreleri
    mu, sigma = norm.fit(residuals)
    
    # Grafik
    plt.figure(figsize=(6,4))
    sns.histplot(residuals, bins=30, kde=True, color='darkblue', stat="density", edgecolor='black')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'r--', linewidth=2, label=f"N({mu:.2f}, {sigma:.2f}²)")
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel("Tahmin Hatası (Pred - True) [°C]")
    plt.ylabel("Yoğunluk")
    plt.title("Tahmin Hatalarının (Residuals) Dağılımı")
    plt.legend()
    plt.tight_layout()
    plt.savefig("residual_dist.svg")
    
    # =======================
    #  Shapiro–Wilk normalite testi
    # =======================
    stat, p_value = shapiro(residuals)
    print(f"Shapiro–Wilk testi: stat={stat:.3f}, p={p_value:.3f}")
    if p_value > 0.05:
        print("✅ Hatalar normal dağılmış olabilir (Gaussian kabul edilebilir).")
    else:
        print("⚠️ Hatalar normal dağılmıyor olabilir (modelde sistematik hata var).")
        


    ##################################################################################   
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import random
    from captum.attr import IntegratedGradients
    from torchvision import transforms
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import re

    model.eval()

    # --- 1. Integrated Gradients Hesaplaması için Hazırlık ---
    ig = IntegratedGradients(model)

    # ViT için standart ImageNet normalizasyon değerleri (tersine çevirmek için)
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # --- 2. 2x2 Grid ve Rastgele Örnekleri Hazırlama ---
    num_rows = 2
    num_cols = 2
    num_samples = num_rows * num_cols

    # Rastgele 4 örnek indeksi seç
    # Her çalıştığında farklı olması için random.seed() kullanmıyorum, ama istersen eklenebilir.
    random_indices = random.sample(range(len(dataset)), num_samples)

    # Görselleştirme için verileri depolama
    original_images = []
    attr_maps = []
    true_labels = []

    print("Calculating Integrated Gradients for 4 random samples...")
    for idx in random_indices:
        data = dataset[idx]
        inputs = data['pixel_values'].unsqueeze(0).to(device)
        label = data['labels'].item()
        baseline = torch.zeros_like(inputs).to(device)

        # IG hesapla
        attributions, delta = ig.attribute(inputs, baselines=baseline, return_convergence_delta=True)
        
        # --- Görüntüyü Ters Normalizasyon Yap ---
        normalized_image = inputs.squeeze(0)
        unnormalized_image = normalized_image * std[:, None, None] + mean[:, None, None]
        original_image_np = np.clip(unnormalized_image[0].cpu().detach().numpy(), 0, 1)
        
        # --- Öznitelik Haritasını Hazırla ---
        attr = attributions.squeeze(0).cpu().detach().numpy()
        attr_map = np.sum(np.abs(attr), axis=0)
        
        # Verileri listeye ekle
        original_images.append(original_image_np)
        attr_maps.append(attr_map)
        true_labels.append(label)

    print("Plotting results...")

    # --- 3. Tüm Grafikler için Tutarlı Renk Skalası ---
    # Tüm haritalardaki min/max değerlere göre bir renk normalizasyonu oluştur
    vmin = min(m.min() for m in attr_maps)
    vmax = max(m.max() for m in attr_maps)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # --- 4. 2x2 Grid Üzerine Çizdirme ---
    # Her alt grafik arasında daha fazla boşluk bırakmak için hspace ve wspace ayarlandı
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12), gridspec_kw={'wspace': 0.1, 'hspace': 0.3})
    fig.suptitle("Integrated Gradients Overlay for Random Samples", fontsize=18, y=0.98) # y=0.98 başlığı biraz daha yukarı taşır

    im = None # colorbar oluşturmak için kullanılacak image object'ini saklamak için

    for i, ax in enumerate(axes.flatten()):
        # 1. Katman: Orijinal görüntü (gri)
        ax.imshow(original_images[i], cmap='gray')
        
        # 2. Katman: Isı haritası (transparan)
        # 'norm=norm' tüm haritaların aynı skalada olmasını sağlar
        im = ax.imshow(attr_maps[i], cmap='hot', alpha=0.65, norm=norm)
        
        ax.set_title(f"Sample (True Temp: {true_labels[i]:.1f}°C)", fontsize=12)
        ax.axis('off')

    # --- 5. Tek, Ortak Colorbar Ekleme ---
    # Colorbar'ı figür seviyesinde eklemek ve konumunu daha iyi kontrol etmek için
    # make_axes_locatable veya fig.add_axes gibi yöntemler yerine basitçe ax= ile figürün sağına yerleştiriyoruz.
    # shrink ve aspect değerleriyle boyutunu ayarlıyoruz.
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label='Attribution Magnitude')


    plt.tight_layout(rect=[0.05, 0.03, 0.9, 0.93]) # Başlık ve colorbar'a yer açmak için rect ayarlandı
    # rect parametresi: [left, bottom, right, top] - bu koordinatların dışına çizim yapılmaz.

    plt.savefig("ig_xai_multi_corrected.svg", dpi=300) # Daha yüksek DPI ile kaydetmek iyi olabilir
    plt.show()