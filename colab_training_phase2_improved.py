"""
COLAB PRO TRAINING - PHASE 2 IMPROVED
Обучение модели на Colab Pro с улучшениями

ОСОБЕННОСТИ COLAB PRO:
- GPU: T4, V100, A100 (зависит от доступности)
- Сессия: 24 часа максимум (но могут прервать раньше)
- RAM: 25-50 GB
- Стратегия: Сохранение каждые 4 эпохи для безопасности

УЛУЧШЕНИЯ В ЭТОЙ ВЕРСИИ:
✅ Анизотропный patch [64, 192, 192]
✅ Deep supervision
✅ EMA (Exponential Moving Average)
✅ Умный патч-сэмплинг (70/20/10)
✅ Расширенные аугментации
✅ Автосохранение каждые 4 эпохи

ИСПОЛЬЗОВАНИЕ:
1. Запустить в Colab Pro
2. Подключить Google Cloud bucket
3. Настроить MODEL_SEED
4. Run All
5. При прерывании - можно продолжить с последнего checkpoint
"""

# =============================================================================
# 🔥 ПАРАМЕТРЫ - ИЗМЕНИТЬ ЗДЕСЬ
# =============================================================================
MODEL_SEED = 42  # 🔥 ИЗМЕНИТЬ: 42, 123, или 777
RESUME_FROM = None  # None для нового обучения, или путь к checkpoint для продолжения

# Google Cloud настройки
GCS_BUCKET = "your-bucket-name"  # 🔥 ИЗМЕНИТЬ: имя вашего bucket
GCS_DATA_PATH = "vesuvius-data"  # 🔥 ИЗМЕНИТЬ: путь к данным в bucket

# =============================================================================
# CELL 1: Environment Setup для Colab
# =============================================================================
import os
import sys

# Проверка что мы в Colab
try:
    from google.colab import drive, auth
    IN_COLAB = True
    print("✓ Running in Google Colab")
except:
    IN_COLAB = False
    print("❌ Not in Colab - script designed for Colab Pro")
    sys.exit(1)

# =============================================================================
# CELL 2: Google Cloud Authentication
# =============================================================================
print("🔐 Authenticating with Google Cloud...")

# Аутентификация
auth.authenticate_user()

# Установка gsutil если нужно
!pip install -q gcsfs

from google.cloud import storage
import gcsfs

# Создаем filesystem для работы с GCS
fs = gcsfs.GCSFileSystem(project='your-project-id')  # 🔥 ИЗМЕНИТЬ на ваш project

print("✓ Google Cloud authenticated")

# =============================================================================
# CELL 3: Install Dependencies
# =============================================================================
print("📦 Installing packages...")

!pip install -q monai
!pip install -q torch-ema

print("✓ Packages installed")

# =============================================================================
# CELL 4: Imports
# =============================================================================
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from monai.networks.nets import DynUNet
from monai.losses import DiceCELoss

try:
    from torch_ema import ExponentialMovingAverage
    HAS_TORCH_EMA = True
except:
    HAS_TORCH_EMA = False
    print("⚠️ torch-ema not available, using custom EMA")

print("✓ Imports successful")

# =============================================================================
# CELL 5: Custom EMA (если torch-ema недоступен)
# =============================================================================
class SimpleEMA:
    """Простая реализация EMA для весов модели"""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
    
    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']

class EMAContextManager:
    """Context manager для использования EMA весов"""
    def __init__(self, ema, model):
        self.ema = ema
        self.model = model
    
    def __enter__(self):
        self.ema.apply_shadow(self.model)
        return self
    
    def __exit__(self, *args):
        self.ema.restore(self.model)

print("✓ EMA classes defined")

# =============================================================================
# CELL 6: Configuration
# =============================================================================
# Локальная директория для работы
WORK_DIR = Path("/content/work")
WORK_DIR.mkdir(exist_ok=True)

# Директория для моделей (будет синхронизироваться с GCS)
OUTPUT_DIR = WORK_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

# GCS пути
GCS_IMG_DIR = f"gs://{GCS_BUCKET}/{GCS_DATA_PATH}/train_images"
GCS_MSK_DIR = f"gs://{GCS_BUCKET}/{GCS_DATA_PATH}/train_labels"
GCS_OUTPUT_DIR = f"gs://{GCS_BUCKET}/models/phase2_improved"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Улучшенная конфигурация
CONFIG = {
    'model_seed': MODEL_SEED,
    'split_seed': 42,
    
    # Архитектура
    'filters': [28, 56, 112, 224, 280],
    'patch_size': [64, 192, 192],  # ✨ Анизотропный
    
    # Обучение
    'batch_size': 1,
    'accum_steps': 4,
    'total_epochs': 20,
    'n_patches_per_vol': 12,  # Можно увеличить до 15-20 если есть время
    
    # Optimizer
    'base_lr': 5e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-4,
    
    # EMA
    'ema_decay': 0.999,  # ✨ Новое
    
    # Патч-сэмплинг (улучшенный)
    'surface_focus_prob': 0.70,   # ✨ 70% - на поверхности
    'hard_negative_prob': 0.20,   # ✨ 20% - рядом с поверхностью
    'random_prob': 0.10,          # ✨ 10% - случайные
    
    # Loss веса
    'alpha_dice': 0.45,
    'alpha_cldice': 0.35,
    'alpha_boundary': 0.20,
    
    # Deep supervision веса
    'ds_weights': [0.5, 0.25, 0.15, 0.1],  # ✨ Новое
    
    # Checkpoint settings
    'save_every_n_epochs': 4,  # Сохранять каждые 4 эпохи
}

print(f"\n{'='*70}")
print(f"CONFIGURATION")
print(f"{'='*70}")
print(f"Model seed: {MODEL_SEED}")
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"Patch size: {CONFIG['patch_size']} (anisotropic)")
print(f"GCS bucket: {GCS_BUCKET}")
print(f"{'='*70}\n")

torch.manual_seed(MODEL_SEED)
np.random.seed(MODEL_SEED)

# =============================================================================
# CELL 7: Data Discovery from GCS
# =============================================================================
print("🔍 Discovering data files on GCS...")

# Получить список файлов из GCS
img_files = fs.glob(f"{GCS_IMG_DIR}/*.tif")
msk_files = fs.glob(f"{GCS_MSK_DIR}/*.tif")

# Извлечь ID
available_imgs = set([Path(f).stem for f in img_files])
available_msks = set([Path(f).stem for f in msk_files])

print(f"Found {len(available_imgs)} image files")
print(f"Found {len(available_msks)} mask files")

# Найти пересечение
valid_ids = available_imgs.intersection(available_msks)
valid_ids = sorted([int(vid) for vid in valid_ids])

print(f"\n✓ Valid paired files: {len(valid_ids)}")

if len(valid_ids) == 0:
    raise FileNotFoundError("No training data found in GCS bucket!")

print(f"Sample IDs: {valid_ids[:5]}...")

# =============================================================================
# CELL 8: Topology-Aware Loss (с Deep Supervision)
# =============================================================================
def soft_erode(img):
    kernel = torch.ones(1, 1, 3, 3, 3, device=img.device) / 27.0
    return 1.0 - F.conv3d(1.0 - img, kernel, padding=1)

def soft_skeletonize(img, iterations=3):
    for _ in range(iterations):
        img = soft_erode(img)
    return img

class TopologyAwareLoss(nn.Module):
    def __init__(self, alpha_dice=0.45, alpha_cldice=0.35, alpha_boundary=0.20,
                 ds_weights=None):
        super().__init__()
        self.dice_ce = DiceCELoss(to_onehot_y=True, softmax=True)
        self.alpha_dice = alpha_dice
        self.alpha_cldice = alpha_cldice
        self.alpha_boundary = alpha_boundary
        self.ds_weights = ds_weights or [1.0]  # Веса для deep supervision
        
    def compute_cldice(self, pred_probs, target):
        pred = pred_probs[:, 1:2]
        target_binary = (target == 1).float().unsqueeze(1)
        
        pred_skel = soft_skeletonize(pred, iterations=3)
        target_skel = soft_skeletonize(target_binary, iterations=3)
        
        tprec = (pred_skel * target_binary).sum() / (pred_skel.sum() + 1e-8)
        tsens = (target_skel * pred).sum() / (target_skel.sum() + 1e-8)
        
        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + 1e-8)
        return 1.0 - cl_dice
    
    def compute_boundary_loss(self, pred_probs, target):
        pred = pred_probs[:, 1:2]
        target_binary = (target == 1).float().unsqueeze(1)
        
        pred_boundary = pred - soft_erode(pred)
        target_boundary = target_binary - soft_erode(target_binary)
        
        return F.mse_loss(pred_boundary, target_boundary)
    
    def _compute_single_loss(self, logits, target, valid_mask):
        """Вычислить loss для одного выхода"""
        target_onehot = target.unsqueeze(1) * valid_mask.unsqueeze(1).long()
        dice_ce = self.dice_ce(logits, target_onehot)
        
        if (target == 1).sum() > 100:
            pred_probs = F.softmax(logits, dim=1)
            cldice = self.compute_cldice(pred_probs, target)
            boundary = self.compute_boundary_loss(pred_probs, target)
            
            total = (self.alpha_dice * dice_ce + 
                     self.alpha_cldice * cldice + 
                     self.alpha_boundary * boundary)
        else:
            total = dice_ce
        
        return total
    
    def forward(self, logits, target, valid_mask):
        # ✨ DEEP SUPERVISION SUPPORT
        if isinstance(logits, list):
            # Несколько выходов от deep supervision
            total_loss = 0
            
            for i, output in enumerate(logits):
                weight = self.ds_weights[i] if i < len(self.ds_weights) else 0.1
                
                # Масштабировать target если нужно
                if output.shape[2:] != logits[0].shape[2:]:
                    target_scaled = F.interpolate(
                        target.float().unsqueeze(1), 
                        size=output.shape[2:], 
                        mode='nearest'
                    ).long().squeeze(1)
                    vm_scaled = F.interpolate(
                        valid_mask.float().unsqueeze(1),
                        size=output.shape[2:],
                        mode='nearest'
                    ).bool().squeeze(1)
                else:
                    target_scaled = target
                    vm_scaled = valid_mask
                
                # Вычислить loss для этого уровня
                loss_i = self._compute_single_loss(output, target_scaled, vm_scaled)
                total_loss += weight * loss_i
            
            return total_loss
        else:
            # Один выход (deep supervision отключен)
            return self._compute_single_loss(logits, target, valid_mask)

print("✓ Loss defined (with Deep Supervision support)")

# =============================================================================
# CELL 9: Dataset (с улучшенным сэмплингом)
# =============================================================================
def load_tif_from_gcs(gcs_path):
    """Загрузить .tif файл из GCS"""
    with fs.open(gcs_path, 'rb') as f:
        return tifffile.imread(f)

def pad_to_shape(arr, shape, value=0):
    D, H, W = arr.shape
    pd, ph, pw = shape
    out = np.full(shape, value, dtype=arr.dtype)
    out[:min(D,pd), :min(H,ph), :min(W,pw)] = arr[:min(D,pd), :min(H,ph), :min(W,pw)]
    return out

class ColabDataset(Dataset):
    def __init__(self, ids, config, gcs_img_dir, gcs_msk_dir, augment=True):
        self.ids = list(ids)
        self.config = config
        self.gcs_img_dir = gcs_img_dir
        self.gcs_msk_dir = gcs_msk_dir
        self.augment = augment
        self._surface_cache = {}
        self.patch_size = tuple(config['patch_size'])
        
        # Стратегия сэмплинга
        self.surface_prob = config['surface_focus_prob']
        self.hard_neg_prob = config['hard_negative_prob']
        self.random_prob = config['random_prob']
        
        print(f"✓ Dataset initialized: {len(self.ids)} volumes")
        print(f"  Sampling strategy: {self.surface_prob:.0%} surface, "
              f"{self.hard_neg_prob:.0%} hard-neg, {self.random_prob:.0%} random")
    
    def _get_surface_coords(self, vid):
        """Получить координаты поверхности (кэшируются)"""
        if vid in self._surface_cache:
            return self._surface_cache[vid]
        
        try:
            gcs_path = f"{self.gcs_msk_dir}/{vid}.tif"
            msk = load_tif_from_gcs(gcs_path)
            surf_coords = np.argwhere(msk == 1)
            
            # Сэмплировать для экономии памяти
            if len(surf_coords) > 100:
                indices = np.random.choice(len(surf_coords), 100, replace=False)
                surf_coords = surf_coords[indices]
            
            self._surface_cache[vid] = surf_coords if len(surf_coords) > 0 else None
            del msk
        except Exception as e:
            print(f"⚠️ Warning: Could not load surface for {vid}: {e}")
            self._surface_cache[vid] = None
        
        return self._surface_cache[vid]
    
    def _random_center(self, vol_shape):
        """Случайный центр патча"""
        D, H, W = vol_shape
        pd, ph, pw = self.patch_size
        
        z = np.random.randint(pd//2, D - pd//2)
        y = np.random.randint(ph//2, H - ph//2)
        x = np.random.randint(pw//2, W - pw//2)
        
        return np.array([z, y, x])
    
    def _get_patch_center(self, vid, vol_shape):
        """
        ✨ УЛУЧШЕННЫЙ СЭМПЛИНГ:
        70% - на поверхности
        20% - рядом с поверхностью (hard negatives)
        10% - случайные
        """
        rand = np.random.rand()
        surf = self._get_surface_coords(vid)
        
        if rand < self.surface_prob:
            # 70% - surface patches
            if surf is not None and len(surf) > 0:
                center = surf[np.random.randint(len(surf))]
            else:
                center = self._random_center(vol_shape)
        
        elif rand < (self.surface_prob + self.hard_neg_prob):
            # 20% - hard negatives (near surface)
            if surf is not None and len(surf) > 0:
                center = surf[np.random.randint(len(surf))]
                # Смещаем на 20-40 вокселей
                offset = np.random.randint(20, 40, size=3) * np.random.choice([-1, 1], size=3)
                center = center + offset
                # Clip to valid range
                D, H, W = vol_shape
                pd, ph, pw = self.patch_size
                center[0] = np.clip(center[0], pd//2, D - pd//2 - 1)
                center[1] = np.clip(center[1], ph//2, H - ph//2 - 1)
                center[2] = np.clip(center[2], pw//2, W - pw//2 - 1)
            else:
                center = self._random_center(vol_shape)
        
        else:
            # 10% - random patches
            center = self._random_center(vol_shape)
        
        return center
    
    def _normalize(self, img):
        p2, p98 = np.percentile(img, [2, 98])
        img = np.clip(img, p2, p98)
        img = (img - p2) / (p98 - p2 + 1e-8)
        return img.astype(np.float32)
    
    def _augment(self, img, msk):
        """✨ РАСШИРЕННЫЕ АУГМЕНТАЦИИ"""
        # Базовые flips
        for axis in [0, 1, 2]:
            if np.random.rand() < 0.5:
                img = np.flip(img, axis).copy()
                msk = np.flip(msk, axis).copy()
        
        # Rotation 90° в X-Y плоскости
        if np.random.rand() < 0.5:
            k = np.random.randint(1, 4)
            img = np.rot90(img, k, axes=(1, 2)).copy()
            msk = np.rot90(msk, k, axes=(1, 2)).copy()
        
        # ✨ Histogram shift (эмулирует разное оборудование)
        if np.random.rand() < 0.3:
            shift = np.random.uniform(-0.1, 0.1)
            img = np.clip(img + shift, 0, 1)
        
        # ✨ Coarse dropout (эмулирует артефакты сканирования)
        if np.random.rand() < 0.3:
            d, h, w = img.shape
            for _ in range(np.random.randint(3, 8)):
                sz = 8
                z = np.random.randint(0, max(1, d - sz))
                y = np.random.randint(0, max(1, h - sz))
                x = np.random.randint(0, max(1, w - sz))
                img[z:z+sz, y:y+sz, x:x+sz] = 0
        
        # ✨ Additive noise
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1).astype(np.float32)
        
        return img, msk
    
    def __len__(self):
        return len(self.ids) * self.config['n_patches_per_vol']
    
    def __getitem__(self, idx):
        vol_idx = idx // self.config['n_patches_per_vol']
        vid = self.ids[vol_idx]
        
        # Загрузить из GCS
        img_path = f"{self.gcs_img_dir}/{vid}.tif"
        msk_path = f"{self.gcs_msk_dir}/{vid}.tif"
        
        img = load_tif_from_gcs(img_path)
        msk = load_tif_from_gcs(msk_path)
        
        # Получить центр патча (улучшенный сэмплинг)
        center = self._get_patch_center(vid, img.shape)
        
        # Извлечь патч
        pd, ph, pw = self.patch_size
        z, y, x = center
        
        img_patch = img[z-pd//2:z+pd//2, y-ph//2:y+ph//2, x-pw//2:x+pw//2]
        msk_patch = msk[z-pd//2:z+pd//2, y-ph//2:y+ph//2, x-pw//2:x+pw//2]
        
        # Pad если нужно
        if img_patch.shape != self.patch_size:
            img_patch = pad_to_shape(img_patch, self.patch_size, value=0)
            msk_patch = pad_to_shape(msk_patch, self.patch_size, value=0)
        
        # Нормализация
        img_patch = self._normalize(img_patch)
        
        # Аугментации
        if self.augment:
            img_patch, msk_patch = self._augment(img_patch, msk_patch)
        
        # Valid mask
        valid_mask = (msk_patch >= 0).astype(np.uint8)
        msk_patch = np.clip(msk_patch, 0, 1).astype(np.uint8)
        
        # To tensors
        img_t = torch.from_numpy(img_patch).unsqueeze(0).float()
        msk_t = torch.from_numpy(msk_patch).long()
        vm_t = torch.from_numpy(valid_mask).bool()
        
        del img, msk, img_patch, msk_patch
        
        return img_t, msk_t, vm_t

print("✓ Dataset defined (with improved sampling)")

# =============================================================================
# CELL 10: Model
# =============================================================================
def build_model(config):
    kernels = [[3,3,3]] * 5
    strides = [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2]]
    
    model = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        filters=config['filters'],
        dropout=0.1,
        norm_name="instance",
        deep_supervision=True,  # ✨ ВКЛЮЧЕНО!
        res_block=True
    )
    return model

print("✓ Model builder defined (Deep Supervision enabled)")

# =============================================================================
# CELL 11: Data Preparation
# =============================================================================
train_ids = np.array(valid_ids)

# Split
np.random.seed(CONFIG['split_seed'])
np.random.shuffle(train_ids)
split = int(0.9 * len(train_ids))
train_vols = train_ids[:split]
val_vols = train_ids[split:]

print(f"\n{'='*70}")
print("DATA SPLIT")
print(f"{'='*70}")
print(f"Total valid volumes: {len(train_ids)}")
print(f"Train volumes: {len(train_vols)}")
print(f"Val volumes: {len(val_vols)}")

# Datasets
train_ds = ColabDataset(train_vols, CONFIG, GCS_IMG_DIR, GCS_MSK_DIR, augment=True)
val_ds = ColabDataset(val_vols, CONFIG, GCS_IMG_DIR, GCS_MSK_DIR, augment=False)

# Loaders
train_loader = DataLoader(
    train_ds,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"{'='*70}\n")

# =============================================================================
# CELL 12: Model, Optimizer, Scheduler, EMA
# =============================================================================
model = build_model(CONFIG).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params/1e6:.1f}M")

criterion = TopologyAwareLoss(
    alpha_dice=CONFIG['alpha_dice'],
    alpha_cldice=CONFIG['alpha_cldice'],
    alpha_boundary=CONFIG['alpha_boundary'],
    ds_weights=CONFIG['ds_weights']  # ✨ Deep supervision веса
)

opt = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['base_lr'],
    weight_decay=CONFIG['weight_decay']
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt,
    T_max=CONFIG['total_epochs'],
    eta_min=CONFIG['min_lr']
)

scaler = torch.amp.GradScaler("cuda", enabled=True)

# ✨ EMA SETUP
if HAS_TORCH_EMA:
    ema = ExponentialMovingAverage(model.parameters(), decay=CONFIG['ema_decay'])
    print(f"✓ EMA enabled (torch-ema, decay={CONFIG['ema_decay']})")
else:
    ema = SimpleEMA(model, decay=CONFIG['ema_decay'])
    print(f"✓ EMA enabled (custom, decay={CONFIG['ema_decay']})")

print("✓ Model, optimizer, scheduler, EMA ready")

# =============================================================================
# CELL 13: Checkpoint Management
# =============================================================================
def save_checkpoint(path, epoch, model, opt, scheduler, ema, val_loss, train_loss, config):
    """Сохранить checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': opt.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'ema_state': ema.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
        'config': config,
    }
    torch.save(checkpoint, path)
    print(f"  💾 Checkpoint saved: {path.name}")
    
    # Также загрузить на GCS
    gcs_path = f"{GCS_OUTPUT_DIR}/{path.name}"
    with fs.open(gcs_path, 'wb') as f:
        torch.save(checkpoint, f)
    print(f"  ☁️  Uploaded to GCS: {gcs_path}")

def load_checkpoint(path, model, opt, scheduler, ema):
    """Загрузить checkpoint"""
    if path.startswith('gs://'):
        # Загрузить из GCS
        with fs.open(path, 'rb') as f:
            checkpoint = torch.load(f, map_location=DEVICE)
    else:
        checkpoint = torch.load(path, map_location=DEVICE)
    
    model.load_state_dict(checkpoint['model_state'])
    opt.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    ema.load_state_dict(checkpoint['ema_state'])
    
    print(f"✓ Checkpoint loaded from epoch {checkpoint['epoch']}")
    return checkpoint['epoch'], checkpoint['val_loss']

# =============================================================================
# CELL 14: Validation Function
# =============================================================================
@torch.no_grad()
def validate(model, loader, criterion, max_batches=30, use_ema=True):
    """Валидация с опциональным использованием EMA весов"""
    model.eval()
    losses = []
    
    # Контекст для EMA
    if use_ema:
        if HAS_TORCH_EMA:
            ctx = ema.average_parameters()
        else:
            ctx = EMAContextManager(ema, model)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()
    
    with ctx:
        for i, (x, y, vm) in enumerate(loader):
            if i >= max_batches:
                break
            
            x, y, vm = x.to(DEVICE), y.to(DEVICE), vm.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                out = model(x)
                # Deep supervision возвращает список, берем финальный выход
                if isinstance(out, list):
                    out = out[0]
                loss = criterion(out, y, vm)
            
            losses.append(loss.item())
            del x, y, vm, out, loss
    
    model.train()
    torch.cuda.empty_cache()
    gc.collect()
    
    return np.mean(losses) if losses else 999.0

print("✓ Validation function defined")

# =============================================================================
# CELL 15: Resume from Checkpoint (если нужно)
# =============================================================================
start_epoch = 0
best_val_loss = 999.0
history = []

if RESUME_FROM is not None:
    print(f"\n{'='*70}")
    print(f"RESUMING FROM CHECKPOINT")
    print(f"{'='*70}")
    print(f"Loading: {RESUME_FROM}")
    
    start_epoch, best_val_loss = load_checkpoint(
        RESUME_FROM, model, opt, scheduler, ema
    )
    
    print(f"Resuming from epoch {start_epoch}")
    print(f"Best val loss so far: {best_val_loss:.4f}")
    print(f"{'='*70}\n")

# =============================================================================
# CELL 16: TRAINING LOOP
# =============================================================================
print(f"\n{'='*70}")
print(f"🚀 TRAINING START - SEED {MODEL_SEED}")
print(f"Total epochs: {CONFIG['total_epochs']} (starting from {start_epoch})")
print(f"Checkpoints every {CONFIG['save_every_n_epochs']} epochs")
print(f"{'='*70}\n")

for ep in range(start_epoch, CONFIG['total_epochs']):
    epoch_start = time.time()
    
    print(f"\n{'='*70}")
    print(f"EPOCH {ep+1}/{CONFIG['total_epochs']} (Seed {MODEL_SEED})")
    print(f"{'='*70}")
    
    model.train()
    epoch_losses = []
    
    for step, (x, y, vm) in enumerate(train_loader):
        x, y, vm = x.to(DEVICE), y.to(DEVICE), vm.to(DEVICE)
        
        with torch.amp.autocast('cuda'):
            out = model(x)            
            loss = criterion(out, y, vm) / CONFIG['accum_steps']
        
        scaler.scale(loss).backward()
        del out
        
        if (step + 1) % CONFIG['accum_steps'] == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            
            # ✨ EMA UPDATE
            ema.update(model)
            
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        
        epoch_losses.append(loss.item() * CONFIG['accum_steps'])
        del x, y, vm, loss
        
        if (step + 1) % 100 == 0:
            avg_loss = np.mean(epoch_losses[-100:])
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - epoch_start
            print(f"  Step {step+1:04d}/{len(train_loader)} "
                  f"Loss: {avg_loss:.4f} LR: {lr:.2e} Time: {elapsed/60:.1f}m")
        
        if (step + 1) % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    scheduler.step()
    
    # Validation (с EMA весами)
    val_loss = validate(model, val_loader, criterion, use_ema=True)
    avg_train = np.mean(epoch_losses)
    
    epoch_time = time.time() - epoch_start
    
    print(f"\n✓ Epoch {ep+1} completed in {epoch_time/60:.1f} minutes:")
    print(f"  Train Loss: {avg_train:.4f}")
    print(f"  Val Loss (EMA): {val_loss:.4f}")
    print(f"  Gap: {abs(avg_train - val_loss):.4f}")
    
    # Сохранить checkpoint каждые N эпох
    if (ep + 1) % CONFIG['save_every_n_epochs'] == 0:
        checkpoint_path = OUTPUT_DIR / f"checkpoint_epoch{ep+1}_seed{MODEL_SEED}.pt"
        save_checkpoint(
            checkpoint_path, ep + 1, model, opt, scheduler, ema,
            val_loss, avg_train, CONFIG
        )
    
    # Сохранить лучшую модель
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_path = OUTPUT_DIR / f"best_model_seed{MODEL_SEED}.pt"
        save_checkpoint(
            best_path, ep + 1, model, opt, scheduler, ema,
            val_loss, avg_train, CONFIG
        )
        print(f"  🏆 New best val loss! Saved to {best_path.name}")
    
    history.append({
        'epoch': ep + 1,
        'train_loss': avg_train,
        'val_loss': val_loss,
        'time_minutes': epoch_time / 60,
    })
    
    # Сохранить историю
    with open(OUTPUT_DIR / f'history_seed{MODEL_SEED}.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"{'-'*70}")
    gc.collect()
    torch.cuda.empty_cache()

# =============================================================================
# CELL 17: Save Final Model
# =============================================================================
final_model_path = OUTPUT_DIR / f"final_model_seed{MODEL_SEED}.pt"

save_checkpoint(
    final_model_path, CONFIG['total_epochs'], model, opt, scheduler, ema,
    best_val_loss, avg_train, CONFIG
)

print(f"\n{'='*70}")
print(f"✅ TRAINING COMPLETED")
print(f"{'='*70}")
print(f"Best val loss: {best_val_loss:.4f}")
print(f"Final model: {final_model_path}")
print(f"File size: {final_model_path.stat().st_size / 1e6:.1f} MB")
print(f"\nAll checkpoints uploaded to: {GCS_OUTPUT_DIR}")
print(f"{'='*70}\n")

print("📋 Next steps:")
print("1. Download best model from GCS")
print("2. Run inference on Kaggle")
print("3. Repeat for other seeds (123, 777) if results are good")
print("\n🎉 Expected score with improvements: 0.60-0.70")
