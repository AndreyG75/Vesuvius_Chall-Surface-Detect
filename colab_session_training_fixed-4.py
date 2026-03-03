"""
COLAB PRO - SESSION-BASED TRAINING (12-HOUR SAFE)
Обучение с автоматической защитой от прерывания сессий

ОСОБЕННОСТИ:
✅ Адаптировано под 12-часовые сессии Colab Pro
✅ Автоматическое сохранение после каждой эпохи
✅ Early stopping для экономии времени
✅ Timeout protection (11ч 40мин)
✅ Resume capability между сессиями

ИСПОЛЬЗОВАНИЕ:
Сессия 1: CURRENT_SESSION = 1, RESUME_FROM = None
Сессия 2: CURRENT_SESSION = 2, RESUME_FROM = "gs://.../checkpoint_epoch5.pt"
Сессия 3: CURRENT_SESSION = 3, RESUME_FROM = "gs://.../checkpoint_epoch10.pt"
Сессия 4: CURRENT_SESSION = 4, RESUME_FROM = "gs://.../checkpoint_epoch15.pt"
"""

# =============================================================================
# 🔥 КОНФИГУРАЦИЯ СЕССИИ - ИЗМЕНЯТЬ ПЕРЕД КАЖДЫМ ЗАПУСКОМ
# =============================================================================

# === ИЗМЕНИТЬ ЭТИ 3 ПАРАМЕТРА ПЕРЕД КАЖДОЙ СЕССИЕЙ ===
CURRENT_SESSION = 1  # 1, 2, 3, или 4
MODEL_SEED = 42      # 42, 123, или 777
RESUME_FROM = None   # None для первой сессии, путь к checkpoint для остальных

# Примеры для следующих сессий:
# Сессия 2: RESUME_FROM = "gs://your-bucket/models/phase2_improved/checkpoint_epoch5_seed42.pt"
# Сессия 3: RESUME_FROM = "gs://your-bucket/models/phase2_improved/checkpoint_epoch10_seed42.pt"
# Сессия 4: RESUME_FROM = "gs://your-bucket/models/phase2_improved/checkpoint_epoch15_seed42.pt"

# === GOOGLE CLOUD НАСТРОЙКИ ===
GCS_BUCKET = "your-bucket-name"  # 🔥 ИЗМЕНИТЬ
GCS_DATA_PATH = ""               # Оставить пустым если данные в корне bucket
PROJECT_ID = "your-project-id"   # 🔥 ИЗМЕНИТЬ

# =============================================================================
# СЕССИОННАЯ КОНФИГУРАЦИЯ
# =============================================================================

SESSION_EPOCHS = {
    1: {'start': 0,  'end': 5},   # Эпохи 1-5
    2: {'start': 5,  'end': 10},  # Эпохи 6-10
    3: {'start': 10, 'end': 15},  # Эпохи 11-15
    4: {'start': 15, 'end': 20},  # Эпохи 16-20
}

session_config = SESSION_EPOCHS[CURRENT_SESSION]
START_EPOCH = session_config['start']
TARGET_EPOCH = session_config['end']
EPOCHS_THIS_SESSION = TARGET_EPOCH - START_EPOCH

# Safety settings
SESSION_TIMEOUT_MINUTES = 700  # 11ч 40мин (оставляем 20мин на сохранение)
ENABLE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 3

print(f"\n{'='*70}")
print(f"🎯 SESSION {CURRENT_SESSION} OF 4")
print(f"{'='*70}")
print(f"Training epochs: {START_EPOCH+1} → {TARGET_EPOCH}")
print(f"Episodes this session: {EPOCHS_THIS_SESSION}")
print(f"Expected duration: ~{EPOCHS_THIS_SESSION * 2.17:.1f} hours")
print(f"Timeout protection: {SESSION_TIMEOUT_MINUTES/60:.1f} hours")
print(f"Resume from: {RESUME_FROM if RESUME_FROM else 'New training'}")
print(f"{'='*70}\n")

# =============================================================================
# CELL 1: Environment Setup
# =============================================================================
import os
import sys
import time

try:
    from google.colab import drive, auth
    IN_COLAB = True
    print("✓ Running in Google Colab")
except:
    IN_COLAB = False
    print("❌ Not in Colab")
    sys.exit(1)

# Засечь время старта сессии
SESSION_START_TIME = time.time()

def get_elapsed_minutes():
    return (time.time() - SESSION_START_TIME) / 60

def check_timeout():
    elapsed = get_elapsed_minutes()
    if elapsed > SESSION_TIMEOUT_MINUTES:
        return True, elapsed
    return False, elapsed

# =============================================================================
# CELL 2: Authentication
# =============================================================================
print("🔐 Authenticating...")
auth.authenticate_user()

!pip install -q gcsfs
from google.cloud import storage
import gcsfs

fs = gcsfs.GCSFileSystem(project=PROJECT_ID)
print("✓ Authenticated")

# =============================================================================
# CELL 3: Install Dependencies
# =============================================================================
print("📦 Installing packages...")
!pip install -q monai
!pip install -q imagecodecs  # Для чтения сжатых TIFF (LZW compression)

# Попробовать установить torch-ema
try:
    !pip install -q torch-ema
    HAS_TORCH_EMA = True
except:
    HAS_TORCH_EMA = False

print("✓ Packages installed")

# =============================================================================
# CELL 4: Imports
# =============================================================================
import gc
import json
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

print("✓ Imports successful")

# =============================================================================
# CELL 5: EMA Implementation
# =============================================================================
if not HAS_TORCH_EMA:
    class SimpleEMA:
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
        def __init__(self, ema, model):
            self.ema = ema
            self.model = model
        
        def __enter__(self):
            self.ema.apply_shadow(self.model)
            return self
        
        def __exit__(self, *args):
            self.ema.restore(self.model)
else:
    from torch_ema import ExponentialMovingAverage
    SimpleEMA = None
    EMAContextManager = None

print("✓ EMA classes ready")

# =============================================================================
# CELL 6: Configuration
# =============================================================================
WORK_DIR = Path("/content/work")
WORK_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = WORK_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

GCS_IMG_DIR = f"gs://{GCS_BUCKET}/{GCS_DATA_PATH}/train_images" if GCS_DATA_PATH else f"gs://{GCS_BUCKET}/train_images"
GCS_MSK_DIR = f"gs://{GCS_BUCKET}/{GCS_DATA_PATH}/train_labels" if GCS_DATA_PATH else f"gs://{GCS_BUCKET}/train_labels"
GCS_OUTPUT_DIR = f"gs://{GCS_BUCKET}/models/phase2_improved"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    'model_seed': MODEL_SEED,
    'split_seed': 42,
    
    'filters': [28, 56, 112, 224, 280],
    'patch_size': [64, 192, 192],
    
    'batch_size': 1,
    'accum_steps': 4,
    'total_epochs': TARGET_EPOCH,  # Целевое количество для сессии
    'n_patches_per_vol': 12,
    
    'base_lr': 5e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-4,
    
    'ema_decay': 0.999,
    
    'surface_focus_prob': 0.70,
    'hard_negative_prob': 0.20,
    'random_prob': 0.10,
    
    'alpha_dice': 0.45,
    'alpha_cldice': 0.35,
    'alpha_boundary': 0.20,
    
    'ds_weights': [0.5, 0.25, 0.15, 0.1],
    
    # Session-specific
    'current_session': CURRENT_SESSION,
    'start_epoch': START_EPOCH,
}

print(f"✓ Configuration ready")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

torch.manual_seed(MODEL_SEED)
np.random.seed(MODEL_SEED)

# =============================================================================
# CELL 7: Topology-Aware Loss
# =============================================================================
def soft_erode(img):
    kernel = torch.ones(1, 1, 3, 3, 3, device=img.device) / 27.0
    return 1.0 - F.conv3d(1.0 - img, kernel, padding=1)

def soft_skeletonize(img, iterations=3):
    for _ in range(iterations):
        img = soft_erode(img)
    return img

class TopologyAwareLoss(nn.Module):
    def __init__(self, alpha_dice=0.45, alpha_cldice=0.35, alpha_boundary=0.20, ds_weights=None):
        super().__init__()
        self.dice_ce = DiceCELoss(to_onehot_y=True, softmax=True)
        self.alpha_dice = alpha_dice
        self.alpha_cldice = alpha_cldice
        self.alpha_boundary = alpha_boundary
        self.ds_weights = ds_weights or [1.0]
        
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
        target_onehot = target.unsqueeze(1) * valid_mask.unsqueeze(1).long()
        dice_ce = self.dice_ce(logits, target_onehot)
        
        if (target == 1).sum() > 100:
            pred_probs = F.softmax(logits, dim=1)
            cldice = self.compute_cldice(pred_probs, target)
            boundary = self.compute_boundary_loss(pred_probs, target)
            
            total = (self.alpha_dice * dice_ce + self.alpha_cldice * cldice + self.alpha_boundary * boundary)
        else:
            total = dice_ce
        
        return total
    
    def forward(self, logits, target, valid_mask):
        if isinstance(logits, list):
            total_loss = 0
            
            for i, output in enumerate(logits):
                weight = self.ds_weights[i] if i < len(self.ds_weights) else 0.1
                
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
                
                loss_i = self._compute_single_loss(output, target_scaled, vm_scaled)
                total_loss += weight * loss_i
            
            return total_loss
        else:
            return self._compute_single_loss(logits, target, valid_mask)

print("✓ TopologyAwareLoss defined")

# =============================================================================
# CELL 8: Dataset
# =============================================================================
def load_tif_from_gcs(gcs_path, fs):
    with fs.open(gcs_path, 'rb') as f:
        return tifffile.imread(f)

def pad_to_shape(arr, shape, value=0):
    D, H, W = arr.shape
    pd, ph, pw = shape
    out = np.full(shape, value, dtype=arr.dtype)
    out[:min(D,pd), :min(H,ph), :min(W,pw)] = arr[:min(D,pd), :min(H,ph), :min(W,pw)]
    return out

class ColabDataset(Dataset):
    def __init__(self, ids, config, gcs_img_dir, gcs_msk_dir, fs, augment=True):
        self.ids = list(ids)
        self.config = config
        self.gcs_img_dir = gcs_img_dir
        self.gcs_msk_dir = gcs_msk_dir
        self.fs = fs
        self.augment = augment
        self._surface_cache = {}
        self.patch_size = tuple(config['patch_size'])
        
        self.surface_prob = config['surface_focus_prob']
        self.hard_neg_prob = config['hard_negative_prob']
        self.random_prob = config['random_prob']
        
        print(f"✓ Dataset initialized: {len(self.ids)} volumes")
    
    def _get_surface_coords(self, vid):
        if vid in self._surface_cache:
            return self._surface_cache[vid]
        
        try:
            gcs_path = f"{self.gcs_msk_dir}/{vid}.tif"
            msk = load_tif_from_gcs(gcs_path, self.fs)
            surf_coords = np.argwhere(msk == 1)
            
            if len(surf_coords) > 100:
                indices = np.random.choice(len(surf_coords), 100, replace=False)
                surf_coords = surf_coords[indices]
            
            self._surface_cache[vid] = surf_coords if len(surf_coords) > 0 else None
            del msk
        except Exception as e:
            self._surface_cache[vid] = None
        
        return self._surface_cache[vid]
    
    def _random_center(self, vol_shape):
        D, H, W = vol_shape
        pd, ph, pw = self.patch_size
        
        z = np.random.randint(pd//2, D - pd//2)
        y = np.random.randint(ph//2, H - ph//2)
        x = np.random.randint(pw//2, W - pw//2)
        
        return np.array([z, y, x])
    
    def _get_patch_center(self, vid, vol_shape):
        rand = np.random.rand()
        surf = self._get_surface_coords(vid)
        
        if rand < self.surface_prob:
            if surf is not None and len(surf) > 0:
                center = surf[np.random.randint(len(surf))]
            else:
                center = self._random_center(vol_shape)
        elif rand < (self.surface_prob + self.hard_neg_prob):
            if surf is not None and len(surf) > 0:
                center = surf[np.random.randint(len(surf))]
                offset = np.random.randint(20, 40, size=3) * np.random.choice([-1, 1], size=3)
                center = center + offset
                D, H, W = vol_shape
                pd, ph, pw = self.patch_size
                center[0] = np.clip(center[0], pd//2, D - pd//2 - 1)
                center[1] = np.clip(center[1], ph//2, H - ph//2 - 1)
                center[2] = np.clip(center[2], pw//2, W - pw//2 - 1)
            else:
                center = self._random_center(vol_shape)
        else:
            center = self._random_center(vol_shape)
        
        return center
    
    def _normalize(self, img):
        p2, p98 = np.percentile(img, [2, 98])
        img = np.clip(img, p2, p98)
        img = (img - p2) / (p98 - p2 + 1e-8)
        return img.astype(np.float32)
    
    def _augment(self, img, msk):
        for axis in [0, 1, 2]:
            if np.random.rand() < 0.5:
                img = np.flip(img, axis).copy()
                msk = np.flip(msk, axis).copy()
        
        if np.random.rand() < 0.5:
            k = np.random.randint(1, 4)
            img = np.rot90(img, k, axes=(1, 2)).copy()
            msk = np.rot90(msk, k, axes=(1, 2)).copy()
        
        if np.random.rand() < 0.3:
            shift = np.random.uniform(-0.1, 0.1)
            img = np.clip(img + shift, 0, 1)
        
        if np.random.rand() < 0.3:
            d, h, w = img.shape
            for _ in range(np.random.randint(3, 8)):
                sz = 8
                z = np.random.randint(0, max(1, d - sz))
                y = np.random.randint(0, max(1, h - sz))
                x = np.random.randint(0, max(1, w - sz))
                img[z:z+sz, y:y+sz, x:x+sz] = 0
        
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1).astype(np.float32)
        
        return img, msk
    
    def __len__(self):
        return len(self.ids) * self.config['n_patches_per_vol']
    
    def __getitem__(self, idx):
        vol_idx = idx // self.config['n_patches_per_vol']
        vid = self.ids[vol_idx]
        
        img_path = f"{self.gcs_img_dir}/{vid}.tif"
        msk_path = f"{self.gcs_msk_dir}/{vid}.tif"
        
        img = load_tif_from_gcs(img_path, self.fs)
        msk = load_tif_from_gcs(msk_path, self.fs)
        
        center = self._get_patch_center(vid, img.shape)
        
        pd, ph, pw = self.patch_size
        z, y, x = center
        
        img_patch = img[z-pd//2:z+pd//2, y-ph//2:y+ph//2, x-pw//2:x+pw//2]
        msk_patch = msk[z-pd//2:z+pd//2, y-ph//2:y+ph//2, x-pw//2:x+pw//2]
        
        if img_patch.shape != self.patch_size:
            img_patch = pad_to_shape(img_patch, self.patch_size, value=0)
            msk_patch = pad_to_shape(msk_patch, self.patch_size, value=0)
        
        img_patch = self._normalize(img_patch)
        
        if self.augment:
            img_patch, msk_patch = self._augment(img_patch, msk_patch)
        
        valid_mask = (msk_patch >= 0).astype(np.uint8)
        msk_patch = np.clip(msk_patch, 0, 1).astype(np.uint8)
        
        img_t = torch.from_numpy(img_patch).unsqueeze(0).float()
        msk_t = torch.from_numpy(msk_patch).long()
        vm_t = torch.from_numpy(valid_mask).bool()
        
        del img, msk, img_patch, msk_patch
        
        return img_t, msk_t, vm_t

print("✓ ColabDataset defined")

# =============================================================================
# CELL 9: Model
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
        deep_supervision=True,
        res_block=True
    )
    return model

print("✓ build_model defined")

# =============================================================================
# CELL 10: Data Discovery
# =============================================================================
print("🔍 Discovering data...")
img_files = fs.glob(f"{GCS_IMG_DIR}/*.tif")
msk_files = fs.glob(f"{GCS_MSK_DIR}/*.tif")

available_imgs = set([Path(f).stem for f in img_files])
available_msks = set([Path(f).stem for f in msk_files])

valid_ids = sorted([int(vid) for vid in available_imgs.intersection(available_msks)])

print(f"✓ Found {len(valid_ids)} valid volumes")

# =============================================================================
# CELL 11: Data Split
# =============================================================================
train_ids = np.array(valid_ids)
np.random.seed(CONFIG['split_seed'])
np.random.shuffle(train_ids)

split = int(0.9 * len(train_ids))
train_vols = train_ids[:split]
val_vols = train_ids[split:]

print(f"Train: {len(train_vols)}, Val: {len(val_vols)}")

# Создать datasets
train_ds = ColabDataset(train_vols, CONFIG, GCS_IMG_DIR, GCS_MSK_DIR, fs=fs, augment=True)
val_ds = ColabDataset(val_vols, CONFIG, GCS_IMG_DIR, GCS_MSK_DIR, fs=fs, augment=False)

# Создать dataloaders
# ВАЖНО: num_workers=0 потому что gcsfs не поддерживает multiprocessing
train_loader = DataLoader(
    train_ds,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=0,  # GCS не fork-safe, используем single process
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0,  # GCS не fork-safe
    pin_memory=True
)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# =============================================================================
# CELL 12: Model, Optimizer, Scheduler, EMA
# =============================================================================
model = build_model(CONFIG).to(DEVICE)
criterion = TopologyAwareLoss(
    alpha_dice=CONFIG['alpha_dice'],
    alpha_cldice=CONFIG['alpha_cldice'],
    alpha_boundary=CONFIG['alpha_boundary'],
    ds_weights=CONFIG['ds_weights']
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

if HAS_TORCH_EMA:
    ema = ExponentialMovingAverage(model.parameters(), decay=CONFIG['ema_decay'])
else:
    ema = SimpleEMA(model, decay=CONFIG['ema_decay'])

print("✓ Model, optimizer, scheduler, EMA ready")

# =============================================================================
# CELL 13: Checkpoint Functions
# =============================================================================
def save_checkpoint(path, epoch, model, opt, scheduler, ema, val_loss, train_loss, config, history):
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': opt.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'ema_state': ema.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
        'config': config,
        'history': history,
    }
    
    # Сохранить локально
    torch.save(checkpoint, path)
    print(f"  💾 Saved: {path.name}")
    
    # Загрузить в GCS
    try:
        gcs_path = f"{GCS_OUTPUT_DIR}/{path.name}"
        with fs.open(gcs_path, 'wb') as f:
            torch.save(checkpoint, f)
        print(f"  ☁️  Uploaded to GCS")
    except Exception as e:
        print(f"  ⚠️ Failed to upload to GCS: {e}")

def load_checkpoint(path, model, opt, scheduler, ema):
    if path.startswith('gs://'):
        with fs.open(path, 'rb') as f:
            checkpoint = torch.load(f, map_location=DEVICE)
    else:
        checkpoint = torch.load(path, map_location=DEVICE)
    
    model.load_state_dict(checkpoint['model_state'])
    opt.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    ema.load_state_dict(checkpoint['ema_state'])
    
    history = checkpoint.get('history', [])
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint['epoch'], checkpoint['val_loss'], history

# =============================================================================
# CELL 14: Validation Function
# =============================================================================
@torch.no_grad()
def validate(model, loader, criterion, max_batches=30):
    model.eval()
    losses = []
    
    if HAS_TORCH_EMA:
        ctx = ema.average_parameters()
    else:
        ctx = EMAContextManager(ema, model)
    
    with ctx:
        for i, (x, y, vm) in enumerate(loader):
            if i >= max_batches:
                break
            
            x, y, vm = x.to(DEVICE), y.to(DEVICE), vm.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                out = model(x)
                if isinstance(out, list):
                    out = out[0]
                loss = criterion(out, y, vm)
            
            losses.append(loss.item())
            del x, y, vm, out, loss
    
    model.train()
    torch.cuda.empty_cache()
    gc.collect()
    
    return np.mean(losses) if losses else 999.0

# =============================================================================
# CELL 15: Resume from Checkpoint
# =============================================================================
start_epoch = START_EPOCH
best_val_loss = 999.0
history = []

if RESUME_FROM is not None:
    print(f"\n{'='*70}")
    print("📂 RESUMING FROM CHECKPOINT")
    print(f"{'='*70}")
    
    start_epoch, best_val_loss, history = load_checkpoint(
        RESUME_FROM, model, opt, scheduler, ema
    )
    
    print(f"Resuming from epoch {start_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Previous history: {len(history)} epochs")
    print(f"{'='*70}\n")

# =============================================================================
# CELL 16: TRAINING LOOP WITH SESSION PROTECTION
# =============================================================================
print(f"\n{'='*70}")
print(f"🚀 SESSION {CURRENT_SESSION} - TRAINING START")
print(f"{'='*70}")
print(f"Target: Epochs {start_epoch+1} → {TARGET_EPOCH}")
print(f"Early stopping: {'Enabled' if ENABLE_EARLY_STOPPING else 'Disabled'}")
print(f"Timeout protection: {SESSION_TIMEOUT_MINUTES/60:.1f} hours")
print(f"{'='*70}\n")

patience_counter = 0
session_interrupted = False

for ep in range(start_epoch, TARGET_EPOCH):
    epoch_start = time.time()
    
    # ⚠️ CHECK TIMEOUT BEFORE STARTING EPOCH
    timeout, elapsed = check_timeout()
    if timeout:
        print(f"\n🚨 SESSION TIMEOUT APPROACHING")
        print(f"Elapsed: {elapsed:.0f}/{SESSION_TIMEOUT_MINUTES} minutes")
        print(f"Stopping before epoch {ep+1} to safely save")
        session_interrupted = True
        break
    
    print(f"\n{'='*70}")
    print(f"EPOCH {ep+1}/{TARGET_EPOCH} (Session {CURRENT_SESSION})")
    print(f"Time elapsed: {elapsed:.0f}/{SESSION_TIMEOUT_MINUTES} min")
    print(f"{'='*70}")
    
    model.train()
    epoch_losses = []
    
    # === TRAINING LOOP ===
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
            ema.update(model)
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        
        epoch_losses.append(loss.item() * CONFIG['accum_steps'])
        del x, y, vm, loss
        
        if (step + 1) % 100 == 0:
            avg_loss = np.mean(epoch_losses[-100:])
            lr = scheduler.get_last_lr()[0]
            print(f"  Step {step+1:04d}/{len(train_loader)} "
                  f"Loss: {avg_loss:.4f} LR: {lr:.2e}")
        
        # ⚠️ PERIODIC TIMEOUT CHECK
        if (step + 1) % 500 == 0:
            timeout, elapsed = check_timeout()
            if timeout:
                print(f"\n🚨 TIMEOUT during epoch {ep+1}")
                print("Saving emergency checkpoint...")
                emergency_path = OUTPUT_DIR / f"emergency_epoch{ep+1}_seed{MODEL_SEED}.pt"
                save_checkpoint(emergency_path, ep+1, model, opt, scheduler, ema,
                              999.0, np.mean(epoch_losses), CONFIG, history)
                session_interrupted = True
                break
            
            gc.collect()
            torch.cuda.empty_cache()
    
    if session_interrupted:
        break
    
    scheduler.step()
    
    # === VALIDATION ===
    val_loss = validate(model, val_loader, criterion)
    avg_train = np.mean(epoch_losses)
    epoch_time = time.time() - epoch_start
    
    print(f"\n✓ Epoch {ep+1} completed in {epoch_time/60:.1f} minutes:")
    print(f"  Train Loss: {avg_train:.4f}")
    print(f"  Val Loss (EMA): {val_loss:.4f}")
    print(f"  Gap: {abs(avg_train - val_loss):.4f}")
    
    # === EARLY STOPPING ===
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        print(f"  🏆 New best val loss!")
        
        best_path = OUTPUT_DIR / f"best_model_seed{MODEL_SEED}.pt"
        save_checkpoint(best_path, ep+1, model, opt, scheduler, ema,
                       val_loss, avg_train, CONFIG, history)
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{EARLY_STOP_PATIENCE})")
        
        if ENABLE_EARLY_STOPPING and patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️ EARLY STOPPING triggered")
            print(f"Val loss hasn't improved for {EARLY_STOP_PATIENCE} epochs")
            break
    
    # === SAVE CHECKPOINT ===
    checkpoint_path = OUTPUT_DIR / f"checkpoint_epoch{ep+1}_seed{MODEL_SEED}.pt"
    save_checkpoint(checkpoint_path, ep+1, model, opt, scheduler, ema,
                   val_loss, avg_train, CONFIG, history)
    
    # === HISTORY ===
    history.append({
        'epoch': ep+1,
        'train_loss': avg_train,
        'val_loss': val_loss,
        'time_minutes': epoch_time / 60,
        'session': CURRENT_SESSION,
    })
    
    # Сохранить историю
    with open(OUTPUT_DIR / f'history_seed{MODEL_SEED}.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    try:
        gcs_history = f"{GCS_OUTPUT_DIR}/history_seed{MODEL_SEED}.json"
        with fs.open(gcs_history, 'w') as f:
            json.dump(history, f, indent=2)
    except:
        pass
    
    gc.collect()
    torch.cuda.empty_cache()

# =============================================================================
# CELL 17: Session Summary
# =============================================================================
total_elapsed = get_elapsed_minutes()

print(f"\n{'='*70}")
print(f"✅ SESSION {CURRENT_SESSION} FINISHED")
print(f"{'='*70}")
print(f"Total time: {total_elapsed/60:.1f} hours")
print(f"Epochs completed: {start_epoch+1} → {ep+1}")
print(f"Best val loss: {best_val_loss:.4f}")

if ep+1 < TARGET_EPOCH:
    print(f"\n⚠️ Session stopped early")
    print(f"Target was epoch {TARGET_EPOCH}, reached {ep+1}")
    if session_interrupted:
        print("Reason: Timeout protection")
    else:
        print("Reason: Early stopping")

if ep+1 < 20:
    next_session = CURRENT_SESSION + 1
    next_checkpoint = f"checkpoint_epoch{ep+1}_seed{MODEL_SEED}.pt"
    print(f"\n📋 NEXT SESSION SETTINGS:")
    print(f"CURRENT_SESSION = {next_session}")
    print(f'RESUME_FROM = "gs://{GCS_BUCKET}/models/phase2_improved/{next_checkpoint}"')
else:
    print(f"\n🎉 ALL TRAINING COMPLETE!")
    print(f"Use best_model_seed{MODEL_SEED}.pt for inference")

print(f"{'='*70}\n")
