"""
KAGGLE INFERENCE - PHASE 2 IMPROVED
Инференс с улучшенной моделью + постпроцессинг

ИСПОЛЬЗОВАНИЕ:
1. Создать Kaggle Notebook
2. Add Data: 
   - vesuvius-challenge-surface-detection (test data)
   - your-models-dataset (модели с Colab)
3. Settings: GPU P100
4. Настроить пути к моделям
5. Run All

ОСОБЕННОСТИ:
✅ Sliding window с overlap
✅ TTA (Test Time Augmentation)
✅ Topology-aware постпроцессинг
✅ Ансамбль (опционально)
"""

# =============================================================================
# 🔥 ПАРАМЕТРЫ - НАСТРОИТЬ ЗДЕСЬ
# =============================================================================
# Пути к моделям (загруженным из Colab)
MODEL_PATHS = [
    "/kaggle/input/your-models-dataset/best_model_seed42.pt",
    # "/kaggle/input/your-models-dataset/best_model_seed123.pt",  # Раскомментировать для ансамбля
    # "/kaggle/input/your-models-dataset/best_model_seed777.pt",
]

# Inference настройки
INFERENCE_CONFIG = {
    'overlap': 0.5,  # Overlap для sliding window (0.5 = 50%)
    'use_tta': True,  # Test Time Augmentation
    'tta_flips': ['x', 'y'],  # Какие оси флипать
    
    # Постпроцессинг
    'smooth_sigma': 1.0,  # Сглаживание вероятностей
    'threshold_high': 0.70,  # Высокий порог (core)
    'threshold_low': 0.45,   # Низкий порог (candidates)
    'bridge_erosion': 1,     # Радиус эрозии для bridge-killing
    'closing_radius': 1,      # Радиус closing для заполнения разрывов
    'min_component_size': 8000,  # Минимальный размер компонента
}

# =============================================================================
# CELL 1: Imports and Setup
# =============================================================================
import os
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from skimage import morphology, measure
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import DynUNet
from monai.inferers import sliding_window_inference

print("✓ Imports successful")

# =============================================================================
# CELL 2: Environment
# =============================================================================
DATA_DIR = Path("/kaggle/input/vesuvius-challenge-surface-detection")
TEST_IMG_DIR = DATA_DIR / "test_images"
OUTPUT_DIR = Path("/kaggle/working")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n{'='*70}")
print("INFERENCE CONFIGURATION")
print(f"{'='*70}")
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"Models to ensemble: {len(MODEL_PATHS)}")
print(f"TTA enabled: {INFERENCE_CONFIG['use_tta']}")
print(f"Sliding window overlap: {INFERENCE_CONFIG['overlap']}")
print(f"{'='*70}\n")

# =============================================================================
# CELL 3: Model Definition
# =============================================================================
def build_model(config):
    """Та же архитектура что и при обучении"""
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
        deep_supervision=True,  # Было включено при обучении
        res_block=True
    )
    return model

def load_model_with_ema(model_path, device):
    """Загрузить модель с EMA весами"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = build_model(config).to(device)
    
    # Попытаться загрузить EMA веса
    if 'ema_state' in checkpoint:
        print(f"  Loading EMA weights from {Path(model_path).name}")
        ema_state = checkpoint['ema_state']
        
        # Применить EMA веса к модели
        if 'shadow' in ema_state:
            # Custom EMA format
            state_dict = model.state_dict()
            for name, param in model.named_parameters():
                if name in ema_state['shadow']:
                    state_dict[name] = ema_state['shadow'][name]
            model.load_state_dict(state_dict)
        else:
            # torch-ema format - уже в правильном формате
            model.load_state_dict(checkpoint['model_state'])
    else:
        print(f"  Loading regular weights from {Path(model_path).name}")
        model.load_state_dict(checkpoint['model_state'])
    
    model.eval()
    return model, config

print("✓ Model functions defined")

# =============================================================================
# CELL 4: Load Models
# =============================================================================
models = []
configs = []

print("\n📦 Loading models...")
for i, path in enumerate(MODEL_PATHS):
    print(f"\nModel {i+1}/{len(MODEL_PATHS)}: {Path(path).name}")
    model, config = load_model_with_ema(path, DEVICE)
    models.append(model)
    configs.append(config)
    print(f"  Config: patch_size={config['patch_size']}, filters={config['filters']}")

print(f"\n✓ {len(models)} model(s) loaded\n")

# =============================================================================
# CELL 5: Data Processing Functions
# =============================================================================
def normalize_volume(vol):
    """Нормализация как при обучении"""
    p2, p98 = np.percentile(vol, [2, 98])
    vol = np.clip(vol, p2, p98)
    vol = (vol - p2) / (p98 - p2 + 1e-8)
    return vol.astype(np.float32)

def pad_to_multiple(vol, multiple=16):
    """Pad volume чтобы размеры были кратны multiple"""
    D, H, W = vol.shape
    
    new_D = int(np.ceil(D / multiple) * multiple)
    new_H = int(np.ceil(H / multiple) * multiple)
    new_W = int(np.ceil(W / multiple) * multiple)
    
    if (new_D, new_H, new_W) == (D, H, W):
        return vol, (0, 0, 0)
    
    padded = np.zeros((new_D, new_H, new_W), dtype=vol.dtype)
    padded[:D, :H, :W] = vol
    
    return padded, (new_D - D, new_H - H, new_W - W)

def unpad(vol, padding):
    """Убрать padding"""
    pad_d, pad_h, pad_w = padding
    D, H, W = vol.shape
    
    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return vol
    
    return vol[:D-pad_d if pad_d > 0 else D, 
               :H-pad_h if pad_h > 0 else H, 
               :W-pad_w if pad_w > 0 else W]

print("✓ Processing functions defined")

# =============================================================================
# CELL 6: Sliding Window Inference
# =============================================================================
@torch.no_grad()
def predict_volume_sliding(volume, model, config, overlap=0.5, use_tta=False):
    """
    Sliding window inference с optional TTA
    
    Returns: probability map (float32, shape=volume.shape)
    """
    # Normalize
    volume = normalize_volume(volume)
    
    # Pad to multiple of 16
    volume_padded, padding = pad_to_multiple(volume, multiple=16)
    
    # To tensor
    vol_tensor = torch.from_numpy(volume_padded).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    
    # Patch size from config
    patch_size = config['patch_size']
    
    # Базовый predict
    def predict_once(vol_t):
        # Sliding window с gaussian weights
        probs = sliding_window_inference(
            inputs=vol_t,
            roi_size=patch_size,
            sw_batch_size=1,
            predictor=lambda x: model(x)[0] if isinstance(model(x), list) else model(x),
            overlap=overlap,
            mode='gaussian',
            device=DEVICE
        )
        
        # Softmax для вероятностей
        probs = F.softmax(probs, dim=1)
        return probs[:, 1:2]  # Класс 1 (поверхность)
    
    # Базовый predict
    prob_map = predict_once(vol_tensor)
    
    # TTA (Test Time Augmentation)
    if use_tta and INFERENCE_CONFIG['use_tta']:
        tta_count = 1
        
        for flip_axis in INFERENCE_CONFIG['tta_flips']:
            if flip_axis == 'x':
                axis = 4  # W axis в tensor (B, C, D, H, W)
            elif flip_axis == 'y':
                axis = 3  # H axis
            else:
                continue
            
            # Flip, predict, flip back
            vol_flipped = torch.flip(vol_tensor, dims=[axis])
            prob_flipped = predict_once(vol_flipped)
            prob_flipped = torch.flip(prob_flipped, dims=[axis])
            
            prob_map += prob_flipped
            tta_count += 1
        
        # Average
        prob_map = prob_map / tta_count
    
    # To numpy
    prob_map = prob_map.squeeze().cpu().numpy()
    
    # Unpad
    prob_map = unpad(prob_map, padding)
    
    return prob_map

print("✓ Sliding window inference defined")

# =============================================================================
# CELL 7: Ensemble Prediction
# =============================================================================
@torch.no_grad()
def predict_volume_ensemble(volume, models, configs, overlap=0.5, use_tta=False):
    """
    Ансамбль предсказаний от нескольких моделей
    
    Returns: averaged probability map
    """
    prob_maps = []
    
    for i, (model, config) in enumerate(zip(models, configs)):
        print(f"  Model {i+1}/{len(models)}...", end=' ')
        
        prob = predict_volume_sliding(
            volume, model, config,
            overlap=overlap,
            use_tta=use_tta
        )
        
        prob_maps.append(prob)
        print(f"Done (shape: {prob.shape})")
        
        # Очистить кэш
        torch.cuda.empty_cache()
        gc.collect()
    
    # Усреднить
    ensemble_prob = np.mean(prob_maps, axis=0)
    
    del prob_maps
    gc.collect()
    
    return ensemble_prob

print("✓ Ensemble prediction defined")

# =============================================================================
# CELL 8: Postprocessing Functions
# =============================================================================
def smooth_probabilities(prob_map, sigma=1.0):
    """Сглаживание вероятностной карты"""
    if sigma > 0:
        return ndimage.gaussian_filter(prob_map, sigma=sigma)
    return prob_map

def dual_threshold_segmentation(prob_map, threshold_high, threshold_low):
    """
    Двухпороговая сегментация:
    1. Высокий порог → надежное "ядро"
    2. Низкий порог → кандидаты
    3. Grow от ядра внутри кандидатов
    """
    # Core (высокий порог)
    core = prob_map >= threshold_high
    
    # Candidates (низкий порог)
    candidates = prob_map >= threshold_low
    
    # Морфологическая операция: соединить ядро с близкими кандидатами
    # (это geodesic dilation)
    result = morphology.reconstruction(core, candidates, method='dilation')
    
    return result.astype(np.uint8)

def kill_bridges(mask, erosion_radius=1, top_k_components=5):
    """
    Bridge-killing: убрать тонкие перемычки между слоями
    
    Стратегия:
    1. Легкая эрозия → разбивает тонкие мосты
    2. Оставить top-K компонент
    3. Geodesic grow обратно
    """
    if erosion_radius <= 0:
        return mask
    
    # Эрозия
    struct = morphology.ball(erosion_radius)
    eroded = morphology.binary_erosion(mask, struct)
    
    # Label components
    labeled = measure.label(eroded, connectivity=3)
    regions = measure.regionprops(labeled)
    
    if len(regions) == 0:
        return mask
    
    # Сортировать по размеру
    regions = sorted(regions, key=lambda r: r.area, reverse=True)
    
    # Оставить top-K
    keep_labels = [r.label for r in regions[:top_k_components]]
    
    # Маска top-K компонент
    core = np.isin(labeled, keep_labels)
    
    # Geodesic grow обратно внутри исходной маски
    result = morphology.reconstruction(core, mask, method='dilation')
    
    return result.astype(np.uint8)

def fill_small_holes(mask, max_hole_size=1000):
    """Заполнить маленькие дыры"""
    return morphology.remove_small_holes(mask, area_threshold=max_hole_size)

def remove_small_components(mask, min_size=8000):
    """Убрать маленькие компоненты"""
    return morphology.remove_small_objects(mask, min_size=min_size, connectivity=3)

def closing_operation(mask, radius=1):
    """Морфологическое closing"""
    if radius <= 0:
        return mask
    struct = morphology.ball(radius)
    return morphology.binary_closing(mask, struct)

def postprocess_mask(prob_map, config):
    """
    Полный постпроцессинг pipeline
    
    Pipeline:
    1. Сгладить вероятности
    2. Двухпороговая бинаризация
    3. Bridge-killing
    4. Closing (заполнить мелкие разрывы)
    5. Убрать маленькие компоненты
    """
    print("  🔧 Postprocessing...")
    
    # 1. Сглаживание
    prob_smooth = smooth_probabilities(prob_map, sigma=config['smooth_sigma'])
    print(f"    1. Smoothed (sigma={config['smooth_sigma']})")
    
    # 2. Двухпороговая сегментация
    mask = dual_threshold_segmentation(
        prob_smooth,
        threshold_high=config['threshold_high'],
        threshold_low=config['threshold_low']
    )
    print(f"    2. Dual threshold ({config['threshold_high']}/{config['threshold_low']})")
    
    # 3. Bridge-killing
    mask = kill_bridges(
        mask,
        erosion_radius=config['bridge_erosion'],
        top_k_components=5
    )
    print(f"    3. Bridge-killing (erosion={config['bridge_erosion']})")
    
    # 4. Closing
    mask = closing_operation(mask, radius=config['closing_radius'])
    print(f"    4. Closing (radius={config['closing_radius']})")
    
    # 5. Убрать маленькие компоненты
    mask = remove_small_components(mask, min_size=config['min_component_size'])
    print(f"    5. Remove small components (min_size={config['min_component_size']})")
    
    # Статистика
    n_components = len(np.unique(measure.label(mask, connectivity=3))) - 1
    coverage = mask.sum() / mask.size
    print(f"    ✓ Components: {n_components}, Coverage: {coverage:.1%}")
    
    return mask

print("✓ Postprocessing functions defined")

# =============================================================================
# CELL 9: Get Test Volume IDs
# =============================================================================
print("\n🔍 Finding test volumes...")

test_files = sorted(TEST_IMG_DIR.glob("*.tif"))
test_ids = [f.stem for f in test_files]

print(f"✓ Found {len(test_ids)} test volumes")
print(f"Test IDs: {test_ids[:5]}...")

# =============================================================================
# CELL 10: Run Inference
# =============================================================================
print(f"\n{'='*70}")
print("🚀 STARTING INFERENCE")
print(f"{'='*70}\n")

predictions = {}

for i, vid in enumerate(test_ids):
    print(f"\n[{i+1}/{len(test_ids)}] Processing volume: {vid}")
    print("-" * 70)
    
    # Загрузить volume
    img_path = TEST_IMG_DIR / f"{vid}.tif"
    volume = tifffile.imread(str(img_path))
    print(f"  Loaded volume: shape={volume.shape}, dtype={volume.dtype}")
    
    # Predict (ensemble)
    print(f"  Predicting with {len(models)} model(s)...")
    prob_map = predict_volume_ensemble(
        volume,
        models,
        configs,
        overlap=INFERENCE_CONFIG['overlap'],
        use_tta=INFERENCE_CONFIG['use_tta']
    )
    
    print(f"  Probability range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")
    
    # Postprocess
    mask = postprocess_mask(prob_map, INFERENCE_CONFIG)
    
    # Сохранить
    predictions[vid] = mask
    
    # Сохранить на диск (временно)
    output_path = OUTPUT_DIR / f"pred_{vid}.npy"
    np.save(output_path, mask)
    print(f"  💾 Saved to {output_path.name}")
    
    # Очистить память
    del volume, prob_map, mask
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"  ✓ Done\n")

print(f"{'='*70}")
print(f"✅ INFERENCE COMPLETED")
print(f"{'='*70}\n")

# =============================================================================
# CELL 11: Create Submission
# =============================================================================
print("📝 Creating submission file...")

# Формат submission: каждая строка = id,prediction
# prediction = список (z,y,x) координат где mask == 1

submission_rows = []

for vid in test_ids:
    mask = predictions[vid]
    
    # Найти координаты surface voxels
    coords = np.argwhere(mask == 1)
    
    # Конвертировать в строку: "[[z1,y1,x1],[z2,y2,x2],...]"
    coords_str = '[' + ','.join([f'[{z},{y},{x}]' for z, y, x in coords]) + ']'
    
    submission_rows.append({
        'id': vid,
        'prediction': coords_str
    })
    
    print(f"  {vid}: {len(coords)} surface voxels")

# Создать DataFrame
submission = pd.DataFrame(submission_rows)

# Сохранить
submission_path = OUTPUT_DIR / 'submission.csv'
submission.to_csv(submission_path, index=False)

print(f"\n✓ Submission saved: {submission_path}")
print(f"  Shape: {submission.shape}")
print(f"\nFirst few rows:")
print(submission.head())

# =============================================================================
# CELL 12: Optional - Save Configurations
# =============================================================================
# Сохранить конфигурацию inference для воспроизводимости
inference_log = {
    'models': [str(p) for p in MODEL_PATHS],
    'inference_config': INFERENCE_CONFIG,
    'test_volumes': test_ids,
    'predictions_summary': {
        vid: {
            'n_surface_voxels': int(predictions[vid].sum()),
            'volume_shape': predictions[vid].shape
        }
        for vid in test_ids
    }
}

with open(OUTPUT_DIR / 'inference_log.json', 'w') as f:
    json.dump(inference_log, f, indent=2)

print("\n✓ Inference log saved")

print(f"\n{'='*70}")
print("🎉 ALL DONE!")
print(f"{'='*70}")
print("\nFiles created:")
print(f"  - {submission_path}")
print(f"  - inference_log.json")
print(f"  - pred_*.npy (intermediate files)")
print("\n📤 Ready to submit!")
