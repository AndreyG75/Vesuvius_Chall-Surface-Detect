"""
KAGGLE INFERENCE EXPERIMENTS - RAPID TESTING
Быстрое тестирование разных конфигураций inference

Этот скрипт позволяет быстро протестировать разные:
- Пороги бинаризации
- Постпроцессинг стратегии
- TTA настройки

ИСПОЛЬЗОВАНИЕ:
1. Запустить базовый inference (полный)
2. Потом запустить этот скрипт для экспериментов
3. Сравнить результаты
"""

# =============================================================================
# ЭКСПЕРИМЕНТЫ ДЛЯ ТЕСТИРОВАНИЯ
# =============================================================================

EXPERIMENTS = {
    # Базовый (как в основном inference)
    'baseline': {
        'smooth_sigma': 1.0,
        'threshold_high': 0.70,
        'threshold_low': 0.45,
        'bridge_erosion': 1,
        'closing_radius': 1,
        'min_component_size': 8000,
    },
    
    # Conservative (меньше false positives, больше precision)
    'conservative': {
        'smooth_sigma': 1.5,  # Больше сглаживание
        'threshold_high': 0.75,  # Выше порог
        'threshold_low': 0.50,
        'bridge_erosion': 2,  # Агрессивнее убирать мосты
        'closing_radius': 1,
        'min_component_size': 10000,  # Убрать больше мелочи
    },
    
    # Aggressive (больше recall, меньше пропусков)
    'aggressive': {
        'smooth_sigma': 0.5,  # Меньше сглаживание
        'threshold_high': 0.65,  # Ниже порог
        'threshold_low': 0.40,
        'bridge_erosion': 0,  # Не убирать мосты
        'closing_radius': 2,  # Больше заполнять разрывы
        'min_component_size': 5000,  # Оставить больше
    },
    
    # Balanced (сбалансированный подход)
    'balanced': {
        'smooth_sigma': 1.2,
        'threshold_high': 0.68,
        'threshold_low': 0.42,
        'bridge_erosion': 1,
        'closing_radius': 1,
        'min_component_size': 7000,
    },
    
    # Minimal processing (минимум постпроцессинга)
    'minimal': {
        'smooth_sigma': 0.5,
        'threshold_high': 0.70,
        'threshold_low': 0.70,  # Одинаковые пороги = простая бинаризация
        'bridge_erosion': 0,
        'closing_radius': 0,
        'min_component_size': 5000,
    },
}

# =============================================================================
# IMPORTS
# =============================================================================
import os
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import morphology, measure

print("✓ Imports successful")

# =============================================================================
# LOAD PREPROCESSED PROBABILITIES
# =============================================================================
print("\n📂 Loading preprocessed probability maps...")

OUTPUT_DIR = Path("/kaggle/working")

# Предполагаем что базовый inference уже запущен и сохранил prob_maps
# Если нет - нужно сначала запустить основной inference

# Проверить наличие probability maps
prob_files = list(OUTPUT_DIR.glob("prob_*.npy"))

if len(prob_files) == 0:
    print("❌ Error: No probability maps found!")
    print("Please run main inference first to generate prob_*.npy files")
    print("\nTo save prob maps in main inference, add after line 'prob_map = predict_volume_ensemble(...)':")
    print("  np.save(OUTPUT_DIR / f'prob_{vid}.npy', prob_map)")
    exit(1)

print(f"✓ Found {len(prob_files)} probability maps")

# Загрузить все prob_maps
prob_maps = {}
for prob_file in prob_files:
    vid = prob_file.stem.replace('prob_', '')
    prob_maps[vid] = np.load(prob_file)
    print(f"  {vid}: shape={prob_maps[vid].shape}")

# =============================================================================
# POSTPROCESSING FUNCTIONS (копия из основного inference)
# =============================================================================
def smooth_probabilities(prob_map, sigma=1.0):
    if sigma > 0:
        return ndimage.gaussian_filter(prob_map, sigma=sigma)
    return prob_map

def dual_threshold_segmentation(prob_map, threshold_high, threshold_low):
    core = prob_map >= threshold_high
    candidates = prob_map >= threshold_low
    result = morphology.reconstruction(core, candidates, method='dilation')
    return result.astype(np.uint8)

def kill_bridges(mask, erosion_radius=1, top_k_components=5):
    if erosion_radius <= 0:
        return mask
    
    struct = morphology.ball(erosion_radius)
    eroded = morphology.binary_erosion(mask, struct)
    labeled = measure.label(eroded, connectivity=3)
    regions = measure.regionprops(labeled)
    
    if len(regions) == 0:
        return mask
    
    regions = sorted(regions, key=lambda r: r.area, reverse=True)
    keep_labels = [r.label for r in regions[:top_k_components]]
    core = np.isin(labeled, keep_labels)
    result = morphology.reconstruction(core, mask, method='dilation')
    
    return result.astype(np.uint8)

def remove_small_components(mask, min_size=8000):
    return morphology.remove_small_objects(mask, min_size=min_size, connectivity=3)

def closing_operation(mask, radius=1):
    if radius <= 0:
        return mask
    struct = morphology.ball(radius)
    return morphology.binary_closing(mask, struct)

def postprocess_mask(prob_map, config):
    prob_smooth = smooth_probabilities(prob_map, sigma=config['smooth_sigma'])
    mask = dual_threshold_segmentation(
        prob_smooth,
        threshold_high=config['threshold_high'],
        threshold_low=config['threshold_low']
    )
    mask = kill_bridges(mask, erosion_radius=config['bridge_erosion'], top_k_components=5)
    mask = closing_operation(mask, radius=config['closing_radius'])
    mask = remove_small_components(mask, min_size=config['min_component_size'])
    return mask

print("✓ Postprocessing functions loaded")

# =============================================================================
# RUN EXPERIMENTS
# =============================================================================
print(f"\n{'='*70}")
print("🔬 RUNNING EXPERIMENTS")
print(f"{'='*70}\n")

results = {}

for exp_name, exp_config in EXPERIMENTS.items():
    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name.upper()}")
    print(f"{'='*70}")
    print(f"Config: {exp_config}")
    print()
    
    exp_predictions = {}
    exp_stats = {}
    
    for vid in prob_maps:
        print(f"  Processing {vid}...", end=' ')
        
        # Применить постпроцессинг с этой конфигурацией
        mask = postprocess_mask(prob_maps[vid], exp_config)
        
        # Статистика
        n_voxels = mask.sum()
        n_components = len(np.unique(measure.label(mask, connectivity=3))) - 1
        coverage = mask.sum() / mask.size
        
        exp_predictions[vid] = mask
        exp_stats[vid] = {
            'n_voxels': int(n_voxels),
            'n_components': n_components,
            'coverage': float(coverage),
        }
        
        print(f"voxels={n_voxels:,}, components={n_components}, coverage={coverage:.2%}")
    
    # Сохранить результаты эксперимента
    results[exp_name] = {
        'config': exp_config,
        'predictions': exp_predictions,
        'stats': exp_stats,
    }
    
    # Создать submission для этого эксперимента
    submission_rows = []
    
    for vid in exp_predictions:
        mask = exp_predictions[vid]
        coords = np.argwhere(mask == 1)
        coords_str = '[' + ','.join([f'[{z},{y},{x}]' for z, y, x in coords]) + ']'
        submission_rows.append({'id': vid, 'prediction': coords_str})
    
    submission = pd.DataFrame(submission_rows)
    submission_path = OUTPUT_DIR / f'submission_{exp_name}.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"\n✓ Saved: {submission_path.name}")
    print(f"  Total surface voxels across all volumes: {sum(s['n_voxels'] for s in exp_stats.values()):,}")

# =============================================================================
# COMPARE EXPERIMENTS
# =============================================================================
print(f"\n{'='*70}")
print("📊 EXPERIMENT COMPARISON")
print(f"{'='*70}\n")

# Создать сравнительную таблицу
comparison = []

for exp_name in EXPERIMENTS:
    stats = results[exp_name]['stats']
    
    total_voxels = sum(s['n_voxels'] for s in stats.values())
    avg_components = np.mean([s['n_components'] for s in stats.values()])
    avg_coverage = np.mean([s['coverage'] for s in stats.values()])
    
    comparison.append({
        'experiment': exp_name,
        'total_voxels': total_voxels,
        'avg_components': avg_components,
        'avg_coverage': avg_coverage,
        'threshold_high': EXPERIMENTS[exp_name]['threshold_high'],
        'threshold_low': EXPERIMENTS[exp_name]['threshold_low'],
    })

comparison_df = pd.DataFrame(comparison)
print(comparison_df.to_string(index=False))
print()

# Сохранить сравнение
comparison_df.to_csv(OUTPUT_DIR / 'experiments_comparison.csv', index=False)
print("✓ Saved: experiments_comparison.csv")

# Детальная статистика по объемам
print(f"\n{'='*70}")
print("📈 PER-VOLUME STATISTICS")
print(f"{'='*70}\n")

for vid in prob_maps:
    print(f"\nVolume: {vid}")
    print("-" * 50)
    
    for exp_name in EXPERIMENTS:
        stats = results[exp_name]['stats'][vid]
        print(f"  {exp_name:15s}: {stats['n_voxels']:8,} voxels, "
              f"{stats['n_components']:2d} components, "
              f"{stats['coverage']:5.1%} coverage")

# =============================================================================
# SAVE DETAILED RESULTS
# =============================================================================
print(f"\n{'='*70}")
print("💾 SAVING DETAILED RESULTS")
print(f"{'='*70}\n")

# Сохранить полный отчет
report = {
    'experiments': {
        exp_name: {
            'config': EXPERIMENTS[exp_name],
            'summary': {
                'total_voxels': sum(s['n_voxels'] for s in results[exp_name]['stats'].values()),
                'avg_components': float(np.mean([s['n_components'] for s in results[exp_name]['stats'].values()])),
                'avg_coverage': float(np.mean([s['coverage'] for s in results[exp_name]['stats'].values()])),
            },
            'per_volume': results[exp_name]['stats'],
        }
        for exp_name in EXPERIMENTS
    }
}

with open(OUTPUT_DIR / 'experiments_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("✓ Saved: experiments_report.json")

print(f"\n{'='*70}")
print("✅ ALL EXPERIMENTS COMPLETED")
print(f"{'='*70}\n")

print("📋 Files created:")
print("  - submission_baseline.csv")
print("  - submission_conservative.csv")
print("  - submission_aggressive.csv")
print("  - submission_balanced.csv")
print("  - submission_minimal.csv")
print("  - experiments_comparison.csv")
print("  - experiments_report.json")

print("\n🎯 Next steps:")
print("1. Submit all CSV files to Kaggle")
print("2. Compare scores on leaderboard")
print("3. Pick best configuration")
print("4. Fine-tune parameters further if needed")

print("\n💡 Tips:")
print("  - 'conservative' usually good for high precision")
print("  - 'aggressive' usually good for high recall")
print("  - 'balanced' is a good starting point")
print("  - Compare which config gives best leaderboard score")
