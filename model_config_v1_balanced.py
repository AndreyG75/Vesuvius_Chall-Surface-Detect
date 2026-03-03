"""
МОДЕЛЬ v1: BALANCED (Сбалансированный подход)
==============================================
Фокус: исправить валидацию + усилить регуляризацию + улучшить сэмплинг

Основные изменения:
1. ✅ Валидация по всему val set (убрали max_batches)
2. ✅ Усиленная регуляризация (dropout, weight_decay)
3. ✅ Улучшенный сэмплинг данных (больше surface focus)
4. ✅ Усиленные аугментации (gamma, contrast, blur)
5. ✅ Сниженный LR для стабильности
"""

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

CONFIG = {
    'model_seed': 123,
    'split_seed': 42,
    
    # МОДЕЛЬ - немного упрощена
    'filters': [24, 48, 96, 192, 240],  # Было [28, 56, 112, 224, 280]
    
    # ДАННЫЕ
    'patch_size': [64, 192, 192],
    'batch_size': 1,
    'accum_steps': 4,
    'n_patches_per_vol': 12,
    
    # ОБУЧЕНИЕ - консервативное
    'total_epochs': 5,
    'base_lr': 1e-4,  # Было 3e-4 → снижено в 3 раза
    'min_lr': 1e-6,
    'weight_decay': 1e-3,  # Было 3e-4 → увеличено в ~3 раза
    'ema_decay': 0.999,
    
    # СЭМПЛИНГ - критично важно!
    'surface_focus_prob': 0.85,  # Было 0.70 → увеличено
    'hard_negative_prob': 0.10,  # Было 0.20 → уменьшено
    'random_prob': 0.05,         # Было 0.10 → уменьшено
    
    # LOSS - упрощенные веса
    'alpha_dice': 0.55,      # Было 0.45 → увеличен фокус на Dice
    'alpha_cldice': 0.25,    # Было 0.35 → уменьшен топологический вес
    'alpha_boundary': 0.20,
    'ds_weights': [0.6, 0.25, 0.10, 0.05],  # Больше веса на главный выход
    
    # ВАЛИДАЦИЯ
    'val_max_batches': None,  # ← КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: весь val set
}

# =============================================================================
# ИЗМЕНЕНИЯ В КОДЕ
# =============================================================================

"""
1. BUILD MODEL - увеличить dropout:

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
        dropout=0.25,  # ← БЫЛО 0.1, СТАЛО 0.25
        norm_name="instance",
        deep_supervision=True,
        res_block=True
    )
    return model

2. VALIDATE - весь val set:

@torch.no_grad()
def validate(model, loader, criterion, max_batches=None):  # ← БЫЛО 30
    model.eval()
    losses = []
    
    if HAS_TORCH_EMA:
        ctx = ema.average_parameters()
    else:
        ctx = EMAContextManager(ema, model)
    
    with ctx:
        for i, (x, y, vm) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y, vm = x.to(DEVICE), y.to(DEVICE), vm.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                out = model(x)
                if isinstance(out, torch.Tensor) and out.dim() == 6:
                    out = [out[:, :, j, ...] for j in range(out.shape[2])]
                loss = criterion(out, y, vm)
            
            losses.append(loss.item())
            del x, y, vm, out, loss
    
    model.train()
    torch.cuda.empty_cache()
    gc.collect()
    
    # ← ДОБАВИТЬ: печать статистики
    print(f"  Val batches used: {len(losses)}/{len(loader)}")
    
    return float(np.mean(losses)) if losses else 999.0

3. ВЫЗОВ VALIDATE:

val_loss = validate(model, val_loader, criterion, max_batches=None)  # ← None вместо 30

4. AUGMENT - усилить аугментации:

def _augment(self, img, msk):
    # Существующие flip/rotate
    for axis in [0, 1, 2]:
        if np.random.rand() < 0.5:
            img = np.flip(img, axis).copy()
            msk = np.flip(msk, axis).copy()
    
    if np.random.rand() < 0.5:
        k = np.random.randint(1, 4)
        img = np.rot90(img, k, axes=(1, 2)).copy()
        msk = np.rot90(msk, k, axes=(1, 2)).copy()
    
    # ← НОВЫЕ АУГМЕНТАЦИИ
    
    # Intensity shift - увеличена вероятность и диапазон
    if np.random.rand() < 0.4:  # было 0.3
        shift = np.random.uniform(-0.15, 0.15)  # было (-0.1, 0.1)
        img = np.clip(img + shift, 0, 1)
    
    # NEW: Contrast
    if np.random.rand() < 0.35:
        contrast = np.random.uniform(0.75, 1.25)
        img = np.clip((img - 0.5) * contrast + 0.5, 0, 1)
    
    # NEW: Gamma
    if np.random.rand() < 0.35:
        gamma = np.random.uniform(0.7, 1.4)
        img = np.clip(img ** gamma, 0, 1)
    
    # Cutout - увеличена вероятность
    if np.random.rand() < 0.4:  # было 0.3
        d, h, w = img.shape
        for _ in range(np.random.randint(3, 10)):  # было (3, 8)
            sz = np.random.randint(6, 16)  # было фикс 8
            z = np.random.randint(0, max(1, d - sz))
            y = np.random.randint(0, max(1, h - sz))
            x = np.random.randint(0, max(1, w - sz))
            img[z:z+sz, y:y+sz, x:x+sz] = 0
    
    # Gaussian noise - усилен
    if np.random.rand() < 0.4:  # было 0.3
        noise = np.random.normal(0, 0.07, img.shape).astype(np.float32)  # было 0.05
        img = np.clip(img + noise, 0, 1).astype(np.float32)
    
    # NEW: Простой blur (box filter)
    if np.random.rand() < 0.25:
        from scipy.ndimage import uniform_filter
        img = uniform_filter(img, size=3).astype(np.float32)
    
    return img, msk
"""

# =============================================================================
# ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ
# =============================================================================

"""
Эта конфигурация должна показать:
1. Более стабильный val loss (без скачков из-за малой выборки)
2. Меньший gap между train и val (регуляризация работает)
3. Более плавное обучение (низкий LR)
4. Лучшее обобщение (сильные аугментации + правильный сэмплинг)

Ожидаемая динамика на 3-5 эпохах:
- Эпоха 1: Train ~0.62, Val ~0.85 (gap ~0.23)
- Эпоха 2: Train ~0.58, Val ~0.80 (gap ~0.22) ← val должна снижаться!
- Эпоха 3: Train ~0.56, Val ~0.76 (gap ~0.20)

Если val не снижается - проблема в данных (дисбаланс классов).
"""
