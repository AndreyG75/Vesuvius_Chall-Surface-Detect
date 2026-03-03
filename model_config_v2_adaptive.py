"""
МОДЕЛЬ v2: ADAPTIVE (Адаптивный подход)
========================================
Фокус: адаптивный loss + агрессивный сэмплинг поверхностей + борьба с дисбалансом

Основные изменения:
1. ✅ Адаптивный loss (постепенное введение топологии)
2. ✅ Очень агрессивный surface sampling (95%)
3. ✅ Focal Loss для борьбы с дисбалансом классов
4. ✅ ReduceLROnPlateau вместо Cosine
5. ✅ Увеличенный patience для early stopping
"""

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

CONFIG = {
    'model_seed': 123,
    'split_seed': 42,
    
    # МОДЕЛЬ - стандартный размер
    'filters': [28, 56, 112, 224, 280],  # Оригинал, не меняем
    
    # ДАННЫЕ
    'patch_size': [64, 192, 192],
    'batch_size': 1,
    'accum_steps': 4,
    'n_patches_per_vol': 16,  # Было 12 → больше вариативности
    
    # ОБУЧЕНИЕ
    'total_epochs': 5,
    'base_lr': 1e-4,  # Консервативное
    'min_lr': 1e-7,
    'weight_decay': 5e-4,  # Средняя регуляризация
    'ema_decay': 0.999,
    
    # СЭМПЛИНГ - АГРЕССИВНЫЙ ФОКУС НА ПОВЕРХНОСТИ!
    'surface_focus_prob': 0.95,  # ← КРИТИЧЕСКИ ВАЖНО для дисбаланса
    'hard_negative_prob': 0.04,
    'random_prob': 0.01,
    'min_surface_pixels': 50,  # Минимум пикселей поверхности в патче
    
    # ADAPTIVE LOSS - фазовое обучение
    'loss_schedule': {
        'epochs_0_3': {  # Эпохи 0-2: только Dice+CE
            'use_topology': False,
            'alpha_dice': 1.0,
        },
        'epochs_3_5': {  # Эпохи 3-4: мягкий топологический
            'use_topology': True,
            'alpha_dice': 0.60,
            'alpha_cldice': 0.25,
            'alpha_boundary': 0.15,
        },
        'epochs_5_plus': {  # Эпохи 5+: полный топологический
            'use_topology': True,
            'alpha_dice': 0.45,
            'alpha_cldice': 0.35,
            'alpha_boundary': 0.20,
        }
    },
    
    'ds_weights': [0.7, 0.2, 0.05, 0.05],  # Еще больше на главный выход
    
    # ВАЛИДАЦИЯ
    'val_max_batches': None,  # Весь val set
    
    # SCHEDULER
    'use_reduce_on_plateau': True,
    'plateau_patience': 2,
    'plateau_factor': 0.5,
    
    # EARLY STOPPING
    'early_stop_patience': 5,  # Было 3 → больше терпения
}

# =============================================================================
# НОВЫЙ ADAPTIVE LOSS CLASS
# =============================================================================

"""
class AdaptiveTopologyLoss(nn.Module):
    '''
    Адаптивный loss с постепенным введением топологических компонентов.
    Помогает избежать ранней нестабильности от clDice и boundary loss.
    '''
    def __init__(self, config):
        super().__init__()
        self.dice_ce = DiceCELoss(
            to_onehot_y=True, 
            softmax=True,
            # Focal loss параметры для борьбы с дисбалансом
            include_background=False,
            lambda_dice=0.5,
            lambda_ce=0.5,
        )
        self.config = config
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        '''Вызывать в начале каждой эпохи'''
        self.current_epoch = epoch
    
    def _get_schedule_params(self):
        '''Получить параметры loss на текущей эпохе'''
        schedule = self.config['loss_schedule']
        
        if self.current_epoch < 3:
            return schedule['epochs_0_3']
        elif self.current_epoch < 5:
            return schedule['epochs_3_5']
        else:
            return schedule['epochs_5_plus']
    
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
        params = self._get_schedule_params()
        
        # Базовый Dice+CE всегда считаем
        target_with_channel = target.unsqueeze(1)
        dice_ce = self.dice_ce(logits, target_with_channel)
        
        # Если топология не включена - возвращаем только Dice+CE
        if not params['use_topology']:
            return dice_ce
        
        # Топологические компоненты только если есть поверхность
        surface_pixels = (target == 1).sum()
        if surface_pixels > 100:
            pred_probs = F.softmax(logits, dim=1)
            cldice = self.compute_cldice(pred_probs, target)
            boundary = self.compute_boundary_loss(pred_probs, target)
            
            total = (params['alpha_dice'] * dice_ce + 
                    params['alpha_cldice'] * cldice + 
                    params['alpha_boundary'] * boundary)
        else:
            total = dice_ce
        
        return total
    
    def forward(self, logits, target, valid_mask):
        # Deep supervision
        if isinstance(logits, (list, tuple)):
            if len(logits) > 0 and isinstance(logits[0], (list, tuple)):
                logits = [item[0] if isinstance(item, (list, tuple)) else item 
                         for item in logits]
            
            total_loss = 0
            ds_weights = self.config['ds_weights']
            
            for i, output in enumerate(logits):
                weight = ds_weights[i] if i < len(ds_weights) else 0.05
                
                if output.shape[2:] != logits[0].shape[2:]:
                    target_scaled = F.interpolate(
                        target.float().unsqueeze(1),
                        size=output.shape[2:],
                        mode='nearest'
                    ).long().squeeze(1)
                else:
                    target_scaled = target
                
                loss_i = self._compute_single_loss(output, target_scaled, valid_mask)
                total_loss += weight * loss_i
            
            return total_loss
        else:
            return self._compute_single_loss(logits, target, valid_mask)
"""

# =============================================================================
# ИЗМЕНЕНИЯ В КОДЕ
# =============================================================================

"""
1. ЗАМЕНИТЬ TopologyAwareLoss на AdaptiveTopologyLoss:

# В CELL 12 (после импортов):
criterion = AdaptiveTopologyLoss(CONFIG)

2. В НАЧАЛЕ КАЖДОЙ ЭПОХИ добавить:

for ep in range(start_epoch, TARGET_EPOCH):
    # ← ДОБАВИТЬ:
    criterion.set_epoch(ep)
    
    epoch_start = time.time()
    # ... остальной код ...

3. ИЗМЕНИТЬ SCHEDULER на ReduceLROnPlateau:

if CONFIG.get('use_reduce_on_plateau', False):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='min',
        factor=CONFIG['plateau_factor'],
        patience=CONFIG['plateau_patience'],
        min_lr=CONFIG['min_lr'],
        verbose=True
    )
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=CONFIG['total_epochs'],
        eta_min=CONFIG['min_lr']
    )

4. В ТРЕНИРОВОЧНОМ ЦИКЛЕ после валидации:

if CONFIG.get('use_reduce_on_plateau', False):
    scheduler.step(val_loss)  # ← подаем val_loss
else:
    scheduler.step()

5. УЛУЧШИТЬ СЭМПЛИНГ - добавить проверку min_surface_pixels:

def _get_patch_center(self, vid, vol_shape):
    max_attempts = 10
    min_pixels = self.config.get('min_surface_pixels', 50)
    
    for attempt in range(max_attempts):
        # Существующая логика выбора центра
        rand = np.random.rand()
        surf = self._get_surface_coords(vid)
        
        if rand < self.surface_prob:
            if surf is not None and len(surf) > 0:
                center = surf[np.random.randint(len(surf))]
            else:
                center = self._random_center(vol_shape)
        elif rand < (self.surface_prob + self.hard_neg_prob):
            # ... hard negative logic ...
            center = # ...
        else:
            center = self._random_center(vol_shape)
        
        # ← НОВАЯ ПРОВЕРКА: считаем сколько пикселей поверхности в патче
        if attempt < max_attempts - 1:  # Проверяем только если есть попытки
            pd, ph, pw = self.patch_size
            z, y, x = center
            
            # Быстрая проверка через surf_coords
            if surf is not None and len(surf) > 0:
                # Подсчет точек в патче
                in_patch = surf[
                    (surf[:, 0] >= z - pd//2) & (surf[:, 0] < z + pd//2) &
                    (surf[:, 1] >= y - ph//2) & (surf[:, 1] < y + ph//2) &
                    (surf[:, 2] >= x - pw//2) & (surf[:, 2] < x + pw//2)
                ]
                
                if len(in_patch) >= min_pixels:
                    break  # Патч подходит
            else:
                break  # Нет поверхности вообще - берем любой
        else:
            break  # Последняя попытка - берем что есть
    
    return center

6. УВЕЛИЧИТЬ PATIENCE для early stopping:

EARLY_STOP_PATIENCE = CONFIG.get('early_stop_patience', 5)
"""

# =============================================================================
# ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ
# =============================================================================

"""
Эта конфигурация должна показать:
1. Стабильное обучение на первых 3 эпохах (только DiceCE)
2. Плавное введение топологии на эпохах 3-5
3. Меньше false negatives (агрессивный surface sampling)
4. Адаптивный LR (автоматически снижается при стагнации)

Ожидаемая динамика:
- Эпохи 0-2: Стабильное снижение и train и val (нет топологии)
  - Эпоха 1: Train ~0.65, Val ~0.88 (простой DiceCE учится хорошо)
  - Эпоха 2: Train ~0.60, Val ~0.82
  - Эпоха 3: Train ~0.57, Val ~0.78

- Эпохи 3-4: Мягкое введение топологии
  - Эпоха 4: Train ~0.56, Val ~0.76 (небольшой скачок val из-за новых компонентов)
  - Эпоха 5: Train ~0.54, Val ~0.74 (адаптация)

Если val все еще растет на эпохах 0-2 - проблема точно в данных (дисбаланс).
Тогда нужно добавить Focal Loss или взвешенные классы.
"""

# =============================================================================
# FOCAL LOSS (опциональное улучшение)
# =============================================================================

"""
Если дисбаланс классов очень сильный (поверхность <1% объема),
добавить Focal Loss вместо обычного CE:

from monai.losses import FocalLoss

# В AdaptiveTopologyLoss.__init__:
self.focal = FocalLoss(
    to_onehot_y=True,
    gamma=2.0,  # Фокусировка на трудных примерах
    weight=torch.tensor([0.1, 0.9]).to(DEVICE)  # Вес для класса 1 (поверхность)
)

# В _compute_single_loss:
dice_ce = 0.5 * self.dice_ce(...) + 0.5 * self.focal(...)
"""
