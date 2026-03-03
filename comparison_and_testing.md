# СРАВНЕНИЕ ДВУХ ВАРИАНТОВ МОДЕЛЕЙ

## 📊 Быстрое сравнение

| Параметр | Оригинал | v1: BALANCED | v2: ADAPTIVE |
|----------|----------|--------------|--------------|
| **Философия** | Базовая | Консервативная регуляризация | Адаптивное обучение |
| **Dropout** | 0.1 | 0.25 | 0.1 |
| **Weight Decay** | 3e-4 | 1e-3 | 5e-4 |
| **Learning Rate** | 3e-4 | 1e-4 | 1e-4 |
| **Filters** | [28,56,112,224,280] | [24,48,96,192,240] | [28,56,112,224,280] |
| **Surface Focus** | 0.70 | 0.85 | 0.95 |
| **Val Batches** | 30 | ALL | ALL |
| **Loss** | Статичный топологический | Статичный (простой веса) | **Адаптивный по эпохам** |
| **Scheduler** | Cosine | Cosine | **ReduceLROnPlateau** |
| **Early Stop Patience** | 3 | 3 | 5 |
| **Патчей/объем** | 12 | 12 | 16 |

---

## 🎯 Когда выбрать какую модель

### v1: BALANCED - выбирайте если:
✅ Хотите консервативный, стабильный подход
✅ Подозреваете, что модель слишком сложная
✅ Нужна максимальная регуляризация
✅ Готовы пожертвовать немного емкостью модели

**Плюсы:**
- Простота (меньше новых компонентов)
- Сильная регуляризация
- Усиленные аугментации готовы к использованию

**Минусы:**
- Уменьшенная модель может недообучиться
- Нет адаптации loss

---

### v2: ADAPTIVE - выбирайте если:
✅ Проблема в дисбалансе классов (поверхность <1% объема)
✅ Хотите постепенно вводить сложные компоненты loss
✅ Нужен автоматический контроль LR
✅ Готовы к более сложной реализации

**Плюсы:**
- Умная стратегия обучения
- Агрессивный surface sampling (помогает при дисбалансе)
- Adaptive loss (меньше ранней нестабильности)
- ReduceLROnPlateau (авто-снижение LR)

**Минусы:**
- Более сложный код (новый loss class)
- Требует больше внимания к фазам обучения

---

## 🧪 ПЛАН ТЕСТИРОВАНИЯ

### Шаг 1: Базовая диагностика (перед обучением)
```python
# Проверить дисбаланс классов
def analyze_class_balance(train_ids, gcs_msk_dir, fs):
    total_fg = 0
    total_bg = 0
    
    for vid in train_ids[:10]:  # Выборка
        msk = load_tif_from_gcs(f"{gcs_msk_dir}/{vid}.tif", fs)
        total_fg += (msk == 1).sum()
        total_bg += (msk == 0).sum()
    
    ratio = total_fg / (total_fg + total_bg)
    print(f"Foreground ratio: {ratio:.4f} ({ratio*100:.2f}%)")
    
    if ratio < 0.01:
        print("⚠️ SEVERE CLASS IMBALANCE - recommend v2: ADAPTIVE")
    elif ratio < 0.05:
        print("⚠️ MODERATE CLASS IMBALANCE - either model OK")
    else:
        print("✅ Reasonable balance - v1: BALANCED preferred")
    
    return ratio

# Запустить перед обучением
class_ratio = analyze_class_balance(train_vols, GCS_MSK_DIR, fs)
```

### Шаг 2: Параллельное обучение (3 эпохи)
Запустите обе модели одновременно (на разных машинах или последовательно).

**Метрики для отслеживания:**
1. Train Loss (должна стабильно падать)
2. Val Loss (должна падать, а не расти!)
3. Gap = abs(Train - Val) (должна уменьшаться или стабилизироваться)
4. Val Loss variance (насколько скачет между эпохами)

### Шаг 3: Критерии успеха после 3 эпох

**МОДЕЛЬ УСПЕШНА если:**
```python
def evaluate_model_success(history):
    # Получить последние 3 эпохи
    last_3 = history[-3:]
    
    val_losses = [e['val_loss'] for e in last_3]
    train_losses = [e['train_loss'] for e in last_3]
    
    # 1. Val loss должна снижаться или хотя бы стабилизироваться
    val_trend = val_losses[-1] <= val_losses[0] * 1.05  # Допуск 5%
    
    # 2. Gap не должна расти
    gaps = [abs(t - v) for t, v in zip(train_losses, val_losses)]
    gap_trend = gaps[-1] <= gaps[0] * 1.1  # Допуск 10%
    
    # 3. Val loss должна быть разумной (не >2x от train)
    reasonable_gap = val_losses[-1] < train_losses[-1] * 2.0
    
    success = val_trend and gap_trend and reasonable_gap
    
    print(f"Val trend OK: {val_trend}")
    print(f"Gap trend OK: {gap_trend}")
    print(f"Reasonable gap: {reasonable_gap}")
    print(f"\nOVERALL: {'✅ SUCCESS' if success else '❌ NEEDS WORK'}")
    
    return success

# После 3 эпох:
success = evaluate_model_success(history)
```

**МОДЕЛЬ ПРОВАЛЕНА если:**
- Val loss растет на всех 3 эпохах
- Gap > 0.5 и продолжает расти
- Val loss > 2 * Train loss

### Шаг 4: Выбор победителя

Сравните после 3 эпох:

```python
def compare_models(history_v1, history_v2):
    v1_val = history_v1[-1]['val_loss']
    v2_val = history_v2[-1]['val_loss']
    
    v1_gap = abs(history_v1[-1]['train_loss'] - v1_val)
    v2_gap = abs(history_v2[-1]['train_loss'] - v2_val)
    
    print(f"v1 BALANCED: Val={v1_val:.4f}, Gap={v1_gap:.4f}")
    print(f"v2 ADAPTIVE: Val={v2_val:.4f}, Gap={v2_gap:.4f}")
    
    if v1_val < v2_val * 0.95 and v1_gap < v2_gap:
        print("\n🏆 v1 BALANCED WINS - use for full training")
        return "v1"
    elif v2_val < v1_val * 0.95 and v2_gap < v1_gap:
        print("\n🏆 v2 ADAPTIVE WINS - use for full training")
        return "v2"
    else:
        print("\n🤝 TIE - choose based on stability (less variance)")
        return "tie"
```

---

## 🔬 ЧТО МОЖНО ПРОВЕРИТЬ НА ТЕСТАХ

### 1. Тест дисбаланса классов
```python
# Проверить: насколько несбалансированы данные
analyze_class_balance(train_ids, GCS_MSK_DIR, fs)
```
**Результат:** Если <1% - точно нужна v2 с агрессивным sampling

### 2. Тест стабильности валидации
```python
# Запустить валидацию 3 раза на одной модели
val_losses = []
for _ in range(3):
    val_loss = validate(model, val_loader, criterion, max_batches=30)
    val_losses.append(val_loss)

std = np.std(val_losses)
print(f"Val loss std with max_batches=30: {std:.4f}")

# Теперь с max_batches=None
val_losses_full = []
for _ in range(3):
    val_loss = validate(model, val_loader, criterion, max_batches=None)
    val_losses_full.append(val_loss)

std_full = np.std(val_losses_full)
print(f"Val loss std with max_batches=None: {std_full:.4f}")
```
**Результат:** Если std с 30 батчами намного больше - проблема в валидации

### 3. Тест эффекта dropout
```python
# Сравнить с dropout=0.1 vs 0.25 на 1 эпохе
# Смотреть: насколько меняется gap
```

### 4. Тест сэмплинга
```python
# Проверить: сколько патчей имеют >50 пикселей поверхности
count_with_surface = 0
for _ in range(100):  # 100 случайных патчей
    img, msk, vm = train_ds[np.random.randint(len(train_ds))]
    if (msk == 1).sum() >= 50:
        count_with_surface += 1

print(f"Patches with ≥50 surface pixels: {count_with_surface}/100")
```
**Результат:** Если <70% - sampling работает плохо, увеличить surface_focus_prob

---

## 📋 ЧЕКЛИСТ ПЕРЕД ЗАПУСКОМ

### Для v1: BALANCED
- [ ] Изменить `filters` в CONFIG
- [ ] Увеличить `dropout` до 0.25 в build_model()
- [ ] Увеличить `weight_decay` до 1e-3
- [ ] Снизить `base_lr` до 1e-4
- [ ] Изменить `surface_focus_prob` до 0.85
- [ ] Изменить validate() на max_batches=None
- [ ] Обновить _augment() с новыми методами (gamma, contrast, blur)
- [ ] Добавить print валидационной статистики

### Для v2: ADAPTIVE
- [ ] Добавить новый класс AdaptiveTopologyLoss
- [ ] Заменить criterion на AdaptiveTopologyLoss(CONFIG)
- [ ] Добавить criterion.set_epoch(ep) в цикл
- [ ] Изменить scheduler на ReduceLROnPlateau
- [ ] Изменить `surface_focus_prob` до 0.95
- [ ] Добавить проверку min_surface_pixels в _get_patch_center()
- [ ] Увеличить `early_stop_patience` до 5
- [ ] Изменить validate() на max_batches=None
- [ ] Добавить scheduler.step(val_loss) после валидации

---

## 🎲 МОЯ РЕКОМЕНДАЦИЯ

**Если не уверены - начните с v2: ADAPTIVE, потому что:**
1. Адаптивный loss дает больше контроля
2. Агрессивный sampling критичен при дисбалансе
3. ReduceLROnPlateau автоматически реагирует на проблемы
4. Больше patience = меньше ложных early stopping

**Но используйте v1: BALANCED если:**
- Хотите минимум изменений в коде
- Нужна максимальная стабильность
- Модель явно переобучается (слишком большая)

**Идеальный вариант:**
Запустите обе параллельно на 3 эпохи и выберите лучшую!
