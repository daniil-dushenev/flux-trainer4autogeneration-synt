# Генерация синтетических данных через FLUX + LoRA + ControlNetUnion

Пайплайн для генерации синтетических данных через обучение LoRA-адаптера для модели FLUX с использованием ControlNetUnion (ControlNet++ из [ControlNetPlus](https://github.com/xinsir6/ControlNetPlus)).

ControlNetUnion поддерживает множественные условия одновременно (canny edges, depth maps и другие).

Это вторая часть общего пайплайна:
1. **vlm-markup-anything/** - разметка данных с помощью VLM
2. **synthetic-generation/** (этот модуль) - генерация синтетических данных
3. (планируется) - обучение моделей и сравнение качества

## Архитектура

Пайплайн состоит из следующих модулей:

- `data_loader.py` - загрузка размеченных данных из predictions.jsonl
- `data_preparation.py` - подготовка данных для обучения (промпты, ControlNet условия)
- `lora_training.py` - обучение LoRA адаптера через diffusion-pipe
- `generation.py` - генерация синтетических изображений
- `output_handler.py` - сохранение результатов (изображения + аннотации)
- `runner.py` - главный скрипт для запуска пайплайна
- `utils/controlnet.py` - утилиты для генерации ControlNet условий (legacy, для обратной совместимости)
- `utils/controlnet_union.py` - утилиты для генерации ControlNetUnion условий
- `utils/prompts.py` - генерация текстовых промптов из аннотаций

## Установка

```bash
pip install -r requirements.txt
```

### Установка ControlNetPlus

Для работы с ControlNetUnion необходимо установить ControlNetPlus:

```bash
git clone https://github.com/xinsir6/ControlNetPlus.git
cd ControlNetPlus
pip install -e .
```

Или установить зависимости вручную (см. requirements.txt в репозитории ControlNetPlus).

### Скачивание весов модели

Веса ControlNetUnion доступны на Hugging Face:
- https://huggingface.co/xinsir/controlnet-union-sdxl-1.0

**Примечание**: Для обучения LoRA рекомендуется использовать `diffusion-pipe` (tdrussell/diffusion-pipe), который может потребовать отдельной установки. В качестве fallback используется библиотека `diffusers` от Hugging Face.

## Использование

### Базовый пример

```python
from pathlib import Path
from synthetic_generation.runner import run_synthetic_generation_pipeline

run_synthetic_generation_pipeline(
    jsonl_path="path/to/predictions.jsonl",
    output_dir="outputs/synthetic_data",
    images_root="path/to/images",  # опционально
)
```

### Из командной строки

```bash
python -m synthetic_generation.runner \
    path/to/predictions.jsonl \
    outputs/synthetic_data \
    --images-root path/to/images \
    --num-samples 100 \
    --num-epochs 10 \
    --learning-rate 1e-4
```

### Использование уже обученного адаптера

```bash
python -m synthetic_generation.runner \
    path/to/predictions.jsonl \
    outputs/synthetic_data \
    --lora-path outputs/lora_checkpoints/final_lora \
    --skip-training \
    --num-samples 50
```

## Формат входных данных

Пайплайн ожидает файл `predictions.jsonl` в формате, совместимом с выводом первой части (vlm-markup-anything):

```json
{
  "image_id": "image001.png",
  "task": "detection",
  "prediction": {
    "detections": [
      {"bbox": [10, 20, 100, 200], "label": "cat"},
      {"bbox": [150, 50, 250, 300], "label": "dog"}
    ]
  },
  "raw": {
    "meta": {
      "abs_path": "/full/path/to/image001.png",
      "rel_path": "image001.png",
      "root": "/path/to/images"
    }
  }
}
```

Поддерживаются задачи:
- `classification` - с полем `prediction.label`
- `detection` - с полем `prediction.detections`

## Формат выходных данных

Результаты сохраняются в структурированную папку:

```
output/
├── images/
│   ├── synthetic_0001.png
│   ├── synthetic_0002.png
│   └── ...
├── controlnet_conditions/
│   ├── canny/
│   │   ├── synthetic_0001_canny.png
│   │   └── ...
│   └── depth/
│       ├── synthetic_0001_depth.png
│       └── ...
└── predictions.jsonl
```

Формат `predictions.jsonl` совместим с форматом первой части:

```json
{
  "image_id": "synthetic_0001.png",
  "task": "detection",
  "prediction": {...},
  "raw": {
    "meta": {
      "generated": true,
      "source_image_id": "original_001.png",
      "lora_path": "outputs/lora_checkpoints/final_lora",
      "controlnet_types": ["canny", "depth"],
      "prompt": "a photo with cat and dog"
    }
  }
}
```

## Конфигурация

### Параметры обучения LoRA

- `rank` - ранг LoRA адаптера (по умолчанию 16)
- `alpha` - альфа параметр LoRA (по умолчанию 32)
- `learning_rate` - learning rate (по умолчанию 1e-4)
- `batch_size` - размер батча (по умолчанию 1)
- `num_epochs` - количество эпох (по умолчанию 10)

### Параметры генерации

- `num_inference_steps` - количество шагов генерации (по умолчанию 50)
- `guidance_scale` - guidance scale (по умолчанию 7.5)
- `seed` - seed для воспроизводимости (опционально)

### ControlNet типы

Поддерживаются:
- `canny` - Canny edge maps
- `depth` - Depth maps (использует Intel/dpt-large для оценки глубины)

## Важные замечания

1. **diffusion-pipe**: Точный API библиотеки diffusion-pipe может отличаться от предполагаемого. Код содержит гибкую архитектуру для адаптации под реальный API.

2. **ControlNetUnion для FLUX**: ControlNetUnion разработан для SDXL, но может использоваться с FLUX через адаптацию. FLUX использует архитектуру, отличную от Stable Diffusion. Может потребоваться адаптация ControlNetUnion или использование FLUX-native подходов к контролю генерации.

3. **Генерация аннотаций**: Для синтетических изображений используются исходные аннотации из размеченных данных. Опционально можно запустить модель разметки из первой части на сгенерированных изображениях.

4. **Требования к ресурсам**: Обучение LoRA и генерация требуют значительных вычислительных ресурсов (рекомендуется GPU с достаточным объемом памяти).

## Структура проекта

```
synthetic-generation/
├── __init__.py
├── data_loader.py          # Загрузка данных
├── data_preparation.py     # Подготовка данных
├── lora_training.py        # Обучение LoRA
├── generation.py           # Генерация изображений
├── output_handler.py       # Сохранение результатов
├── runner.py               # Главный скрипт
├── requirements.txt        # Зависимости
├── README.md              # Документация
└── utils/
    ├── __init__.py
    ├── controlnet.py       # ControlNet утилиты
    └── prompts.py          # Генерация промптов
```

## Примеры использования

### Полный цикл: обучение + генерация

```python
from synthetic_generation.runner import run_synthetic_generation_pipeline
from synthetic_generation.lora_training import LoRATrainingConfig

config = LoRATrainingConfig(
    rank=16,
    alpha=32,
    learning_rate=1e-4,
    batch_size=1,
    num_epochs=10,
)

run_synthetic_generation_pipeline(
    jsonl_path="data/predictions.jsonl",
    output_dir="outputs/synthetic",
    training_config=config,
    num_samples_to_generate=100,
)
```

### Только генерация с существующим адаптером

```python
from synthetic_generation.runner import run_synthetic_generation_pipeline

run_synthetic_generation_pipeline(
    jsonl_path="data/predictions.jsonl",
    output_dir="outputs/synthetic",
    lora_path="outputs/lora_checkpoints/final_lora",
    skip_training=True,
    num_samples_to_generate=50,
)
```

## Лицензия

См. LICENSE файл в корне проекта.

