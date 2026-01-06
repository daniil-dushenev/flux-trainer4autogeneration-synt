"""
Скрипт для создания тестового датасета.
Создает простые тестовые изображения и predictions.jsonl в правильном формате.
"""
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np

# Настраиваем окружение для скачивания на диск D
try:
    from setup_env import *
except ImportError:
    # Если setup_env не найден, создаем переменные окружения вручную
    D_DRIVE_BASE = Path("D:/")
    HF_HOME = D_DRIVE_BASE / "huggingface_cache"
    HF_HOME.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["TRANSFORMERS_CACHE"] = str(HF_HOME / "transformers")

# Пути для тестового датасета
TEST_DATA_DIR = Path("D:/test_synthetic_data")
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = TEST_DATA_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Тестовые данные - создаем простые изображения
TEST_IMAGES = [
    {
        "image_id": "cat_001.png",
        "task": "classification",
        "label": "cat",
        "color": (200, 150, 100)  # Оранжевый
    },
    {
        "image_id": "dog_001.png",
        "task": "classification",
        "label": "dog",
        "color": (100, 150, 200)  # Синий
    },
    {
        "image_id": "dog_002.png",
        "task": "classification",
        "label": "dog",
        "color": (150, 200, 100)  # Зеленый
    },
    {
        "image_id": "detection_001.png",
        "task": "detection",
        "detections": [
            {"bbox": [50, 50, 200, 200], "label": "cat"},
            {"bbox": [250, 100, 400, 350], "label": "dog"}
        ],
        "color": (200, 200, 100)  # Желтый
    },
]

def create_test_image(save_path: Path, color=(100, 150, 200), pattern="solid"):
    """Создает тестовое изображение с простым паттерном."""
    img = Image.new("RGB", (512, 512), color)
    
    if pattern == "gradient":
        # Создаем градиент
        arr = np.array(img)
        for y in range(512):
            factor = y / 512.0
            arr[y, :] = (
                int(color[0] * (1 - factor) + color[0] * 0.5 * factor),
                int(color[1] * (1 - factor) + color[1] * 0.5 * factor),
                int(color[2] * (1 - factor) + color[2] * 0.5 * factor)
            )
        img = Image.fromarray(arr)
    elif pattern == "checkerboard":
        # Простая шахматная доска
        arr = np.array(img)
        size = 64
        for y in range(0, 512, size):
            for x in range(0, 512, size):
                if (x // size + y // size) % 2 == 0:
                    arr[y:y+size, x:x+size] = color
                else:
                    arr[y:y+size, x:x+size] = tuple(c // 2 for c in color)
        img = Image.fromarray(arr)
    
    img.save(save_path, "PNG")
    return True

def create_test_dataset():
    """Создает тестовый датасет."""
    print(f"Creating test dataset in {TEST_DATA_DIR}")
    
    predictions = []
    
    for item in TEST_IMAGES:
        image_path = IMAGES_DIR / item["image_id"]
        
        # Создаем тестовое изображение
        color = item.get("color", (100, 150, 200))
        pattern = item.get("pattern", "solid")
        create_test_image(image_path, color=color, pattern=pattern)
        print(f"Created test image: {image_path}")
        
        # Создаем запись для predictions.jsonl
        if item["task"] == "classification":
            prediction_entry = {
                "image_id": item["image_id"],
                "task": "classification",
                "prediction": {
                    "label": item["label"]
                },
                "raw": {
                    "meta": {
                        "abs_path": str(image_path.absolute()),
                        "rel_path": item["image_id"],
                        "root": str(IMAGES_DIR.absolute())
                    }
                }
            }
        else:
            # Для detection задачи
            prediction_entry = {
                "image_id": item["image_id"],
                "task": "detection",
                "prediction": {
                    "detections": item.get("detections", [])
                },
                "raw": {
                    "meta": {
                        "abs_path": str(image_path.absolute()),
                        "rel_path": item["image_id"],
                        "root": str(IMAGES_DIR.absolute())
                    }
                }
            }
        
        predictions.append(prediction_entry)
    
    # Сохраняем predictions.jsonl
    jsonl_path = TEST_DATA_DIR / "predictions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for entry in predictions:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"Created test dataset:")
    print(f"  Images: {IMAGES_DIR}")
    print(f"  Predictions: {jsonl_path}")
    print(f"  Total samples: {len(predictions)}")
    
    return jsonl_path, IMAGES_DIR

if __name__ == "__main__":
    jsonl_path, images_dir = create_test_dataset()
    print(f"\nTest dataset ready!")
    print(f"JSONL path: {jsonl_path}")
    print(f"Images dir: {images_dir}")

