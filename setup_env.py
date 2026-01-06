"""
Скрипт для настройки переменных окружения для скачивания моделей на диск D.
"""
import os
from pathlib import Path

# Устанавливаем пути на диск D
D_DRIVE_BASE = Path("D:/")
HF_HOME = D_DRIVE_BASE / "huggingface_cache"
TRANSFORMERS_CACHE = HF_HOME / "transformers"
HF_DATASETS_CACHE = HF_HOME / "datasets"

# Создаем директории если их нет
HF_HOME.mkdir(parents=True, exist_ok=True)
TRANSFORMERS_CACHE.mkdir(parents=True, exist_ok=True)
HF_DATASETS_CACHE.mkdir(parents=True, exist_ok=True)

# Устанавливаем переменные окружения
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["TRANSFORMERS_CACHE"] = str(TRANSFORMERS_CACHE)
os.environ["HF_DATASETS_CACHE"] = str(HF_DATASETS_CACHE)

# Также для huggingface_hub
os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")

print(f"Environment variables set:")
print(f"  HF_HOME = {HF_HOME}")
print(f"  TRANSFORMERS_CACHE = {TRANSFORMERS_CACHE}")
print(f"  HF_DATASETS_CACHE = {HF_DATASETS_CACHE}")
print(f"  HF_HUB_CACHE = {HF_HOME / 'hub'}")

