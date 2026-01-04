"""
Модуль для загрузки размеченных данных из predictions.jsonl.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Tuple
from PIL import Image
from dataclasses import dataclass


@dataclass
class AnnotatedSample:
    """Структурированные данные одного сэмпла с аннотациями."""
    image: Image.Image
    image_id: str
    annotations: Dict[str, Any]
    metadata: Dict[str, Any]
    task: str  # "classification" | "detection"


class AnnotatedDatasetLoader:
    """
    Класс для загрузки размеченных данных из predictions.jsonl.
    
    Поддерживает форматы classification и detection из первой части пайплайна.
    """
    
    def __init__(self, jsonl_path: str | Path, images_root: Optional[str | Path] = None):
        """
        Инициализация загрузчика.
        
        Args:
            jsonl_path: Путь к файлу predictions.jsonl
            images_root: Корневая папка с изображениями (опционально, 
                        если не указан, используется root из meta)
        """
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")
        
        self.images_root = Path(images_root) if images_root else None
        
    def _load_image(self, meta: Dict[str, Any]) -> Image.Image:
        """
        Загружает изображение по метаданным.
        
        Пытается использовать abs_path, если не доступен - rel_path + root.
        """
        abs_path = meta.get("abs_path")
        rel_path = meta.get("rel_path")
        root = meta.get("root")
        
        image_path = None
        
        # Сначала пробуем абсолютный путь
        if abs_path:
            image_path = Path(abs_path)
            if image_path.exists():
                return Image.open(image_path).convert("RGB")
        
        # Потом пробуем относительный путь
        if rel_path:
            if self.images_root:
                image_path = self.images_root / rel_path
            elif root:
                image_path = Path(root) / rel_path
            else:
                raise ValueError(f"Cannot determine image path: missing root or images_root")
            
            if image_path.exists():
                return Image.open(image_path).convert("RGB")
        
        # Если ничего не сработало
        raise FileNotFoundError(
            f"Image not found. abs_path={abs_path}, rel_path={rel_path}, "
            f"root={root}, images_root={self.images_root}"
        )
    
    def _parse_annotations(self, item: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Парсит аннотации из записи JSONL.
        
        Returns:
            (task, annotations, metadata)
        """
        task = item.get("task", "unknown")
        prediction = item.get("prediction", {})
        raw = item.get("raw", {})
        meta = raw.get("meta", {})
        
        return task, prediction, meta
    
    def load_all(self) -> List[AnnotatedSample]:
        """
        Загружает все данные из JSONL файла.
        
        Returns:
            Список AnnotatedSample
        """
        samples = []
        for sample in self:
            samples.append(sample)
        return samples
    
    def __iter__(self) -> Iterator[AnnotatedSample]:
        """
        Итератор по данным из JSONL файла.
        """
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue
                
                try:
                    image_id = item.get("image_id", f"unknown_{line_num}")
                    task, annotations, metadata = self._parse_annotations(item)
                    
                    # Загружаем изображение
                    image = self._load_image(metadata)
                    
                    yield AnnotatedSample(
                        image=image,
                        image_id=image_id,
                        annotations=annotations,
                        metadata=metadata,
                        task=task,
                    )
                except Exception as e:
                    print(f"Warning: Failed to process item at line {line_num}: {e}")
                    continue
    
    def __len__(self) -> int:
        """Подсчитывает количество записей в JSONL файле."""
        count = 0
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    count += 1
        return count

