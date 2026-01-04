"""
Модуль для сохранения результатов генерации (изображения + аннотации).
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from PIL import Image

from .data_preparation import TrainingSample


class OutputHandler:
    """
    Класс для сохранения сгенерированных изображений и аннотаций.
    """
    
    def __init__(self, output_dir: str | Path):
        """
        Инициализация обработчика вывода.
        
        Args:
            output_dir: Корневая директория для сохранения результатов
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем поддиректории
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        self.conditions_dir = self.output_dir / "controlnet_conditions"
        self.canny_dir = self.conditions_dir / "canny"
        self.depth_dir = self.conditions_dir / "depth"
        self.canny_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictions_path = self.output_dir / "predictions.jsonl"
        
        # Счетчик для имен файлов
        self.counter = 0
    
    def save(
        self,
        image: Image.Image,
        sample: TrainingSample,
        generated_image_id: Optional[str] = None,
        source_image_id: Optional[str] = None,
        lora_path: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Сохраняет одно сгенерированное изображение и его аннотации.
        
        Args:
            image: Сгенерированное изображение
            sample: Исходный TrainingSample с промптом и условиями
            generated_image_id: ID для сгенерированного изображения (авто если None)
            source_image_id: ID исходного изображения
            lora_path: Путь к использованному LoRA адаптеру
            additional_metadata: Дополнительные метаданные
        
        Returns:
            ID сохраненного изображения
        """
        # Генерируем ID если не указан
        if generated_image_id is None:
            self.counter += 1
            generated_image_id = f"synthetic_{self.counter:04d}"
        
        # Определяем имя файла
        image_filename = f"{generated_image_id}.png"
        image_path = self.images_dir / image_filename
        
        # Сохраняем изображение
        image.save(image_path, "PNG")
        
        # Сохраняем ControlNet условия если есть
        conditions_saved = []
        if sample.canny_condition:
            canny_filename = f"{generated_image_id}_canny.png"
            canny_path = self.canny_dir / canny_filename
            sample.canny_condition.save(canny_path, "PNG")
            conditions_saved.append("canny")
        
        if sample.depth_condition:
            depth_filename = f"{generated_image_id}_depth.png"
            depth_path = self.depth_dir / depth_filename
            sample.depth_condition.save(depth_path, "PNG")
            conditions_saved.append("depth")
        
        # Формируем метаданные
        metadata = {
            "generated": True,
            "source_image_id": source_image_id or sample.image_id,
            "lora_path": str(lora_path) if lora_path else None,
            "controlnet_types": conditions_saved,
            "prompt": sample.prompt,
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Формируем запись для predictions.jsonl
        # Используем исходные аннотации
        prediction_entry = {
            "image_id": image_filename,
            "task": self._detect_task_from_annotations(sample.annotations),
            "prediction": sample.annotations,
            "raw": {
                "meta": metadata,
            },
        }
        
        # Сохраняем в JSONL
        with self.predictions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(prediction_entry, ensure_ascii=False) + "\n")
        
        return generated_image_id
    
    def _detect_task_from_annotations(self, annotations: Dict[str, Any]) -> str:
        """Определяет тип задачи по аннотациям."""
        if "label" in annotations:
            return "classification"
        elif "detections" in annotations:
            return "detection"
        else:
            return "unknown"
    
    def save_batch(
        self,
        images: List[Image.Image],
        samples: List[TrainingSample],
        lora_path: Optional[str] = None,
        prefix: str = "synthetic",
    ) -> List[str]:
        """
        Сохраняет батч сгенерированных изображений.
        
        Args:
            images: Список сгенерированных изображений
            samples: Список исходных TrainingSample
            lora_path: Путь к использованному LoRA адаптеру
            prefix: Префикс для имен файлов
        
        Returns:
            Список ID сохраненных изображений
        """
        assert len(images) == len(samples), "Number of images must match number of samples"
        
        saved_ids = []
        for i, (image, sample) in enumerate(zip(images, samples)):
            image_id = f"{prefix}_{i+1:04d}"
            saved_id = self.save(
                image=image,
                sample=sample,
                generated_image_id=image_id,
                source_image_id=sample.image_id,
                lora_path=lora_path,
            )
            saved_ids.append(saved_id)
        
        return saved_ids
    
    def clear(self):
        """Очищает выходную директорию (удаляет все файлы)."""
        import shutil
        
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        
        # Пересоздаем структуру
        self.__init__(self.output_dir)

