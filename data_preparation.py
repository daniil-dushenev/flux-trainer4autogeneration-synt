"""
Модуль для подготовки данных для обучения LoRA адаптера.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from dataclasses import dataclass

from data_loader import AnnotatedDatasetLoader, AnnotatedSample
from utils.prompts import generate_prompt_from_annotations
from utils.controlnet_union import prepare_controlnet_union_conditions


@dataclass
class TrainingSample:
    """Структурированный сэмпл для обучения."""
    image: Image.Image
    prompt: str
    canny_condition: Optional[Image.Image]
    depth_condition: Optional[Image.Image]
    controlnet_union_image_list: List[Image.Image | int]
    controlnet_union_type_list: List[int]
    annotations: Dict[str, Any]
    metadata: Dict[str, Any]
    image_id: str
    task: str
    class_label: Optional[str]


class TrainingDataPreparator:
    """
    Класс для подготовки данных для обучения LoRA адаптера.
    
    Объединяет загрузку данных, генерацию промптов и ControlNet условий.
    """
    
    def __init__(
        self,
        jsonl_path: str | Path,
        images_root: Optional[str | Path] = None,
        controlnet_types: List[str] = None,
        prompt_template: Optional[str] = None,
        canny_thresholds: Optional[Tuple[int, int]] = None,
        depth_model: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Инициализация подготовителя данных.
        
        Args:
            jsonl_path: Путь к predictions.jsonl
            images_root: Корневая папка с изображениями
            controlnet_types: Типы ControlNet условий (["canny", "depth"])
            prompt_template: Шаблон для генерации промптов
            canny_thresholds: Пороги для Canny (low, high)
            depth_model: Модель для depth estimation
            image_size: Размер для ресайза изображений (width, height), None = без ресайза
        """
        self.loader = AnnotatedDatasetLoader(jsonl_path, images_root)
        self.controlnet_types = controlnet_types or ["canny", "depth"]
        self.prompt_template = prompt_template
        self.canny_thresholds = canny_thresholds
        self.depth_model = depth_model
        self.image_size = image_size
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Ресайзит изображение если указан image_size."""
        if self.image_size:
            return image.resize(self.image_size, Image.Resampling.LANCZOS)
        return image
    
    def _prepare_sample(self, annotated_sample: AnnotatedSample) -> TrainingSample:
        """
        Подготавливает один сэмпл для обучения.
        
        Args:
            annotated_sample: Загруженный сэмпл с аннотациями
        
        Returns:
            Подготовленный TrainingSample
        """
        # Ресайзим изображение если нужно
        image = self._resize_image(annotated_sample.image)
        
        # Генерируем промпт
        prompt = generate_prompt_from_annotations(
            annotated_sample.task,
            annotated_sample.annotations,
            template=self.prompt_template
        )
        
        class_label = None
        if annotated_sample.task == "classification":
            label_value = annotated_sample.annotations.get("label")
            if label_value is not None:
                class_label = str(label_value)
        
        # Генерируем ControlNetUnion условия
        union_image_list, union_type_list = prepare_controlnet_union_conditions(
            image,
            self.controlnet_types,
            canny_thresholds=self.canny_thresholds,
            depth_model=self.depth_model
        )
        
        # Извлекаем отдельные условия для обратной совместимости
        # Находим индексы canny и depth в списке controlnet_types
        canny_condition = None
        depth_condition = None
        
        for idx, ctrl_type in enumerate(self.controlnet_types):
            if idx < len(union_image_list) and union_type_list[idx] == 1:
                condition_img = union_image_list[idx]
                if isinstance(condition_img, Image.Image):
                    if ctrl_type == "canny":
                        canny_condition = condition_img
                    elif ctrl_type == "depth":
                        depth_condition = condition_img
        
        return TrainingSample(
            image=image,
            prompt=prompt,
            canny_condition=canny_condition,
            depth_condition=depth_condition,
            controlnet_union_image_list=union_image_list,
            controlnet_union_type_list=union_type_list,
            annotations=annotated_sample.annotations,
            metadata=annotated_sample.metadata,
            image_id=annotated_sample.image_id,
            task=annotated_sample.task,
            class_label=class_label,
        )
    
    def prepare_all(self) -> List[TrainingSample]:
        """
        Подготавливает все сэмплы из датасета.
        
        Returns:
            Список подготовленных TrainingSample
        """
        samples = []
        for annotated_sample in self.loader:
            try:
                training_sample = self._prepare_sample(annotated_sample)
                samples.append(training_sample)
            except Exception as e:
                print(f"Warning: Failed to prepare sample {annotated_sample.image_id}: {e}")
                continue
        
        return samples
    
    def __iter__(self):
        """Итератор по подготовленным сэмплам."""
        for annotated_sample in self.loader:
            try:
                yield self._prepare_sample(annotated_sample)
            except Exception as e:
                print(f"Warning: Failed to prepare sample {annotated_sample.image_id}: {e}")
                continue
    
    def __len__(self) -> int:
        """Количество сэмплов в датасете."""
        return len(self.loader)
