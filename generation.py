"""
Модуль для генерации синтетических изображений с использованием обученного LoRA + ControlNetUnion.
"""
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import torch
from PIL import Image
import numpy as np

from data_preparation import TrainingSample


class SyntheticImageGenerator:
    """
    Класс для генерации синтетических изображений с использованием LoRA + ControlNetUnion (ControlNet++).
    """
    
    def __init__(
        self,
        lora_path: str | Path,
        base_model: str = "black-forest-labs/FLUX.1-dev",
        device: Optional[str] = None,
        controlnet_types: List[str] = None,
        cpu_mode: str = "mock",
    ):
        """
        Инициализация генератора.
        
        Args:
            lora_path: Путь к обученному LoRA адаптеру
            base_model: Базовая модель FLUX
            device: Устройство для вычислений ("cuda" или "cpu")
            controlnet_types: Типы ControlNet условий (["canny", "depth"]) для ControlNetUnion
        """
        self.lora_path = Path(lora_path)
        self.base_model = base_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.controlnet_types = controlnet_types or ["canny", "depth"]
        self.cpu_mode = cpu_mode
        
        # Модель загружается при первом использовании
        self.pipe = None
        self._load_model()
    
    def _load_model(self):
        """Загружает модель с LoRA адаптером."""
        if self.device == "cpu":
            print("Note: Running on CPU - model loading will be skipped for testing")
            print("      Real generation requires GPU and FLUX model")
            print(f"      CPU mode: {self.cpu_mode}")
            # Создаем mock pipe для тестирования логики
            self.pipe = MockPipeline()
            return
        
        try:
            # Попытка загрузить через diffusion-pipe
            try:
                from diffusion_pipe import load_pipeline_with_lora
                
                self.pipe = load_pipeline_with_lora(
                    base_model=self.base_model,
                    lora_path=str(self.lora_path),
                    device=self.device,
                )
                return
            except ImportError:
                pass
            
            # Fallback через diffusers
            try:
                from diffusers import DiffusionPipeline
                import torch
                
                print("Loading model with diffusers (fallback mode)")
                
                # Загружаем базовую модель
                self.pipe = DiffusionPipeline.from_pretrained(
                    self.base_model,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
                self.pipe = self.pipe.to(self.device)
                
                # Загружаем LoRA адаптер если доступен
                if self.lora_path.exists():
                    try:
                        self.pipe.load_lora_weights(str(self.lora_path))
                    except Exception as e:
                        print(f"Warning: Could not load LoRA weights: {e}")
                
                print("Note: FLUX may require specialized loading. Refer to FLUX documentation.")
                return
                
            except ImportError:
                raise ImportError(
                    "Neither diffusion-pipe nor diffusers available. "
                    "Please install one of them."
                )
        
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        canny_condition: Optional[Image.Image] = None,
        depth_condition: Optional[Image.Image] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
    ) -> Image.Image:
        """
        Генерирует одно изображение по промпту с ControlNet условиями.
        
        Args:
            prompt: Текстовый промпт
            canny_condition: Canny edge map (опционально)
            depth_condition: Depth map (опционально)
            num_inference_steps: Количество шагов генерации
            guidance_scale: Guidance scale
            seed: Seed для воспроизводимости
            width: Ширина изображения
            height: Высота изображения
        
        Returns:
            Сгенерированное изображение (PIL Image)
        """
        if self.pipe is None:
            self._load_model()
        
        # Устанавливаем seed если указан
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Подготавливаем ControlNetUnion условия
        # ControlNetUnion поддерживает множественные условия через image_list и control_type
        try:
            # Попытка генерации через ControlNetUnion (если поддерживается)
            if hasattr(self.pipe, "generate_with_controlnet_union"):
                from utils.controlnet_union import prepare_controlnet_union_conditions
                
                union_image_list, union_type_list = prepare_controlnet_union_conditions(
                    canny_condition if canny_condition else depth_condition if depth_condition else Image.new("RGB", (width, height)),
                    self.controlnet_types
                )
                
                kwargs = {
                    "prompt": prompt,
                    "image_list": union_image_list,
                    "control_type": union_type_list,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "generator": generator,
                    "width": width,
                    "height": height,
                }
                
                result = self.pipe.generate_with_controlnet_union(**kwargs)
                return result.images[0]
            
            # Попытка генерации через стандартный ControlNet (fallback)
            elif hasattr(self.pipe, "generate_with_controlnet"):
                kwargs = {
                    "prompt": prompt,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "generator": generator,
                    "width": width,
                    "height": height,
                }
                
                if canny_condition and "canny" in self.controlnet_types:
                    kwargs["canny_image"] = canny_condition
                if depth_condition and "depth" in self.controlnet_types:
                    kwargs["depth_image"] = depth_condition
                
                result = self.pipe.generate_with_controlnet(**kwargs)
                return result.images[0]
            
            # Fallback: генерация без ControlNet (только промпт)
            elif hasattr(self.pipe, "__call__"):
                result = self.pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    width=width,
                    height=height,
                )
                return result.images[0]
            else:
                raise RuntimeError("Pipeline does not support generation")
        
        except Exception as e:
            print(f"Error during generation: {e}")
            raise
    
    def generate_batch(
        self,
        samples: List[TrainingSample],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Генерирует батч изображений.
        
        Args:
            samples: Список TrainingSample с промптами и ControlNet условиями
            num_inference_steps: Количество шагов генерации
            guidance_scale: Guidance scale
            seed: Базовый seed (будет инкрементироваться для каждого изображения)
        
        Returns:
            Список сгенерированных изображений
        """
        if self.device == "cpu" and self.cpu_mode == "copy":
            return [sample.image.copy() for sample in samples]

        images = []
        current_seed = seed
        
        for i, sample in enumerate(samples):
            if current_seed is not None:
                current_seed = seed + i
            
            image = self.generate(
                prompt=sample.prompt,
                canny_condition=sample.canny_condition,
                depth_condition=sample.depth_condition,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=current_seed,
            )
            images.append(image)
        
        return images


class MockPipeline:
    """Mock pipeline для тестирования на CPU."""
    def __call__(self, **kwargs):
        from PIL import Image
        # Возвращаем mock изображение
        return type('MockResult', (), {'images': [Image.new("RGB", (512, 512), (128, 128, 128))]})()
    
    def generate_with_controlnet_union(self, **kwargs):
        return self(**kwargs)
    
    def generate_with_controlnet(self, **kwargs):
        return self(**kwargs)
