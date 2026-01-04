"""
Утилиты для генерации ControlNet условий (canny edges и depth maps).
"""
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict
import cv2


def generate_canny_edges(
    image: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200
) -> Image.Image:
    """
    Генерирует Canny edge map из изображения.
    
    Args:
        image: Входное изображение (PIL Image)
        low_threshold: Нижний порог для детектора Кэнни
        high_threshold: Верхний порог для детектора Кэнни
    
    Returns:
        PIL Image с Canny edges (grayscale)
    """
    # Конвертируем PIL в numpy array
    img_array = np.array(image)
    
    # Если изображение RGB, конвертируем в grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Применяем Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Конвертируем обратно в PIL Image
    edges_image = Image.fromarray(edges).convert("RGB")
    
    return edges_image


def generate_depth_map(
    image: Image.Image,
    model_name: str = "Intel/dpt-large",
    device: Optional[str] = None
) -> Image.Image:
    """
    Генерирует depth map из изображения используя модель depth estimation.
    
    Args:
        image: Входное изображение (PIL Image)
        model_name: Название модели depth estimation (Intel/dpt-large, MiDaS и т.д.)
        device: Устройство для вычислений ("cuda" или "cpu", None для auto)
    
    Returns:
        PIL Image с depth map (grayscale, нормализованный)
    """
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch
    except ImportError:
        raise ImportError(
            "transformers library is required for depth estimation. "
            "Install it with: pip install transformers"
        )
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Загружаем модель и процессор
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Подготовка изображения
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Получение depth map
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Конвертация в numpy и нормализация
    depth = predicted_depth.cpu().numpy()
    depth_min = depth.min()
    depth_max = depth.max()
    
    if depth_max > depth_min:
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = depth
    
    # Масштабирование до 0-255 и конвертация в PIL
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    depth_uint8 = depth_uint8.squeeze()  # Убираем лишние размерности
    
    depth_image = Image.fromarray(depth_uint8, mode="L").convert("RGB")
    
    return depth_image


def prepare_controlnet_conditions(
    image: Image.Image,
    controlnet_types: list[str],
    canny_thresholds: Optional[Tuple[int, int]] = None,
    depth_model: Optional[str] = None
) -> Dict[str, Image.Image]:
    """
    Генерирует все необходимые ControlNet условия для изображения.
    
    Args:
        image: Входное изображение
        controlnet_types: Список типов условий (["canny", "depth"])
        canny_thresholds: Пороги для Canny (low, high), по умолчанию (100, 200)
        depth_model: Модель для depth estimation, по умолчанию "Intel/dpt-large"
    
    Returns:
        Словарь с условиями: {"canny": Image, "depth": Image}
    """
    conditions = {}
    
    if canny_thresholds is None:
        canny_thresholds = (100, 200)
    
    if depth_model is None:
        depth_model = "Intel/dpt-large"
    
    for ctrl_type in controlnet_types:
        if ctrl_type == "canny":
            conditions["canny"] = generate_canny_edges(
                image,
                low_threshold=canny_thresholds[0],
                high_threshold=canny_thresholds[1]
            )
        elif ctrl_type == "depth":
            conditions["depth"] = generate_depth_map(image, model_name=depth_model)
        else:
            print(f"Warning: Unknown controlnet type: {ctrl_type}")
    
    return conditions

