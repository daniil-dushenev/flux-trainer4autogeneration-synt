"""
Утилиты для генерации ControlNetUnion условий (ControlNet++ из ControlNetPlus).

Поддержка унифицированного ControlNet с множественными условиями.
Типы контроля:
0 -- openpose
1 -- depth
2 -- thick line (scribble/hed/softedge/ted-512)
3 -- thin line (canny/mlsd/lineart/animelineart/ted-1280)
4 -- normal
5 -- segment
"""
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, List
import cv2


# Маппинг имен условий на control type id
CONTROL_TYPE_MAPPING = {
    "openpose": 0,
    "depth": 1,
    "scribble": 2,
    "hed": 2,
    "softedge": 2,
    "ted-512": 2,
    "canny": 3,
    "mlsd": 3,
    "lineart": 3,
    "animelineart": 3,
    "ted-1280": 3,
    "normal": 4,
    "segment": 5,
}


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
        PIL Image с Canny edges (RGB)
    """
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    edges = cv2.Canny(gray, low_threshold, high_threshold)
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
        model_name: Название модели depth estimation
        device: Устройство для вычислений ("cuda" или "cpu", None для auto)
    
    Returns:
        PIL Image с depth map (RGB, нормализованный)
    """
    import torch
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # На CPU используем mock depth map для тестирования
    if device == "cpu":
        print("Note: Running on CPU - using mock depth map for testing")
        # Создаем простой градиент как mock depth map
        width, height = image.size
        arr = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            # Градиент от светлого к темному
            arr[y, :] = int(255 * (1 - y / height))
        depth_image = Image.fromarray(arr, mode="L").convert("RGB")
        return depth_image
    
    # На GPU загружаем реальную модель
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError:
        raise ImportError(
            "transformers library is required for depth estimation. "
            "Install it with: pip install transformers"
        )
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    depth = predicted_depth.cpu().numpy()
    depth_min = depth.min()
    depth_max = depth.max()
    
    if depth_max > depth_min:
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = depth
    
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    depth_uint8 = depth_uint8.squeeze()
    
    depth_image = Image.fromarray(depth_uint8, mode="L").convert("RGB")
    
    return depth_image


def get_control_type_id(control_type: str) -> int:
    """
    Получает control type id для заданного типа контроля.
    
    Args:
        control_type: Имя типа контроля (например, "canny", "depth")
    
    Returns:
        Control type id (0-5)
    """
    control_type_lower = control_type.lower()
    if control_type_lower in CONTROL_TYPE_MAPPING:
        return CONTROL_TYPE_MAPPING[control_type_lower]
    else:
        raise ValueError(
            f"Unknown control type: {control_type}. "
            f"Supported types: {list(CONTROL_TYPE_MAPPING.keys())}"
        )


def prepare_controlnet_union_conditions(
    image: Image.Image,
    controlnet_types: List[str],
    canny_thresholds: Optional[Tuple[int, int]] = None,
    depth_model: Optional[str] = None
) -> Tuple[List[Image.Image], List[int]]:
    """
    Подготавливает условия для ControlNetUnion (ControlNet++).
    
    Args:
        image: Входное изображение
        controlnet_types: Список типов условий (например, ["canny", "depth"])
        canny_thresholds: Пороги для Canny (low, high), по умолчанию (100, 200)
        depth_model: Модель для depth estimation, по умолчанию "Intel/dpt-large"
    
    Returns:
        Tuple of (image_list, control_type_list):
        - image_list: Список изображений условий в порядке [cond1, cond2, 0, 0, 0, 0]
        - control_type: Список control type id в формате [1, 1, 0, 0, 0, 0]
        
    Пример:
        Для canny и depth:
        image_list = [canny_img, depth_img, 0, 0, 0, 0]
        control_type = [1, 1, 0, 0, 0, 0]  # где 1 означает использование условия
    """
    if canny_thresholds is None:
        canny_thresholds = (100, 200)
    
    if depth_model is None:
        depth_model = "Intel/dpt-large"
    
    # Генерируем изображения условий
    condition_images = []
    control_type_ids = []
    
    for ctrl_type in controlnet_types:
        if ctrl_type == "canny":
            canny_img = generate_canny_edges(
                image,
                low_threshold=canny_thresholds[0],
                high_threshold=canny_thresholds[1]
            )
            condition_images.append(canny_img)
            control_type_ids.append(get_control_type_id("canny"))
        
        elif ctrl_type == "depth":
            depth_img = generate_depth_map(image, model_name=depth_model)
            condition_images.append(depth_img)
            control_type_ids.append(get_control_type_id("depth"))
        
        else:
            print(f"Warning: Control type '{ctrl_type}' not implemented yet, skipping")
            continue
    
    # Формируем список изображений в формате ControlNetUnion
    # Формат: [cond1, cond2, 0, 0, 0, 0] (всего 6 элементов)
    image_list = condition_images.copy()
    while len(image_list) < 6:
        image_list.append(0)  # Заполняем нулями
    
    # Формируем control_type в формате [1, 1, 0, 0, 0, 0]
    control_type = [1 if i < len(condition_images) else 0 for i in range(6)]
    
    return image_list, control_type


def prepare_controlnet_union_single_condition(
    image: Image.Image,
    control_type: str,
    canny_thresholds: Optional[Tuple[int, int]] = None,
    depth_model: Optional[str] = None
) -> Tuple[List[Image.Image], List[int]]:
    """
    Подготавливает одно условие для ControlNetUnion.
    
    Args:
        image: Входное изображение
        control_type: Тип условия ("canny" или "depth")
        canny_thresholds: Пороги для Canny (low, high)
        depth_model: Модель для depth estimation
    
    Returns:
        Tuple of (image_list, control_type_list) для одного условия
    """
    return prepare_controlnet_union_conditions(
        image,
        [control_type],
        canny_thresholds=canny_thresholds,
        depth_model=depth_model
    )


