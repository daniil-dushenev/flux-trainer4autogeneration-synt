"""
Утилиты для генерации текстовых промптов из аннотаций.
"""
from typing import Dict, Any, List, Optional


def generate_classification_prompt(
    label: str,
    template: Optional[str] = None
) -> str:
    """
    Генерирует промпт для classification задачи.
    
    Args:
        label: Предсказанный класс
        template: Шаблон промпта (по умолчанию: "a photo of a {label}")
    
    Returns:
        Текстовый промпт
    """
    if template is None:
        template = "a photo of a {label}"
    
    return template.format(label=label)


def generate_detection_prompt(
    detections: List[Dict[str, Any]],
    template: Optional[str] = None
) -> str:
    """
    Генерирует промпт для detection задачи на основе детекций.
    
    Args:
        detections: Список детекций, каждая содержит "label" и "bbox"
        template: Шаблон промпта (по умолчанию: "a photo with {labels}")
    
    Returns:
        Текстовый промпт
    """
    if not detections:
        return "a photo"
    
    # Извлекаем уникальные метки
    labels = []
    seen = set()
    for det in detections:
        label = det.get("label", "")
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
    
    if not labels:
        return "a photo"
    
    if template is None:
        if len(labels) == 1:
            template = "a photo with {labels}"
        elif len(labels) == 2:
            template = "a photo with {label1} and {label2}"
        else:
            # Для множества объектов: "a photo with X, Y, and Z"
            template = "a photo with " + ", ".join(["{label" + str(i+1) + "}" for i in range(len(labels)-1)]) + " and {label" + str(len(labels)) + "}"
    
    # Форматируем шаблон
    format_dict = {f"label{i+1}": labels[i] for i in range(min(len(labels), 10))}
    format_dict["labels"] = ", ".join(labels)
    
    try:
        return template.format(**format_dict)
    except KeyError:
        # Если шаблон не поддерживает все метки, используем простой формат
        return f"a photo with {', '.join(labels)}"


def generate_prompt_from_annotations(
    task: str,
    annotations: Dict[str, Any],
    template: Optional[str] = None
) -> str:
    """
    Генерирует промпт из аннотаций в зависимости от типа задачи.
    
    Args:
        task: Тип задачи ("classification" | "detection")
        annotations: Словарь с аннотациями
        template: Опциональный шаблон промпта
    
    Returns:
        Текстовый промпт
    """
    if task == "classification":
        label = annotations.get("label", "")
        return generate_classification_prompt(label, template)
    
    elif task == "detection":
        detections = annotations.get("detections", [])
        return generate_detection_prompt(detections, template)
    
    else:
        # Для неизвестных задач возвращаем общий промпт
        return "a photo"

