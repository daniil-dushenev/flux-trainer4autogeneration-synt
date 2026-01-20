"""
Главный скрипт для запуска пайплайна генерации синтетических данных.
"""
import argparse
from pathlib import Path
from typing import Optional, Dict, List
import logging
from dataclasses import replace
import itertools
import re
import hashlib

# Настраиваем окружение для скачивания на диск D
try:
    from setup_env import *
except ImportError:
    # Если setup_env не найден, пытаемся импортировать из текущей директории
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from setup_env import *

from data_preparation import TrainingDataPreparator
from lora_training import FluxLoRATrainer, LoRATrainingConfig
from generation import SyntheticImageGenerator
from output_handler import OutputHandler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

FLUX_MODEL_ALIASES = {
    "flux1-dev": "black-forest-labs/FLUX.1-dev",
    "flux1-schnell": "black-forest-labs/FLUX.1-schnell",
    "flux2-dev": "black-forest-labs/FLUX.2-dev",
    "flux2-klein-4b": "black-forest-labs/FLUX.2-klein-4B",
    "flux2-klein-9b": "black-forest-labs/FLUX.2-klein-9B",
    "nano": "black-forest-labs/FLUX.2-klein-4B",
}


def resolve_flux_model(model_id_or_alias: str) -> str:
    """Resolve a known alias to an HF model ID, otherwise pass through."""
    key = model_id_or_alias.strip().lower()
    return FLUX_MODEL_ALIASES.get(key, model_id_or_alias)


def resolve_device(device_arg: str) -> Optional[str]:
    key = device_arg.strip().lower()
    if key == "auto":
        return None
    if key in {"cpu", "cuda"}:
        return key
    return None


def resolve_cpu_mode(cpu_mode_arg: str) -> str:
    key = cpu_mode_arg.strip().lower()
    if key in {"mock", "copy", "error"}:
        return key
    return "mock"


def _slugify_label(label: str) -> str:
    """Make a filesystem-safe label for output directories."""
    stripped = label.strip()
    if re.fullmatch(r"[a-zA-Z0-9._-]+", stripped):
        return stripped
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", stripped)
    safe = safe.strip("._-")
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:8]
    if not safe:
        return f"class_{digest}"
    return f"{safe}_{digest}"


def _group_samples_by_class(samples) -> Dict[str, List]:
    grouped: Dict[str, List] = {}
    for sample in samples:
        if sample.class_label:
            grouped.setdefault(sample.class_label, []).append(sample)
    return grouped


def _expand_samples_for_count(samples: List, count: int) -> List:
    if count <= 0 or not samples:
        return []
    if count <= len(samples):
        return samples[:count]
    return list(itertools.islice(itertools.cycle(samples), count))


def _distribute_counts(total: int, group_sizes: Dict[str, int]) -> Dict[str, int]:
    if total <= 0:
        return {k: 0 for k in group_sizes}
    total_size = sum(group_sizes.values())
    if total_size == 0:
        return {k: 0 for k in group_sizes}
    raw = {k: total * (v / total_size) for k, v in group_sizes.items()}
    counts = {k: int(round(v)) for k, v in raw.items()}
    diff = total - sum(counts.values())
    keys = sorted(group_sizes.keys())
    idx = 0
    while diff != 0 and keys:
        key = keys[idx % len(keys)]
        if diff > 0:
            counts[key] += 1
            diff -= 1
        else:
            if counts[key] > 0:
                counts[key] -= 1
                diff += 1
        idx += 1
    return counts


def run_synthetic_generation_pipeline(
    jsonl_path: str | Path,
    output_dir: str | Path,
    images_root: Optional[str | Path] = None,
    lora_output_dir: Optional[str | Path] = None,
    lora_path: Optional[str | Path] = None,
    skip_training: bool = False,
    num_samples_to_generate: Optional[int] = None,
    training_config: Optional[LoRATrainingConfig] = None,
    controlnet_types: Optional[list[str]] = None,
    generation_params: Optional[dict] = None,
    flux_model: str = "flux1-dev",
    per_class_num_samples: Optional[int] = None,
    per_class_multiplier: Optional[float] = None,
    device: str = "auto",
    cpu_mode: str = "mock",
):
    """
    Запускает полный пайплайн генерации синтетических данных.
    
    Args:
        jsonl_path: Путь к predictions.jsonl с размеченными данными
        output_dir: Директория для сохранения результатов
        images_root: Корневая папка с изображениями
        lora_output_dir: Директория для сохранения LoRA адаптера
        lora_path: Путь к уже обученному LoRA адаптеру (если skip_training=True)
        skip_training: Пропустить обучение и использовать существующий адаптер
        num_samples_to_generate: Количество сэмплов для генерации (None = все)
        training_config: Конфигурация обучения (по умолчанию используется стандартная)
        controlnet_types: Типы ControlNet условий (["canny", "depth"])
        generation_params: Параметры генерации (num_inference_steps, guidance_scale, seed)
        flux_model: HF модель или алиас (flux1-dev, flux1-schnell, flux2-dev, flux2-klein-4b, flux2-klein-9b, nano)
        per_class_num_samples: Фиксированное количество изображений на класс (classification only)
        per_class_multiplier: Мультипликатор количества изображений на класс (classification only)
        device: Устройство для вычислений ("auto", "cpu", "cuda")
        cpu_mode: Режим работы на CPU ("mock", "copy", "error")
    """
    logger.info("Starting synthetic data generation pipeline")
    
    # Инициализируем параметры по умолчанию
    controlnet_types = controlnet_types or ["canny", "depth"]
    generation_params = generation_params or {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "seed": None,
    }
    
    # Шаг 1: Подготовка данных
    logger.info("Step 1: Preparing data")
    preparator = TrainingDataPreparator(
        jsonl_path=jsonl_path,
        images_root=images_root,
        controlnet_types=controlnet_types,
    )
    
    samples = preparator.prepare_all()
    logger.info(f"Prepared {len(samples)} training samples")
    
    classification_only = all(sample.task == "classification" for sample in samples)
    resolved_flux_model = resolve_flux_model(flux_model)
    resolved_device = resolve_device(device)
    resolved_cpu_mode = resolve_cpu_mode(cpu_mode)

    # Шаг 2: Обучение LoRA адаптера (если нужно)
    if classification_only:
        logger.info("Step 2: Training LoRA adapters per class (classification)")
        grouped = _group_samples_by_class(samples)
        if not grouped:
            raise ValueError("No classification samples with class labels found")

        lora_paths: Dict[str, Path] = {}
        base_output_dir = Path(lora_output_dir) if lora_output_dir else None

        if skip_training:
            if lora_path is None:
                raise ValueError("lora_path must be provided when skip_training=True")
            lora_base_dir = Path(lora_path)
            if not lora_base_dir.is_dir():
                raise ValueError("For classification mode, lora_path must be a directory with per-class LoRA adapters")
            for class_label in grouped:
                class_dir = lora_base_dir / _slugify_label(class_label)
                if not class_dir.exists():
                    raise ValueError(f"Missing LoRA adapter for class {class_label}: {class_dir}")
                lora_paths[class_label] = class_dir
        else:
            if training_config is None:
                training_config = LoRATrainingConfig(
                    controlnet_types=controlnet_types,
                )
            if base_output_dir is None:
                base_output_dir = Path(training_config.output_dir)
            for class_label, class_samples in grouped.items():
                class_dir = base_output_dir / _slugify_label(class_label)
                class_config = replace(
                    training_config,
                    output_dir=str(class_dir),
                    base_model=resolved_flux_model,
                    device=resolved_device,
                    cpu_mode=resolved_cpu_mode,
                )
                trainer = FluxLoRATrainer(class_config)
                lora_path_for_class = trainer.train(class_samples)
                trainer.save_config()
                lora_paths[class_label] = lora_path_for_class
                logger.info(f"LoRA adapter for class {class_label} saved to {lora_path_for_class}")

        # Шаг 3: Генерация синтетических изображений
        logger.info("Step 3: Generating synthetic images per class")
        output_handler = OutputHandler(output_dir)

        group_sizes = {k: len(v) for k, v in grouped.items()}
        if per_class_num_samples is not None:
            class_counts = {k: per_class_num_samples for k in grouped}
        elif per_class_multiplier is not None:
            class_counts = {k: max(1, int(round(len(v) * per_class_multiplier))) for k, v in grouped.items()}
        elif num_samples_to_generate is not None:
            class_counts = _distribute_counts(num_samples_to_generate, group_sizes)
        else:
            class_counts = {k: len(v) for k, v in grouped.items()}

        total_generated = 0
        for class_label, class_samples in grouped.items():
            count = class_counts.get(class_label, 0)
            samples_to_generate = _expand_samples_for_count(class_samples, count)
            if not samples_to_generate:
                continue
            generator = SyntheticImageGenerator(
                lora_path=lora_paths[class_label],
                base_model=resolved_flux_model,
                device=resolved_device,
                controlnet_types=controlnet_types,
                cpu_mode=resolved_cpu_mode,
            )
            generated_images = generator.generate_batch(
                samples_to_generate,
                num_inference_steps=generation_params.get("num_inference_steps", 50),
                guidance_scale=generation_params.get("guidance_scale", 7.5),
                seed=generation_params.get("seed"),
            )
            for image, sample in zip(generated_images, samples_to_generate):
                output_handler.save(
                    image=image,
                    sample=sample,
                    source_image_id=sample.image_id,
                    lora_path=str(lora_paths[class_label]),
                    additional_metadata={
                        "class_label": class_label,
                        "flux_model": resolved_flux_model,
                    },
                )
            total_generated += len(generated_images)
        logger.info(f"Generated {total_generated} images in classification mode")
    else:
        logger.info("Step 2: Training LoRA adapter (single, detection or mixed)")
        if num_samples_to_generate:
            samples = samples[:num_samples_to_generate]
            logger.info(f"Limiting to {num_samples_to_generate} samples for generation")
        if not skip_training:
            if training_config is None:
                training_config = LoRATrainingConfig(
                    controlnet_types=controlnet_types,
                )
            training_config = replace(
                training_config,
                base_model=resolved_flux_model,
                device=resolved_device,
                cpu_mode=resolved_cpu_mode,
            )
            if lora_output_dir:
                training_config.output_dir = str(lora_output_dir)

            trainer = FluxLoRATrainer(training_config)
            lora_path = trainer.train(samples)
            trainer.save_config()
            logger.info(f"LoRA adapter trained and saved to {lora_path}")
        else:
            if lora_path is None:
                raise ValueError("lora_path must be provided when skip_training=True")
            logger.info(f"Step 2: Skipping training, using existing LoRA at {lora_path}")

        # Шаг 3: Генерация синтетических изображений
        logger.info("Step 3: Generating synthetic images")

        generator = SyntheticImageGenerator(
            lora_path=lora_path,
            base_model=resolved_flux_model,
            device=resolved_device,
            controlnet_types=controlnet_types,
            cpu_mode=resolved_cpu_mode,
        )

        generated_images = generator.generate_batch(
            samples,
            num_inference_steps=generation_params.get("num_inference_steps", 50),
            guidance_scale=generation_params.get("guidance_scale", 7.5),
            seed=generation_params.get("seed"),
        )
        logger.info(f"Generated {len(generated_images)} images")

        # Шаг 4: Сохранение результатов
        logger.info("Step 4: Saving results")

        output_handler = OutputHandler(output_dir)
        saved_ids = output_handler.save_batch(
            images=generated_images,
            samples=samples,
            lora_path=str(lora_path),
        )
        logger.info(f"Saved {len(saved_ids)} images to {output_dir}")
    
    logger.info("Pipeline completed successfully!")


def main():
    """Точка входа для запуска из командной строки."""
    parser = argparse.ArgumentParser(
        description="Генерация синтетических данных через FLUX + LoRA + ControlNet"
    )
    
    parser.add_argument(
        "jsonl_path",
        type=str,
        help="Путь к predictions.jsonl с размеченными данными"
    )
    
    parser.add_argument(
        "output_dir",
        type=str,
        help="Директория для сохранения результатов"
    )
    
    parser.add_argument(
        "--images-root",
        type=str,
        default=None,
        help="Корневая папка с изображениями (опционально)"
    )
    
    parser.add_argument(
        "--lora-output-dir",
        type=str,
        default=None,
        help="Директория для сохранения LoRA адаптера"
    )
    
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Путь к уже обученному LoRA адаптеру (если --skip-training)"
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Пропустить обучение и использовать существующий адаптер"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Количество сэмплов для генерации (по умолчанию все; в классификации распределяется пропорционально)"
    )

    parser.add_argument(
        "--per-class-num-samples",
        type=int,
        default=None,
        help="Фиксированное количество изображений для каждого класса (classification only)"
    )

    parser.add_argument(
        "--per-class-multiplier",
        type=float,
        default=None,
        help="Мультипликатор числа изображений на класс (classification only)"
    )
    
    parser.add_argument(
        "--controlnet-types",
        nargs="+",
        default=["canny", "depth"],
        choices=["canny", "depth"],
        help="Типы ControlNet условий"
    )
    
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Количество шагов генерации"
    )

    parser.add_argument(
        "--flux-model",
        type=str,
        default="flux1-dev",
        help="HF модель или алиас (flux1-dev, flux1-schnell, flux2-dev, flux2-klein-4b, flux2-klein-9b, nano)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Устройство для вычислений"
    )

    parser.add_argument(
        "--cpu-mode",
        type=str,
        default="mock",
        choices=["mock", "copy", "error"],
        help="Режим CPU: mock = заглушки, copy = копия исходных изображений, error = запрет CPU"
    )
    
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale для генерации"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed для воспроизводимости"
    )
    
    # Параметры обучения
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate для обучения LoRA"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size для обучения"
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Количество эпох обучения"
    )
    
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="Rank для LoRA адаптера"
    )
    
    args = parser.parse_args()
    
    # Формируем конфигурацию обучения
    training_config = LoRATrainingConfig(
        rank=args.lora_rank,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        controlnet_types=args.controlnet_types,
        base_model=resolve_flux_model(args.flux_model),
        device=resolve_device(args.device),
        cpu_mode=resolve_cpu_mode(args.cpu_mode),
    )
    
    # Параметры генерации
    generation_params = {
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
    }
    
    # Запускаем пайплайн
    run_synthetic_generation_pipeline(
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        images_root=args.images_root,
        lora_output_dir=args.lora_output_dir,
        lora_path=args.lora_path,
        skip_training=args.skip_training,
        num_samples_to_generate=args.num_samples,
        training_config=training_config,
        controlnet_types=args.controlnet_types,
        generation_params=generation_params,
        flux_model=args.flux_model,
        per_class_num_samples=args.per_class_num_samples,
        per_class_multiplier=args.per_class_multiplier,
        device=args.device,
        cpu_mode=args.cpu_mode,
    )


if __name__ == "__main__":
    main()
