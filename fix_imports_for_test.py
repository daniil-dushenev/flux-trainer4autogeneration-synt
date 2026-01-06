"""
Вспомогательный скрипт для исправления относительных импортов в модулях.
Создает временные версии файлов с абсолютными импортами.
"""
import shutil
from pathlib import Path
import re

def fix_imports_in_file(file_path: Path, output_path: Path = None):
    """Исправляет относительные импорты в файле."""
    if output_path is None:
        output_path = file_path
    
    content = file_path.read_text(encoding='utf-8')
    
    # Заменяем относительные импорты
    # from .module import X -> from module import X (для текущей директории)
    content = re.sub(r'from \.(\w+) import', r'from \1 import', content)
    # from .utils.module import X -> from utils.module import X
    content = re.sub(r'from \.utils\.(\w+) import', r'from utils.\1 import', content)
    
    output_path.write_text(content, encoding='utf-8')
    return output_path

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    
    files_to_fix = [
        'data_preparation.py',
        'lora_training.py',
        'generation.py',
        'output_handler.py',
        'runner.py',
    ]
    
    # Создаем резервные копии и исправляем
    backups = {}
    for file_name in files_to_fix:
        file_path = current_dir / file_name
        if file_path.exists():
            backup_path = current_dir / f"{file_name}.bak"
            shutil.copy2(file_path, backup_path)
            backups[file_name] = backup_path
            fix_imports_in_file(file_path)
            print(f"Fixed imports in {file_name}")
    
    print("Files fixed. Run tests, then restore with restore_imports.py")

