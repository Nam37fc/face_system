import sys
from pathlib import Path

# Ép kiểu encoding UTF-8 để in được tiếng Việt trên console Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from src.core.trainer import ModelTrainer
from src.utils.logger import handle_main_exception

if __name__ == "__main__":
    try:
        trainer = ModelTrainer()
        trainer.train_svm()
    except Exception as e:
        handle_main_exception(e)
        sys.exit(1)
