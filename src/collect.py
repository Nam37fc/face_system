import os
import sys
from pathlib import Path

# Ép kiểu encoding UTF-8 để in được tiếng Việt trên console Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Thêm dự án vào sys.path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from src.services.collection import FaceDataCollector
from src.utils.logger import handle_main_exception

if __name__ == "__main__":
    try:
        user_name = os.environ.get("FACE_USER_NAME", "Unknown")
        num_images = int(os.environ.get("FACE_NUM_IMAGES", 100))
        
        collector = FaceDataCollector(num_images=num_images)
        collector.collect(user_name)
    except Exception as e:
        handle_main_exception(e)
        sys.exit(1)
