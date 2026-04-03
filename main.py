import sys
import tkinter as tk
from pathlib import Path

# Thêm đường dẫn gốc vào sys.path để import được src
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Ép kiểu encoding UTF-8 để in được tiếng Việt trên console Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from src.utils.logger import AppLogger, handle_main_exception
from src.utils.config_manager import ConfigManager
from src.ui.desktop.gui_app import FaceRecognitionGUI

def main():
    """Điểm khởi đầu chính của toàn bộ hệ thống."""
    try:
        # 1. Khởi tạo cấu hình
        ConfigManager.load()
        AppLogger.info("Hệ thống điểm nhận diện khuôn mặt đang khởi động...")
        
        # 2. Khởi tạo Giao diện Tkinter
        root = tk.Tk()
        app = FaceRecognitionGUI(root)
        
        # 3. Xử lý đóng cửa sổ an toàn
        def on_close():
            if app._running_process:
                app._running_process.terminate()
            if app._dashboard_process:
                app._dashboard_process.terminate()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)
        
        # 4. Chạy vòng lặp ứng dụng
        root.mainloop()
        
    except Exception as e:
        handle_main_exception(e)

if __name__ == "__main__":
    main()
