import sys
import logging
import colorama
from colorama import Fore, Style

colorama.init()

class AppLogger:
    """Hệ thống Log tập trung cho toàn bộ ứng dụng."""
    
    @staticmethod
    def info(msg: str):
        print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} {msg}", flush=True)

    @staticmethod
    def success(msg: str):
        print(f"{Fore.GREEN}[✓ THANH CONG]{Style.RESET_ALL} {msg}", flush=True)

    @staticmethod
    def warning(msg: str):
        print(f"{Fore.YELLOW}[! CANH BAO]{Style.RESET_ALL} {msg}", flush=True)

    @staticmethod
    def error(msg: str, error_obj=None):
        error_msg = f"{Fore.RED}[LỖI NGHIÊM TRỌNG]{Style.RESET_ALL} {msg}"
        if error_obj:
            error_msg += f" | Chi tiết: {str(error_obj).encode('utf-8', 'ignore').decode('utf-8')}"
        print(error_msg, flush=True)

    @staticmethod
    def critical(msg: str):
        """Dành cho lỗi cực kỳ nghiêm trọng, GUI sẽ hiện Popup."""
        print(f"{Fore.RED}{Style.BRIGHT}[CRITICAL_ERROR]{Style.RESET_ALL} {msg}", flush=True)

# ─── Custom Exceptions ──────────────────────────────────────────

class CameraError(Exception):
    """Lỗi liên quan đến phần cứng camera."""
    pass

class ModelError(Exception):
    """Lỗi liên quan đến việc tải hoặc chạy AI models."""
    pass

class DatabaseError(Exception):
    """Lỗi kết nối hoặc thao tác cơ sở dữ liệu."""
    pass

# ─── Global Exception Handler ───────────────────────────────────

def handle_main_exception(e):
    """Xử lý lỗi ở cấp độ cao nhất của ứng dụng."""
    if isinstance(e, CameraError):
        AppLogger.critical(f"Lỗi Camera: {e}")
    elif isinstance(e, ModelError):
        AppLogger.critical(f"Lỗi mô hình AI: {e}")
    elif isinstance(e, DatabaseError):
        AppLogger.critical(f"Lỗi cơ sở dữ liệu: {e}")
    else:
        AppLogger.error("Lỗi không mong muốn", e)
    
    # Có thể thêm logic gửi thông báo khẩn cấp ở đây
