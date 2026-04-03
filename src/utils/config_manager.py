import os
import yaml
from pathlib import Path
from .logger import AppLogger

class ConfigManager:
    """Quản lý cấu hình YAML tập trung cho toàn ứng dụng."""
    
    _config = None
    CONFIG_FILE = "config.yaml"

    @classmethod
    def load(cls):
        """Tải cấu hình từ config.yaml."""
        if cls._config is None:
            config_path = Path(cls.CONFIG_FILE)
            if not config_path.exists():
                AppLogger.warning(f"Không tìm thấy {cls.CONFIG_FILE}. Sử dụng cấu hình mặc định.")
                cls._config = {}
            else:
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        cls._config = yaml.safe_load(f) or {}
                except Exception as e:
                    AppLogger.error(f"Lỗi khi đọc cấu hình: {e}")
                    cls._config = {}
        return cls._config

    @classmethod
    def get(cls, section, key=None, default=None):
        """Lấy giá trị từ cấu hình."""
        config = cls.load()
        if section not in config:
            return default
        
        if key is None:
            return config[section]
            
        return config[section].get(key, default)

    @classmethod
    def save(cls, new_config: dict):
        """Lưu cấu hình mới vào config.yaml."""
        try:
            with open(cls.CONFIG_FILE, "w", encoding="utf-8") as f:
                yaml.dump(new_config, f, allow_unicode=True, sort_keys=False)
            cls._config = new_config
            return True
        except Exception as e:
            AppLogger.error(f"Lỗi khi lưu cấu hình: {e}")
            return False

    @staticmethod
    def get_env_or_config(env_name, section, key, default):
        """Ưu tiên lấy từ biến môi trường (do GUI truyền), nếu không có thì lấy từ Config."""
        val = os.environ.get(env_name)
        if val is not None:
            # Tự động convert type nếu là số
            if val.isdigit():
                return int(val)
            if val in ["0", "1"]: # Boolean / Flag
                return val == "1"
            return val
        
        return ConfigManager.get(section, key, default)
