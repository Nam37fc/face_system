import os
import re
import urllib.parse
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from ..utils.logger import AppLogger
from ..utils.config_manager import ConfigManager
from dotenv import load_dotenv

# Tải các biến môi trường từ .env
load_dotenv()

class DatabaseManager:
    """Quản lý kết nối MongoDB Atlas (Singleton)."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._init_db()
        return cls._instance

    def _init_db(self):
        """Khởi tạo cấu trúc cơ sở dữ liệu và các Index cần thiết."""
        self.client = None
        self.db = None
        self.attendance_col = None
        
        db = self.get_database()
        if db is not None:
            try:
                self.attendance_col = db["attendance"]
                
                # Khởi tạo các Index tương ứng với PostgreSQL để tăng tốc độ truy vấn
                self.attendance_col.create_index("name")
                self.attendance_col.create_index("date")
                self.attendance_col.create_index([("timestamp", -1)])
                
                AppLogger.success("Cấu trúc Database 'attendance' trên MongoDB Atlas đã được đồng bộ.")
            except Exception as e:
                AppLogger.warning(f"Lỗi khởi tạo index: {e}")

    def get_connection_uri(self):
        """Lấy URI kết nối từ biến môi trường .env hoặc config.yaml và tự động chuẩn hóa."""
        # Ưu tiên lấy từ .env
        uri = os.getenv("MONGODB_URI")
        if not uri:
            # Nếu không có, thử lấy từ config.yaml
            config = ConfigManager.load()
            db_cfg = config.get("database", {})
            uri = db_cfg.get("mongodb_uri", "")
            
        if not uri:
            return uri

        # Tự động mã hóa ký tự đặc biệt trong username/password nếu có để tránh lỗi InvalidURI
        try:
            if "://" in uri and "@" in uri:
                scheme_part, rest = uri.split("://", 1)
                r_parts = rest.rsplit("@", 1)
                if len(r_parts) == 2:
                    creds, host_part = r_parts
                    if ":" in creds:
                        user, password = creds.split(":", 1)
                        # Kiểm tra xem password đã được percent-encode chưa
                        is_quoted = re.search(r'%[0-9a-fA-F]{2}', password)
                        if not is_quoted:
                            quoted_user = urllib.parse.quote_plus(user)
                            quoted_password = urllib.parse.quote_plus(password)
                            uri = f"{scheme_part}://{quoted_user}:{quoted_password}@{host_part}"
        except Exception as e:
            AppLogger.warning(f"Lỗi tự động chuẩn hóa URI: {e}")
            
        return uri

    def get_database(self):
        """Trả về instance database của MongoDB."""
        if self.db is not None:
            return self.db
            
        uri = self.get_connection_uri()
        if not uri:
            AppLogger.error("Không tìm thấy cấu hình MONGODB_URI trong .env hoặc config.yaml!")
            return None
            
        try:
            # Khởi tạo MongoClient với timeout 5 giây
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Kiểm tra kết nối
            self.client.admin.command('ping')
            
            # Phân tách tên database từ URI hoặc mặc định là 'face_recognition'
            db_name = "face_recognition"
            try:
                parsed = urllib.parse.urlparse(uri)
                path = parsed.path.strip("/")
                if path:
                    db_name = path
            except Exception:
                pass
                
            self.db = self.client[db_name]
            return self.db
        except ConnectionFailure as e:
            AppLogger.error(f"Không thể kết nối đến MongoDB Atlas: {e}")
            return None
        except Exception as e:
            AppLogger.error(f"Lỗi kết nối MongoDB: {e}")
            return None

# Singleton instance
db_manager = DatabaseManager()
