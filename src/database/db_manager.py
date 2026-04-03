import psycopg2
import os
from ..utils.logger import AppLogger
from ..utils.config_manager import ConfigManager

class DatabaseManager:
    """Quản lý kết nối PostgreSQL (Singleton)."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._init_db()
        return cls._instance

    def _init_db(self):
        """Khởi tạo cấu trúc bảng nếu chưa có."""
        self.conn = None
        conn = self.get_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    # Tạo bảng attendance với cấu trúc đầy đủ cho Dashboard
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS attendance (
                            id SERIAL PRIMARY KEY,
                            name VARCHAR(100) NOT NULL,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            date DATE DEFAULT CURRENT_DATE,
                            time TIME DEFAULT CURRENT_TIME,
                            confidence FLOAT NOT NULL,
                            snapshot_path TEXT
                        );
                    """)
                    # Tự động nâng cấp Schema nếu là bảng cũ
                    cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS date DATE")
                    cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS time TIME")
                    cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS confidence FLOAT")
                    cur.execute("ALTER TABLE attendance ADD COLUMN IF NOT EXISTS snapshot_path TEXT")
                    
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_attendance_name ON attendance(name);
                        CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
                        CREATE INDEX IF NOT EXISTS idx_attendance_time ON attendance(timestamp);
                    """)
                AppLogger.success("Cấu trúc Database 'attendance' đã được đồng bộ.")
            except Exception as e:
                AppLogger.warning(f"Lỗi khởi tạo bảng: {e}")

    def _get_params(self):
        """Lấy thông số mới nhất từ ConfigManager."""
        config = ConfigManager.load()
        db_cfg = config.get("database", {})
        
        return {
            "host": db_cfg.get("host", "localhost"),
            "port": db_cfg.get("port", 5433), # Mặc định 5433 cho máy bạn
            "dbname": db_cfg.get("dbname", "face_recognition"),
            "user": db_cfg.get("user", "postgres"),
            "password": db_cfg.get("password", "")
        }

    def get_connection(self):
        """Trả về kết nối hiện có hoặc tạo mới."""
        params = self._get_params()
        if self.conn is None or self.conn.closed:
            try:
                self.conn = psycopg2.connect(
                    host=params["host"],
                    port=params["port"],
                    database=params["dbname"],
                    user=params["user"],
                    password=params["password"]
                )
                self.conn.autocommit = True
            except Exception as e:
                # Chỉ in warning lần đầu để tránh spam console
                return None
        return self.conn

# Singleton instance
db_manager = DatabaseManager()
