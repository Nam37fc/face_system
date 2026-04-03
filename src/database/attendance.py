import csv
import time
import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime, date, timedelta
from threading import Lock
from .db_manager import DatabaseManager
from ..utils.logger import AppLogger
from ..services.notifier import TelegramNotifier

class AttendanceLogger:
    """
    Quản lý nghiệp vụ điểm danh: Ghi log, thống kê, xuất báo cáo.
    Thread-safe: dùng Lock cho các thao tác ghi.
    """

    def __init__(
        self,
        snapshots_dir: str = "data/snapshots",
        cooldown_seconds: int = 30,
        log_unknown: bool = False,
        send_notify: bool = True,
    ):
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown_seconds = cooldown_seconds
        self.log_unknown = log_unknown
        self.send_notify = send_notify
        
        self._lock = Lock()
        self._last_log: dict[str, float] = {}  # {name: last_log_timestamp}
        self._last_global_log = 0.0            # Cooldown toàn hệ thống
        self._db = DatabaseManager()
        self.notifier = TelegramNotifier()

    def _save_snapshot(self, name: str, frame: np.ndarray) -> str | None:
        """Lưu ảnh snapshot khi nhận diện thành công."""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            filename = f"{name}_{ts}.jpg"
            filepath = self.snapshots_dir / filename
            cv2.imwrite(str(filepath), frame)
            return filename
        except Exception as e:
            AppLogger.warning(f"Không thể lưu snapshot: {e}")
            return None

    def log(self, name: str, confidence: float, frame: np.ndarray = None) -> bool:
        """Ghi nhận điểm danh vào database + lưu snapshot + bản sao lưu."""
        if name == "Unknown" and not self.log_unknown:
            return False

        now = time.time()
        
        # 1. Kiểm tra Cooldown Toàn hệ thống và theo Tên
        if now - self._last_global_log < self.cooldown_seconds:
            return False
        if name in self._last_log and (now - self._last_log[name] < self.cooldown_seconds):
            return False

        # 2. Lưu snapshot
        snapshot_file = self._save_snapshot(name, frame) if frame is not None else None

        # 3. Ghi vào PostgreSQL
        dt = datetime.now()
        sql = "INSERT INTO attendance (name, timestamp, date, time, confidence, snapshot_path) VALUES (%s, %s, %s, %s, %s, %s)"
        
        with self._lock:
            # Re-check inside lock
            if now - self._last_global_log < self.cooldown_seconds:
                return False
            
            self._last_global_log = now
            self._last_log[name] = now
            
            conn = self._db.get_connection()
            if conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(sql, (str(name), dt, dt.date(), dt.time(), float(confidence), snapshot_file))
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    AppLogger.error(f"Không thể ghi attendance vào DB: {e}")
                    return False
            else:
                AppLogger.warning("DB không sẵn sàng, bỏ qua ghi log SQL.")

        # 3. Cập nhật cooldown và thông báo
        self._last_log[name] = now
        AppLogger.success(f"{name} — {dt.strftime('%H:%M:%S')} — {confidence:.1%}")
        
        # 4. Gửi thông báo Telegram
        if self.send_notify:
            msg = f"🔔 ĐIỂM DANH: {name}\n⏰ Lúc: {dt.strftime('%H:%M:%S')}\n🎯 Độ tin cậy: {confidence:.1%}"
            if snapshot_file:
                photo_path = str(self.snapshots_dir / snapshot_file)
                self.notifier.send_photo(photo_path, caption=msg)
            else:
                self.notifier.send_message(msg)

        # 5. Lưu bản sao dự phòng (CSV và LOG)
        self._backup_to_files(name, dt, confidence)
        return True

    def _backup_to_files(self, name, dt, confidence):
        """Lưu bản sao ra CSV và TXT để an toàn dữ liệu."""
        try:
            # Ghi CSV hàng ngày
            csv_path = Path("data") / f"attendance_{dt.strftime('%Y-%m-%d')}.csv"
            exists = csv_path.exists()
            with open(csv_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(["Name", "Timestamp", "Confidence"])
                writer.writerow([name, dt.strftime("%Y-%m-%d %H:%M:%S"), f"{confidence:.2%}"])
            
            # Ghi file .log tổng hợp
            log_path = Path("data") / "attendance_history.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] {name} - Confidence: {confidence:.2%}\n")
        except:
             pass

    def get_all_logs(self, limit=100) -> list:
        """Lấy danh sách điểm danh mới nhất cho Dashboard."""
        sql = "SELECT name, TO_CHAR(timestamp, 'DD/MM HH24:MI:SS'), confidence, snapshot_path FROM attendance ORDER BY timestamp DESC LIMIT %s"
        conn = self._db.get_connection()
        if not conn: return []
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (limit,))
                return [{"name": r[0], "time": r[1], "confidence": f"{r[2]:.1%}", "snapshot": r[3] or ""} for r in cur.fetchall()]
        finally:
            conn.close()

    def get_stats_today(self) -> dict:
        """Thống kê tổng quan ngày hôm nay."""
        today = date.today().isoformat()
        sql_total = "SELECT COUNT(*), COUNT(DISTINCT name) FROM attendance WHERE date = %s"
        sql_breakdown = "SELECT name, COUNT(*) as count, TO_CHAR(MAX(time), 'HH24:MI') as last_seen FROM attendance WHERE date = %s GROUP BY name ORDER BY last_seen DESC"
        
        conn = self._db.get_connection()
        if not conn: return {"total": 0, "persons": 0, "date": today, "breakdown": []}
        try:
            with conn.cursor() as cur:
                cur.execute(sql_total, (today,))
                total, persons = cur.fetchone()
                cur.execute(sql_breakdown, (today,))
                breakdown = [{"name": r[0], "count": r[1], "last_seen": r[2]} for r in cur.fetchall()]
            return {"total": total or 0, "persons": persons or 0, "date": today, "breakdown": breakdown}
        finally:
            conn.close()

    def get_daily_stats(self, days=7) -> list:
        """Thống kê 7 ngày gần nhất cho biểu đồ."""
        res = []
        conn = self._db.get_connection()
        if not conn: return []
        try:
            with conn.cursor() as cur:
                for i in range(days-1, -1, -1):
                    d = (date.today() - timedelta(days=i)).isoformat()
                    cur.execute("SELECT COUNT(*), COUNT(DISTINCT name) FROM attendance WHERE date = %s", (d,))
                    total, persons = cur.fetchone()
                    res.append({"date": d, "total": total or 0, "persons": persons or 0})
            return res
        finally:
            conn.close()

    def export_csv(self, output_dir: str = "data") -> str:
        """Xuất toàn bộ lịch sử điểm danh ra CSV."""
        out_path = Path(output_dir) / "attendance_export.csv"
        sql = "SELECT id, name, timestamp, confidence FROM attendance ORDER BY timestamp DESC"
        conn = self._db.get_connection()
        if not conn: return ""
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                with open(out_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["ID", "Name", "Timestamp", "Confidence"])
                    writer.writerows(rows)
            return str(out_path)
        finally:
            conn.close()
