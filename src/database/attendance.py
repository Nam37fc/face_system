import csv
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta
from threading import Lock
from .db_manager import DatabaseManager
from ..utils.logger import AppLogger
from ..services.notifier import TelegramNotifier

class AttendanceLogger:
    """
    Quản lý nghiệp vụ điểm danh: Ghi log, thống kê, xuất báo cáo sử dụng MongoDB Atlas.
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
        """Ghi nhận điểm danh vào MongoDB Atlas + lưu snapshot + bản sao lưu."""
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

        # 3. Ghi vào MongoDB Atlas
        dt = datetime.now()
        
        with self._lock:
            # Re-check inside lock
            if now - self._last_global_log < self.cooldown_seconds:
                return False
            
            self._last_global_log = now
            self._last_log[name] = now
            
            col = self._db.attendance_col
            if col is not None:
                try:
                    doc = {
                        "name": str(name),
                        "timestamp": dt,
                        "date": dt.date().isoformat(),
                        "time": dt.strftime("%H:%M:%S"),
                        "confidence": float(confidence),
                        "snapshot_path": snapshot_file
                    }
                    col.insert_one(doc)
                except Exception as e:
                    AppLogger.error(f"Không thể ghi attendance vào MongoDB: {e}")
                    return False
            else:
                AppLogger.warning("MongoDB không sẵn sàng, bỏ qua ghi log.")
                return False

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

        # 5. Lưu bản sao dự phòng (CSV và LOG) để an toàn dữ liệu cục bộ
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
        col = self._db.attendance_col
        if col is None:
            return []
        try:
            cursor = col.find().sort("timestamp", -1).limit(limit)
            logs = []
            for r in cursor:
                ts = r.get("timestamp")
                time_str = ts.strftime("%d/%m %H:%M:%S") if ts else ""
                logs.append({
                    "name": r.get("name"),
                    "time": time_str,
                    "confidence": f"{r.get('confidence', 0.0):.1%}",
                    "snapshot": r.get("snapshot_path") or ""
                })
            return logs
        except Exception as e:
            AppLogger.error(f"Lỗi lấy logs từ MongoDB: {e}")
            return []

    def get_stats_today(self) -> dict:
        """Thống kê tổng quan ngày hôm nay."""
        today = date.today().isoformat()
        col = self._db.attendance_col
        if col is None:
            return {"total": 0, "persons": 0, "date": today, "breakdown": []}
        try:
            total = col.count_documents({"date": today})
            distinct_names = col.distinct("name", {"date": today})
            persons = len(distinct_names)
            
            # Breakdown: group by name, count logs, and get max time
            pipeline = [
                {"$match": {"date": today}},
                {"$group": {
                    "_id": "$name",
                    "count": {"$sum": 1},
                    "last_seen_dt": {"$max": "$timestamp"}
                }},
                {"$sort": {"last_seen_dt": -1}}
            ]
            breakdown = []
            for r in col.aggregate(pipeline):
                last_seen_dt = r.get("last_seen_dt")
                last_seen_str = last_seen_dt.strftime("%H:%M") if last_seen_dt else ""
                breakdown.append({
                    "name": r["_id"],
                    "count": r["count"],
                    "last_seen": last_seen_str
                })
            return {"total": total, "persons": persons, "date": today, "breakdown": breakdown}
        except Exception as e:
            AppLogger.error(f"Lỗi thống kê ngày hôm nay từ MongoDB: {e}")
            return {"total": 0, "persons": 0, "date": today, "breakdown": []}

    def get_daily_stats(self, days=7) -> list:
        """Thống kê 7 ngày gần nhất cho biểu đồ."""
        res = []
        col = self._db.attendance_col
        if col is None:
            return []
        try:
            for i in range(days-1, -1, -1):
                d = (date.today() - timedelta(days=i)).isoformat()
                total = col.count_documents({"date": d})
                distinct_names = col.distinct("name", {"date": d})
                persons = len(distinct_names)
                res.append({"date": d, "total": total, "persons": persons})
            return res
        except Exception as e:
            AppLogger.error(f"Lỗi thống kê 7 ngày từ MongoDB: {e}")
            return []

    def export_csv(self, output_dir: str = "data") -> str:
        """Xuất toàn bộ lịch sử điểm danh ra CSV."""
        out_path = Path(output_dir) / "attendance_export.csv"
        col = self._db.attendance_col
        if col is None:
            return ""
        try:
            cursor = col.find().sort("timestamp", -1)
            with open(out_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Name", "Timestamp", "Confidence"])
                for r in cursor:
                    ts = r.get("timestamp")
                    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else ""
                    writer.writerow([
                        str(r.get("_id")),
                        r.get("name"),
                        ts_str,
                        r.get("confidence")
                    ])
            return str(out_path)
        except Exception as e:
            AppLogger.error(f"Lỗi xuất CSV từ MongoDB: {e}")
            return ""
