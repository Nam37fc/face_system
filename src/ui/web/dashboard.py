"""
Module 7: Web Dashboard — Giao diện xem điểm danh
Flask app hiển thị lịch sử nhận diện real-time, thống kê và ảnh snapshot.
Chạy song song với real_time_app.py.

Cách chạy:
    python src/dashboard.py
Rồi mở trình duyệt: http://localhost:5000
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory, abort

# Ép kiểu encoding UTF-8 để in được tiếng Việt trên console Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào sys.path để import được src
ROOT = Path(__file__).parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.database.attendance import AttendanceLogger

# Chỉ định thư mục templates ngay trong ui/web
template_dir = Path(__file__).parent / "templates"
app = Flask(__name__, template_folder=str(template_dir))

# Khởi tạo logger — kết nối PostgreSQL qua ConfigManager
logger = AttendanceLogger(
    snapshots_dir=str(ROOT / "data" / "snapshots"),
)

SNAPSHOTS_DIR = ROOT / "data" / "snapshots"


# ─── Routes ──────────────────────────────────────────────────

@app.route("/")
def index():
    """Trang chính — Dashboard điểm danh."""
    stats = logger.get_stats_today()
    daily = logger.get_daily_stats(days=7)
    return render_template("index.html", stats=stats, daily=daily)


@app.route("/api/logs")
def api_logs():
    """API: Lấy N bản ghi mới nhất (JSON)."""
    logs = logger.get_all_logs(limit=50)
    return jsonify(logs)


@app.route("/api/stats")
def api_stats():
    """API: Thống kê hôm nay (JSON)."""
    return jsonify(logger.get_stats_today())


@app.route("/api/daily")
def api_daily():
    """API: Thống kê theo 7 ngày gần nhất (JSON)."""
    return jsonify(logger.get_daily_stats(days=7))


@app.route("/snapshots/<filename>")
def snapshot(filename: str):
    """Trả về ảnh snapshot."""
    if not SNAPSHOTS_DIR.exists():
        abort(404)
    return send_from_directory(str(SNAPSHOTS_DIR), filename)


@app.route("/api/export")
def export_csv():
    """Xuất CSV hôm nay và trả về đường dẫn."""
    path = logger.export_csv(output_dir=str(ROOT / "data"))
    return jsonify({"file": path, "status": "ok"})


# ─── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    from waitress import serve
    print("\n" + "=" * 55)
    print("   FACE RECOGNITION – WEB DASHBOARD (PRODUCTION)")
    print("=" * 55)
    print(f"\n  🌐 Mở trình duyệt: http://localhost:5000\n")
    
    # Sử dụng Waitress thay cho Flask Development Server để chạy bền bỉ hơn trên Windows
    serve(app, host="0.0.0.0", port=5000, threads=6)
