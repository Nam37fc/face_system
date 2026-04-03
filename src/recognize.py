import cv2
import sys
from pathlib import Path
from datetime import datetime

# Ép kiểu encoding UTF-8 để in được tiếng Việt trên console Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from src.core.recognizer import FaceRecognizer
from src.core.liveness import LivenessDetector
from src.database.attendance import AttendanceLogger
from src.utils.logger import AppLogger
from src.utils.config_manager import ConfigManager

def run_identification():
    """Luồng nhận diện thời gian thực với chống giả mạo đa mục tiêu."""
    config = ConfigManager.load()
    cam_cfg = config.get("camera", {})
    det_cfg = config.get("detection", {})
    att_cfg = config.get("attendance", {})
    liv_cfg = config.get("liveness", {})   # Phải có cái này trước khi dùng
    
    # 1. Khởi tạo các thành phần (Ưu tiên Environment Variables từ GUI)
    import os
    cam_index = int(os.environ.get("FACE_CAMERA_INDEX", cam_cfg.get("index", 0)))
    cooldown = int(os.environ.get("FACE_COOLDOWN", att_cfg.get("cooldown_seconds", 30)))
    liveness_enabled = os.environ.get("FACE_LIVENESS", "1") == "1"
    min_blinks = int(os.environ.get("FACE_MIN_BLINKS", liv_cfg.get("min_blinks", 2)))
    send_notify = os.environ.get("FACE_NOTIFY", "1") == "1"

    try:
        # Tăng cửa sổ làm mượt (smoothing) để tránh nhảy tên
        recognizer = FaceRecognizer(smoothing_window=det_cfg.get("smoothing_window", 5))
        
        liveness = LivenessDetector(
            min_blinks=min_blinks,
            timeout=liv_cfg.get("timeout_seconds", 15)
        )
        
        logger = AttendanceLogger(
            cooldown_seconds=cooldown,
            log_unknown=att_cfg.get("unknown_log", False),
            send_notify=send_notify
        )
    except Exception as e:
        AppLogger.error(f"Lỗi khởi động AI: {e}")
        return

    # 2. Mở Camera
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 480))

    AppLogger.info("🎥 Hệ thống nhận diện đang hoạt động. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 3. Xử lý nhận diện và Tracking
        results = recognizer.process_frame(frame)
        
        # [NEW] Dọn dẹp các ID đã biến mất khỏi khung hình để giải phóng bộ nhớ và trạng thái
        tracked_ids = set(recognizer.tracker.objects.keys())
        for fid in list(liveness.face_states.keys()):
            if fid not in tracked_ids:
                liveness.reset_id(fid)

        for res in results:
            box = res["box"]
            name = res["name"]
            conf = res["confidence"]
            face_img = res["face_img"]
            face_id = res["face_id"]

            # 4. Kiểm tra Liveness (Nếu được bật)
            if liveness_enabled:
                is_verified, blinks, status_msg, entropy, b_score = liveness.check_liveness(face_img, face_id, name)
            else:
                is_verified, blinks, status_msg, entropy, b_score = True, 0, "LIVENESS_OFF", 0, 0

            # 5. Vẽ giao diện
            color = (0, 255, 0) if is_verified else (0, 165, 255) # Green if real, Orange if pending
            if status_msg == "FAKE_TEXTURE": 
                color = (0, 0, 255) # Red if fake
                status_msg = "FAKE! (NON-HUMAN)"
            elif status_msg == "NAME_CHANGED":
                color = (0, 0, 255)
                status_msg = "ID SWAP DETECTED!"

            # Vẽ khung và tên
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            label = f"ID:{face_id} {name} ({conf:.1%})"
            if not is_verified:
                label = f"ID:{face_id} {status_msg}"
                # Hiển thị số lần nháy mắt và các chỉ số DEBUG
                cv2.putText(frame, f"Blinks:{blinks}/{liveness.min_blinks} | E:{entropy:.1f} B:{b_score:.1f}", 
                            (box[0], box[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"{conf:.1%}", (box[0], box[3] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 6. Ghi log (Logger đã có cooldown 30s)
            if is_verified and name != recognizer.UNKNOWN_LABEL:
                logger.log(name, conf, frame)

        # 7. Hiển thị
        cv2.imshow("Nhan dien Khuon mat (Multi-Face Liveness)", frame)
        
        # 8. Xử lý phím tắt
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Nhấn 'q' để thoát
            AppLogger.info("Đang đóng hệ thống nhận diện...")
            break
        elif key == ord(' '): # Nhấn 'Space' để tạm dừng
            AppLogger.info("TẠM DỪNG. Nhấn Space lần nữa để tiếp tục...")
            while True:
                k = cv2.waitKey(100) & 0xFF
                if k == ord(' ') or k == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_identification()
