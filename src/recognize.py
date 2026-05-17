from torch.fx.experimental import symbolic_shapes
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

def draw_vietnamese_text(img, text, pos, font_size=16, color=(255, 255, 255)):
    """Vẽ chữ tiếng Việt có dấu lên ảnh OpenCV sử dụng Pillow."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    
    # Chuyển OpenCV (BGR) sang PIL (RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # Danh sách các font hỗ trợ tiếng Việt tốt trên Windows
    font_paths = [
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\tahoma.ttf",
        "C:\\Windows\\Fonts\\Segoe UI\\segoeuib.ttf",
        "arial.ttf",
        "tahoma.ttf"
    ]
    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
        
    draw.text(pos, text, font=font, fill=color)
    
    # Chuyển ngược lại OpenCV (BGR)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

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
    # min_blinks = int(os.environ.get("FACE_MIN_BLINKS", liv_cfg.get("min_blinks", 2)))
    min_blinks = int(os.environ.get("FACE_MIN_BLINKS", liv_cfg.get("min_blinks", 1)))
    
    send_notify = os.environ.get("FACE_NOTIFY", "1") == "1"

    try:
        # Tăng cửa sổ làm mượt (smoothing) để tránh nhảy tên
        # recognizer = FaceRecognizer(smoothing_window=det_cfg.get("smoothing_window", 5))
        recognizer = FaceRecognizer(smoothing_window=3)
        
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
    # cap = cv2.VideoCapture(cam_index)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 640))
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 480))

    # cap = cv2.VideoCapture(cam_index)

    # Tăng tốc camera
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # # Độ phân giải tối ưu realtime
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # # FPS cao hơn
    # cap.set(cv2.CAP_PROP_FPS, 30)

    # 2. Mở Camera
    cap = cv2.VideoCapture(cam_index)
    
    # Độ phân giải tối ưu cho việc phân tích Texture khuôn mặt
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 480))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm tối đa độ trễ frame
    cap.set(cv2.CAP_PROP_FPS, 30)        # Đảm bảo FPS mượt để bắt entropy chính xác

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
            # color = (0, 255, 0) if is_verified else (0, 165, 255) # Green if real, Orange if pending
            # if status_msg == "FAKE_TEXTURE": 
            #     color = (0, 0, 255) # Red if fake
            #     status_msg = "FAKE! (NON-HUMAN)"
            # elif status_msg == "NAME_CHANGED":
            #     color = (0, 0, 255)
            #     status_msg = "ID SWAP DETECTED!"
            # Người lạ -> đỏ
            # if name == recognizer.UNKNOWN_LABEL:
            #     color = (0, 0, 255)

            # # Đúng người + xác thực thành công -> xanh
            # elif is_verified:
            #     color = (0, 255, 0)

            # # Đang xác thực -> cam
            # else:
            #     color = (0, 165, 255)

            # # Fake ảnh/video -> đỏ
            # if status_msg == "FAKE_TEXTURE":
            #     color = (0, 0, 255)
            #     status_msg = "FAKE! (NON-HUMAN)"

            # # Đổi ID tracking -> đỏ
            # elif status_msg == "NAME_CHANGED":
            #     color = (0, 0, 255)
            #     status_msg = "ID SWAP DETECTED!"
            # 5. Vẽ giao diện (Đã gộp thành 1 khối để sửa lỗi nhảy màu)
            # if liveness_enabled and status_msg == "FAKE_TEXTURE":
            #     color = (0, 0, 255)  # Đỏ cho ảnh/video giả mạo
            #     status_msg = "FAKE! (NON-HUMAN)"
                
            # elif liveness_enabled and status_msg == "NAME_CHANGED":
            #     color = (0, 0, 255)  # Đỏ khi bị nhảy ID lỗi tracker
            #     status_msg = "ID SWAP DETECTED!"
                
            # elif name == recognizer.UNKNOWN_LABEL:
            #     color = (0, 0, 255)  # Đỏ cho người lạ chưa đăng ký
                
            # elif is_verified:
            #     color = (0, 255, 0)  # Xanh lá khi đã xác thực + nháy mắt thành công
                
            # else:
            #     color = (0, 165, 255) # Cam cố định khi đang chờ người dùng nháy mắt
            #     if status_msg == "STABLE":
            #         status_msg = "SCANNING LIVENESS..."
        

            # # Vẽ khung và tên
            # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # label = f"ID:{face_id} {name} ({conf:.1%})"
            # if not is_verified:
            #     label = f"ID:{face_id} {status_msg}"
            #     # Hiển thị số lần nháy mắt và các chỉ số DEBUG
            #     cv2.putText(frame, f"Blinks:{blinks}/{liveness.min_blinks} | E:{entropy:.1f} B:{b_score:.1f}", 
            #                 (box[0], box[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # cv2.putText(frame, f"{conf:.1%}", (box[0], box[3] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # --- 5. VẼ GIAO DIỆN (TỐI ƯU HÓA HOÀN TOÀN BẬT/TẮT LIVENESS) ---
            
            # Nếu người lạ (Chưa đăng ký) -> Luôn luôn hiển thị Đỏ (Bất kể bật hay tắt liveness)
            if name == recognizer.UNKNOWN_LABEL:
                color = (0, 0, 255)  # BGR: Đỏ
                if not liveness_enabled: 
                    status_msg = "UNKNOWN"

            # TRƯỜNG HỢP 1: NẾU BẬT CHẾ ĐỘ CHỐNG GIẢ MẠO (LIVENESS ON)
            elif liveness_enabled:
                if status_msg == "FAKE_TEXTURE":
                    color = (0, 0, 255)  # Đỏ
                    status_msg = "FAKE! (NON-HUMAN)"
                elif status_msg == "NAME_CHANGED":
                    color = (0, 0, 255)  # Đỏ
                    status_msg = "ID SWAP DETECTED!"
                elif is_verified:
                    color = (0, 255, 0)  # Xanh lá khi nháy mắt xong
                else:
                    color = (0, 165, 255) # Cam khi đang chờ nháy mắt
                    if status_msg == "STABLE":
                        status_msg = "SCANNING LIVENESS..."

            # TRƯỜNG HỢP 2: NẾU TẮT CHẾ ĐỘ CHỐNG GIẢ MẠO (LIVENESS OFF)
            else:
                # Đã là người quen thì gán thẳng màu Xanh lá, không bắt quét liveness
                color = (0, 255, 0)  # BGR: Xanh lá
                status_msg = "VERIFIED"

            # --- TIẾN HÀNH VẼ LÊN FRAME ---
            # Vẽ khung chữ nhật quanh mặt
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # Cấu hình nhãn hiển thị tên và độ tự tin (Dọn dẹp hiển thị tiếng Việt đẹp hơn)
            display_name = name.replace("ca_sĩ_", "").replace("ca_sĩ_", "").replace("_", " ")
            label = f"ID:{face_id} {display_name} ({conf:.1%})"
            
            # Nếu đang bật liveness và chưa quét xong -> Hiển thị trạng thái quét (Quét texture / Nháy mắt)
            if liveness_enabled and not is_verified:
                label = f"ID:{face_id} {status_msg}"
                # Hiển thị các thông số nháy mắt phục vụ debug
                cv2.putText(frame, f"Blinks:{blinks}/{min_blinks} | E:{entropy:.1f} B:{b_score:.1f}", 
                            (box[0], box[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Nếu tắt liveness nhưng là người lạ -> Hiển thị nhãn người lạ
            elif name == recognizer.UNKNOWN_LABEL:
                label = f"ID:{face_id} UNKNOWN"

            # Vẽ chữ nhãn tiếng Việt có dấu lên phía trên khung hình bằng Pillow
            rgb_color = (color[2], color[1], color[0])
            frame = draw_vietnamese_text(frame, label, (box[0], box[1] - 25), font_size=16, color=rgb_color)
            
            # Hiển thị % tự tin của FaceNet phía dưới góc khung
            if name != recognizer.UNKNOWN_LABEL:
                cv2.putText(frame, f"Conf: {conf:.1%}", (box[0], box[3] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # 6. Ghi log (Logger đã có cooldown 30s)
            if is_verified:
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
