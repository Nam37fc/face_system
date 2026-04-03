import os
import cv2
import time
import numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN
from ..utils.logger import AppLogger, CameraError

class FaceDataCollector:
    """Thu thập ảnh khuôn mặt từ webcam, tự động phát hiện và cắt."""
    
    def __init__(self, output_dir="data/raw", num_images=100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_images = num_images
        
        # Nhúng nhận diện khuôn mặt MTCNN để thu thập ảnh chuẩn
        self.mtcnn = MTCNN(keep_all=False, device='cpu')

    def collect(self, user_name: str):
        """Khởi chạy quy trình thu thập ảnh cho người dùng."""
        save_dir = self.output_dir / user_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise CameraError("Không thể mở Webcam để thu thập ảnh.")
            
        count = len(list(save_dir.glob("*.jpg")))
        AppLogger.info(f"Đã có {count} ảnh. Chụp tiếp đến {self.num_images}...")
        
        while count < self.num_images:
            ret, frame = cap.read()
            if not ret: break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = self.mtcnn.detect(rgb)
            
            display_frame = frame.copy()
            if boxes is not None:
                box = boxes[0].astype(int)
                cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                # Tự động lưu mỗi 0.2s
                face = frame[max(0, box[1]):box[3], max(0, box[0]):box[2]]
                if face.size > 0:
                    count += 1
                    cv2.imwrite(str(save_dir / f"{user_name}_{count:03d}.jpg"), face)
                    AppLogger.info(f"Đã lưu ảnh {count}/{self.num_images}")
                    time.sleep(0.2)

            cv2.putText(display_frame, f"User: {user_name} | Images: {count}/{self.num_images}", (20, 40), 2, 0.7, (255, 255, 255), 2)
            cv2.imshow("Data Collection", display_frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            
        cap.release()
        cv2.destroyAllWindows()
        AppLogger.success(f"Hoàn thành thu thập ảnh cho {user_name}!")
        return count
