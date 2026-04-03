import cv2
import time
import numpy as np
from pathlib import Path
from collections import deque
from skimage.feature import local_binary_pattern
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ..utils.logger import AppLogger

class FaceState:
    """Lưu trữ trạng thái liveness cho một khuôn mặt cụ thể (theo ID)."""
    def __init__(self, entropy_threshold):
        self.blink_count = 0
        self.is_verified = False
        self.verified_name = None 
        self.last_verified_time = 0
        self.entropy_history = deque(maxlen=3)
        self.closed_frame_count = 0 # Đếm số frame nhắm mắt liên tiếp
        self.last_blink_time = 0
        self.status_msg = "STABLE"
        self.entropy_threshold = entropy_threshold

    def reset(self):
        self.blink_count = 0
        self.is_verified = False
        self.verified_name = None
        self.entropy_history.clear()
        self.closed_frame_count = 0
        self.status_msg = "STABLE"

    def update_texture(self, entropy):
        self.entropy_history.append(entropy)
        avg_entropy = sum(self.entropy_history) / len(self.entropy_history)
        return len(self.entropy_history) < 2 or avg_entropy >= self.entropy_threshold

class LivenessDetector:
    def __init__(self, min_blinks=2, timeout=15):
        root_dir = Path(__file__).parent.parent.parent
        model_path = root_dir / "models" / "face_landmarker.task"
        
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True, # Bật Blendshapes để lấy điểm số nháy mắt xịn hơn
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.min_blinks = min_blinks
        self.timeout = timeout
        self.blink_threshold = 0.30 # Hạ xuống 0.30 cho người đeo kính dễ nháy hơn
        self.entropy_threshold = 4.00 # Tăng lên 4.00 để chặn điện thoại (vốn là 4.1 ở người thật)
        
        self.face_states = {}

    def get_state(self, face_id) -> FaceState:
        if face_id not in self.face_states:
            self.face_states[face_id] = FaceState(self.entropy_threshold)
        return self.face_states[face_id]

    def reset_id(self, face_id):
        if face_id in self.face_states:
            del self.face_states[face_id]

    def check_liveness(self, face_img, face_id=0, name=None):
        """Trả về tuple (is_verified, blink_count, status_msg, entropy, blink_score) cho một khuôn mặt."""
        if face_img is None or face_img.size == 0:
            return False, 0, "NO_FACE", 0, 0

        state = self.get_state(face_id)
        now = time.time()
        avg_blink_score = 0 # Default if no landmarks

        # Kiểm tra tính đồng nhất của tên (NAME LOCK)
        if state.is_verified and state.verified_name and name:
            if name != state.verified_name:
                state.reset()
                return False, 0, "NAME_CHANGED", 0, 0

        if face_img.dtype != np.uint8:
            face_img = (np.clip(face_img if face_img.min() >=0 else (face_img + 1.0)/2.0, 0, 1) * 255).astype(np.uint8)

        # 1. Texture Check
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        radius = 3
        lbp = local_binary_pattern(gray, 8*radius, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), density=True)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        if not state.update_texture(entropy):
            return False, state.blink_count, "FAKE_TEXTURE", entropy, avg_blink_score

        # 2. Blink Detection (Sử dụng AI Blendshapes của Google)
        rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        result = self.detector.detect(mp_image)

        if result.face_blendshapes:
            # Lấy điểm số nhắm mắt từ AI (eyeBlinkLeft: index 9, eyeBlinkRight: index 10)
            # Hoặc tìm theo tên trong category_name
            blendshapes = result.face_blendshapes[0]
            left_blink = 0
            right_blink = 0
            for category in blendshapes:
                if category.category_name == "eyeBlinkLeft": left_blink = category.score
                if category.category_name == "eyeBlinkRight": right_blink = category.score
            
            avg_blink_score = (left_blink + right_blink) / 2.0

            if avg_blink_score > self.blink_threshold:
                state.closed_frame_count += 1
                if state.closed_frame_count >= 2:
                    state.status_msg = f"BLINKING... ({state.closed_frame_count})"
            else:
                if state.closed_frame_count >= 2:
                    if now - state.last_blink_time > 0.3:
                        state.blink_count += 1
                        state.last_blink_time = now
                    state.status_msg = "STABLE"
                state.closed_frame_count = 0

        # 3. Kết quả xác thực
        if state.blink_count >= self.min_blinks:
            state.is_verified = True
            state.last_verified_time = now
            if name: state.verified_name = name

        # [XÓA] Bỏ cơ chế TIMEOUT - Giữ trạng thái xanh mãi mãi khi còn đứng trước cam
        return state.is_verified, state.blink_count, state.status_msg, entropy, avg_blink_score

    def _calculate_ear(self, landmarks, eye_indices):
        try:
            pts = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
            v1 = np.linalg.norm(pts[1] - pts[5])
            v2 = np.linalg.norm(pts[2] - pts[4])
            h = np.linalg.norm(pts[0] - pts[3])
            return (v1 + v2) / (2.0 * h)
        except: return 1.0
