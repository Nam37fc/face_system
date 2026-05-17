import pickle
import numpy as np
import torch
import cv2
from pathlib import Path
from collections import deque
from ..utils.tracker import CentroidTracker
from facenet_pytorch import MTCNN, InceptionResnetV1
from ..utils.logger import AppLogger, ModelError
from ..utils.config_manager import ConfigManager

class FaceRecognizer:
    """Chip nhận diện: Kết hợp MTCNN + FaceNet + SVM + Tracker."""

    def __init__(self, model_path=None, smoothing_window=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNKNOWN_LABEL = "Unknown"
        
        # 0. Khởi tạo Tracker và lịch sử
        self.tracker = CentroidTracker(max_disappeared=15)
        self._predictions_history = {} # {face_id: deque([names])}
        
        # Lấy từ Config nếu không truyền vào
        if model_path is None:
            model_path = ConfigManager.load().get("paths", {}).get("model_svm", "models/svm_classifier.pkl")
        
        config = ConfigManager.load()
        if smoothing_window is None:
            self.smoothing_window = config.get("detection", {}).get("smoothing_window", 3)
        else:
            self.smoothing_window = smoothing_window
        
        self.svm_threshold = config.get("detection", {}).get("svm_threshold", 0.60)

        # Đảm bảo đường dẫn tuyệt đối từ gốc dự án
        self.root_dir = Path(__file__).parent.parent.parent
        self.full_model_path = self.root_dir / model_path
        
        # 1. MTCNN – phát hiện khuôn mặt
        self.mtcnn = MTCNN(
            image_size=160, margin=20, min_face_size=40,
            thresholds=[0.6, 0.7, 0.7], factor=0.709,
            keep_all=True, device=self.device
        )

        # 2. FaceNet – trích xuất embedding
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # 3. Tải SVM
        self._load_svm(self.full_model_path)
        self._predictions_history = {} # {face_id: deque([names])}
        AppLogger.success("Các mô hình AI (MTCNN, FaceNet, SVM) đã sẵn sàng.")

    def _load_svm(self, model_path):
        m_path = Path(model_path)
        if not m_path.exists():
            raise ModelError(f"Không tìm thấy mô hình SVM tại {m_path}")
        
        try:
            with open(m_path, "rb") as f:
                saved = pickle.load(f, encoding='latin1')
            self.svm = saved["model"]
            self.label_encoder = saved["label_encoder"]
            self.class_names = saved["class_names"]
            self.centroids = saved.get("centroids", {})  # Tải centroids vật lý nếu có
        except Exception as e:
            raise ModelError(f"Lỗi khi đọc mô hình SVM: {e}")

    @torch.no_grad()
    def process_frame(self, frame_bgr):
        """Xử lý frame và trả về danh sách kết quả nhận diện."""
        rgb_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs = self.mtcnn.detect(rgb_img)
        
        results = []
        if boxes is not None:
            # Cập nhật Tracker để lấy IDs duy nhất cho các boxes
            face_mappings = self.tracker.update(boxes)
            
            faces = self.mtcnn.extract(rgb_img, boxes, save_path=None)
            if faces is not None:
                for i, face_tensor in enumerate(faces):
                    # Lấy ID từ tracker (nếu không có thì bỏ qua)
                    face_id = face_mappings.get(i)
                    if face_id is None: continue

                    # 1. Trích xuất embedding
                    face_tensor = face_tensor.to(self.device).unsqueeze(0)
                    emb = self.facenet(face_tensor).cpu().numpy()
                    
                    # 2. SVM dự đoán nhãn dựa trên xác xuất phân loại
                    proba = self.svm.predict_proba(emb)[0]
                    best_idx = np.argmax(proba)
                    raw_name = self.label_encoder.inverse_transform([best_idx])[0]
                    raw_conf = proba[best_idx]
                    
                    # 3. Làm mượt nhãn dự đoán (Smoothing) bằng cửa sổ trượt
                    name, confidence = self._smooth_prediction(face_id, raw_name, raw_conf)
                    
                    # 4. Tính toán Cosine Similarity vật lý thực tế để hiển thị phần trăm "uy tín" (80% - 95%)
                    # Ta chỉ tính khi người đó đã được xác nhận (không phải UNKNOWN) và có centroids
                    if name != self.UNKNOWN_LABEL and hasattr(self, "centroids") and self.centroids and name in self.centroids:
                        # L2-normalize vector đặc trưng của mặt hiện tại
                        live_emb = emb[0] / np.linalg.norm(emb[0])
                        centroid = self.centroids[name]
                        # Tích vô hướng của hai vector đơn vị chính là Cosine Similarity vật lý
                        cosine_sim = float(np.dot(live_emb, centroid))
                        # Giới hạn trong khoảng [0, 1] cho an toàn
                        confidence = float(max(0.0, min(1.0, cosine_sim)))
                    
                    results.append({
                        "face_id": face_id,
                        "box": boxes[i].astype(int),
                        "name": name,
                        "confidence": confidence,
                        # Trả về ảnh khuôn mặt (đã chuyển về float32 BGR cho Liveness)
                        "face_img": cv2.cvtColor(face_tensor.squeeze().permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
                    })
        else:
            # Gọi update với danh sách rỗng để tracker cleanup các ID đã mất
            self.tracker.update([])
            
        return results

    def _smooth_prediction(self, face_id, name, confidence):
        """Giảm rung (flicker) bằng cửa sổ trượt."""
        if face_id not in self._predictions_history:
            self._predictions_history[face_id] = deque(maxlen=self.smoothing_window)
        if not hasattr(self, "_confidences_history"):
            self._confidences_history = {}
        if face_id not in self._confidences_history:
            self._confidences_history[face_id] = deque(maxlen=self.smoothing_window)
        
        # Nếu điểm tin cậy hiện tại dưới ngưỡng, coi như là Unknown trong lịch sử để vote
        resolved_name = name if confidence >= self.svm_threshold else self.UNKNOWN_LABEL
        self._predictions_history[face_id].append(resolved_name)
        self._confidences_history[face_id].append(confidence)
        
        # Lấy tên phổ biến nhất trong lịch sử
        names = list(self._predictions_history[face_id])
        final_name = max(set(names), key=names.count)
        
        # Tính điểm tự tin trung bình trong lịch sử để hiển thị mượt mà hơn
        avg_confidence = sum(self._confidences_history[face_id]) / len(self._confidences_history[face_id])
        
        return final_name, avg_confidence

    def reset_history(self):
        """Xóa lịch sử làm mượt."""
        self._predictions_history.clear()
        if hasattr(self, "_confidences_history"):
            self._confidences_history.clear()
