import os
import pickle
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from facenet_pytorch import InceptionResnetV1
from ..utils.logger import AppLogger, ModelError

class ModelTrainer:
    """Quản lý trích xuất đặc trưng và huấn luyện mô hình SVM."""

    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        
        # Khởi tạo MTCNN để tự động phát hiện và cắt mặt chuẩn trong huấn luyện
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(
            image_size=160, margin=20, min_face_size=40,
            thresholds=[0.6, 0.7, 0.7], factor=0.709,
            keep_all=False, device=self.device
        )
        AppLogger.info(f"Khởi tạo Trainer trên thiết bị: {self.device}")

    @torch.no_grad()
    def extract_features(self, output_path="models/embeddings.pkl"):
        """Bước 2: Quét thư mục ảnh và trích xuất vector 512 chiều."""
        if not self.data_dir.exists():
            raise ModelError(f"Không tìm thấy thư mục ảnh tại {self.data_dir}")

        all_embeddings = []
        all_labels = []
        
        folders = [f for f in self.data_dir.iterdir() if f.is_dir()]
        for person_dir in folders:
            images = [p for p in person_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            if not images: continue
            
            AppLogger.info(f"Đang trích xuất '{person_dir.name}': {len(images)} ảnh")
            for img_path in images:
                try:
                    img = Image.open(img_path).convert("RGB")
                    
                    # Phát hiện và cắt sát khuôn mặt bằng MTCNN (Khử nhiễu hậu cảnh, tóc, vai áo)
                    face_tensor = self.mtcnn(img)
                    if face_tensor is not None:
                        emb = self.facenet(face_tensor.unsqueeze(0).to(self.device)).cpu().numpy()
                    else:
                        # Fallback: Nếu ảnh nghiêng quá sâu không detect được, resize trực tiếp như cũ
                        resized_img = img.resize((160, 160))
                        img_t = (torch.from_numpy(np.array(resized_img)).permute(2, 0, 1).float() - 127.5) / 128.0
                        emb = self.facenet(img_t.unsqueeze(0).to(self.device)).cpu().numpy()
                        
                    all_embeddings.append(emb)
                    all_labels.append(person_dir.name)
                except Exception as e:
                    AppLogger.warning(f"Lỗi khi đọc/trích xuất ảnh {img_path.name}: {e}")

        if not all_embeddings:
            raise ModelError("Không có ảnh nào được trích xuất. Hãy thu thập ảnh trước.")

        data = {
            "embeddings": np.vstack(all_embeddings),
            "labels": all_labels,
            "label_names": sorted(set(all_labels))
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        
        AppLogger.success(f"Đã lưu embeddings vào {output_path}")
        return data

    def train_svm(self, embeddings_path="models/embeddings.pkl", model_save_path="models/svm_classifier.pkl"):
        """Bước 3: Huấn luyện bộ phân loại SVM từ embeddings."""
        if not os.path.exists(embeddings_path):
            raise ModelError("Chưa có embeddings.pkl. Hãy chạy trích xuất đặc trưng trước.")

        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)

        X = data["embeddings"]
        labels = data["labels"]
        
        if len(set(labels)) < 2:
            raise ModelError("Cần ít nhất 2 người khác nhau trong dữ liệu để huấn luyện SVM.")

        le = LabelEncoder()
        y = le.fit_transform(labels)
        
        # Grid Search tối ưu C và gamma
        pipe = Pipeline([
            ("norm", Normalizer(norm="l2")),
            ("svm", SVC(kernel="rbf", probability=True))
        ])
        
        param_grid = {"svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", "auto", 0.01]}
        cv = StratifiedKFold(n_splits=min(5, np.min(np.bincount(y))), shuffle=True)
        
        grid = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, verbose=0)
        grid.fit(X, y)
        
        # Tính toán centroid (mean embedding) cho mỗi người để phục vụ tính Cosine Similarity vật lý khi nhận diện
        centroids = {}
        for label_name in set(labels):
            idxs = [i for i, l in enumerate(labels) if l == label_name]
            class_embs = X[idxs]
            # Tính trung bình cộng các embeddings của lớp
            mean_emb = np.mean(class_embs, axis=0)
            # Chuẩn hóa L2 để có vector đơn vị chuẩn
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            centroids[label_name] = mean_emb

        save_data = {
            "model": grid.best_estimator_,
            "label_encoder": le,
            "class_names": list(le.classes_),
            "centroids": centroids  # Lưu centroids phục vụ tính toán độ tự tin
        }
        
        with open(model_save_path, "wb") as f:
            pickle.dump(save_data, f)
            
        AppLogger.success(f"Huấn luyện thành công. Accuracy: {grid.best_score_*100:.2f}%")
        return grid.best_score_
