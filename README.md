# Hệ Thống Nhận Diện Khuôn Mặt (Face Recognition System)

> **Pipeline:** `OpenCV + MTCNN → FaceNet (512-dim) → SVM Classifier`

---

## 📁 Cấu trúc thư mục

```
He-thong-nhan-dien-khuon-mat/
├── src/
│   ├── data_collection.py   # Module 1: Thu thập ảnh khuôn mặt
│   ├── extract_features.py  # Module 2: Trích xuất embedding FaceNet
│   ├── train_svm.py         # Module 3: Huấn luyện SVM classifier
│   └── real_time_app.py     # Module 4: Nhận diện real-time
├── data/
│   └── raw/
│       └── {ten_nguoi}/     # Ảnh khuôn mặt đã cắt (160x160)
├── models/
│   ├── embeddings.pkl       # Vector embeddings + labels
│   ├── svm_classifier.pkl   # Mô hình SVM đã huấn luyện
│   └── confusion_matrix.png # Biểu đồ đánh giá
├── requirements.txt
└── README.md
```

---

## 🚀 Cài đặt

### 1. Tạo môi trường Python (khuyến nghị)
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# hoặc: source venv/bin/activate  (Linux/macOS)
```

### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

> **Lưu ý:** Nếu có GPU NVIDIA, cài PyTorch với CUDA:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 🔄 Hướng dẫn sử dụng (theo thứ tự)

### Bước 1 — Thu thập dữ liệu khuôn mặt
```bash
python src/data_collection.py
```
- Nhập tên người dùng khi được hỏi
- Nhìn thẳng vào camera, xoay mặt sang các góc khác nhau
- Hệ thống tự động chụp **100 ảnh** và lưu vào `data/raw/{ten}/`
- Phím tắt: `q` = thoát | `s` = tạm dừng/tiếp tục
- **Lặp lại** cho mỗi người cần thêm vào hệ thống

### Bước 2 — Trích xuất đặc trưng
```bash
python src/extract_features.py
```
- Đọc toàn bộ ảnh trong `data/raw/`
- Dùng FaceNet (VGGFace2) để tạo vector **512 chiều/ảnh**
- Lưu kết quả vào `models/embeddings.pkl`

### Bước 3 — Huấn luyện mô hình SVM
```bash
python src/train_svm.py
```
- Tải embeddings từ `models/embeddings.pkl`
- Tự động tối ưu siêu tham số (C, gamma) qua **Grid Search + Cross-Validation**
- In ra báo cáo chi tiết (accuracy, precision, recall, F1)
- Lưu mô hình vào `models/svm_classifier.pkl`
- Tạo ảnh confusion matrix tại `models/confusion_matrix.png`

### Bước 4 — Chạy nhận diện real-time
```bash
python src/real_time_app.py
```
- Mở camera và nhận diện khuôn mặt theo thời gian thực
- Hiển thị tên + độ tin cậy (%) trên màn hình
- Phím tắt: `q` = thoát | `r` = reset lịch sử dự đoán

---

## ⚡ Tối ưu hiệu suất

| Tình huống | Giải pháp |
|---|---|
| FPS thấp | Giảm `skip_frames` trong `real_time_app.py` hoặc dùng GPU |
| Accuracy thấp (<85%) | Thêm ảnh (tối thiểu 80-100/người), đa dạng góc & ánh sáng |
| False Positive cao | Tăng `CONFIDENCE_THRESHOLD` lên 0.65-0.70 |
| MTCNN bỏ sót khuôn mặt | Giảm `min_face_size` hoặc `thresholds` |

---

## 🛠️ Yêu cầu hệ thống

- **Python:** 3.9+
- **Camera:** Webcam USB hoặc tích hợp
- **RAM:** Tối thiểu 4GB (khuyến nghị 8GB+)
- **GPU:** Tùy chọn (NVIDIA CUDA để tăng tốc)

---

## 📦 Thư viện chính

| Thư viện | Mục đích |
|---|---|
| `opencv-python` | Đọc/xử lý video từ webcam |
| `mtcnn` | Phát hiện và localise khuôn mặt |
| `facenet-pytorch` | Trích xuất embedding 512 chiều |
| `scikit-learn` | Huấn luyện và đánh giá SVM |
| `torch` | Chạy mô hình deep learning |
| `tqdm` | Thanh tiến trình |
