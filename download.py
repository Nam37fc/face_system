import os
import requests
# pyrefly: ignore [missing-import]
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from io import BytesIO

# == CHỈNH SỬA Ở ĐÂY: Lưu vào thư mục data/raw ==
output_dir = os.path.join("data", "raw")
os.makedirs(output_dir, exist_ok=True)

print("Đang tải chỉ mục dataset từ Hugging Face...")
# Tải dataset dạng streaming để tiết kiệm RAM
dataset = load_dataset("fptudsc/face-celeb-vietnamese", split="train", streaming=True)

# Cấu hình mục tiêu
TARGET_CELEBS = 100
IMAGES_PER_CELEB = 20

# Từ điển theo dõi số lượng ảnh
celeb_counts = {}
total_downloaded = 0

print("Bắt đầu quét và tải ảnh...")
for item in dataset:
    celeb_name = item.get('label') or item.get('name')
    image_data = item.get('image')
    
    if not celeb_name or not image_data:
        continue
        
    # Chuẩn hóa tên thư mục (Thay khoảng trắng bằng dấu gạch dưới)
    celeb_folder_name = str(celeb_name).strip().replace(" ", "_")
    
    # Điều kiện 1: Đủ 20 ảnh -> Bỏ qua
    if celeb_counts.get(celeb_folder_name, 0) >= IMAGES_PER_CELEB:
        continue
        
    # Điều kiện 2: Đủ 100 người -> Bỏ qua người mới
    if celeb_folder_name not in celeb_counts and len(celeb_counts) >= TARGET_CELEBS:
        continue

    # Tạo thư mục riêng cho từng người: data/raw/ten_nguoi
    celeb_dir = os.path.join(output_dir, celeb_folder_name)
    os.makedirs(celeb_dir, exist_ok=True)
    
    try:
        current_count = celeb_counts.get(celeb_folder_name, 0)
        file_name = f"{celeb_folder_name}_{current_count + 1}.jpg"
        file_path = os.path.join(celeb_dir, file_name)
        
        # Xử lý lưu ảnh
        if isinstance(image_data, Image.Image):
            image_data.convert('RGB').save(file_path, "JPEG")
        elif isinstance(image_data, str) and image_data.startswith("http"):
            response = requests.get(image_data, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img.convert('RGB').save(file_path, "JPEG")
            else:
                continue
        else:
            continue
            
        # Cập nhật tiến độ
        celeb_counts[celeb_folder_name] = current_count + 1
        total_downloaded += 1
        
        print(f"Đã tải: {celeb_folder_name} ({celeb_counts[celeb_folder_name]}/{IMAGES_PER_CELEB}) | Tổng số người: {len(celeb_counts)}/{TARGET_CELEBS}", end="\r")
        
    except Exception as e:
        continue

    # Dừng khi đạt mục tiêu hoàn toàn
    if len(celeb_counts) == TARGET_CELEBS and all(count == IMAGES_PER_CELEB for count in celeb_counts.values()):
        print("\nĐã đạt mục tiêu! Dừng tải.")
        break

print(f"\nHoàn thành! Đã tải tổng cộng {total_downloaded} ảnh vào thư mục /{output_dir}")
