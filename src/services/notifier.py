import os
import urllib.request
import urllib.parse
from ..utils.logger import AppLogger
from ..utils.config_manager import ConfigManager

class TelegramNotifier:
    """Dịch vụ gửi thông báo qua Telegram Bot API."""
    
    def __init__(self):
        self.refresh()

    def refresh(self):
        """Cập nhật lại cấu hình từ file config.yaml."""
        self.token = ConfigManager.get("telegram", "token", "")
        self.chat_id = ConfigManager.get("telegram", "chat_id", "")
        self.enabled = ConfigManager.get("telegram", "enabled", False)

    def send_message(self, text: str):
        if not self.enabled or not self.token or not self.chat_id:
            return False
            
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": self.chat_id, "text": text}).encode("utf-8")
        
        try:
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req) as res:
                return res.status == 200
        except Exception as e:
            AppLogger.warning(f"Lỗi gửi tin nhắn Telegram: {e}")
            return False

    def send_photo(self, photo_path: str, caption: str = ""):
        if not self.enabled or not self.token or not self.chat_id or not os.path.exists(photo_path):
            return False

        url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
        try:
            with open(photo_path, "rb") as f:
                image_data = f.read()

            boundary = '----TelegramFormBoundary'
            body = (
                f'--{boundary}\r\n'
                f'Content-Disposition: form-data; name="chat_id"\r\n\r\n{self.chat_id}\r\n'
                f'--{boundary}\r\n'
                f'Content-Disposition: form-data; name="caption"\r\n\r\n{caption}\r\n'
                f'--{boundary}\r\n'
                f'Content-Disposition: form-data; name="photo"; filename="snapshot.jpg"\r\n'
                f'Content-Type: image/jpeg\r\n\r\n'
            ).encode('utf-8') + image_data + f'\r\n--{boundary}--\r\n'.encode('utf-8')

            req = urllib.request.Request(url, data=body)
            req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
            with urllib.request.urlopen(req) as res:
                return res.status == 200
        except Exception as e:
            AppLogger.warning(f"Lỗi gửi ảnh Telegram: {e}")
            return False
