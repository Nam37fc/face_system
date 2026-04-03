"""
GUI App — Hệ thống nhận diện khuôn mặt
Giao diện tích hợp 4 bước: Thu thập → Trích xuất → Huấn luyện → Nhận diện
Chạy: python src/gui_app.py
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import subprocess
import sys
import os
import queue
import time
import shutil
from tkinter import simpledialog
from pathlib import Path

# Thêm đường dẫn gốc vào sys.path để import được src
ROOT = Path(__file__).parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.logger import AppLogger, handle_main_exception
from src.utils.config_manager import ConfigManager
from src.services.notifier import TelegramNotifier

# Ép kiểu encoding UTF-8 để in được tiếng Việt trên console Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Global notifier
notifier = TelegramNotifier()

# ─── Màu sắc & theme ────────────────────────────────────────
BG          = "#0d1117"
SURFACE     = "#161b22"
SURFACE2    = "#1c2128"
BORDER      = "#30363d"
ACCENT      = "#00e5a0"
ACCENT2     = "#3b82f6"
ACCENT3     = "#f59e0b"
TEXT        = "#e6edf3"
TEXT_MUTED  = "#8b949e"
DANGER      = "#f85149"
SUCCESS     = "#3fb950"
PYTHON_EXE  = sys.executable


class LogRedirector:
    """Chuyển hướng stdout/stderr vào queue để update GUI thread-safe."""
    def __init__(self, queue: queue.Queue):
        self.queue = queue

    def write(self, text: str):
        if text.strip():
            self.queue.put(text)

    def flush(self):
        pass


class FaceRecognitionGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎯 Face Recognition System")
        self.root.geometry("1000x700")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.minsize(800, 600)

        self._log_queue: queue.Queue = queue.Queue()
        self._running_process: subprocess.Popen | None = None
        self._dashboard_process: subprocess.Popen | None = None
        self._step_running = False
        self._manually_stopped = False
        self._auto_chain_var = tk.BooleanVar(value=True)

        # ─── Đọc cấu hình từ config.yaml ───
        self._config = self._load_config()

        self._setup_styles()
        self._build_header()
        self._build_main()
        self._build_log_panel()
        self._build_statusbar()

        # Bắt đầu polling log queue
        self._poll_log()

        # Quét dữ liệu hiện có
        self._refresh_data_list()

        # Tự động khởi động Web Dashboard ngầm (không mở trình duyệt ngay)
        self._start_dashboard_silently()

    def _load_config(self) -> dict:
        """Sử dụng ConfigManager tập trung."""
        return ConfigManager.load()

    # ─── Styles ────────────────────────────────────────────────

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab",
            background=SURFACE, foreground=TEXT_MUTED,
            padding=[18, 8], font=("Segoe UI", 10),
        )
        style.map("TNotebook.Tab",
            background=[("selected", SURFACE2)],
            foreground=[("selected", ACCENT)],
        )
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=SURFACE, relief="flat")

        style.configure("Accent.TButton",
            background=ACCENT, foreground="#000000",
            font=("Segoe UI", 10, "bold"), padding=[14, 8],
            relief="flat", borderwidth=0,
        )
        style.map("Accent.TButton",
            background=[("active", "#00c97a"), ("disabled", BORDER)],
            foreground=[("disabled", TEXT_MUTED)],
        )
        style.configure("Outline.TButton",
            background=SURFACE, foreground=TEXT,
            font=("Segoe UI", 10), padding=[14, 8],
            relief="flat", borderwidth=1,
        )
        style.map("Outline.TButton",
            background=[("active", SURFACE2)],
        )
        style.configure("Danger.TButton",
            background=DANGER, foreground="white",
            font=("Segoe UI", 10, "bold"), padding=[14, 8],
            relief="flat",
        )
        style.map("Danger.TButton",
            background=[("active", "#c73b35")],
        )
        style.configure("TProgressbar",
            background=ACCENT, troughcolor=BORDER,
            thickness=8, borderwidth=0,
        )
        style.configure("TEntry",
            fieldbackground=SURFACE2, foreground=TEXT,
            insertcolor=ACCENT, relief="flat", padding=8,
        )

    # ─── Header ────────────────────────────────────────────────

    def _build_header(self):
        header = tk.Frame(self.root, bg=SURFACE, height=60)
        header.pack(fill="x")
        header.pack_propagate(False)

        # Logo + title
        tk.Label(
            header, text="👁️  Face Recognition System",
            bg=SURFACE, fg=ACCENT,
            font=("Segoe UI", 16, "bold"),
        ).pack(side="left", padx=20, pady=15)

        # Trạng thái
        self._status_var = tk.StringVar(value="● Sẵn sàng")
        tk.Label(
            header, textvariable=self._status_var,
            bg=SURFACE, fg=SUCCESS,
            font=("Segoe UI", 10),
        ).pack(side="right", padx=20)

        # Separator
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

    # ─── Main Content ──────────────────────────────────────────

    def _build_main(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=0, pady=0)

        # Sidebar bên trái (pipeline steps)
        self._build_sidebar(main)

        # Content area (notebook tabs)
        content = tk.Frame(main, bg=BG)
        content.pack(side="left", fill="both", expand=True, padx=0)

        self.notebook = ttk.Notebook(content)
        self.notebook.pack(fill="both", expand=True, padx=0, pady=0)

        self._build_tab_collect()
        self._build_tab_extract()
        self._build_tab_train()
        self._build_tab_realtime()
        self._build_tab_settings()

    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=SURFACE, width=220)
        sidebar.pack(side="left", fill="y")

        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x")

        steps = [
            ("1", "Thu thập ảnh", "📸", 0),
            ("2", "Trích xuất", "🧠", 1),
            ("3", "Huấn luyện", "🎯", 2),
            ("4", "Nhận diện", "🎥", 3),
            ("5", "Cài đặt Bot", "⚙️", 4),
        ]

        tk.Label(
            sidebar, text="PIPELINE",
            bg=SURFACE, fg=TEXT_MUTED,
            font=("Segoe UI", 9, "bold"),
        ).pack(pady=(16, 8))

        self._step_btns = []
        for num, label, icon, tab_idx in steps:
            btn = tk.Button(
                sidebar,
                text=f"  {icon}  Bước {num}: {label}",
                bg=SURFACE2, fg=TEXT,
                font=("Segoe UI", 10),
                relief="flat", bd=0,
                padx=12, pady=10,
                anchor="w",
                cursor="hand2",
                command=lambda i=tab_idx: self.notebook.select(i),
            )
            btn.pack(fill="x", padx=8, pady=2)
            self._step_btns.append(btn)

        # Separator
        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", pady=12)

        # Nút Dashboard
        tk.Button(
            sidebar,
            text="  🌐  Web Dashboard",
            bg=ACCENT2, fg="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat", bd=0,
            padx=12, pady=10,
            anchor="w",
            cursor="hand2",
            command=self._open_dashboard,
        ).pack(fill="x", padx=8, pady=2)

        # Dữ liệu hiện có
        tk.Frame(sidebar, bg=BORDER, height=1).pack(fill="x", pady=12)
        tk.Label(
            sidebar, text="DỮ LIỆU HIỆN CÓ",
            bg=SURFACE, fg=TEXT_MUTED,
            font=("Segoe UI", 9, "bold"),
        ).pack()

        self._data_listbox = tk.Listbox(
            sidebar,
            bg=SURFACE2, fg=TEXT,
            font=("Segoe UI", 9),
            relief="flat", bd=0,
            selectbackground=ACCENT, selectforeground="#000",
            height=6,
        )
        self._data_listbox.pack(fill="x", padx=8, pady=(4, 8))

        # Nút Sửa/Xóa dữ liệu (Sử dụng tk.Button để có màu sắc tốt hơn trên nền tối)
        action_row = tk.Frame(sidebar, bg=SURFACE)
        action_row.pack(fill="x", padx=8)

        tk.Button(action_row, text="✏️ Đổi tên", 
                  bg=SURFACE2, fg="#00e5a0", font=("Segoe UI", 9, "bold"),
                  relief="flat", bd=0, padx=5, pady=6, cursor="hand2",
                  command=self._confirm_rename_user).pack(side="left", expand=True, fill="x", padx=2)
        
        tk.Button(action_row, text="🗑️ Xóa", 
                  bg=SURFACE2, fg="#f85149", font=("Segoe UI", 9, "bold"),
                  relief="flat", bd=0, padx=5, pady=6, cursor="hand2",
                  command=self._delete_user).pack(side="right", expand=True, fill="x", padx=2)

        tk.Button(
            sidebar, text="↻ Làm mới danh sách",
            bg=SURFACE, fg=TEXT_MUTED,
            font=("Segoe UI", 9),
            relief="flat", bd=0, pady=10,
            cursor="hand2",
            command=self._refresh_data_list,
        ).pack()

    # ─── Tab 1: Thu thập dữ liệu ──────────────────────────────

    def _build_tab_collect(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📸  Thu thập")

        self._card(tab, "Thông tin người dùng", self._collect_form, expand=False)
        self._card(tab, "Hướng dẫn", self._collect_guide, expand=False)

    def _collect_form(self, parent):
        # Nhập tên
        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill="x", padx=16, pady=(12, 4))

        tk.Label(row, text="Tên người dùng:", bg=SURFACE, fg=TEXT_MUTED,
                 font=("Segoe UI", 10)).pack(side="left")

        self._name_var = tk.StringVar()
        name_entry = ttk.Entry(row, textvariable=self._name_var,
                               font=("Segoe UI", 11), width=24)
        name_entry.pack(side="left", padx=12)
        name_entry.bind("<Return>", lambda e: self._start_collection())

        # Nút Kiểm tra khuôn mặt
        ttk.Button(row, text="🔍 Kiểm tra khuôn mặt",
                   style="Outline.TButton",
                   command=self._check_existing_face).pack(side="left", padx=5)

        # Số ảnh
        tk.Label(row, text="Số ảnh:", bg=SURFACE, fg=TEXT_MUTED,
                 font=("Segoe UI", 10)).pack(side="left", padx=(20, 0))
        
        collect_default = self._config.get("collection", {}).get("num_images", 100)
        self._num_images_var = tk.StringVar(value=str(collect_default))
        ttk.Entry(row, textvariable=self._num_images_var,
                  font=("Segoe UI", 11), width=6).pack(side="left", padx=8)

        # Checkbox Tự động hóa
        tk.Checkbutton(row, text="Tự động trích xuất & huấn luyện khi xong",
                       variable=self._auto_chain_var,
                       bg=SURFACE, fg=ACCENT, activebackground=SURFACE,
                       selectcolor=BG, font=("Segoe UI", 9)).pack(side="left", padx=20)

        # Buttons
        btn_row = tk.Frame(parent, bg=SURFACE)
        btn_row.pack(fill="x", padx=16, pady=12)

        ttk.Button(btn_row, text="▶  Bắt đầu chụp ảnh",
                   style="Accent.TButton",
                   command=self._start_collection).pack(side="left")

        ttk.Button(btn_row, text="⏹  Dừng",
                   style="Danger.TButton",
                   command=self._stop_process).pack(side="left", padx=8)

        # Progress
        self._collect_progress = ttk.Progressbar(
            parent, mode="indeterminate", style="TProgressbar")
        self._collect_progress.pack(fill="x", padx=16, pady=(0, 12))

    def _collect_guide(self, parent):
        guide = """✅  Nhập tên (không dấu, không khoảng trắng)  →  Nhấn "Bắt đầu chụp ảnh"
📷  Webcam sẽ tự động mở — nhìn thẳng vào camera
🔄  Xoay mặt: trái → phải → ngẩng → cúi để đa dạng góc độ
⌨️   Phím tắt trong cửa sổ camera:   'q' = thoát   |   Space = tạm dừng
👥  Lặp lại cho mỗi người cần thêm vào hệ thống"""
        tk.Label(parent, text=guide, bg=SURFACE, fg=TEXT_MUTED,
                 font=("Segoe UI", 10), justify="left",
                 anchor="w").pack(fill="x", padx=16, pady=12)

    def _start_collection(self):
        name = self._name_var.get().strip()
        if not name:
            messagebox.showwarning("Thiếu tên", "Vui lòng nhập tên người dùng!")
            return
        # Chuẩn hoá tên
        safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        num = self._num_images_var.get().strip() or "100"

        # Patch data_collection.py với tên và số ảnh qua env var
        env = os.environ.copy()
        env["FACE_USER_NAME"] = safe_name
        env["FACE_NUM_IMAGES"] = num

        self._log(f"\n🚀 Bắt đầu thu thập ảnh cho '{safe_name}' ({num} ảnh)...\n")
        self._collect_progress.start(10)
        self._run_script("collect.py", env=env,
                         on_done=self._on_collect_done)

    def _check_existing_face(self):
        """Khởi động nhận diện nhanh để xem người này đã có trong DB chưa."""
        self._log("\n🔍 Đang kiểm tra xem khuôn mặt này đã có trong hệ thống chưa...\n", "accent")
        self._set_status("🔍 Đang kiểm tra...")
        
        # Biến để đánh dấu xem đã tìm thấy tên chưa
        self._found_name = None
        
        def on_done(rc):
            self._set_status("● Sẵn sàng")
            if self._found_name:
                self._name_var.set(self._found_name)
                self._log(f"\n✅ Đã tìm thấy! Người này là: '{self._found_name}'\n", "success")
                
                # Hỏi người dùng muốn làm gì tiếp theo
                ans = messagebox.askyesnocancel(
                    "Đã có dữ liệu",
                    f"Người này đã có trong hệ thống với tên: '{self._found_name}'\n\n"
                    "Nhấn YES (Có): Để chụp thêm ảnh (giúp AI thông minh hơn)\n"
                    "Nhấn NO (Không): Để chuyển sang Tab NHẬN DIỆN ngay lập tức",
                    default=messagebox.YES
                )
                
                if ans is True: # YES
                    self._log("Bạn chọn chụp thêm ảnh. Tên đã được tự động điền.\n", "success")
                elif ans is False: # NO
                    self._log("Bạn chọn Điểm danh. Đang chuyển Tab...\n", "accent")
                    self.notebook.select(3) # Tab Nhận diện
                    # Tự động nhấn nút bắt đầu nhận diện (tùy chọn)
                else: # CANCEL
                    self._name_var.set("") # Xóa tên nếu hủy
            else:
                self._log("\n❓ Không tìm thấy khuôn mặt này trong dữ liệu cũ (Người lạ).\n", "accent")
                messagebox.showinfo("Người lạ", "Người này chưa có trong hệ thống (Unknown). Bạn có thể đặt tên mới.")

        # Override log để bắt tên
        original_log_queue_put = self._log_queue.put
        def intercept_log(msg):
            # Log format: [✓ THANH CONG] Bac_Pham — 08:31 — 93.0%
            if "[✓ THANH CONG]" in msg and "—" in msg:
                try:
                    # Tách tên từ chuỗi: Sau dấu ']' và trước dấu '—'
                    name = msg.split("]")[1].split("—")[0].strip()
                    if name and name != "Unknown":
                        self._found_name = name
                        # Dừng ngay khi tìm thấy
                        if self._running_process:
                            self._running_process.terminate()
                except: pass
            original_log_queue_put(msg)
        
        self._log_queue.put = intercept_log
        
        # BẮT BUỘC bật chống giả mạo để kiểm tra bảo mật, TẮT thông báo Telegram ở tab này
        env = os.environ.copy()
        env["FACE_LIVENESS"] = "1"
        env["FACE_NOTIFY"] = "0"
        env["FACE_MIN_BLINKS"] = self._min_blinks_var.get()

        # Gọi recognize.py thay cho real_time_app.py cũ
        self._run_script("recognize.py", env=env, on_done=lambda rc: [
            setattr(self._log_queue, 'put', original_log_queue_put), # Trả lại log
            on_done(rc)
        ])

    def _on_collect_done(self, returncode: int):
        self._collect_progress.stop()
        if returncode == 0 and self._auto_chain_var.get() and not self._manually_stopped:
            self._log("\n🔄 Tự động chuyển sang bước 2: Trích xuất đặc trưng...\n", "accent")
            self.notebook.select(1)  # Chuyển sang Tab Trích xuất
            self.root.after(1000, self._run_extract)
        elif returncode != 0 and not self._manually_stopped:
            self._log("\n⚠️ Quá trình thu thập bị dừng hoặc có lỗi. Đã hủy tự động hóa.\n", "error")

    # ─── Tab 2: Trích xuất ─────────────────────────────────────

    def _build_tab_extract(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🧠  Trích xuất")

        self._card(tab, "Trích xuất embeddings FaceNet", self._extract_form, expand=False)

    def _extract_form(self, parent):
        tk.Label(parent,
                 text="Đọc toàn bộ ảnh trong data/raw/ và tạo vector 512 chiều bằng FaceNet (VGGFace2).",
                 bg=SURFACE, fg=TEXT_MUTED, font=("Segoe UI", 10),
                 wraplength=600, justify="left").pack(anchor="w", padx=16, pady=(12, 4))

        btn_row = tk.Frame(parent, bg=SURFACE)
        btn_row.pack(fill="x", padx=16, pady=12)

        ttk.Button(btn_row, text="▶  Trích xuất embeddings",
                   style="Accent.TButton",
                   command=self._run_extract).pack(side="left")

        ttk.Button(btn_row, text="⏹  Dừng",
                   style="Danger.TButton",
                   command=self._stop_process).pack(side="left", padx=8)

        self._extract_progress = ttk.Progressbar(
            parent, mode="indeterminate", style="TProgressbar")
        self._extract_progress.pack(fill="x", padx=16, pady=(0, 12))

    def _run_extract(self):
        self._log("\n🧠 Bắt đầu trích xuất đặc trưng FaceNet...\n")
        self._extract_progress.start(10)
        self._run_script("extract.py",
                         on_done=self._on_extract_done)

    def _on_extract_done(self, returncode: int):
        self._extract_progress.stop()
        if returncode == 0 and self._auto_chain_var.get() and not self._manually_stopped:
            self._log("\n🎯 Tự động chuyển sang bước 3: Huấn luyện SVM...\n", "accent")
            self.notebook.select(2)  # Chuyển sang Tab Huấn luyện
            self.root.after(1000, self._run_train)
        elif returncode != 0 and not self._manually_stopped:
            self._log("\n⚠️ Quá trình trích xuất bị lỗi. Đã hủy tự động hóa.\n", "error")

    # ─── Tab 3: Huấn luyện ─────────────────────────────────────

    def _build_tab_train(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🎯  Huấn luyện")

        self._card(tab, "Huấn luyện SVM Classifier", self._train_form, expand=False)

    def _train_form(self, parent):
        # Options
        opt_row = tk.Frame(parent, bg=SURFACE)
        opt_row.pack(fill="x", padx=16, pady=(12, 4))

        tk.Label(opt_row, text="Test size:", bg=SURFACE, fg=TEXT_MUTED,
                 font=("Segoe UI", 10)).pack(side="left")
        self._test_size_var = tk.StringVar(value="0.2")
        ttk.Entry(opt_row, textvariable=self._test_size_var,
                  width=6, font=("Segoe UI", 10)).pack(side="left", padx=8)

        tk.Label(opt_row, text="(0.1 – 0.3 khuyến nghị)",
                 bg=SURFACE, fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(side="left")

        btn_row = tk.Frame(parent, bg=SURFACE)
        btn_row.pack(fill="x", padx=16, pady=12)

        ttk.Button(btn_row, text="▶  Bắt đầu huấn luyện",
                   style="Accent.TButton",
                   command=self._run_train).pack(side="left")

        ttk.Button(btn_row, text="⏹  Dừng",
                   style="Danger.TButton",
                   command=self._stop_process).pack(side="left", padx=8)

        ttk.Button(btn_row, text="📊  Xem confusion matrix",
                   style="Outline.TButton",
                   command=self._open_confusion_matrix).pack(side="left")

        self._train_progress = ttk.Progressbar(
            parent, mode="indeterminate", style="TProgressbar")
        self._train_progress.pack(fill="x", padx=16, pady=(0, 12))

        # Result box
        tk.Label(parent, text="Kết quả:", bg=SURFACE, fg=TEXT_MUTED,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=16)
        self._train_result = tk.Label(
            parent, text="—", bg=SURFACE2, fg=ACCENT,
            font=("Segoe UI", 14, "bold"),
            pady=12,
        )
        self._train_result.pack(fill="x", padx=16, pady=8)

    def _run_train(self):
        self._log("\n🎯 Bắt đầu huấn luyện bộ phân loại SVM...\n")
        self._train_result.config(text="Đang huấn luyện...", fg=ACCENT3)
        
        # Lấy test size từ UI
        env = os.environ.copy()
        test_size = "0.2"
        try:
            test_size = self._test_size_var.get() or "0.2"
        except: pass
        env["FACE_TEST_SIZE"] = test_size

        self._train_progress.start(10)
        self._run_script("train.py", env=env,
                         on_done=self._on_train_done)

    def _on_train_done(self, returncode: int):
        self._train_progress.stop()
        if returncode == 0:
            self._train_result.config(
                text="✅ Huấn luyện hoàn tất!",
                fg=SUCCESS,
            )
            if self._auto_chain_var.get() and not self._manually_stopped:
                self._log("\n✅ TOÀN BỘ QUY TRÌNH TỰ ĐỘNG HOÀN TẤT!\n", "success")
                messagebox.showinfo("Thành công", "Quy trình Tự động (Chụp -> Trích xuất -> Huấn luyện) đã hoàn tất!")
        else:
            self._train_result.config(text="❌ Lỗi huấn luyện", fg=DANGER)

    def _open_confusion_matrix(self):
        path = ROOT / "models" / "confusion_matrix.png"
        if path.exists():
            os.startfile(str(path))
        else:
            messagebox.showinfo("Chưa có file",
                                "Chưa có confusion_matrix.png.\nHãy huấn luyện mô hình trước.")

    # ─── Tab 4: Nhận diện Real-time ───────────────────────────

    def _build_tab_realtime(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🎥  Nhận diện")

        self._card(tab, "Nhận diện khuôn mặt Real-time", self._realtime_form, expand=False)

    def _realtime_form(self, parent):
        info = """🎥  Mở webcam và nhận diện khuôn mặt theo thời gian thực
📋  Tự động ghi nhận điểm danh vào PostgreSQL
⌨️   'q' = thoát   |   'r' = reset lịch sử   |   'e' = xuất CSV"""
        tk.Label(parent, text=info, bg=SURFACE, fg=TEXT_MUTED,
                 font=("Segoe UI", 10), justify="left",
                 anchor="w").pack(fill="x", padx=16, pady=(12, 4))

        opt_row = tk.Frame(parent, bg=SURFACE)
        opt_row.pack(fill="x", padx=16, pady=4)
        tk.Label(opt_row, text="Camera index:", bg=SURFACE, fg=TEXT_MUTED,
                 font=("Segoe UI", 10)).pack(side="left")
        cam_default = self._config.get("camera", {}).get("index", 0)
        self._camera_idx_var = tk.StringVar(value=str(cam_default))
        ttk.Entry(opt_row, textvariable=self._camera_idx_var,
                  width=4, font=("Segoe UI", 10)).pack(side="left", padx=8)
        
        tk.Label(opt_row, text="Cooldown (giây):", bg=SURFACE, fg=TEXT_MUTED,
                 font=("Segoe UI", 10)).pack(side="left", padx=(16, 0))
        
        cd_default = self._config.get("attendance", {}).get("cooldown_seconds", 30)
        self._cooldown_var = tk.StringVar(value=str(cd_default))
        ttk.Entry(opt_row, textvariable=self._cooldown_var,
                  width=6, font=("Segoe UI", 10)).pack(side="left", padx=8)

        # Liveness Toggle
        live_default = self._config.get("liveness", {}).get("enabled", True)
        self._liveness_var = tk.BooleanVar(value=live_default)
        tk.Checkbutton(
            opt_row, text="Chống giả mạo",
            variable=self._liveness_var,
            bg=SURFACE, fg=ACCENT,
            selectcolor=SURFACE2, activebackground=SURFACE,
            activeforeground=ACCENT, font=("Segoe UI", 10)
        ).pack(side="left", padx=(16, 0))

        tk.Label(opt_row, text="Nháy mắt:", bg=SURFACE, fg=TEXT_MUTED,
                 font=("Segoe UI", 10)).pack(side="left", padx=(16, 0))
        
        blink_default = self._config.get("liveness", {}).get("min_blinks", 1)
        self._min_blinks_var = tk.StringVar(value=str(blink_default))
        ttk.Entry(opt_row, textvariable=self._min_blinks_var,
                  width=4, font=("Segoe UI", 10)).pack(side="left", padx=8)

        btn_row = tk.Frame(parent, bg=SURFACE)
        btn_row.pack(fill="x", padx=16, pady=12)

        ttk.Button(btn_row, text="▶  Bắt đầu nhận diện",
                   style="Accent.TButton",
                   command=self._run_realtime).pack(side="left")

        ttk.Button(btn_row, text="⏹  Dừng camera",
                   style="Danger.TButton",
                   command=self._stop_process).pack(side="left", padx=8)

    def _run_realtime(self):
        self._log("\n🎥 Khởi động nhận diện khuôn mặt real-time...\n")
        env = os.environ.copy()
        env["FACE_CAMERA_INDEX"] = self._camera_idx_var.get()
        env["FACE_COOLDOWN"]     = self._cooldown_var.get()
        env["FACE_LIVENESS"]     = "1" if self._liveness_var.get() else "0"
        env["FACE_MIN_BLINKS"]   = self._min_blinks_var.get()
        env["FACE_NOTIFY"]       = "1" # BẬT thông báo Telegram ở tab Nhận diện chính
        # Chạy module mới
        self._run_script("recognize.py", env=env)

    # ─── Log Panel ─────────────────────────────────────────────

    def _build_log_panel(self):
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        log_frame = tk.Frame(self.root, bg=SURFACE2, height=200)
        log_frame.pack(fill="both", expand=False)
        log_frame.pack_propagate(False)

        header = tk.Frame(log_frame, bg=SURFACE2)
        header.pack(fill="x", padx=12, pady=(6, 0))

        tk.Label(header, text="📋 Log",
                 bg=SURFACE2, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(side="left")

        tk.Button(header, text="🗑 Xoá log",
                  bg=SURFACE2, fg=TEXT_MUTED,
                  font=("Segoe UI", 9),
                  relief="flat", bd=0, cursor="hand2",
                  command=self._clear_log).pack(side="right")

        self._log_text = tk.Text(
            log_frame,
            bg="#0a0e14", fg="#c9d1d9",
            font=("Cascadia Code", 9),
            relief="flat", bd=0,
            wrap="word",
            state="disabled",
            height=10,
        )
        self._log_text.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        # Tag màu
        self._log_text.tag_configure("success",  foreground=SUCCESS)
        self._log_text.tag_configure("error",    foreground=DANGER)
        self._log_text.tag_configure("accent",   foreground=ACCENT)
        self._log_text.tag_configure("warning",  foreground=ACCENT3)
        self._log_text.tag_configure("critical", foreground="#fff", background=DANGER)
        self._log_text.tag_configure("muted",    foreground=TEXT_MUTED)

        scrollbar = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=scrollbar.set)

    # ─── Status bar ────────────────────────────────────────────

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg=SURFACE, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        tk.Frame(bar, bg=BORDER, height=1).pack(fill="x", side="top")

        self._statusbar_var = tk.StringVar(value="Hệ thống sẵn sàng")
        tk.Label(bar, textvariable=self._statusbar_var,
                 bg=SURFACE, fg=TEXT_MUTED,
                 font=("Segoe UI", 9)).pack(side="left", padx=12)

        tk.Label(bar, text="Face Recognition System v2.0",
                 bg=SURFACE, fg=TEXT_MUTED,
                 font=("Segoe UI", 9)).pack(side="right", padx=12)

    # ─── Helpers ───────────────────────────────────────────────

    def _card(self, parent, title: str, content_fn, expand=True):
        frame = tk.Frame(parent, bg=SURFACE, bd=0)
        frame.pack(fill="x" if not expand else "both",
                   expand=expand, padx=16, pady=10)

        tk.Label(frame, text=f"  {title}",
                 bg=SURFACE2, fg=ACCENT,
                 font=("Segoe UI", 11, "bold"),
                 pady=8, anchor="w").pack(fill="x")

        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x")
        content_fn(frame)

    def _log(self, msg: str, tag: str = None):
        self._log_text.configure(state="normal")
        if tag:
            self._log_text.insert("end", msg, tag)
        else:
            self._log_text.insert("end", msg)
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _clear_log(self):
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    def _poll_log(self):
        """Poll queue 50ms một lần để cập nhật log an toàn từ thread khác."""
        try:
            while True:
                msg = self._log_queue.get_nowait()
                tag = None
                
                # 1. Nhận diện Tag màu dựa trên chuẩn AppLogger
                if "[✓ THANH CONG]" in msg or any(w in msg for w in ["THÀNH CÔNG", "OK"]):
                    tag = "success"
                elif "[CRITICAL_ERROR]" in msg:
                    tag = "critical"
                    # Tự động hiện Popup cho lỗi nghiêm trọng
                    error_msg = msg.split(":", 1)[1].strip() if ":" in msg else msg
                    messagebox.showerror("Lỗi hệ thống", error_msg)
                elif any(w in msg for w in ["[LỖI NGHIÊM TRỌNG]", "Error:", "Traceback"]):
                    tag = "error"
                elif "[! CANH BAO]" in msg or "CẢNH BÁO" in msg:
                    tag = "warning"
                elif any(w in msg for w in ["[INFO]", "BẮT ĐẦU", "KẾT QUẢ"]):
                    tag = "accent"
                
                self._log(msg, tag)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_log)

    def _run_script(self, script_name: str, env=None, on_done=None):
        """Chạy script Python trong thread riêng, stream output vào log."""
        if self._step_running:
            messagebox.showwarning("Đang chạy",
                                   "Một tác vụ đang chạy. Hãy dừng trước!")
            return

        script_path = ROOT / "src" / script_name
        if not script_path.exists():
            self._log(f"[LỖI] Không tìm thấy: {script_path}\n", "error")
            return

        self._step_running = True
        self._manually_stopped = False
        self._set_status("⏳ Đang chạy...")

        def worker():
            try:
                # Tạo bản sao env để tránh lỗi UnboundLocalError
                current_env = env.copy() if env else os.environ.copy()
                current_env["PYTHONUNBUFFERED"] = "1"
                
                proc = subprocess.Popen(
                    [PYTHON_EXE, "-u", str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    cwd=str(ROOT),
                    env=current_env,
                    bufsize=1,
                )
                self._running_process = proc

                for line in proc.stdout:
                    self._log_queue.put(line)

                proc.wait()
                self._running_process = None

                if proc.returncode == 0 or self._manually_stopped:
                    self._log_queue.put("\n✅ Hoàn thành!\n")
                    self._set_status("✅ Hoàn thành")
                else:
                    self._log_queue.put(f"\n❌ Lỗi (exit code {proc.returncode})\n")
                    self._set_status("❌ Có lỗi xảy ra")

            except Exception as e:
                self._log_queue.put(f"[LỖI] {e}\n")
                self._set_status("❌ Có lỗi")
            finally:
                self._step_running = False
                if on_done:
                    # Đảm bảo rc có giá trị kể cả khi proc không khởi tạo được
                    _rc = proc.returncode if 'proc' in locals() and proc else 1
                    self.root.after(0, lambda: on_done(_rc))

        threading.Thread(target=worker, daemon=True).start()

    def _stop_process(self):
        if self._running_process:
            self._manually_stopped = True
            self._running_process.terminate()
            self._log_queue.put("\n⏹ Đã dừng tiến trình.\n")
            self._step_running = False
            self._set_status("⏹ Đã dừng")

    def _set_status(self, msg: str):
        self.root.after(0, lambda: self._status_var.set(msg))
        self.root.after(0, lambda: self._statusbar_var.set(msg))

    # ─── Tab 5: Cài đặt (Telegram) ───────────────────────────

    def _build_tab_settings(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="⚙️ Cài đặt")

        scroll = scrolledtext.ScrolledText(tab, bg=BG, fg=TEXT, font=("Consolas", 10), bd=0)
        scroll.pack(fill="both", expand=True, padx=20, pady=20)

        # Container cho form
        form = tk.Frame(tab, bg=SURFACE, padx=30, pady=30)
        form.place(relx=0.5, rely=0.4, anchor="center")

        tk.Label(form, text="⚙️ Cấu hình Hệ thống", bg=SURFACE, fg=ACCENT,
                 font=("Segoe UI", 14, "bold")).pack(pady=(0, 20))

        # Telegram Token
        tk.Label(form, text="Telegram Bot Token:", bg=SURFACE, fg=TEXT_MUTED).pack(anchor="w")
        self._tele_token_var = tk.StringVar(value=notifier.token)
        ttk.Entry(form, textvariable=self._tele_token_var, width=50).pack(pady=(0, 15))

        # Chat ID
        tk.Label(form, text="Telegram Chat ID:", bg=SURFACE, fg=TEXT_MUTED).pack(anchor="w")
        self._tele_chat_id_var = tk.StringVar(value=notifier.chat_id)
        ttk.Entry(form, textvariable=self._tele_chat_id_var, width=50).pack(pady=(0, 15))

        # Checkbox Enable
        self._tele_enabled_var = tk.BooleanVar(value=notifier.enabled)
        ttk.Checkbutton(form, text="Bật thông báo Telegram",
                        variable=self._tele_enabled_var).pack(pady=10)

        # Nút Lưu
        ttk.Button(form, text="💾 Lưu cấu hình",
                   style="Accent.TButton",
                   command=self._save_settings).pack(pady=20)

        # Hướng dẫn
        guide = (
            "💡 Hướng dẫn lấy thông tin:\n"
            "1. Token: Tạo bot qua @BotFather và lấy 'API Token'\n"
            "2. Chat ID: Gửi tin nhắn cho bot @userinfobot để lấy ID của bạn\n"
            "3. Sau khi lưu, hệ thống sẽ tự động gửi cảnh báo khi nhận diện"
        )
        tk.Label(form, text=guide, bg=SURFACE, fg=TEXT_MUTED,
                 font=("Segoe UI", 9), justify="left").pack()

    def _save_settings(self):
        token = self._tele_token_var.get().strip()
        chat_id = self._tele_chat_id_var.get().strip()
        enabled = self._tele_enabled_var.get()

        # Lưu qua ConfigManager
        config = ConfigManager.load()
        if "telegram" not in config: config["telegram"] = {}
        config["telegram"]["token"] = token
        config["telegram"]["chat_id"] = chat_id
        config["telegram"]["enabled"] = enabled
        
        if ConfigManager.save(config):
            notifier.refresh() # Cập nhật instance đang chạy
            if enabled:
                success = notifier.send_message("🔔 Cấu hình Telegram đã được cập nhật thành công!")
                if success:
                    messagebox.showinfo("Thành công", "Đã lưu cấu hình và gửi tin nhắn thử nghiệm thành công!")
                else:
                    messagebox.showwarning("Chú ý", "Đã lưu cấu hình nhưng không gửi được tin nhắn thử. Hãy kiểm tra lại Token/ChatID.")
            else:
                messagebox.showinfo("Thành công", "Đã lưu cấu hình (Thông báo đang tắt).")
        else:
            messagebox.showerror("Lỗi", "Không thể lưu tệp cấu hình.")

    def _delete_user(self):
        """Xóa vĩnh viễn thư mục người dùng."""
        selection = self._data_listbox.curselection()
        if not selection:
            messagebox.showwarning("Chú ý", "Hãy chọn một người dùng trong danh sách trước!")
            return

        text = self._data_listbox.get(selection[0])
        # Format: "  👤 BacPham (100 ảnh)"
        name = text.split("👤")[1].split("(")[0].strip()

        if messagebox.askyesno("Xác nhận xóa", f"Bạn có chắc chắn muốn xóa vĩnh viễn dữ liệu của '{name}' không?\n\nHành động này không thể hoàn tác."):
            user_dir = ROOT / "data" / "raw" / name
            if user_dir.exists():
                shutil.rmtree(user_dir)
                self._log(f"\n🗑️ Đã xóa vĩnh viễn dữ liệu của '{name}'.\n", "error")
                
                # Xóa cache huấn luyện để buộc huấn luyện lại
                self._invalidate_training_cache()
                self._refresh_data_list()
            else:
                messagebox.showerror("Lỗi", "Không tìm thấy thư mục dữ liệu.")

    def _confirm_rename_user(self):
        """Mở dialog để đổi tên người dùng."""
        selection = self._data_listbox.curselection()
        if not selection:
            messagebox.showwarning("Chú ý", "Hãy chọn một người dùng trong danh sách trước!")
            return

        old_text = self._data_listbox.get(selection[0])
        old_name = old_text.split("👤")[1].split("(")[0].strip()

        new_name = simpledialog.askstring("Đổi tên", f"Nhập tên mới cho '{old_name}':", initialvalue=old_name)
        
        if new_name and new_name != old_name:
            # Kiểm tra tên mới hợp lệ
            new_name = new_name.strip().replace(" ", "_")
            old_dir = ROOT / "data" / "raw" / old_name
            new_dir = ROOT / "data" / "raw" / new_name

            if new_dir.exists():
                messagebox.showerror("Lỗi", f"Tên '{new_name}' đã tồn tại!")
                return

            try:
                os.rename(old_dir, new_dir)
                self._log(f"\n✏️ Đã đổi tên '{old_name}' thành '{new_name}'.\n", "accent")
                self._invalidate_training_cache()
                self._refresh_data_list()
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể đổi tên: {e}")

    def _invalidate_training_cache(self):
        """Xóa embeddings và model cũ để buộc người dùng trích xuất/huấn luyện lại."""
        files_to_delete = [
            ROOT / "models" / "embeddings.pkl",
            ROOT / "models" / "svm_classifier.pkl",
            ROOT / "models" / "label_encoder.pkl"
        ]
        for f in files_to_delete:
            if f.exists():
                f.unlink()
        self._log("⚠️ Dữ liệu đã thay đổi. Bạn cần 'Trích xuất' và 'Huấn luyện' lại để cập nhật bộ não AI.\n", "accent")

    def _refresh_data_list(self):
        self._data_listbox.delete(0, "end")
        raw_dir = ROOT / "data" / "raw"
        if raw_dir.exists():
            for d in sorted(raw_dir.iterdir()):
                if d.is_dir():
                    count = len(list(d.glob("*.jpg")))
                    self._data_listbox.insert("end", f"  👤 {d.name} ({count} ảnh)")
        else:
            self._data_listbox.insert("end", "  (Chưa có dữ liệu)")

    def _start_dashboard_silently(self):
        """Khởi động máy chủ Web ngầm mà không mở trình duyệt."""
        if self._dashboard_process and self._dashboard_process.poll() is None:
            return

        def start_dash():
            try:
                proc = subprocess.Popen(
                    [PYTHON_EXE, str(ROOT / "src" / "ui" / "web" / "dashboard.py")],
                    cwd=str(ROOT),
                    text=True,
                    encoding='utf-8',
                )
                self._dashboard_process = proc
            except: pass

        threading.Thread(target=start_dash, daemon=True).start()

    def _open_dashboard(self):
        """Mở trình duyệt truy cập Dashboard."""
        self._start_dashboard_silently()
        time.sleep(0.5)
        import webbrowser
        webbrowser.open("http://localhost:5000")
        self._log("\n🌐 Đã mở trình duyệt tại http://localhost:5000\n", "accent")


# ─── Script an toàn cho data_collection (không cần input terminal) ──



# ─── Main ────────────────────────────────────────────────────

if __name__ == "__main__":

    root = tk.Tk()
    app = FaceRecognitionGUI(root)

    # Xử lý đóng cửa sổ
    def on_close():
        if app._running_process:
            app._running_process.terminate()
        if app._dashboard_process:
            app._dashboard_process.terminate()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n[INFO] Đang đóng ứng dụng bởi người dùng...")
        on_close()
