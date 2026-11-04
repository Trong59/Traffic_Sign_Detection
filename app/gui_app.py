import os
import sys
import tkinter as tk

# Đảm bảo thư mục gốc dự án luôn nằm trong sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import queue
from utils.config import VIETNAMESE_LABELS

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class TrafficSignGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("HỆ THỐNG NHẬN DIỆN BIỂN BÁO GIAO THÔNG - YOLOv8")
        self.geometry("1100x650")

        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.dirname(self.current_dir)
        self.model_path = os.path.join(self.root_dir, "models", "best.pt")

        print(f"[*] Đang nạp mô hình từ: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"[!] Lỗi nạp mô hình: {e}")
            self.model = None

        self.cap = None
        self.is_running = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self._photo_ref = None  # Giữ tham chiếu PhotoImage tránh GC

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="BẢNG ĐIỀU KHIỂN",
                                       font=ctk.CTkFont(size=18, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 30))

        self.btn_image = ctk.CTkButton(self.sidebar_frame, text="Chọn ảnh", command=self.start_image,
                                       height=40)
        self.btn_image.grid(row=1, column=0, padx=20, pady=10)

        self.btn_video = ctk.CTkButton(self.sidebar_frame, text="Chọn video", command=self.start_video,
                                       height=40)
        self.btn_video.grid(row=2, column=0, padx=20, pady=10)

        self.btn_webcam = ctk.CTkButton(self.sidebar_frame, text="📹 Webcam",
                                        command=self.start_webcam,
                                        fg_color="#1a7a4a", hover_color="#145e38", height=40)
        self.btn_webcam.grid(row=3, column=0, padx=20, pady=10)

        self.btn_stop = ctk.CTkButton(self.sidebar_frame, text="Dừng Xử Lý", command=self.stop_system,
                                      fg_color="crimson", hover_color="#B22222", height=40)
        self.btn_stop.grid(row=4, column=0, padx=20, pady=10)

        self.display_frame = ctk.CTkFrame(self)
        self.display_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.display_frame.grid_columnconfigure(0, weight=1)
        self.display_frame.grid_rowconfigure(0, weight=1)

        # ─── DÙNG tk.Canvas THAY CTkLabel ĐỂ TRÁNH LỖI DPI SCALING ───
        self.canvas = tk.Canvas(self.display_frame, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # ─── BẢNG HIỂN THỊ DANH SÁCH KẾT QUẢ BIỂN BÁO BÊN PHẢI ───
        self.results_frame = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.results_frame.grid(row=0, column=2, sticky="nsew")
        self.results_frame.grid_propagate(False)

        self.results_title = ctk.CTkLabel(self.results_frame, text="BIỂN BÁO PHÁT HIỆN",
                                           font=ctk.CTkFont(size=18, weight="bold"))
        self.results_title.grid(row=0, column=0, padx=20, pady=(20, 5), sticky="w")

        self.results_subtitle = ctk.CTkLabel(self.results_frame, text="Danh sách và độ tin cậy tương ứng",
                                              font=ctk.CTkFont(size=12, slant="italic"), text_color="gray")
        self.results_subtitle.grid(row=1, column=0, padx=20, pady=(0, 15), sticky="w")

        self.scrollable_results = ctk.CTkScrollableFrame(self.results_frame, label_text=None,
                                                          fg_color="transparent")
        self.scrollable_results.grid(row=2, column=0, padx=15, pady=10, sticky="nsew")
        
        self.results_frame.grid_rowconfigure(2, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)

        # Văn bản chờ hiển thị trên canvas
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self._welcome_text_id = None
        self._draw_welcome_text()
        
        # Cập nhật kết quả rỗng ban đầu
        self.update_results_panel([])

    def update_results_panel(self, detections):
        """Cập nhật danh sách biển báo được phát hiện."""
        # Xóa các widget cũ trong scrollable frame
        for widget in self.scrollable_results.winfo_children():
            widget.destroy()

        if not detections:
            # Hiển thị thông báo không có biển báo nào
            no_detect_label = ctk.CTkLabel(
                self.scrollable_results,
                text="Không phát hiện biển báo nào",
                font=ctk.CTkFont(size=13, slant="italic"),
                text_color="gray"
            )
            no_detect_label.pack(pady=40)
            return

        # Sắp xếp các phát hiện theo độ tin cậy giảm dần
        detections = sorted(detections, key=lambda x: x['conf'], reverse=True)

        for index, det in enumerate(detections):
            label_text = det['label']
            conf_val = det['conf']
            class_code = det['class_code']

            # Tạo một Card nhỏ cho mỗi biển báo
            card_frame = ctk.CTkFrame(self.scrollable_results, corner_radius=8, fg_color=("gray90", "gray20"))
            card_frame.pack(fill="x", pady=6, padx=5)

            # Xác định màu sắc theo loại biển báo
            # Cấm (P., DP.): Đỏ
            # Nguy hiểm (W.): Cam/Vàng
            # Hiệu lệnh/Chỉ dẫn (R., I.): Xanh dương
            theme_color = "#999999" # Mặc định xám
            badge_text = "KHÁC"
            if class_code.startswith("P.") or class_code.startswith("DP."):
                theme_color = "#E53935" # Đỏ nguy hiểm/cấm
                badge_text = "CẤM"
            elif class_code.startswith("W."):
                theme_color = "#FB8C00" # Cam/Vàng cảnh báo
                badge_text = "CẢNH BÁO"
            elif class_code.startswith("R.") or class_code.startswith("I."):
                theme_color = "#1E88E5" # Xanh hiệu lệnh/chỉ dẫn
                badge_text = "HIỆU LỆNH"

            # Badge màu hiển thị loại biển báo
            badge_label = ctk.CTkLabel(
                card_frame,
                text=badge_text,
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color="white",
                fg_color=theme_color,
                corner_radius=4,
                width=75,
                height=20
            )
            badge_label.grid(row=0, column=0, padx=(10, 5), pady=8, sticky="w")

            # Mã biển báo (ví dụ: P.102)
            code_label = ctk.CTkLabel(
                card_frame,
                text=class_code,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color="gray"
            )
            code_label.grid(row=0, column=1, padx=5, pady=8, sticky="w")

            # Phần trăm tin cậy
            pct_label = ctk.CTkLabel(
                card_frame,
                text=f"{conf_val * 100:.1f}%",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=theme_color
            )
            pct_label.grid(row=0, column=2, padx=(5, 10), pady=8, sticky="e")
            card_frame.grid_columnconfigure(2, weight=1)

            # Tên biển báo (dòng dưới)
            name_label = ctk.CTkLabel(
                card_frame,
                text=label_text,
                font=ctk.CTkFont(size=13, weight="bold"),
                wraplength=220,
                justify="left"
            )
            name_label.grid(row=1, column=0, columnspan=3, padx=10, pady=(0, 8), sticky="w")

    def _draw_welcome_text(self):
        """Vẽ text chào mừng ở giữa canvas."""
        if self._welcome_text_id:
            self.canvas.delete(self._welcome_text_id)
        cx = self.canvas.winfo_width() // 2 or 400
        cy = self.canvas.winfo_height() // 2 or 300
        self._welcome_text_id = self.canvas.create_text(
            cx, cy,
            text="Hệ thống đã sẵn sàng\nVui lòng chọn nguồn đầu vào",
            fill="gray", font=("Arial", 14), justify="center"
        )

    def _on_canvas_resize(self, event):
        """Cập nhật vị trí text khi cửa sổ thay đổi kích thước."""
        if self._photo_ref is None:
            self._draw_welcome_text()

    def _display_pil_image(self, pil_image):
        """
        Hiển thị PIL image lên canvas, thu nhỏ nếu cần để vừa khung,
        không bao giờ phóng to, không bao giờ crop.
        """
        self.update_idletasks()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            return

        # Thu nhỏ nếu ảnh lớn hơn canvas, giữ nguyên nếu nhỏ hơn
        display_img = pil_image.copy()
        display_img.thumbnail((canvas_w, canvas_h), Image.LANCZOS)

        # Dùng ImageTk.PhotoImage — không có DPI scaling, pixel-perfect
        photo = ImageTk.PhotoImage(display_img)
        self._photo_ref = photo  # Giữ tham chiếu tránh garbage collector

        self.canvas.delete("all")
        self._welcome_text_id = None
        # Đặt ảnh ở giữa canvas
        self.canvas.create_image(canvas_w // 2, canvas_h // 2,
                                  image=photo, anchor="center")

    def start_image(self):
        if self.model is None:
            self.canvas.delete("all")
            self.canvas.create_text(400, 300,
                text="Lỗi: Không tìm thấy file trọng số (best.pt)!",
                fill="red", font=("Arial", 13))
            return

        # Dừng luồng video nếu đang chạy
        self.stop_system()

        file_path = filedialog.askopenfilename(
            title="Chọn hình ảnh",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )

        if not file_path:
            return

        self.canvas.delete("all")
        self.canvas.create_text(400, 300, text="Đang xử lý ảnh...",
                                 fill="white", font=("Arial", 13))
        self.update()

        # Đọc ảnh bằng OpenCV
        frame = cv2.imread(file_path)
        if frame is None:
            self.canvas.delete("all")
            self.canvas.create_text(400, 300,
                text=f"Lỗi: Không thể đọc dữ liệu ảnh!\n({file_path})",
                fill="red", font=("Arial", 13))
            return

        # Nhận diện qua YOLO
        results = self.model.predict(frame, conf=0.545, device='cpu', verbose=False)
        result = results[0]

        # Thu thập thông tin phát hiện
        detections = []
        if result.boxes:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                class_code = self.model.names.get(cls_id, "Unknown")
                label = VIETNAMESE_LABELS.get(class_code, class_code)
                detections.append({
                    "class_code": class_code,
                    "label": label,
                    "conf": conf
                })

        # Ép nhãn tiếng Việt
        for class_id, class_code in result.names.items():
            if class_code in VIETNAMESE_LABELS:
                result.names[class_id] = VIETNAMESE_LABELS[class_code]

        annotated_frame = result.plot()

        # Chuyển BGR → RGB → PIL rồi hiển thị
        cv2_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2_image)
        self._display_pil_image(pil_image)
        
        # Cập nhật kết quả lên bảng điều khiển bên phải
        self.update_results_panel(detections)

    def start_video(self):
        if self.model is None:
            return

        file_path = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")]
        )

        if not file_path:
            return

        self.setup_stream(file_path)

    def start_webcam(self):
        """Mở webcam mặc định (index 0) để nhận diện thời gian thực."""
        if self.model is None:
            return
        # Thử mở webcam, báo lỗi nếu không có
        test_cap = cv2.VideoCapture(0)
        if not test_cap.isOpened():
            self.canvas.delete("all")
            self.canvas.create_text(400, 300,
                text="Không tìm thấy webcam!\nHãy kiểm tra kết nối camera.",
                fill="red", font=("Arial", 13), justify="center")
            return
        test_cap.release()
        self.setup_stream(0)  # 0 = webcam mặc định

    def setup_stream(self, source):
        if self.model is None:
            return

        self.stop_system()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.canvas.delete("all")
            self.canvas.create_text(400, 300,
                text=f"Lỗi: Không thể mở nguồn dữ liệu!\n({source})",
                fill="red", font=("Arial", 13))
            return

        self.is_running = True
        self.stop_event.clear()

        # Kích hoạt Luồng Ngầm (Worker Thread) để chạy AI liên tục cho Video
        self.thread = threading.Thread(target=self.ai_worker, args=(source,), daemon=True)
        self.thread.start()
        self.update_gui_loop()

    # ─── LUỒNG NGẦM: CHUYÊN CHẠY AI (Chỉ dùng cho Video) ───
    def ai_worker(self, source):
        while self.is_running and not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model.predict(frame, conf=0.545, device='cpu', verbose=False)
            result = results[0]

            # Trích xuất thông tin phát hiện trước khi sửa đổi result.names
            detections = []
            if result.boxes:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    class_code = self.model.names.get(cls_id, "Unknown")
                    label = VIETNAMESE_LABELS.get(class_code, class_code)
                    detections.append({
                        "class_code": class_code,
                        "label": label,
                        "conf": conf
                    })

            for class_id, class_code in result.names.items():
                if class_code in VIETNAMESE_LABELS:
                    result.names[class_id] = VIETNAMESE_LABELS[class_code]

            annotated_frame = result.plot()

            cv2_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image)

            # Đẩy ảnh và kết quả phát hiện vào hàng đợi
            if not self.frame_queue.full():
                self.frame_queue.put((pil_image, detections))

        # Giải phóng khi thoát vòng lặp
        if self.cap:
            self.cap.release()

    # ─── LUỒNG CHÍNH: RÚT ẢNH TỪ HÀNG ĐỢI ĐỂ VẼ LÊN GIAO DIỆN ───
    def update_gui_loop(self):
        if not self.is_running:
            return

        try:
            queue_item = self.frame_queue.get_nowait()
            if isinstance(queue_item, tuple):
                pil_image, detections = queue_item
                self._display_pil_image(pil_image)
                self.update_results_panel(detections)
            else:
                self._display_pil_image(queue_item)
        except queue.Empty:
            pass

        if self.thread and not self.thread.is_alive() and self.frame_queue.empty():
            self.stop_system()
            self.canvas.delete("all")
            self.canvas.create_text(400, 300, text="Đã kết thúc luồng Video.",
                                     fill="gray", font=("Arial", 13))
            return

        self.after(15, self.update_gui_loop)

    def stop_system(self):
        self.is_running = False
        self.stop_event.set()
        self._photo_ref = None

        # Làm sạch hàng đợi ảnh
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self.canvas.delete("all")
        self._welcome_text_id = None
        self._draw_welcome_text()
        
        # Reset danh sách phát hiện trên giao diện
        self.update_results_panel([])

if __name__ == "__main__":
    app = TrafficSignGUI()
    app.mainloop()