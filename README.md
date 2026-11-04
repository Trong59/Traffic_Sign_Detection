# Hệ thống Nhận diện Biển báo Giao thông Việt Nam (YOLOv8 & CustomTkinter)

Đề tài xây dựng ứng dụng tự động nhận diện và phân loại 58 loại biển báo giao thông đường bộ tuân thủ theo tiêu chuẩn **QCVN 41:2019/BGTVT**. Dự án áp dụng mô hình kiến trúc học sâu dạng Pipeline, tách biệt hoàn toàn môi trường huấn luyện (Cloud - Google Colab) và môi trường thực thi giao diện điều khiển (Local - PyCharm).

---

## Cấu trúc Dự án

```text
Traffic_Sign_Project/
│
├── app/
│   ├── __init__.py
│   └── gui_app.py          # Giao diện đồ họa (GUI) điều khiển bằng CustomTkinter
│
├── models/
│   └── best.pt             # Trọng số tối ưu nhất sau quá trình huấn luyện bằng GPU
│
├── notebooks/
│   └── Traffic_Sign_Project.ipynb # Source code huấn luyện mô hình trên Google Colab
│
├── outputs/
│   ├── result_image.jpg    # Hình ảnh kết quả sau khi hệ thống đóng khung nhận diện
│   └── result_video.mp4    # Video kết quả nhận diện thực nghiệm
│
├── src/
│   ├── __init__.py
│   └── detect.py           # Logic xử lý hình ảnh và video lõi bằng OpenCV
│
├── tests/
│   ├── sample_image.png    # Hình ảnh dữ liệu đầu vào để kiểm thử
│   └── sample_video.mp4    # Tệp video dữ liệu đầu vào để kiểm thử
│
├── utils/
│   ├── __init__.py
│   └── config.py           # Các hàm bổ trợ và cấu hình hệ thống toàn cục
│
├── .gitignore              # Cấu hình bỏ qua các tệp tin rác khi đẩy lên GitHub
├── README.md               # Tài liệu hướng dẫn sử dụng dự án
└── requirements.txt        # Danh sách các thư viện cần thiết của hệ thống
```

---

## Hướng dẫn Cài đặt và Sử dụng

### 1. Cài đặt môi trường và các thư viện lõi
Mở Terminal tại thư mục gốc của dự án trên PyCharm và chạy lệnh sau để cài đặt toàn bộ các thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```

### 2. Khởi chạy ứng dụng Giao diện đồ họa (GUI)
Để khởi động phần mềm điều khiển trực quan, thực hiện chạy file `gui_app.py`:

```bash
python app/gui_app.py
```

### 3. Khởi chạy Script xử lý qua dòng lệnh (CLI)
Nếu muốn chạy kiểm thử thuần bằng mã script OpenCV không qua giao diện, bạn có thể thực thi file `detect.py`:

```bash
python src/detect.py
```

---
