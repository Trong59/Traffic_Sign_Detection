import os
from ultralytics import YOLO
import cv2
from utils.config import VIETNAMESE_LABELS

def load_model(model_path):
    print(f"[*] Đang nạp mô hình từ: {model_path}...")
    return YOLO(model_path)


def process_image(model, input_path, output_path, conf_thresh=0.5):
    print(f"[*] Đang phân tích ảnh: {input_path}")

    # Chạy nhận diện
    results = model.predict(source=input_path, conf=conf_thresh, verbose=False)
    result = results[0]

    # Ép buộc đổi tên tiếng Việt
    for class_id, class_code in result.names.items():
        if class_code in VIETNAMESE_LABELS:
            result.names[class_id] = VIETNAMESE_LABELS[class_code]

    # Vẽ và lưu ảnh
    annotated_frame = result.plot()
    cv2.imwrite(output_path, annotated_frame)
    print(f"[+] Đã lưu kết quả tại: {output_path}")

    # Hiển thị lên màn hình
    cv2.imshow("Ket qua Anh", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(model, input_path, output_path, conf_thresh=0.5):
    print(f"[*] Đang phân tích video: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[!] Lỗi: Không thể mở video {input_path}")
        return

    # Lấy thông số kỹ thuật của video gốc
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Khởi tạo công cụ ghi video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"\rĐang xử lý frame {frame_count}/{total_frames}...", end="")

        # Nhận diện trên từng frame
        results = model.predict(source=frame, conf=conf_thresh, verbose=False)
        result = results[0]

        # Ép buộc đổi tên tiếng Việt
        for class_id, class_code in result.names.items():
            if class_code in VIETNAMESE_LABELS:
                result.names[class_id] = VIETNAMESE_LABELS[class_code]

        # Vẽ bounding box lên frame
        annotated_frame = result.plot()

        # Ghi frame đã vẽ vào video đầu ra
        out.write(annotated_frame)

        # Hiển thị trực tiếp quá trình
        cv2.imshow("Ket qua Video (Bam 'q' de thoat)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[!] Người dùng đã dừng sớm.")
            break

    # Dọn dẹp bộ nhớ
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n[+] Đã lưu video kết quả tại: {output_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)

    output_dir = os.path.join(root_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(root_dir, "models", "best.pt")

    my_model = load_model(model_path)

    #XỬ LÝ ẢNH
    image_in = os.path.join(root_dir, "tests", "sample_image.png")
    image_out = os.path.join(output_dir, "result_image.jpg")
    process_image(my_model, image_in, image_out, conf_thresh=0.5)

    #XỬ LÝ VIDEO
    video_in = os.path.join(root_dir, "tests", "sample_video.mp4")
    video_out = os.path.join(output_dir, "result_video.mp4")
    process_video(my_model, video_in, video_out, conf_thresh=0.5)