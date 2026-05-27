# 🚀 Zero-shot BOM Pattern Detection

Báo cáo cấu phần và hướng dẫn sử dụng mã nguồn cho hệ thống tự động phát hiện mẫu bản vẽ kỹ thuật (BOM/Schematic) không cần huấn luyện (Zero-shot).

---

## 📌 4.1. GitHub Repository

* **Repository URL:** `<YOUR_GITHUB_REPO_URL>`
* **Mục đích:** Chứa toàn bộ mã nguồn xử lý lõi của hệ thống **Zero-shot BOM Pattern Detection**. Hệ thống được thiết kế để tìm kiếm các vùng khớp với ảnh mẫu (Pattern) trong một bản vẽ kỹ thuật lớn mà không cần training, fine-tuning hoặc sử dụng trọng số của bất kỳ mô hình học máy pretrained nào.

### 📊 Yêu cầu & Trạng thái dự án

| Yêu cầu | Trạng thái | Ghi chú |
| --- | --- | --- |
| **Mã nguồn hoàn chỉnh** | ✅ | Nằm trong thư mục `src/pattern_detector/` |
| **Khả năng chạy ngay** | ✅ | Hỗ trợ cả CLI script và giao diện Gradio |
| **Tài liệu README rõ ràng** | ✅ | Đầy đủ cài đặt, cách chạy, kiểm thử và deploy |
| **Cấu trúc thư mục sạch** | ✅ | Phân tách rõ ràng giữa source, scripts, docs, và tests |
| **Comments / Docstrings** | ✅ | Viết chi tiết tại các module xử lý thuật toán lõi |
| **File quản lý thư viện** | ✅ | Sẵn sàng với file `requirements.txt` |
| **Trọng số mô hình (Weights)** | 🚫 | Không yêu cầu / Không tải lên mạng |
| **Sẵn sàng triển khai** | ✅ | Tương thích hoàn toàn với HuggingFace Spaces |

---

## 📂 Cấu trúc thư mục (Repository Structure)

```text
zero-shot-bom-pattern-detection/
├── app.py                      # Ứng dụng Gradio giao diện web
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── README.md                   # Tài liệu hướng dẫn này
├── configs/
│   └── default.yaml            # Cấu hình tham số mặc định cho pipeline
├── docs/
│   └── system_design.md        # Tài liệu thiết kế hệ thống chi tiết
├── examples/                   # Dữ liệu mẫu phục vụ test nhanh
│   ├── patterns/               # Ảnh mẫu cần tìm
│   ├── drawings/               # Ảnh bản vẽ lớn
│   └── outputs/                # Kết quả đầu ra (Ảnh + JSON)
├── scripts/                    # Các kịch bản thực thi nhanh
│   ├── run_inference.py        # Chạy inference qua dòng lệnh (CLI)
│   ├── benchmark.py            # Đánh giá hiệu năng thuật toán
│   └── make_examples.py        # Tạo dữ liệu demo mẫu
├── src/                        # Mã nguồn lõi của dự án
│   └── pattern_detector/
│       ├── __init__.py
│       ├── detector.py
│       ├── preprocessing.py
│       ├── candidate_generator.py
│       ├── scoring.py
│       ├── chamfer.py
│       ├── nms.py
│       ├── config.py
│       ├── schemas.py
│       ├── geometry.py
│       ├── integral_pruning.py
│       └── visualization.py
└── tests/                      # Thư mục chứa Unit Tests

```

---

## 🛠️ Chi tiết các Module mã nguồn chính (`src/pattern_detector/`)

* **`preprocessing.py` (Tiền xử lý):** Chuyển ảnh sang hệ màu xám (Grayscale), binarization/trích xuất cạnh, crop chặt ảnh mẫu (Pattern), resize bản vẽ giữ nguyên tỉ lệ và theo dõi scale factor để map bounding box ngược lại ảnh gốc.
* **`candidate_generator.py` (Tạo vùng đề xuất):** Sinh các ứng viên thông qua template matching đa tỷ lệ (Multi-scale) và đa góc xoay (Rotation-aware), kết hợp gộp ứng viên và kiểm soát ngân sách vùng gợi ý (Candidate budget control).
* **`chamfer.py` (Xác thực Chamfer):** Cài đặt thuật toán Directional Chamfer matching nâng cao thông qua ước lượng hướng gradient, phân nhóm hướng (binning) và biến đổi khoảng cách hướng.
* **`scoring.py` (Tính điểm ứng viên):** Chịu trách nhiệm chấm điểm toàn diện dựa trên: Edge IoU, Bidirectional edge precision/recall/F1, phạt biên ngoài (Outside-edge penalty), độ nhất quán về mật độ nét vẽ và tỷ lệ khung hình.
* **`detector.py` (Điều phối Pipeline):** Đầu não kết nối toàn bộ luồng từ Tiền xử lý ➡️ Sinh ứng viên ➡️ Đánh giá điểm số ➡️ Lọc ngưỡng động ➡️ NMS (Non-Maximum Suppression) ➡️ Xuất metadata JSON.
* **`nms.py` (Lọc trùng):** Xử lý Non-Maximum Suppression để loại bỏ các bounding box bị đè lên nhau trùng lặp.
* **`visualization.py` (Trực quan hóa):** Vẽ bounding box và ghi điểm tự tin (Confidence score) trực tiếp lên ảnh kết quả đầu ra.

---

## 💻 Hướng dẫn cài đặt & Sử dụng

### 1. Khởi tạo môi trường

```bash
# Clone dự án về máy địa phương
git clone https://github.com/chplay2020/zero-shot-technical-drawing-search
cd zero-shot-bom-pattern-detection

# Khởi tạo Virtual Environment
# Dành cho Linux / macOS / WSL:
python -m venv .venv
source .venv/bin/activate

# Dành cho Windows PowerShell:
python -m venv .venv
.venv\Scripts\activate

# Cập nhật pip và cài đặt thư viện
pip install --upgrade pip
pip install -r requirements.txt

```

> ⚠️ **Lưu ý quan trọng về Dependencies:**
> Hệ thống chỉ sử dụng các thư viện tính toán siêu nhẹ trên CPU (`opencv-python-headless`, `numpy`, `scipy`, `Pillow`, v.v.). Hoàn toàn **KHÔNG** yêu cầu các mô hình nặng như `torch`, `transformers`, `GroundingDINO`, hay `SAM`.

### 2. Sử dụng qua giao diện dòng lệnh (CLI)

Cấu trúc câu lệnh tổng quát:

```bash
python scripts/run_inference.py \
  --pattern <PATH_TO_PATTERN_IMAGE> \
  --drawing <PATH_TO_DRAWING_IMAGE> \
  --output <PATH_TO_OUTPUT_IMAGE> \
  --json-output <PATH_TO_OUTPUT_JSON>

```

**Ví dụ chạy thực tế với dữ liệu mẫu:**

```bash
# Test mẫu số 1
python scripts/run_inference.py \
  --pattern examples/patterns/1-1.png \
  --drawing examples/drawings/1.png \
  --output examples/outputs/1-1_result.png \
  --json-output examples/outputs/1-1_result.json

# Test mẫu số 2
python scripts/run_inference.py \
  --pattern examples/patterns/1-2.png \
  --drawing examples/drawings/1.png \
  --output examples/outputs/1-2_result.png \
  --json-output examples/outputs/1-2_result.json

```

*Các tham số mở rộng tùy chọn:*

* `--config configs/default.yaml`: Chỉ định file cấu hình riêng.
* `--confidence-threshold 0.5`: Thay đổi ngưỡng điểm tin cậy.
* `--advanced-search`: Kích hoạt chế độ tìm kiếm nâng cao nâng độ chính xác.

### 3. Khởi chạy giao diện Web (Gradio Local)

Để kiểm tra trực quan bằng kéo thả chuột trên trình duyệt, khởi chạy:

```bash
python app.py

```

Sau đó truy cập đường dẫn local hiển thị trên terminal (thông thường là `http://127.0.0.1:7860`). Giao diện hỗ trợ tải ảnh bản vẽ, ảnh mẫu và hiển thị ngay kết quả bounding box kèm file JSON metadata tiện lợi.

---

## 🧪 Kiểm thử dự án (Testing)

Trước khi tiến hành nộp bài (Submission), hãy đảm bảo chạy quy trình kiểm tra chất lượng code tự động sau:

```bash
# 1. Biên dịch thử toàn bộ file python xem có lỗi cú pháp không
python -m compileall src scripts app.py

# 2. Chạy bộ Unit Tests bằng pytest
PYTHONPATH=$PWD/src pytest tests/ -q

```