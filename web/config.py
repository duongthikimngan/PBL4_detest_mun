# ================================================================
#  CẤU HÌNH ĐƯỜNG DẪN MODEL
#  - Local Windows : dùng đường dẫn tuyệt đối
#  - Docker/HF     : download từ HF Hub qua biến HF_MODEL_REPO
# ================================================================
import os, sys

_IS_DOCKER = sys.platform != "win32" or os.environ.get("DOCKER")

def _download_models():
    """Download model files từ HF Hub về /tmp/hf_models/ khi chạy trên HF Spaces."""
    from huggingface_hub import snapshot_download
    repo = os.environ["HF_MODEL_REPO"].strip()
    d = "/tmp/hf_models"
    print(f"⬇️  Downloading models from {repo} ...")
    snapshot_download(
        repo_id=repo,
        repo_type="model",
        local_dir=d,
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"   ✅ Models downloaded to {d}")
    return d

if not _IS_DOCKER:
    # ── Local Windows ────────────────────────────────────────────
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ws_dinov2 = os.path.join(ROOT, "dinov2_vitb14_best.pth")
    ws_yolo   = os.path.join(ROOT, "best.pt")
    ws_acne   = os.path.join(ROOT, "acne-lds-main")
    ws_ckpt   = os.path.join(ROOT, "fold_1", "best-epoch=35-youden=0.7596.ckpt")

    abs_base = r"C:\26F\Mun\DetectMun"
    abs_dinov2 = os.path.join(abs_base, "dinov2_vitb14_best.pth")
    abs_yolo   = os.path.join(abs_base, "best.pt")
    abs_acne   = os.path.join(abs_base, "acne-lds-main")
    abs_ckpt   = os.path.join(abs_base, "fold_1", "best-epoch=35-youden=0.7596.ckpt")

    DINOV2_MODEL_PATH = ws_dinov2 if os.path.exists(ws_dinov2) else abs_dinov2
    YOLO_MODEL_PATH   = ws_yolo   if os.path.exists(ws_yolo)   else abs_yolo
    ACNE_LDS_SRC      = ws_acne   if os.path.exists(ws_acne)   else abs_acne
    RESNET_CKPT       = {
        1: ws_ckpt if os.path.exists(ws_ckpt) else abs_ckpt,
    }
else:
    # ── Docker / HF Spaces ───────────────────────────────────────
    d = _download_models()
    DINOV2_MODEL_PATH = f"{d}/dinov2_vitb14_best.pth"
    YOLO_MODEL_PATH   = f"{d}/best.pt"
    ACNE_LDS_SRC      = "/app/acne-lds-main"
    RESNET_CKPT       = {1: f"{d}/fold_1/best-epoch=35-youden=0.7596.ckpt"}


# ================================================================
#  NGƯỠNG & HẰNG SỐ
# ================================================================
YOLO_CONF        = 0.01
YOLO_IMG_SIZE    = 640
ACNE_CLASS_INDEX = 0      # Class_0 = Acne trong DINOv2

# ── Temperature Scaling cho YOLO ─────────────────────────────────
# Giá trị T > 1 nén confidence xuống, tránh overconfident.
# Tune T trên tập val bằng NLL minimisation; mặc định 1.5 là điểm khởi đầu hợp lý.
YOLO_TEMPERATURE = 1.5

# ── Trọng số Weighted Average (từ benchmark val) ─────────────────
# ResNet50 đóng vai anchor grade (tin cậy hơn về mức độ tổng thể).
# YOLO chỉ bổ sung thông tin vị trí/số nốt, không quyết định grade.
# Chỉnh 2 giá trị này sau khi có kết quả benchmark thực tế.
RESNET_WEIGHT = 0.65
YOLO_WEIGHT   = 0.35      # RESNET_WEIGHT + YOLO_WEIGHT phải = 1.0

# ── Ngưỡng confidence DINOv2 để gắn cảnh báo (không chặn pipeline) ──
# DINOv2 luôn trả 1 trong 31 bệnh (không có class rỗng).
# Khi conf < ngưỡng này chỉ thêm warning, KHÔNG bỏ qua kết quả.
DINOV2_LOW_CONF_WARN = 0.5

# Tên bệnh tương ứng với từng class DINOv2 (31 classes, index 0–30)
# Class 0 = Acne — là class duy nhất đi vào luồng YOLO + ResNet
DINOV2_CLASS_NAMES = {
    0:  "Mụn trứng cá (Acne)",
    1:  "Dày sừng ánh sáng (Actinic Keratosis)",
    2:  "Ung thư tế bào đáy (Basal Cell Carcinoma)",
    3:  "Bệnh Darier (Darier's Disease)",
    4:  "U xơ da (Dermatofibroma)",
    5:  "Ly thượng bì bóng nước ngứa (Epidermolysis Bullosa Pruriginosa)",
    6:  "Bệnh Hailey-Hailey (Hailey-Hailey Disease)",
    7:  "Herpes Simplex",
    8:  "Chốc lở (Impetigo)",
    9:  "Ký sinh trùng di chuyển da (Larva Migrans)",
    10: "Phong trung gian (Leprosy Borderline)",
    11: "Phong u (Leprosy Lepromatous)",
    12: "Phong củ (Leprosy Tuberculoid)",
    13: "Lichen phẳng (Lichen Planus)",
    14: "Lupus đỏ mãn tính dạng đĩa (Lupus Erythematosus Chronicus Discoides)",
    15: "U hắc tố ác tính (Melanoma)",
    16: "U mềm lây (Molluscum Contagiosum)",
    17: "Nấm mycosis fungoides (Mycosis Fungoides)",
    18: "U xơ thần kinh (Neurofibromatosis)",
    19: "Nốt ruồi / Nevus (Nevus)",
    20: "Gai đen hợp lưu dạng lưới (Papillomatosis Confluentes and Reticulate)",
    21: "Chấy rận đầu (Pediculosis Capitis)",
    22: "Tổn thương dày sừng lành tính (Benign Keratosis-like Lesions)",
    23: "Vảy phấn hồng (Pityriasis Rosea)",
    24: "Dày sừng vòng (Porokeratosis Actinic)",
    25: "Vảy nến (Psoriasis)",
    26: "Ung thư tế bào vảy (Squamous Cell Carcinoma)",
    27: "Nấm thân (Tinea Corporis)",
    28: "Nấm đen (Tinea Nigra)",
    29: "Bọ cát (Tungiasis)",
    30: "Tổn thương mạch máu (Vascular Lesion)",
}

# Grade đặc biệt: da bình thường (YOLO không detect nốt nào VÀ ResNet grade 0)
# Dùng -1 để phân biệt với Grade 0 (mụn nhẹ)
NORMAL_SKIN_GRADE = -1

# 4 mức độ mụn — ResNet50 quyết định, YOLO bổ sung vị trí
GRADE_LABEL = {
    NORMAL_SKIN_GRADE: "Da bình thường",
    0: "Nhẹ (Grade 0)",
    1: "Trung bình (Grade 1)",
    2: "Nặng (Grade 2)",
    3: "Rất nặng (Grade 3)",
}

GRADE_COLOR_HEX = {
    NORMAL_SKIN_GRADE: "#16a34a",
    0: "#22c55e",
    1: "#eab308",
    2: "#f97316",
    3: "#ef4444",
}

RECOMMENDATIONS = {
    NORMAL_SKIN_GRADE: (
        "Không phát hiện nốt mụn. Da bạn trông bình thường - "
        "tiếp tục duy trì thói quen chăm sóc da hàng ngày."
    ),
    0: "Mụn nhẹ: dùng BHA/AHA không kê đơn, rửa mặt 2 lần/ngày.",
    1: "Mụn trung bình: cân nhắc gặp bác sĩ da liễu, có thể dùng Benzoyl Peroxide.",
    2: "Mụn nặng: cần kê đơn (retinoid, kháng sinh), gặp bác sĩ da liễu.",
    3: "Mụn rất nặng: điều trị chuyên khoa ngay, không tự ý điều trị tại nhà.",
}
