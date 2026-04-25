# ================================================================
#  CẤU HÌNH ĐƯỜNG DẪN MODEL
#  - Khi chạy LOCAL  : đặt đường dẫn tuyệt đối bên dưới
#  - Khi deploy HF   : set biến môi trường HF_MODEL_REPO
#    (app sẽ tự download models từ Hugging Face Hub)
# ================================================================
import os

def _download_models():
    """Download model files từ HF Hub về /tmp/models khi chạy trên HF Spaces."""
    from huggingface_hub import hf_hub_download
    repo = os.environ["HF_MODEL_REPO"]
    d = "/tmp/hf_models"
    os.makedirs(d + "/fold_1", exist_ok=True)
    paths = {}
    for fname, key in [
        ("dinov2_vitb14_best.pth",                     "dinov2"),
        ("best.pt",                                     "yolo"),
        ("fold_1/best-epoch=35-youden=0.7596.ckpt",    "resnet"),
    ]:
        print(f"⬇️  Downloading {fname} from HF Hub...")
        paths[key] = hf_hub_download(repo_id=repo, filename=fname,
                                     local_dir=d, local_dir_use_symlinks=False)
        print(f"   ✅ {fname} → {paths[key]}")
    return paths

_HF_REPO = os.environ.get("HF_MODEL_REPO", "")
if _HF_REPO:
    # ── Hugging Face Spaces deployment ──────────────────────────
    _p = _download_models()
    DINOV2_MODEL_PATH = _p["dinov2"]
    YOLO_MODEL_PATH   = _p["yolo"]
    ACNE_LDS_SRC      = os.path.join(os.path.dirname(__file__), "acne-lds-main")
    RESNET_CKPT       = {1: _p["resnet"]}
else:
    # ── Local development ────────────────────────────────────────
    DINOV2_MODEL_PATH = r"C:\26F\Mun\DetectMun\dinov2_vitb14_best.pth"
    YOLO_MODEL_PATH   = r"C:\26F\Mun\DetectMun\best.pt"
    ACNE_LDS_SRC      = r"C:\26F\Mun\DetectMun\acne-lds-main"
    RESNET_CKPT       = {
        1: r"C:\26F\Mun\DetectMun\fold_1\best-epoch=35-youden=0.7596.ckpt",
    }

# ================================================================
#  NGƯỠNG & HẰNG SỐ
# ================================================================
DINOV2_THRESHOLD = 0.5
YOLO_CONF        = 0.01
YOLO_IMG_SIZE    = 640
ACNE_CLASS_INDEX = 0      # Class_0 = Acne trong DINOv2

# Tên bệnh tương ứng với từng class DINOv2 (31 classes)
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
    30: "Da bình thường / Khác",
    31: "Tổn thương mạch máu (Vascular Lesion)",
}

GRADE_LABEL = {
    0: "Nhẹ (Grade 0)",
    1: "Trung bình (Grade 1)",
    2: "Nặng (Grade 2)",
    3: "Rất nặng (Grade 3)",
}

GRADE_COLOR_HEX = {
    0: "#22c55e",
    1: "#eab308",
    2: "#f97316",
    3: "#ef4444",
}

RECOMMENDATIONS = {
    0: "Mụn nhẹ: dùng BHA/AHA không kê đơn, rửa mặt 2 lần/ngày.",
    1: "Mụn trung bình: cân nhắc gặp bác sĩ da liễu, có thể dùng Benzoyl Peroxide.",
    2: "Mụn nặng: cần kê đơn (retinoid, kháng sinh), gặp bác sĩ da liễu.",
    3: "Mụn rất nặng: điều trị chuyên khoa ngay, không tự ý điều trị tại nhà.",
}
