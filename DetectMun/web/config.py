# ================================================================
#  CẤU HÌNH ĐƯỜNG DẪN MODEL — chỉnh sửa cho khớp máy của bạn
# ================================================================

# Model 1 — DINOv2 ViT-B/14 (phân loại bệnh da)
DINOV2_MODEL_PATH = r"C:\DetectMun\DetectMun\dinov2_vitb14_best.pth"

# Model 2 — YOLOv8 (phát hiện từng nốt mụn)
YOLO_MODEL_PATH = r"C:\DetectMun\DetectMun\best.pt"

# Model 3 — ResNet50 5-fold ensemble (grading toàn ảnh)
# Đường dẫn source code acne-lds (chứa model/resnet50.py, transforms/...)
ACNE_LDS_SRC = r"C:\DetectMun\DetectMun\acne-lds-main"

RESNET_CKPT = {
    1: r"C:\DetectMun\DetectMun\fold_1\best-epoch=35-youden=0.7596.ckpt",
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
