import sys, os, warnings, concurrent.futures
import cv2, numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
from ultralytics import YOLO

from config import *

def _get_class_display_name(idx: int, fallback: str) -> str:
    """Trả về tên bệnh thân thiện; nếu không có mapping thì dùng fallback (Class_N)."""
    return DINOV2_CLASS_NAMES.get(idx, fallback)

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────
#  Biến toàn cục lưu model
# ──────────────────────────────────────────
dinov2_model     = None
dinov2_transform = None
CLASS_NAMES      = []

yolo_model       = None

resnet_models    = {}
resnet_transform = None


# ──────────────────────────────────────────
#  Load model
# ──────────────────────────────────────────
def load_all_models():
    global dinov2_model, dinov2_transform, CLASS_NAMES
    global yolo_model
    global resnet_models, resnet_transform

    errors = []

    # ── Model 1: DINOv2 ──────────────────
    try:
        ckpt       = torch.load(DINOV2_MODEL_PATH, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        num_cls    = state_dict["head.weight"].shape[0] if "head.weight" in state_dict else 2

        dinov2_model = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=False, num_classes=num_cls, img_size=518,
        )
        clean = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        dinov2_model.load_state_dict(clean, strict=False)
        dinov2_model = dinov2_model.to(device).eval()
        CLASS_NAMES  = [f"Class_{i}" for i in range(num_cls)]

        dinov2_transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])
        print(f"✅ DINOv2 loaded ({num_cls} classes)")
    except Exception as e:
        errors.append(f"DINOv2: {e}")
        print(f"❌ DINOv2: {e}")

    # ── Model 2: YOLOv8 ──────────────────
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLOv8 loaded")
    except Exception as e:
        errors.append(f"YOLOv8: {e}")
        print(f"❌ YOLOv8: {e}")

    # ── Model 3: ResNet50 5-fold ──────────
    try:
        if ACNE_LDS_SRC not in sys.path:
            sys.path.insert(0, ACNE_LDS_SRC)
        from model.resnet50 import resnet50 as AcneResNet50
        from transforms.acne_transforms import AcneTransformsTorch

        resnet_transform = AcneTransformsTorch(train=False)

        for fold, ckpt_path in RESNET_CKPT.items():
            try:
                m   = AcneResNet50(num_acne_cls=13)
                ckpt = torch.load(ckpt_path, map_location=device)
                sd  = {k.replace("cnn.", "", 1): v for k, v in ckpt["state_dict"].items()}
                m.load_state_dict(sd)
                resnet_models[fold] = m.to(device).eval()
                print(f"   ✅ ResNet fold {fold}")
            except Exception as e:
                print(f"   ⚠️  ResNet fold {fold}: {e}")

        print(f"✅ ResNet50 ensemble: {len(resnet_models)}/5 folds")
    except Exception as e:
        errors.append(f"ResNet50: {e}")
        print(f"❌ ResNet50: {e}")

    return errors


# ──────────────────────────────────────────
#  Các hàm suy luận
# ──────────────────────────────────────────
def classify_dinov2(img_pil):
    tensor = dinov2_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(dinov2_model(tensor), dim=1)[0]
    idx = probs.argmax().item()
    fallback = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class_{idx}"
    display_name = _get_class_display_name(idx, fallback)
    return idx, probs[idx].item(), display_name


def detect_yolo(img_rgb):
    h, w   = img_rgb.shape[:2]
    scale  = YOLO_IMG_SIZE / max(h, w)
    img_in = cv2.resize(img_rgb, (int(w*scale), int(h*scale))) if scale < 1 else img_rgb
    res    = yolo_model(img_in, conf=YOLO_CONF, verbose=False)[0]
    count  = len(res.boxes)
    confs  = res.boxes.conf.cpu().numpy().tolist() if count > 0 else []
    # Vẽ bounding box thủ công — chỉ hiển thị ô, không có nhãn chữ
    annotated = img_in.copy()
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 50, 50), 2)
    return count, confs, annotated


def yolo_severity(count):
    if count <= 5:  return 0, "Mild (≤5 nốt)",       "#22c55e"
    if count <= 20: return 1, "Moderate (6–20 nốt)",  "#eab308"
    if count <= 50: return 2, "Severe (21–50 nốt)",   "#f97316"
    return              3,    "Very Severe (>50 nốt)","#ef4444"


def grade_resnet(img_pil):
    if not resnet_models:
        return None, None
    tensor    = resnet_transform(img_pil).unsqueeze(0).to(device)
    all_probs = []
    with torch.no_grad():
        for m in resnet_models.values():
            cls_log, cou_log, cou2cls_log = m(tensor)
            merged = torch.stack((
                cls_log[:, :1].sum(1),
                cls_log[:, 1:4].sum(1),
                cls_log[:, 4:10].sum(1),
                cls_log[:, 10:].sum(1),
            ), dim=1)
            combined = 0.5 * (merged + cou2cls_log[:, :4])
            all_probs.append(torch.softmax(combined, dim=1)[0].cpu().numpy())
    mean_probs = np.mean(all_probs, axis=0)
    return int(np.argmax(mean_probs)), mean_probs.tolist()


# ──────────────────────────────────────────
#  Pipeline chính
# ──────────────────────────────────────────
def run_pipeline(img_pil: Image.Image) -> dict:
    img_rgb = np.array(img_pil.convert("RGB"))

    r = dict(
        is_acne=False, dinov2_class="", dinov2_conf=0.0, dinov2_warning=False,
        yolo_count=0, yolo_confs=[], yolo_severity_grade=None,
        yolo_severity_label="", yolo_severity_color="#6b7280",
        resnet_grade=None, resnet_probs=[], resnet_grade_label="",
        final_grade=None, final_label="", final_color="#6b7280",
        inconsistent=False, recommendation="",
        annotated_image_b64="", warnings=[],
    )

    # Step 1 — DINOv2
    if dinov2_model is None:
        r["warnings"].append("DINOv2 chưa được load.")
        return r

    pred_idx, conf, display_name = classify_dinov2(img_pil)
    class_name = display_name
    r["dinov2_class"]   = class_name
    r["dinov2_conf"]    = round(conf, 4)
    r["dinov2_warning"] = conf < DINOV2_THRESHOLD

    if r["dinov2_warning"]:
        r["warnings"].append(f"⚠️ Độ tin cậy phân loại thấp ({conf:.1%}) — kết quả có thể không chính xác.")

    is_acne = (pred_idx == ACNE_CLASS_INDEX) and (conf >= DINOV2_THRESHOLD)
    r["is_acne"] = is_acne

    if not is_acne:
        r["recommendation"] = "Không phát hiện mụn trứng cá. Nếu có vấn đề da liễu, hãy tham khảo bác sĩ."
        return r

    # Step 2 — YOLO + ResNet song song
    import base64, io

    yolo_result  = [None]   # (count, confs, annotated_img)
    resnet_result = [None]  # (grade, probs)

    def _yolo():
        if yolo_model:
            yolo_result[0] = detect_yolo(img_rgb)

    def _resnet():
        if resnet_models:
            resnet_result[0] = grade_resnet(img_pil)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        concurrent.futures.wait([pool.submit(_yolo), pool.submit(_resnet)])

    # ── Xử lý kết quả YOLO ──────────────────────────────────────────
    if yolo_result[0] is not None:
        count, confs, annotated = yolo_result[0]
        yg, yl, yc = yolo_severity(count)
        r.update(yolo_count=count, yolo_confs=confs,
                 yolo_severity_grade=yg, yolo_severity_label=yl,
                 yolo_severity_color=yc)
        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="JPEG", quality=85)
        r["annotated_image_b64"] = base64.b64encode(buf.getvalue()).decode()
    else:
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG", quality=85)
        r["annotated_image_b64"] = base64.b64encode(buf.getvalue()).decode()

    # ── Xử lý kết quả ResNet50 ──────────────────────────────────────
    if resnet_result[0] is not None:
        grade, probs = resnet_result[0]
        if grade is not None:
            r.update(resnet_grade=grade,
                     resnet_probs=[round(p, 4) for p in probs],
                     resnet_grade_label=GRADE_LABEL.get(grade, ""))

    # ── Step 3: Voting 2/3 ──────────────────────────────────────────
    # 3 model độc lập: DINOv2, YOLO, ResNet50
    # Mỗi model 1 phiếu. Kết quả nào có ≥2 phiếu thì thắng.
    #
    # TH đặc biệt: DINOv2 nói "mụn" nhưng
    #   • YOLO detect 0 nốt  → phiếu "bình thường"
    #   • ResNet grade == 0  → phiếu "bình thường"
    # → 2 phiếu "bình thường" vs 1 phiếu "mụn" → kết luận DA BÌNH THƯỜNG

    yg         = r["yolo_severity_grade"]   # None nếu YOLO không chạy
    rg         = r["resnet_grade"]          # None nếu ResNet không chạy
    yolo_count = r["yolo_count"]

    yolo_votes_normal   = (yolo_result[0] is not None) and (yolo_count == 0)
    resnet_votes_normal = (rg is not None) and (rg == 0)

    if yolo_votes_normal and resnet_votes_normal:
        # 2/3 phiếu "bình thường" → override DINOv2
        r["is_acne"] = False
        r["warnings"].append(
            "⚑ DINOv2 nhận định mụn trứng cá nhưng YOLO không phát hiện nốt mụn nào "
            "và ResNet50 đánh giá Grade 0. Kết luận theo đa số (2/3): da bình thường."
        )
        r.update(
            final_grade=None,
            final_label="Da bình thường",
            final_color="#22c55e",
            recommendation="Không phát hiện mụn rõ ràng. Nếu bạn vẫn lo ngại, hãy tham khảo bác sĩ da liễu.",
        )
        return r

    # Các trường hợp còn lại: DINOv2 + ít nhất 1 model đồng ý có mụn
    # Ưu tiên ResNet (grading toàn ảnh), fallback sang YOLO
    final = rg if rg is not None else yg
    inconsistent = (yg is not None and rg is not None and abs(yg - rg) >= 2)

    if inconsistent:
        r["warnings"].append(
            f"⚑ Hai mô hình có nhận định khác nhau "
            f"(YOLO ước tính Grade {yg}, ResNet50 cho Grade {rg}). Nên tham khảo bác sĩ."
        )

    r.update(
        inconsistent=inconsistent,
        final_grade=final,
        final_label=GRADE_LABEL.get(final, "Không xác định"),
        final_color=GRADE_COLOR_HEX.get(final, "#6b7280"),
        recommendation=RECOMMENDATIONS.get(final, ""),
    )
    return r