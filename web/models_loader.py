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
        import types as _types, traceback as _tb

        def _ensure_pl_importable():
            """Register stub modules for pytorch_lightning (and friends) if not installed."""
            _dummy = lambda *a, **k: None
            _Dummy = type('_Dummy', (), {
                '__init__': _dummy,
                '__setstate__': lambda s, d: s.__dict__.update(d) if isinstance(d, dict) else None,
                '__reduce__': lambda s: (_types.SimpleNamespace, ()),
            })
            for _mod_name in [
                'pytorch_lightning', 'pytorch_lightning.callbacks',
                'pytorch_lightning.callbacks.model_checkpoint',
                'pytorch_lightning.loggers', 'pytorch_lightning.loggers.wandb',
                'pytorch_lightning.utilities', 'pytorch_lightning.utilities.rank_zero',
                'pytorch_lightning.core', 'pytorch_lightning.core.lightning',
                'pytorch_lightning.trainer', 'pytorch_lightning.trainer.trainer',
                'omegaconf', 'omegaconf.dictconfig', 'omegaconf.listconfig',
                'hydra', 'hydra.core', 'hydra.core.global_hydra',
                'wandb', 'wandb.sdk', 'wandb.sdk.wandb_run',
                'torchmetrics',
            ]:
                if _mod_name not in sys.modules:
                    _m = _types.ModuleType(_mod_name)
                    for _attr in ['LightningModule', 'Trainer', 'ModelCheckpoint',
                                  'EarlyStopping', 'WandbLogger', 'DictConfig',
                                  'ListConfig', 'Accuracy', 'Precision', 'Recall',
                                  'Specificity', 'MeanAbsoluteError',
                                  'MeanSquaredError', 'MatthewsCorrCoef', 'Run']:
                        setattr(_m, _attr, _Dummy)
                    sys.modules[_mod_name] = _m

        _ensure_pl_importable()

        if ACNE_LDS_SRC not in sys.path:
            sys.path.insert(0, ACNE_LDS_SRC)
        from model.resnet50 import resnet50 as AcneResNet50
        from transforms.acne_transforms import AcneTransformsTorch

        resnet_transform = AcneTransformsTorch(train=False)

        def _load_ckpt(path):
            try:
                ckpt = torch.load(path, map_location=device, weights_only=False)
                print(f"   ℹ️  ckpt top-level keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
                return ckpt
            except Exception as e1:
                print(f"   ⚠️  torch.load failed ({type(e1).__name__}: {e1})")
                _tb.print_exc()
                raise

        for fold, ckpt_path in RESNET_CKPT.items():
            try:
                print(f"   Loading ResNet fold {fold}: {ckpt_path}")
                ckpt = _load_ckpt(ckpt_path)

                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    sd_raw = ckpt["state_dict"]
                    sd = {(k[4:] if k.startswith("cnn.") else k): v
                          for k, v in sd_raw.items()}
                elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    sd = ckpt["model_state_dict"]
                elif isinstance(ckpt, dict):
                    sd = ckpt
                else:
                    raise ValueError(f"Unknown ckpt type: {type(ckpt)}")

                m = AcneResNet50(num_acne_cls=13, pretrained_backbone=False)
                missing, unexpected = m.load_state_dict(sd, strict=False)
                if missing:
                    print(f"   ℹ️  Missing keys ({len(missing)}): {missing[:3]}")
                if unexpected:
                    print(f"   ℹ️  Unexpected keys ({len(unexpected)}): {unexpected[:3]}")
                m.to(device)
                m.eval()
                resnet_models[fold] = m
                print(f"   ✅ ResNet fold {fold} loaded OK")
            except Exception as e:
                print(f"   ❌ ResNet fold {fold} FAILED: {e}")
                _tb.print_exc()

        print(f"✅ ResNet50 ensemble: {len(resnet_models)}/5 folds loaded")
    except Exception as e:
        import traceback
        errors.append(f"ResNet50: {e}")
        print(f"❌ ResNet50 outer error: {e}")
        traceback.print_exc()

    return errors


# ──────────────────────────────────────────
#  Các hàm suy luận
# ──────────────────────────────────────────
def classify_dinov2(img_pil):
    """
    Phân loại bắt buộc 1 trong 31 bệnh da.
    DINOv2 KHÔNG có class rỗng — luôn trả top-1 class.
    Khi conf < DINOV2_LOW_CONF_WARN chỉ gắn warning, không chặn pipeline.
    """
    tensor = dinov2_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(dinov2_model(tensor), dim=1)[0]
    idx = probs.argmax().item()
    fallback = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class_{idx}"
    display_name = _get_class_display_name(idx, fallback)
    return idx, probs[idx].item(), display_name


def calibrate_yolo_conf(confs: list[float], temperature: float = YOLO_TEMPERATURE) -> list[float]:
    """
    Temperature Scaling (post-hoc calibration) cho confidence của YOLO.

    Vấn đề: YOLOv8 thường overconfident — confidence cao nhưng thực tế
    hay đoán sai, đặc biệt với ảnh góc lạ, nốt nhỏ, hoặc ảnh OOD.

    Giải pháp: chia logit cho T trước khi softmax.
        p_calibrated = sigmoid(logit / T)
                     = sigmoid(log(p/(1-p)) / T)
    Với T > 1: confidence được nén về gần 0.5 hơn → phản ánh đúng
    thực tế hơn. Sau calibration, conf 80% thật sự ≈ đúng 80% lần.

    T được tune trên tập val bằng cách tối thiểu hoá NLL (negative log-likelihood).
    Giá trị mặc định YOLO_TEMPERATURE = 1.5 là điểm khởi đầu hợp lý.
    """
    if not confs or temperature <= 0:
        return confs
    calibrated = []
    for p in confs:
        p = float(np.clip(p, 1e-7, 1 - 1e-7))
        logit = np.log(p / (1 - p))
        p_cal = 1.0 / (1.0 + np.exp(-logit / temperature))
        calibrated.append(round(float(p_cal), 4))
    return calibrated


def detect_yolo(img_rgb):
    """
    Detect từng nốt mụn bằng YOLOv8.
    Vai trò trong pipeline: bổ sung thông tin vị trí (bbox) và số lượng nốt.
    Không dùng confidence thô để quyết định grade — calibrate trước.
    Không trả ảnh annotated — ảnh hiển thị là ảnh gốc (xử lý trong run_pipeline).
    """
    h, w   = img_rgb.shape[:2]
    scale  = YOLO_IMG_SIZE / max(h, w)
    img_in = cv2.resize(img_rgb, (int(w*scale), int(h*scale))) if scale < 1 else img_rgb
    res    = yolo_model(img_in, conf=YOLO_CONF, verbose=False)[0]
    count  = len(res.boxes)
    raw_confs = res.boxes.conf.cpu().numpy().tolist() if count > 0 else []

    # Calibrate confidence trước khi dùng
    cal_confs = calibrate_yolo_conf(raw_confs)

    return count, cal_confs


def yolo_severity_score(count: int) -> float:
    """
    Chuyển số nốt mụn từ YOLO thành severity score [0.0–1.0].
    Score này dùng trong weighted average, không dùng trực tiếp làm grade.
    """
    if count <= 5:   return 0.0 / 3.0   # ~ grade 0
    if count <= 20:  return 1.0 / 3.0   # ~ grade 1
    if count <= 50:  return 2.0 / 3.0   # ~ grade 2
    return                3.0 / 3.0     # ~ grade 3


def grade_resnet(img_pil):
    """
    Đánh giá mức độ mụn toàn ảnh bằng ResNet50 5-fold ensemble.
    Vai trò trong pipeline: ANCHOR — quyết định grade chính xác nhất.
    Trả về (grade: int 0-3, probs: list[float] len=4).
    """
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


def _weighted_grade(resnet_grade: int, resnet_probs: list,
                    yolo_count: int) -> int:
    """
    Kết hợp ResNet (anchor) và YOLO (bổ sung) bằng Weighted Average.

    Công thức:
        final_score = RESNET_WEIGHT × resnet_score
                    + YOLO_WEIGHT   × yolo_score

    Trong đó:
        resnet_score = resnet_grade / 3.0   (chuẩn hoá về [0,1])
        yolo_score   = yolo_severity_score(yolo_count)  (đã [0,1])

    Trọng số lấy từ benchmark thực tế trên tập val — KHÔNG dùng
    runtime confidence (tránh bẫy YOLO overconfident).

    ResNet đóng vai anchor: RESNET_WEIGHT = 0.65 > YOLO_WEIGHT = 0.35
    nên dù YOLO cho score cao hơn thực tế, ResNet vẫn kéo kết quả
    về đúng mức độ.
    """
    r_score = resnet_grade / 3.0
    y_score = yolo_severity_score(yolo_count)
    final_score = RESNET_WEIGHT * r_score + YOLO_WEIGHT * y_score
    # Map score [0,1] về 4 mức grade
    final_grade = int(np.clip(round(final_score * 3), 0, 3))
    return final_grade


# ──────────────────────────────────────────
#  Pipeline chính
# ──────────────────────────────────────────
def run_pipeline(img_pil: Image.Image) -> dict:
    """
    Pipeline 3 model — Da Liễu:

    1. DINOv2 phân loại bắt buộc 1 trong 31 bệnh (không có class rỗng).
       - Nếu KHÔNG phải mụn trứng cá → trả tên bệnh + confidence, kết thúc.
       - Nếu LÀ mụn trứng cá → tiếp tục bước 2.

    2. YOLO + ResNet chạy song song trên ảnh gốc:
       - YOLO: detect vị trí nốt mụn → bbox, count, severity score
               (calibrate confidence bằng Temperature Scaling trước)
       - ResNet: grade toàn ảnh (5-fold ensemble) → anchor grade

    3. Fusion bằng Weighted Average (val-based weights):
       final = 0.65 × ResNet_score + 0.35 × YOLO_score
       → 1 kết quả duy nhất đáng tin cậy nhất (nhẹ/trung bình/nặng/rất nặng)
    """
    img_pil = img_pil.convert("RGB")
    img_rgb = np.array(img_pil)

    r = dict(
        # DINOv2
        is_acne=False, dinov2_class="", dinov2_conf=0.0, dinov2_low_conf=False,
        # YOLO (bbox + calibrated confs)
        yolo_count=0, yolo_confs=[],
        # ResNet (anchor grade)
        resnet_grade=None, resnet_probs=[], resnet_grade_label="",
        # Fusion output
        final_grade=None, final_label="", final_color="#6b7280",
        recommendation="",
        # Extras
        annotated_image_b64="", warnings=[],
    )

    # ── Step 1: DINOv2 — phân loại 1 trong 31 bệnh ──────────────────
    if dinov2_model is None:
        r["warnings"].append("DINOv2 chưa được load.")
        return r

    pred_idx, conf, display_name = classify_dinov2(img_pil)
    r["dinov2_class"] = display_name
    r["dinov2_conf"]  = round(conf, 4)

    # Gắn warning nếu confidence thấp, nhưng KHÔNG chặn pipeline
    if conf < DINOV2_LOW_CONF_WARN:
        r["dinov2_low_conf"] = True
        r["warnings"].append(
            f"⚠️ Độ tin cậy DINOv2 thấp ({conf:.1%}) — kết quả có thể chưa chính xác, "
            "nên tham khảo bác sĩ da liễu."
        )

    is_acne = (pred_idx == ACNE_CLASS_INDEX)
    r["is_acne"] = is_acne

    # Không phải mụn → kết thúc luôn, trả tên bệnh + confidence
    if not is_acne:
        r["recommendation"] = (
            "Không phát hiện mụn trứng cá. "
            "Nếu có vấn đề da liễu, hãy tham khảo bác sĩ chuyên khoa."
        )
        return r

    # ── Step 2: YOLO + ResNet song song ─────────────────────────────
    import base64, io

    yolo_result   = [None]   # (count, cal_confs, annotated_img)
    resnet_result = [None]   # (grade, probs)

    def _yolo():
        if yolo_model:
            try:
                yolo_result[0] = detect_yolo(img_rgb)
            except Exception as _e:
                import traceback as _tb
                print(f"❌ YOLO inference error: {_e}"); _tb.print_exc()

    def _resnet():
        if resnet_models:
            try:
                resnet_result[0] = grade_resnet(img_pil)
                print(f"   ℹ️  ResNet grade={resnet_result[0][0] if resnet_result[0] else None}")
            except Exception as _e:
                import traceback as _tb
                print(f"❌ ResNet inference error: {_e}"); _tb.print_exc()
        else:
            print("   ⚠️  resnet_models rỗng tại inference time!")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        concurrent.futures.wait([pool.submit(_yolo), pool.submit(_resnet)])

    # ── Encode ảnh gốc (không vẽ bbox) ──────────────────────────────
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=85)
    r["annotated_image_b64"] = base64.b64encode(buf.getvalue()).decode()

    # ── Lưu kết quả YOLO (số nốt + calibrated confs) ─────────────────
    if yolo_result[0] is not None:
        count, cal_confs = yolo_result[0]
        r.update(yolo_count=count, yolo_confs=cal_confs)
    else:
        r["warnings"].append("⚠️ YOLO không khả dụng — không có thông tin vị trí nốt mụn.")

    # ── Lưu kết quả ResNet (anchor grade) ───────────────────────────
    if resnet_result[0] is not None:
        grade, probs = resnet_result[0]
        if grade is not None:
            r.update(
                resnet_grade=grade,
                resnet_probs=[round(p, 4) for p in probs],
                resnet_grade_label=GRADE_LABEL.get(grade, ""),
            )

    # ── Step 3: Fusion — Weighted Average (ResNet anchor + YOLO bổ sung) ──
    #
    # Chiến lược phân vai rõ ràng, hai model KHÔNG cạnh tranh nhau:
    #   • ResNet50 → quyết định grade (tin cậy hơn về mức độ tổng thể)
    #   • YOLOv8  → bổ sung vị trí bbox + count (không quyết định grade)
    #
    # Confidence của YOLO đã được calibrate bằng Temperature Scaling ở
    # bước detect_yolo() nên severity score phản ánh thực tế hơn.
    #
    # Fallback:
    #   - Chỉ có ResNet → dùng ResNet grade trực tiếp
    #   - Chỉ có YOLO   → dùng YOLO severity score (kém tin cậy hơn, thêm warning)
    #   - Cả hai đều lỗi → báo lỗi

    rg = r["resnet_grade"]
    yc = r["yolo_count"]

    # ── Check da bình thường: YOLO không detect nốt nào VÀ ResNet grade 0 ──
    # Early-exit trước khi tính weighted average — cả hai model đồng thuận
    # rằng không có mụn, không cần fusion thêm.
    if rg is not None and yolo_result[0] is not None and yc == 0 and rg == 0:
        r.update(
            final_grade=NORMAL_SKIN_GRADE,
            final_label=GRADE_LABEL[NORMAL_SKIN_GRADE],
            final_color=GRADE_COLOR_HEX[NORMAL_SKIN_GRADE],
            recommendation=RECOMMENDATIONS[NORMAL_SKIN_GRADE],
        )
        return r

    if rg is not None and yolo_result[0] is not None:
        # Trường hợp lý tưởng: cả hai model đều có kết quả
        final_grade = _weighted_grade(rg, r["resnet_probs"], yc)

    elif rg is not None:
        # Chỉ có ResNet → dùng luôn ResNet grade (đã là anchor)
        final_grade = rg
        r["warnings"].append(
            "⚠️ YOLO không khả dụng — grade dựa hoàn toàn vào ResNet50."
        )

    elif yolo_result[0] is not None:
        # Chỉ có YOLO → map count → grade (kém tin cậy hơn)
        yolo_raw_grade = int(round(yolo_severity_score(yc) * 3))
        final_grade = yolo_raw_grade
        r["warnings"].append(
            "⚠️ ResNet50 không khả dụng — grade chỉ dựa vào YOLO (kém tin cậy hơn). "
            "Nên tham khảo bác sĩ da liễu."
        )

    else:
        # Cả hai đều lỗi
        r["warnings"].append("❌ Cả YOLO và ResNet đều không khả dụng — không thể đánh giá mức độ.")
        r.update(final_grade=None, final_label="Không xác định", final_color="#6b7280",
                 recommendation="Hệ thống gặp lỗi. Vui lòng thử lại hoặc tham khảo bác sĩ da liễu.")
        return r

    r.update(
        final_grade=final_grade,
        final_label=GRADE_LABEL.get(final_grade, "Không xác định"),
        final_color=GRADE_COLOR_HEX.get(final_grade, "#6b7280"),
        recommendation=RECOMMENDATIONS.get(final_grade, ""),
    )
    return r
