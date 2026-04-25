"""
Diagnostic script — chạy: python check_resnet.py
Kiểm tra xem ResNet checkpoint có load được không và lý do thất bại.
"""
import sys, os, types, traceback

CKPT = r"C:\26F\Mun\DetectMun\fold_1\best-epoch=35-youden=0.7596.ckpt"
ACNE_SRC = r"C:\26F\Mun\DetectMun\acne-lds-main"

print("=" * 60)
print("1. Kiểm tra file tồn tại...")
print("  Exists:", os.path.isfile(CKPT))
print("  Size:", os.path.getsize(CKPT) if os.path.isfile(CKPT) else "N/A", "bytes")

print("\n2. Kiểm tra ZIP format (.ckpt)...")
import zipfile
try:
    with zipfile.ZipFile(CKPT, 'r') as zf:
        print("  Files inside ckpt:", zf.namelist()[:10])
    print("  ✅ File is valid ZIP/ckpt")
except Exception as e:
    print("  ❌ Not ZIP:", e)

print("\n3. Thử torch.load bình thường...")
import torch
try:
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    print("  ✅ torch.load OK. Top-level keys:", list(ckpt.keys()))
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        print("  state_dict keys (first 5):", list(sd.keys())[:5])
    elif "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        print("  model_state_dict keys (first 5):", list(sd.keys())[:5])
    else:
        print("  ⚠️  Neither 'state_dict' nor 'model_state_dict' found!")
except Exception as e:
    print("  ❌ torch.load FAILED:", e)
    traceback.print_exc()

print("\n4. Thử mock pytorch_lightning rồi torch.load...")
def inject_mock_pl():
    def _make(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m
    dummy = type('Dummy', (), {'__init__': lambda s,*a,**k: None,
                               '__setstate__': lambda s,d: s.__dict__.update(d) if isinstance(d,dict) else None})
    for mod in ['pytorch_lightning','pytorch_lightning.callbacks','pytorch_lightning.loggers',
                'pytorch_lightning.utilities','pytorch_lightning.core','pytorch_lightning.core.lightning',
                'pytorch_lightning.trainer','pytorch_lightning.trainer.trainer',
                'omegaconf','omegaconf.dictconfig','omegaconf.listconfig',
                'hydra','hydra.core','hydra._internal','wandb']:
        m = _make(mod)
        for attr in ['LightningModule','Trainer','ModelCheckpoint','EarlyStopping',
                     'WandbLogger','DictConfig','ListConfig']:
            setattr(m, attr, dummy)

if 'pytorch_lightning' not in sys.modules:
    inject_mock_pl()
    try:
        ckpt2 = torch.load(CKPT, map_location='cpu', weights_only=False)
        print("  ✅ Mock load OK. Keys:", list(ckpt2.keys()))
        if "state_dict" in ckpt2:
            print("  state_dict prefix sample:", list(ckpt2["state_dict"].keys())[:5])
    except Exception as e:
        print("  ❌ Mock load FAILED:", e)
        traceback.print_exc()
else:
    print("  pytorch_lightning is installed, skipping mock test")

print("\n5. Thử import model/resnet50...")
if ACNE_SRC not in sys.path:
    sys.path.insert(0, ACNE_SRC)
try:
    from model.resnet50 import resnet50 as AcneResNet50
    print("  ✅ Import OK")
    m = AcneResNet50(num_acne_cls=13, pretrained_backbone=False)
    print("  ✅ Model instantiated OK")
    print("  Model keys (first 5):", list(m.state_dict().keys())[:5])
except Exception as e:
    print("  ❌ FAILED:", e)
    traceback.print_exc()

print("\n6. Full load test...")
try:
    from model.resnet50 import resnet50 as AcneResNet50
    ckpt3 = torch.load(CKPT, map_location='cpu', weights_only=False)
    if "state_dict" in ckpt3:
        sd_raw = ckpt3["state_dict"]
        sd = {k[4:] if k.startswith("cnn.") else k: v for k, v in sd_raw.items()}
    elif "model_state_dict" in ckpt3:
        sd = ckpt3["model_state_dict"]
    else:
        sd = ckpt3
    m2 = AcneResNet50(num_acne_cls=13, pretrained_backbone=False)
    missing, unexpected = m2.load_state_dict(sd, strict=False)
    print("  ✅ load_state_dict OK")
    print("  Missing:", len(missing), missing[:3] if missing else "")
    print("  Unexpected:", len(unexpected), unexpected[:3] if unexpected else "")
except Exception as e:
    print("  ❌ FAILED:", e)
    traceback.print_exc()

print("\n" + "=" * 60)
print("Done. Paste output trên vào chat để được hỗ trợ.")
