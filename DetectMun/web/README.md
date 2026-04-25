# AcneDetect AI — Web App

Web phát hiện mụn trứng cá chạy local, sử dụng pipeline 3 model:
**DINOv2 ViT-B/14** → **YOLOv8** + **ResNet50 5-fold ensemble**

## Cấu trúc thư mục

```
web/
├── app.py               # Flask server chính
├── config.py            # Đường dẫn model & hằng số
├── models_loader.py     # Load model & pipeline suy luận
├── requirements.txt
├── templates/
│   └── index.html       # Giao diện web
└── models/              # ← Đặt file model vào đây
    ├── dinov2_vitb14_best.pth
    ├── best.pt                    (YOLO weights)
    ├── acne-lds-main/             (source code ResNet50)
    └── checkpoints/
        ├── fold_0/best.ckpt
        ├── fold_1/best.ckpt
        ├── fold_2/best.ckpt
        ├── fold_3/best.ckpt
        └── fold_4/best.ckpt
```

## Cài đặt & chạy

```bash
cd web

# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
venv\Scripts\activate        # Windows

# Cài thư viện
pip install -r requirements.txt

# Sửa đường dẫn model trong config.py nếu cần, rồi chạy:
python app.py
```

Mở trình duyệt tại **http://localhost:5000**

## Lưu ý

- Sửa đường dẫn trong `config.py` nếu bạn đặt model ở nơi khác.
- Nếu không có model ResNet50, web vẫn chạy — kết quả grade sẽ dùng ước lượng từ YOLO.
- Cần GPU (CUDA) để chạy nhanh hơn; CPU vẫn hoạt động nhưng chậm hơn.
