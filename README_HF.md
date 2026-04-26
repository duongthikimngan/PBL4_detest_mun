---
title: SkinScan AI
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# SkinScan AI — Phân tích mụn trứng cá bằng AI

Ứng dụng phân tích ảnh da liễu sử dụng 3 model AI:
- **DINOv2** — Phân loại bệnh da (31 loại)
- **YOLOv8** — Phát hiện từng nốt mụn
- **ResNet50** — Đánh giá mức độ nghiêm trọng (4 cấp độ)

## Cách deploy

1. Upload model files lên HF Hub (xem hướng dẫn bên dưới)
2. Set biến môi trường `HF_MODEL_REPO = your-username/your-model-repo`
