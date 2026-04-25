from flask import Flask, render_template, request, jsonify
from PIL import Image
import io

from models_loader import load_all_models, run_pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Không có ảnh được gửi lên."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Tên file trống."}), 400

    try:
        img_pil = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Không thể đọc ảnh. Vui lòng thử file khác."}), 400

    result = run_pipeline(img_pil)
    return jsonify(result)


if __name__ == "__main__":
    print("=" * 55)
    print("🔄 Đang load models...")
    load_all_models()
    print("=" * 55)
    print("🌐 Mở trình duyệt tại: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=False, host="0.0.0.0", port=5000)
