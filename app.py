import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# โหลดโมเดล YOLOv8
model = YOLO("best.pt")  # เปลี่ยนเป็น path ที่ถูกต้องหากจำเป็น

@app.route('/')
def home():
    return jsonify({"message": "Cloud Detection API is running"}), 200

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['file']
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # ปรับขนาดภาพถ้าจำเป็น
    max_size = 800
    h, w, _ = image.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    # ตรวจจับ
    results = model(image)
    detections = []
    detected_classes = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cloud_name = model.names[cls]

            detections.append({
                "class": cloud_name,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]  # พิกัดซ้ายบน-ขวาล่าง
            })

            if cloud_name not in detected_classes:
                detected_classes.append(cloud_name)

    return jsonify({
        "detected_classes": detected_classes,
        "detections": detections
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
