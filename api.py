import os
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# === Init Flask ===
app = Flask(__name__)
# == allow anyone to use my api
CORS(app, supports_credentials=True) 

# === Configuration ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
THRESHOLD = 0.6

# === Init Models ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# === Detect and Encode ===
def detect_and_encode(image):
    boxes, _ = mtcnn.detect(image)
    faces = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = image[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face = cv2.resize(face, (160, 160))
            face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
            face_tensor = torch.tensor(face).unsqueeze(0).to(device)
            encoding = resnet(face_tensor).detach().cpu().numpy().flatten()
            faces.append((encoding, box))
    return faces

# === Load Known Faces ===
def load_known_faces():
    known_encodings = []
    known_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            name = os.path.splitext(filename)[0]
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image_bgr = cv2.imread(path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            encodings = detect_and_encode(image_rgb)
            if encodings:
                known_encodings.append(encodings[0][0])
                known_names.append(name)
    return known_encodings, known_names

known_encodings, known_names = load_known_faces()

# === API Route ===
@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)

    faces = detect_and_encode(img_np)

    results = []
    for encoding, _ in faces:
        distances = np.linalg.norm(np.array(known_encodings) - encoding, axis=1)
        min_dist_idx = np.argmin(distances)
        name = known_names[min_dist_idx] if distances[min_dist_idx] < THRESHOLD else "Unknown"
        results.append(name)

    return jsonify({'recognized': results})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
