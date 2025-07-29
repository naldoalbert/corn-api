from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load model klasifikasi jagung
model = tf.keras.models.load_model("model/model_klasifikasi_jagung_DenseNet.h5")

# Label klasifikasi penyakit daun jagung
labels = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# Variabel untuk menyimpan riwayat prediksi di memori (tidak permanen)
history_data = []

# Fungsi untuk preprocessing gambar
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((256, 256)) 
    img_array = np.array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

@app.route('/')
def index():
    return "API Klasifikasi Penyakit Daun Jagung Aktif"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        image_bytes = file.read()
        img = preprocess_image(image_bytes)

        predictions = model.predict(img)
        class_index = np.argmax(predictions)
        confidence = float(np.max(predictions))
        predicted_label = labels[class_index]

        # Simpan ke riwayat sementara (hanya di RAM, tidak ke DB)
        history_data.append({
            "id": len(history_data) + 1,
            "label": predicted_label,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })

        return jsonify({
            'prediction': predicted_label,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    return jsonify(history_data[::-1])  # Ditampilkan dari terbaru ke terlama

if __name__ == '__main__':
    app.run(debug=True)
