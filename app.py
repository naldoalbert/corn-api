import os
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from flask_cors import CORS
import mysql.connector
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Koneksi ke database - Gunakan variabel lingkungan
DB_HOST = os.environ.get("DB_HOST")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_NAME = os.environ.get("DB_NAME")

db = None # Inisialisasi db ke None
cursor = None # Inisialisasi cursor ke None

try:
    db = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    cursor = db.cursor()
    print("Koneksi database berhasil!")
except mysql.connector.Error as err:
    print(f"Error koneksi database: {err}")
    # Di produksi, pertimbangkan untuk menangani error ini lebih serius,
    # mungkin keluar dari aplikasi jika koneksi DB sangat krusial.

# Load model klasifikasi jagung - Gunakan variabel lingkungan atau path default
model_path = os.environ.get("MODEL_PATH", "model/model_klasifikasi_jagung_DenseNet.h5")
model = None # Inisialisasi model ke None
try:
    model = tf.keras.models.load_model(model_path)
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error memuat model dari {model_path}: {e}")
    # Tangani error jika model tidak dapat dimuat

# Label klasifikasi penyakit daun jagung
labels = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

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
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    if db is None or not db.is_connected():
        return jsonify({'error': 'Database connection not available'}), 500

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

        # Simpan hasil prediksi ke database
        insert_query = "INSERT INTO predictions (label, confidence) VALUES (%s, %s)"
        cursor.execute(insert_query, (predicted_label, confidence))
        db.commit()

        return jsonify({
            'prediction': predicted_label,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    if db is None or not db.is_connected():
        return jsonify({'error': 'Database connection not available'}), 500
    try:
        cursor.execute("SELECT id, label, confidence, timestamp FROM predictions ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        result = []
        for row in rows:
            result.append({
                "id": row[0],
                "label": row[1],
                "confidence": row[2],
                "timestamp": row[3].isoformat()
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Gunakan PORT dari variabel lingkungan, default ke 5000 untuk pengembangan lokal
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))