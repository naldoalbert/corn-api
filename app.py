import os
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from flask_cors import CORS
import psycopg2  # Gantikan mysql.connector
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Koneksi ke database PostgreSQL (Railway menyediakan ini sebagai environment variable)
PGHOST = os.environ.get("PGHOST")
PGUSER = os.environ.get("PGUSER")
PGPASSWORD = os.environ.get("PGPASSWORD")
PGDATABASE = os.environ.get("PGDATABASE")
PGPORT = os.environ.get("PGPORT", "5432")  # Default PostgreSQL port

db = None
cursor = None

try:
    db = psycopg2.connect(
        host=PGHOST,
        user=PGUSER,
        password=PGPASSWORD,
        database=PGDATABASE,
        port=PGPORT
    )
    cursor = db.cursor()
    print("Koneksi database PostgreSQL berhasil!")
except psycopg2.Error as err:
    print(f"Error koneksi database PostgreSQL: {err}")

# Load model klasifikasi jagung
model_path = os.environ.get("MODEL_PATH", "model/model_klasifikasi_jagung_DenseNet.h5")
model = None
try:
    model = tf.keras.models.load_model(model_path)
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error memuat model dari {model_path}: {e}")

# Label klasifikasi
labels = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

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
    if db is None:
        return jsonify({'error': 'Database connection not available'}), 500

    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        image_bytes = file.read()
        img = preprocess_image(image_bytes)

        predictions = model.predict(img)
        class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        predicted_label = labels[class_index]

        # Simpan hasil prediksi ke PostgreSQL
        insert_query = "INSERT INTO predictions (label, confidence, timestamp) VALUES (%s, %s, %s);"
        cursor.execute(insert_query, (predicted_label, confidence, datetime.now()))
        db.commit()

        return jsonify({
            'prediction': predicted_label,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    if db is None:
        return jsonify({'error': 'Database connection not available'}), 500
    try:
        cursor.execute("SELECT id, label, confidence, timestamp FROM predictions ORDER BY timestamp DESC;")
        rows = cursor.fetchall()
        result = []
        for row in rows:
            result.append({
                "id": row[0],
                "label": row[1],
                "confidence": float(row[2]),
                "timestamp": row[3].isoformat()
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
