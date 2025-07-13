from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sqlite3
from datetime import datetime
import json

# -----------------------------
# App Setup
# -----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -----------------------------
# Load Trained Model
# -----------------------------
model = load_model('model.h5')

# -----------------------------
# Load Disease Info JSON
# -----------------------------
with open('data/disease_info.json', 'r', encoding='utf-8') as f:
    disease_info = json.load(f)

# -----------------------------
# Labels From Dataset
# -----------------------------
train_classes = list(disease_info.keys())

# -----------------------------
# Initialize DB
# -----------------------------
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        predicted_label TEXT,
        confidence REAL,
        date TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# Home Route
# -----------------------------
@app.route('/')
def home():
    lang = request.args.get('lang', 'ur')
    return render_template('index.html', lang=lang)

# -----------------------------
# Predict Route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    lang = request.args.get('lang', 'ur')
    file = request.files['image']

    if not file:
        return "No file uploaded", 400

    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_label = train_classes[class_index]
    clean_name = class_label.replace('___', ' - ').replace('__', ' ').replace('_', ' ')
    confidence = round(np.max(prediction) * 100, 2)
    info = disease_info.get(class_label, {})

    # Save to DB
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute("INSERT INTO predictions (filename, predicted_label, confidence, date) VALUES (?, ?, ?, ?)",
              (filename, clean_name, confidence, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

    return render_template('index.html',
                           prediction=True,
                           filename=filename,
                           clean_name=clean_name,
                           confidence=confidence,
                           info=info,
                           lang=lang)

# -----------------------------
# History Route
# -----------------------------
@app.route('/history')
def history():
    lang = request.args.get('lang', 'ur')
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return render_template('history.html', history=rows, lang=lang)

# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
