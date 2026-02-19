import os
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# ===== Load Model Safely =====
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "model.pkl")

model = None

try:
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
    else:
        print("❌ Model file not found at:", model_path)
except Exception as e:
    print("❌ Model Load Error:", e)


# ===== Home Route =====
@app.route("/")
def home():
    return render_template("index.html")


# ===== Predict Route =====
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return render_template("index.html", prediction_text="❌ Model not loaded")

        # Get form values
        nitrogen = float(request.form["Nitrogen"])
        phosphorus = float(request.form["phosphorus"])
        potassium = float(request.form["potassium"])
        temperature = float(request.form["Temperature"])
        humidity = float(request.form["Humidity"])
        ph = float(request.form["PH"])
        rainfall = float(request.form["Rainfall"])

        # Prepare input array
        features = np.array([[nitrogen, phosphorus, potassium,
                              temperature, humidity, ph, rainfall]])

        # Prediction
        prediction = model.predict(features)
        output = prediction[0]

        return render_template("index.html", prediction_text=f"🌱 Recommended Crop: {output}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"❌ Error: {str(e)}")


# ===== Run
