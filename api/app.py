import os
import pickle
import numpy as np
from flask import Flask, request, render_template

# Flask initialization - Templates folder bahar hai isliye '../templates' use kiya hai
app = Flask(__name__, template_folder='../templates')

# ===== Load Model Safely =====
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "model.pkl")

model = None
try:
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully from api folder")
    else:
        print(f"❌ Model file NOT found at: {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Error: Model not loaded on server.")

    try:
        # Form se values lena aur numeric mein convert karna
        feature_list = [float(x) for x in request.form.values()]
        features = np.array(feature_list).reshape(1, -1)
        
        # Prediction karna
        prediction = model.predict(features)
        result = prediction[0]
        
        return render_template('index.html', prediction_text=f"Recommended Crop: {result}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error during prediction: {e}")

# Vercel handles the app, but for local testing:
if __name__ == "__main__":
    app.run(debug=True)
