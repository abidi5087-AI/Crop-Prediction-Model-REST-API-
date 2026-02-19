import os
import pickle
import numpy as np
from flask import Flask, request, render_template

# Flask ko batana padega ki templates folder ek step peeche hai
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# ===== Sahi rasta model load karne ke liye =====
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, api_folder="model.pkl")

model = None
try:
    # Aapke purane code mein yahan 'model_path' ki jagah sirf naam tha, ab fix hai
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully from api folder")
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Model not loaded. Check server logs.")

    try:
        # Form se values lekar float mein badalna
        feature_list = [float(x) for x in request.form.values()]
        features = np.array(feature_list).reshape(1, -1)
        
        # Prediction logic
        prediction = model.predict(features)
        result = prediction[0]
        
        # Result ko wapas HTML par bhejna
        return render_template('index.html', prediction_text=f"Recommended Crop: {result}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
