import os
import pickle
import numpy as np
from flask import Flask, request, render_template

# Flask ko batana padega ki templates folder ek step peeche hai
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# ===== Sahi rasta model load karne ke liye =====
# ... baaki imports upar hi rahenge ...

# Yeh line current folder (/api/) ke hisaab se path banayegi
current_dir = os.path.dirname(__file__)   # yeh api/ folder dega
model_path = os.path.join(current_dir, "model.pkl")   # → /api/model.pkl

print("Current working dir:", os.getcwd())
print("Model path trying:", model_path)
print("File exists?", os.path.exists(model_path))

model = None
error_msg = None

try:
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully!")
    else:
        error_msg = f"Model file not found at: {model_path}"
        print(f"❌ {error_msg}")
except Exception as e:
    error_msg = f"Error loading model: {str(e)}"
    print(f"❌ {error_msg}")

# ... baaki code ...

# Predict function mein ye line change karo:
if model is None:
    return render_template('index.html', prediction_text=f"Model Error: {error_msg}")
    

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
