import os
import pickle
import numpy as np
from flask import Flask, request, render_template

# Flask ko batana padega ki templates folder ek step peeche hai
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# ===== Sahi rasta model load karne ke liye =====
# ... baaki imports upar hi rahenge ...

# Yeh line current folder (/api/) ke hisaab se path banayegi
import os
import pickle

model = None
error_msg = "Model not loaded."

# Vercel-safe absolute path (best practice)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /var/task/api
model_path = os.path.join(BASE_DIR, 'model.pkl')

# Extra debug for logs
print(f"[VERCEL DEBUG] BASE_DIR: {BASE_DIR}")
print(f"[VERCEL DEBUG] model_path: {model_path}")
print(f"[VERCEL DEBUG] Current working dir (os.getcwd): {os.getcwd()}")
print(f"[VERCEL DEBUG] Does model.pkl exist? {os.path.exists(model_path)}")
print(f"[VERCEL DEBUG] Is it a file? {os.path.isfile(model_path) if os.path.exists(model_path) else 'No'}")
print(f"[VERCEL DEBUG] List files in BASE_DIR: {os.listdir(BASE_DIR)}")  # Yeh bata dega api/ mein kya files hain!

try:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("[SUCCESS] Model loaded successfully from:", model_path)
        error_msg = ""  # UI pe message clear
    else:
        error_msg = f"Model file NOT found at {model_path}. Check if pushed to Git."
        print("[ERROR]", error_msg)
except Exception as e:
    error_msg = f"Pickle load failed: {str(e)} (corrupt file or version mismatch?)"
    print("[CRITICAL ERROR]", error_msg)

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
