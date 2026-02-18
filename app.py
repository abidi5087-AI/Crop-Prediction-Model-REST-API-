
   import numpy as np
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# 1. Model variable ko pehle khali (None) declare karein
model = None

# 2. Model file ka path set karein
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

# 3. Model load karne ki koshish karein
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Model Load Error: {e}")

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Check karein ki model load hua hai ya nahi
    if model is None:
        return render_template("index.html", prediction_text="Error: Model file not loaded. Check logs.")
    
    try:
        # Aapka purana prediction logic yahan rahega...
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        prediction = model.predict(features)
        return render_template("index.html", prediction_text="The Predicted Crop is {}".format(prediction[0]))
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")
