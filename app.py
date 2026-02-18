import numpy as np
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Model path sahi se set karein
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

# Model load karne ka surakshit tarika
model = None
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
    if model is None:
        return render_template("index.html", prediction_text="Error: Model file not loaded properly.")
    
    try:
        # Form se data nikalna
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        
        # Prediction karna
        prediction = model.predict(features)
        
        return render_template("index.html", prediction_text=f"The Predicted Crop is: {prediction[0]}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error during prediction: {str(e)}")
