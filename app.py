import numpy as np
from flask import Flask, request, render_template
import pickle
import os
app = Flask(__name__)
# Model file dhoondne ka rasta
model_path = 'model.pkl'

# Model load karna
model = None
try:
    if os.path.exists(model_path):
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
        return render_template("index.html", prediction_text="Error: Model file not found or failed to load.")
    
    try:
        input_data = [float(x) for x in request.form.values()]
        features = [np.array(input_data)]
        prediction = model.predict(features)
        return render_template("index.html", prediction_text=f"The Predicted Crop is: {prediction[0]}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")
