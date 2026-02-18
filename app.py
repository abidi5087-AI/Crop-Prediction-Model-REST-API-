import numpy as np
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Model path handling
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'model.pkl')

# Model load karne ka sabse safe tarika
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="The Predicted Crop is {}".format(prediction))

# Vercel ko iski zaroorat nahi hoti par debug ke liye thik hai
if __name__ == "__main__":
    app.run(debug=True)
