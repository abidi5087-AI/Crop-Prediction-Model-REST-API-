import numpy as np
from flask import Flask, request, render_template
import pickle
import os

# Vercel ko 'app' naam hi chahiye hota hai
app = Flask(__name__)

# Model file ka path sahi se set karne ke liye
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(model_path, 'rb'))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="The Predicted Crop is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
