import numpy as np
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Model file ka rasta (path) dhoondne ke liye
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'model.pkl')

# Model load karne ka surakshit tareeka
model = pickle.load(open(model_path, 'rb'))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Input data lena
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    
    # Prediction karna
    prediction = model.predict(features)
    
    return render_template("index.html", prediction_text="The Predicted Crop is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
