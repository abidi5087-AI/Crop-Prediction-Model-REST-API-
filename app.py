import numpy as np
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Model file ka path set karna (Vercel ke liye zaroori)
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

# Model load karna - is line par dhyan dein
try:
    model = pickle.load(open(model_path, 'rb'))
except Exception as e:
    print(f"Model Load Error: {e}")

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Form se data lena
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        
        # Prediction karna
        prediction = model.predict(features)
        
        # Result dikhana
        return render_template("index.html", prediction_text="The Predicted Crop is {}".format(prediction[0]))
    
    except Exception as e:
        # Agar koi galti ho toh screen par error dikhega
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
