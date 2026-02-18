import numpy as np
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Model ko globally load karna best hai
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(model_path, 'rb'))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Form se data lena
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        
        # Prediction
        prediction = model.predict(features)
        
        return render_template("index.html", prediction_text="The Predicted Crop is: {}".format(prediction[0]))
    except Exception as e:
        # Agar abhi bhi version ka panga hua toh yahan dikhega
        return render_template("index.html", prediction_text="Error: {}".format(str(e)))

if __name__ == "__main__":
    app.run(debug=True)
