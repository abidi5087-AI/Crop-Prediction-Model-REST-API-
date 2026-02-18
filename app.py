
   
    #import numpy as np
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Model path globally set kar dete hain
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Jugaad: Model ko function ke andar hi load karo
        if not os.path.exists(model_path):
            return render_template("index.html", prediction_text="Error: model.pkl file nahi mili!")
            
        with open(model_path, 'rb') as f:
            local_model = pickle.load(f)
        
        # Form se data nikalna
        input_data = [float(x) for x in request.form.values()]
        features = [np.array(input_data)]
        
        # Prediction karna
        prediction = local_model.predict(features)
        
        return render_template("index.html", prediction_text="The Predicted Crop is: {}".format(prediction[0]))
    
    except Exception as e:
        # Jo bhi galti hogi ab screen par dikhegi
        return render_template("index.html", prediction_text="Error: {}".format(str(e)))

if __name__ == "__main__":
    app.run(debug=True)
