import os
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Model path setup
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'model.pkl')

# Model loading logic
model = None
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Model file not found!")
