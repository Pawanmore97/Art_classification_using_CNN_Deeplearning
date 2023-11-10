from flask import Flask,render_template,request
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from werkzeug.utils import secure_filename
from keras.preprocessing import image
import numpy as np
import cv2

app = Flask(__name__)

model_path = "Model.h5"

model = load_model(model_path)
model.make_predict_function()
print("Model loaded start serving")


def make_predict(img_path):

    img = cv2.imread(img_path)
    img = cv2.resize(img,(256,256))
    img = img.reshape(1,256,256,3)

    pred = model.predict(img)

    if pred == 0:
        return("AI generated image")
    else:
        return("Non AI generated image")



@app.route("/")
def index():
    return render_template("Images.html")

@app.route("/predict", methods = ["GET","POST"])
def predict():
    if request.method == 'POST':

        f = request.files["file"]
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)

        preds = make_predict(file_path)

    return render_template("Images.html",preds=preds)

if __name__ == "__main__":
    app.run(debug=True,host="127.0.0.1",port=8000)