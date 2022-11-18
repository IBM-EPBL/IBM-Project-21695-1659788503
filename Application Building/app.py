from flask import Flask, request, render_template
from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow as tf

app = Flask(__name__, template_folder=".")

model = load_model("Application Building\mnistCNN.h5")

@app.route('/')
def first():
    return render_template('first.html')

@app.route('/upload')
def second():
    return render_template('second.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        img = Image.open(request.files['file'].stream).convert("L")
        img = img.resize((28, 28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1, 28, 28, 1)
        y_pred = model.predict_classes(im2arr)
        print(y_pred)
        return "Predicted Number: "+str(y_pred)

if __name__ == '__main__':
    app.run(debug=True, threaded=False)
