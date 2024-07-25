import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, request, render_template
import os

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
model = load_model("models/1")
class_names = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    predicted_class = None
    confidence = None

    if request.method == 'POST':
        f = request.files['fruit']
        filename = f.filename
        target = os.path.join(APP_ROOT, 'images/')
        print(target)
        des = os.path.join(target, filename)
        f.save(des)

        test_image = image.load_img(os.path.join(target, filename), target_size=(300, 300))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        prediction = model.predict(test_image)
        print(prediction)

        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])
        print(confidence)

    return render_template('prediction.html', confidence=confidence, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
