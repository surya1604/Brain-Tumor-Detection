import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
h5_model_file = "final.h5"

# Load the model

model = tf.keras.models.load_model(h5_model_file)
    

@app.route('/treatment.html', methods=['GET'])
def index():
    return render_template('treatment.html')

@app.route('/home.html')
def home():
    return render_template('home.html')

@app.route('/work.html')
def work():
    return render_template('work.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'Uploads', secure_filename(f.filename))
        try:
            f.save(file_path)
            value = getResult(file_path)
            result = get_className(value)
            return result
        except Exception as e:
            print(f"Error processing file: {e}")
            return "Error processing file."
    return None

def getResult(img):
    try:
        image = cv2.imread(img)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((120, 120))
        image = np.array(image)
        input_img = np.expand_dims(image, axis=0)
        result = model.predict(input_img)
        max_index = np.argmax(result)
        return max_index
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def get_className(classNo):
    if classNo == 0:
        return "Glioma"
    elif classNo == 1:
        return "Meningioma"
    elif classNo == 2:
        return "notumor"
    elif classNo == 3:
        return "pituitary"
    else:
        return "Unknown"

if __name__ == '__main__':
    app.run(debug=True)
