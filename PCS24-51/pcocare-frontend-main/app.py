from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        file_path = 'static/uploads/' + file.filename
        file.save(file_path)

        # Call your prediction function
        prediction = predict_image(file_path)

        return render_template('result.html', prediction=prediction, image_path=file_path)

def predict_image(path):
    # Load your model
    model = tf.keras.models.load_model('model.h5')

    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    i = tf.keras.utils.img_to_array(img) / 255
    input_arr = np.array([i])
    input_arr.shape

    pred = model.predict(input_arr)
    if pred == 1:
        return "Not Affected"
    else:
        return "Affected"

if __name__ == '__main__':
    app.run(debug=True)
