from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']
    model=request.form['options']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file:
        # Load the saved model
        loaded_model = load_model(model)  # Update with the actual path to your saved model

        # Save the uploaded file to a temporary location
        upload_folder = 'temp_uploads'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Load and preprocess the uploaded image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Assuming your model was trained with rescaling in the range [0, 1]

        # Make predictions
        predictions = loaded_model.predict(img_array)

        # Assuming multi-class classification, you can interpret the result
        class_labels = ['0', '1', '2', '3', '4',  '5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']  # Update with your actual class labels  # Update with your actual class labels
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]
        confidence = predictions[0][predicted_class]

        # Clean up: remove the temporary file
        os.remove(file_path)

        return render_template('result.html', predicted_class=predicted_class, predicted_label=predicted_label,
                               confidence=confidence, model_name=model )


if __name__ == '__main__':
    app.run(debug=True)