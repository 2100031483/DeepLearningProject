from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
def predict_multiclass():
    # Load the saved model
    loaded_model = load_model('model_binary.keras')  # Update with the actual path to your saved model

    # Define the path to the image you want to make predictions on
    image_path = '"C:\Users\HP\PycharmProjects\Projectb\train\Images_Square\image_19.jpg"'  # Update with the path to your image
    img_height, img_width = 224, 224

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Assuming your model was trained with rescaling in the range [0, 1]

    # Make predictions
    predictions = loaded_model.predict(img_array)

    print( predictions )
    # Assuming multi-class classification, you can interpret the result
    class_labels = ['Mild','Moderate','No_DR','Proliferate_DR','Severe']  # Update with your actual class labels
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[predicted_class]

    print(f'Predicted class: {predicted_class} ({predicted_label})')
    print(f'Confidence: {predictions[0][predicted_class]:.2%}')
    return predicted_label, predictions[0][predicted_class]

if __name__ == "__main__":
    pclass,percent=predict_multiclass()
    print(pclass,percent)