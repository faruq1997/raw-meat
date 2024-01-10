import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
from skimage.feature import greycomatrix, greycoprops
import joblib
import matplotlib.pyplot as plt

# Load deep learning model
deep_learning_model = load_model('./mobilenet_86persen.h5')

# Load machine learning model
loaded_model = joblib.load('./best_adaboost.pkl')

labels = ['babi', 'kambing', 'sapi']
img_size_224p = (224, 224)

# Preprocessing function for deep learning model
def preprocess_for_deep_learning(img, input_size):
    nimg = img.convert('RGB').resize(input_size, resample=0)
    img_arr = (np.array(nimg)) / 255
    return img_arr

def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)

# Function to extract features for machine learning model
def extract_features(image):
    image = np.array(image * 255, dtype=np.uint8)

    glcm = greycomatrix(image[..., 0], distances=[1], angles=[0], symmetric=True, normed=True)
    glcm_props = greycoprops(glcm, 'dissimilarity')[0]

    r, g, b = cv2.split(image)
    rgb_features = [
        np.mean(r), np.std(r),
        np.mean(g), np.std(g),
        np.mean(b), np.std(b)
    ]

    return np.hstack((glcm_props, rgb_features))

# Streamlit app
st.title('Red Meat Classification')
st.write('Upload an image of red meat (pig, goat, cow) for classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Load and preprocess the image for deep learning model
    img_for_deep_learning = Image.open(uploaded_file)
    X = preprocess_for_deep_learning(img_for_deep_learning, img_size_224p)
    X = reshape([X])
    y = deep_learning_model.predict(X)

    plt.imshow(img_for_deep_learning)
    plt.show()
    st.write(labels[np.argmax(y)], np.max(y))

    # Load and preprocess the image for machine learning model
    test_image = cv2.cvtColor(np.array(img_for_deep_learning), cv2.COLOR_RGB2BGR)
    test_image = cv2.resize(test_image, (224, 224))
    test_image = (np.array(test_image) / 255.0)
    test_features = extract_features(test_image)

    # Prediction using both models
    predicted_class_deep_learning = np.argmax(y)
    predicted_class_machine_learning = loaded_model.predict(test_features.reshape(1, -1))

    # Get label indices for each model's prediction
    index_deep_learning = predicted_class_deep_learning
    index_machine_learning = predicted_class_machine_learning[0]

    # Combine predictions
    if index_deep_learning == index_machine_learning:
        final_prediction = labels[index_deep_learning]
    else:
        final_prediction = "Prediksi salah dari setidaknya satu model."

    # Calculate average accuracy percentage
    accuracy_deep_learning = np.max(y) * 100
    accuracy_machine_learning = 0
    if index_machine_learning == index_deep_learning:
        accuracy_machine_learning = 100

    average_accuracy = (accuracy_deep_learning + accuracy_machine_learning) / 2

    # Final prediction and average accuracy
    st.write(f'Hasil prediksi: {final_prediction}')
    st.write(f'Akurasi: {average_accuracy}%')
