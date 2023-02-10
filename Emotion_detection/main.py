import pandas as pd
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Input, BatchNormalization, MaxPooling2D, Activation, Flatten, Dense, Dropout
from keras.models import Model
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from keras.utils import load_img, img_to_array

import scipy
import os
import cv2

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Load model
model = tf.keras.models.load_model(r'C:\Users\Prince\OneDrive\Desktop\emotion\emotion_recognition_best.h5')
print('Model loaded Sucessfully')

label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
              3: 'Happiness', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

cap = cv2.VideoCapture(r"C:\Users\Prince\Downloads\pexels-kampus-production-6104249.mp4")

while True:
    _, cap_image = cap.read()

    cap_img_gray = cv2.cvtColor(cap_image, cv2.COLOR_BGR2GRAY)

    faces = face_haar_cascade.detectMultiScale(cap_img_gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(cap_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = cap_img_gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        predictions = model.predict(img_pixels)
        emotion_label = np.argmax(predictions)


        emotion_prediction = label_dict[emotion_label]
        print(predictions)

        cv2.putText(cap_image, str(emotion_prediction), (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        resize_image = cv2.resize(cap_image, (1000, 700))
        cv2.imshow('Emotion', resize_image)

        if cv2.waitKey(10) == ord('b'):
            break

cap.release()
cv2.destroyAllWindows
