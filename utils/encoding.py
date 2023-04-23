from utils.facenet_arch import * 
import numpy as np 
from tensorflow.keras.models import load_model


face_encoder = InceptionResNetV2()
path = "res/facenet_keras_weights.h5"
face_encoder.load_weights(path)


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def get_encode(face):
    face = normalize(face)
    face_d = np.expand_dims(face, axis=0)
    encode = face_encoder.predict(face_d,verbose=0)[0]
    return encode
