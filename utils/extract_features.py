# extract features with pretrained network and save to csv
import dlib
import os
import pandas as pd
import numpy as np
from utils.facenet_arch import InceptionResNetV2

face_encoder = InceptionResNetV2()
path = "res/facenet_keras_weights.h5"
face_encoder.load_weights(path)

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

people_path="train_people/train"

# Create dataframe to hold extracted features
emb_df = pd.DataFrame(columns = range(0,129))
for folder_names in os.listdir(people_path):
    image_folder = os.path.join(people_path,folder_names)
    img_list = (os.listdir(os.path.join(people_path,folder_names)))
    for img in img_list:
        image_path = os.path.join(image_folder,img)
        faces = dlib.load_rgb_image(image_path)
        face = normalize(faces)
        face_d = np.expand_dims(face, axis=0)
        encode = face_encoder.predict(face_d,verbose=0)[0]
        to_append = list(encode)
        to_append.extend([folder_names])
        emb_series = pd.Series(to_append, index = emb_df.columns)
        # emb_df= pd.concat([emb_df, emb_series],ignore_index=True, join="inner")
        emb_df = emb_df.append(emb_series, ignore_index=True)
    
# save the dataframe so you can add more examples and retrain
emb_df.rename(columns = {128:'target'}, inplace = True)
saved_df = emb_df.to_csv('res/face_db.csv',index=False)
print('Saved Embeddings...........')