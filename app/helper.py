import os
import numpy as np
import pickle
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2 , preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input as densenet_preprocess

current_path = os.getcwd()
mri_images = os.path.join(current_path, 'static/classes.pickle')
predictor_model = load_model(r'static/best_model2.h5')

print("Weights loaded")

with open(mri_images, 'rb') as handle:
    images = pickle.load(handle)

def predictor(img_path):
    img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img, (128, 128),interpolation = cv2.INTER_AREA)
    img=np.array(img)
    img = img.astype('float32')
    img /= 255 
    img = img.reshape(1, 128,128,1)
    print(img.shape)
    prediction = predictor_model.predict(img)
    print(prediction)
    prediction = pd.DataFrame(np.round(prediction,1),columns = images).transpose()
    print(prediction)
    prediction.columns = ['values']
    prediction  = prediction.nlargest(5, 'values')
    prediction = prediction.reset_index()
    prediction.columns = ['name', 'values']
    return(prediction)
