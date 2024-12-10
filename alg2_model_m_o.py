import cv2
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

path = "characterset/"

classes = os.listdir(path)
tek_batch = 0

urls = []
sinifs = []

for sinif in classes:
    images = os.listdir(path+sinif)
    for image in images:
        urls.append(path+sinif+"/"+image)
        sinifs.append(sinif)
        tek_batch+=1

df = pd.DataFrame({"adres":urls, "sinif":sinifs})

#200x200
def islem(img):
    yeni_boy = img.reshape((1600, 5, 5))
    orts = []
    for parca in yeni_boy:
        ort = np.mean(parca)
        orts.append(ort)
    orts = np.array(orts)
    orts = orts.reshape(1600,)
    return orts

def on_isle(img):
    return img/255
target_size = (200, 200)
batch_size = tek_batch #500, 500, 1

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=on_isle)

train_set = train_gen.flow_from_dataframe(df,x_col = "adres",y_col="sinif",
                                              target_size=target_size,
                                              color_mode = "grayscale",
                                              shuffle=True,
                                              class_mode='sparse',
                                              batch_size=batch_size)

images, train_y = next(train_set)
train_x = np.array(list(map(islem,images))).astype("float32")
train_y = train_y.astype(int)

print("random forest eğitiliyor")
rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')

rfc.fit(train_x, train_y)
print("Eğitildi")

pred = rfc.predict(train_x)

acc = accuracy_score(pred, train_y)

print("başarılı:", acc)

dosya = "rfc_model.rfc"

pickle.dump(rfc, open(dosya, "wb"))