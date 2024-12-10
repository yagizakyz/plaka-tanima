import os
import matplotlib.pyplot as plt
import cv2
from alg1_plaka_tespit import plaka_konum_don

data = os.listdir("dataset")

#1. Alg inceleme
"""for image_url in data:
    img = cv2.imread("dataset/"+image_url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(500, 500))
    plt.imshow(img)
    plt.show()
    """
#2. Alg inceleme
for image_url in data:
    img = cv2.imread("dataset/"+image_url)

    img = cv2.resize(img, (500, 500))
    plaka = plaka_konum_don(img)
    x, y, w, h = plaka
    if(w>h):
        plaka_bgr = img[y:y+h, x:x+w].copy()
    else:
        plaka_bgr = img[y:y + w, x:x + h].copy()
    img = cv2.cvtColor(plaka_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()