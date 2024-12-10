import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
img_url = os.listdir("dataset")

img = cv2.imread("dataset/"+img_url[10])
img = cv2.resize(img, (500, 500))

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

img_bgr = img
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(img_gray, cmap="gray")
plt.show()

# islem resmi ir_img

ir_img = cv2.medianBlur(img_gray, 5)
ir_img = cv2.medianBlur(ir_img, 5)
plt.imshow(ir_img, cmap="gray")
plt.show()

medyan = np.median(ir_img)

low = 0.67*medyan
high = 1.33*medyan

kenarlık = cv2.Canny(ir_img, low, high)
plt.imshow(kenarlık, cmap="gray")
plt.show()

kenarlık = cv2.dilate(kenarlık, np.ones((3, 3), np.uint8), iterations=1)
plt.imshow(kenarlık, cmap="gray")
plt.show()

cnt = cv2.findContours(kenarlık, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = cnt[0]
cnt = sorted(cnt, key=cv2.contourArea, reverse=True)
H, W, = 500, 500
plaka = None

for c in cnt:
    rect = cv2.minAreaRect(c)
    (x, y), (w, h), r = rect
    if(w>h and w>h*2) or (h>w and h>w*2):
        box = cv2.boxPoints(rect)
        box = np.int64(box)

        minx = np.min(box[:,0])
        miny = np.min(box[:,1])
        maxx = np.max(box[:, 0])
        maxy = np.max(box[:, 1])

        muh_plaka = img_gray[miny:maxy, minx:maxx].copy()
        muh_medyan = np.median(muh_plaka)

        kon1 = muh_medyan > 85 and muh_medyan < 200
        kon2 = h < 50 and w < 150
        kon3 = w < 50 and h < 150

        print(f"muh_plaka medyan:{muh_medyan} genişlik:{w} yükseklik:{h}")
        plt.figure()
        kon = False
        if (kon1 and (kon2 or kon3)):
            #plakadır
            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
            plaka = [int(i) for i in [minx, miny, w, h]]

            plt.title("plaka tespit edildi!!!")
            kon = True
        else:
            #plaka değil
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
            plt.title("plaka tespit edilemedi!!!")

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

        if(kon):
            break
            """

def plaka_konum_don(img):
    img_bgr = img
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.imshow(img_gray, cmap="gray")
    plt.show()

    # islem resmi ir_img

    ir_img = cv2.medianBlur(img_gray, 5)
    ir_img = cv2.medianBlur(ir_img, 5)

    medyan = np.median(ir_img)

    low = 0.67 * medyan
    high = 1.33 * medyan

    kenarlık = cv2.Canny(ir_img, low, high)

    kenarlık = cv2.dilate(kenarlık, np.ones((3, 3), np.uint8), iterations=1)

    cnt = cv2.findContours(kenarlık, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnt[0]
    cnt = sorted(cnt, key=cv2.contourArea, reverse=True)
    H, W, = 500, 500
    plaka = None

    for c in cnt:
        rect = cv2.minAreaRect(c)
        (x, y), (w, h), r = rect
        if (w > h and w > h * 2) or (h > w and h > w * 2):
            box = cv2.boxPoints(rect)
            box = np.int64(box)

            minx = np.min(box[:, 0])
            miny = np.min(box[:, 1])
            maxx = np.max(box[:, 0])
            maxy = np.max(box[:, 1])

            muh_plaka = img_gray[miny:maxy, minx:maxx].copy()
            muh_medyan = np.median(muh_plaka)

            kon1 = muh_medyan > 85 and muh_medyan < 200
            kon2 = h < 50 and w < 150
            kon3 = w < 50 and h < 150

            print(f"muh_plaka medyan:{muh_medyan} genişlik:{w} yükseklik:{h}")
            kon = False
            if (kon1 and (kon2 or kon3)):
                # plakadır
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                plaka = [int(i) for i in [minx, miny, w, h]]

                kon = True
            else:
                # plaka değil
                cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

            if (kon):
                return  plaka
    return []