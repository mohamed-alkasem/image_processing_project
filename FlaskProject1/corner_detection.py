#1-adsoyad:amir elahmed
#1-öğrno:2112721307
#2-adsoyad:mohamad alkassem
#2-öğrno:2212721320


import cv2
import numpy as np

def shi_tomasi_detection(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        corners = np.int32(corners)  # تغيير np.int0 إلى np.int32


        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
    return image
