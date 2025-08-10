#1-adsoyad:amir elahmed
#1-öğrno:2112721307
#2-adsoyad:mohamad alkassem
#2-öğrno:2212721320


import cv2
import numpy as np

def roberts_edge_detection(image):

    kernelx = np.array([[1, 0], [0, -1]], dtype=int)
    kernely = np.array([[0, 1], [-1, 0]], dtype=int)


    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x = cv2.filter2D(img_gray, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img_gray, cv2.CV_16S, kernely)


    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    return Roberts
