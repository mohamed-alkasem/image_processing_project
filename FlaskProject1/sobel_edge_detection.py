#1-adsoyad:amir elahmed
#1-öğrno:2112721307
#2-adsoyad:mohamad alkassem
#2-öğrno:2212721320


import cv2
import numpy as np


def sobel_edge_detection(image):

    Gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)


    G = np.sqrt((Gx ** 2) + (Gy ** 2))


    G = cv2.convertScaleAbs(G)

    return G
