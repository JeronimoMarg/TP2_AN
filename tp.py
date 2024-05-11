import cv2
import numpy as np


def cargarImgHsv(nombre):
    imagen = cv2.imread(nombre)
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    return imagen_hsv

def crearAmarillo():
    print('amarillo')

imagen_hsv_1 = cargarImgHsv('im1_tp2.jpg')
imagen_hsv_2 = cargarImgHsv('im2_tp2.jpg')

#cv2.imwrite('imagen_hsv1.jpg', imagen_hsv_1)
#cv2.imwrite('imagen_hsv2.jpg', imagen_hsv_2)

