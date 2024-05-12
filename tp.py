import cv2
import numpy as np


def cargarImgHsv(nombre):
    imagen = cv2.imread(nombre)
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    return imagen_hsv

'''
Lo que hace esta funcion es leer la imagen y procesarla en HSV.

Lo que retorna cargarImgHsv es una matriz tridimensional de numpy
Tiene la forma de [x, y, 0/1/2]
x , y seria un punto especifico.

hue = frameHSV[y, x, 0]  # Matiz
saturation = frameHSV[y, x, 1]  # Saturación
value = frameHSV[y, x, 2]  # Valor
'''

def cargarVidHsv(nombre):
    cap = cv2.VideoCapture(nombre)
    frames_hsv = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frames_hsv.append(frame_hsv)
        else:
            break
    cap.release()
    return np.array(frames_hsv)

'''
Lo que hace esta funcion es leer el archivo de video y procesarlo en HSV.

Lo que retorna, a diferencia de la anterior, es un array con todos los frames procesados.

Hay que ver si necesitamos todos los frames o solo el ultimo
'''

def crearMascara(imagen):
    amarillo_minimo = np.array([20, 50, 50], np.uint8)
    amarillo_maximo = np.array([40, 255, 255], np.uint8)

    mascaraAmarilla = cv2.inRange(imagen, amarillo_minimo, amarillo_maximo)
    return mascaraAmarilla

'''
Lo que hace esta funcion es establecer los umbrales minimos y maximos del amarillo que queremos buscar

Despues hace la mascara correspondiente y la retorna
'''

imagen_hsv_1 = cargarImgHsv('im1_tp2.jpg')
imagen_hsv_2 = cargarImgHsv('im2_tp2.jpg')
video_hsv = cargarVidHsv('video_tp2.mp4')

mascara_amarilla_1 = crearMascara(imagen_hsv_1)
mascara_amarilla_2 = crearMascara(imagen_hsv_2)

#Visualizacion de imagenes
cv2.imshow('imagenHsv', imagen_hsv_1)
cv2.imshow('mascaraAmarilla', mascara_amarilla_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('imagenHsv', imagen_hsv_2)
cv2.imshow('mascaraAmarilla', mascara_amarilla_2)
cv2.waitKey(0)
cv2.destroyAllWindows()