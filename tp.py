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
    amarillo_minimo = np.array([20, 70, 70], np.uint8)
    amarillo_maximo = np.array([40, 255, 255], np.uint8)

    mascaraAmarilla = cv2.inRange(imagen, amarillo_minimo, amarillo_maximo)
    return mascaraAmarilla

'''
Lo que hace esta funcion es establecer los umbrales minimos y maximos del amarillo que queremos buscar

Despues hace la mascara correspondiente y la retorna
'''

def aplicarMascaraEnSaturacion(imagen, mascara):
    canalSaturacion = imagen[:,:,1]
    mascaraCanalSaturacion = cv2.bitwise_and(canalSaturacion, canalSaturacion, mask=mascara)
    return mascaraCanalSaturacion

'''
Lo que hace esta funcion es aplicar la mascara de antes al canal de saturacion [x,y,1]

La funcion bitwise_and hace un AND logico entre la mascara y la imagen original
y para los valores que coinciden retorna el valor de saturacion y los pone en una imagen

Esa imagen final es la que se retorna
'''

def detectarBordes(mascaraSaturacion):
    sobelx = cv2.Sobel(mascaraSaturacion, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(mascaraSaturacion, cv2.CV_64F, 0, 1, ksize=3)

    #En una solucion usa el laplacian
    #laplacian = cv2.Laplacian(mascaraSaturacion,cv2.CV_64F)

    bordes = np.sqrt(sobelx**2 + sobely**2)
    bordes = cv2.convertScaleAbs(bordes)
    return bordes

'''
Esta funcion detecta los bordes usando la funcion sobel.
Primero saca las derivadas en direccion x e y para despues sacar el vector resultante.

Basicamente es una convolucion con una matriz llamada kernel
El resultado de esta convolucion es el gradiente de la posicionX, posicionY
'''

def dilatacionClausura(bordes):
    kernel = np.ones((3,3), np.uint8)
    bordes_dilatacion = cv2.dilate(bordes, kernel, iterations=1)
    bordes_clausura = cv2.morphologyEx(bordes_dilatacion, cv2.MORPH_CLOSE, kernel, iterations=1)
    return bordes_clausura

'''
Lo que hace esta funcion es primero crear el elemento estructurante (una matriz kernel de 3x3).

Despues se aplica una dilatacion con ese kernel y seguido una clausura o cierre.

Se retorna el resultado de esa operacion
'''

#Cargar info de imagenes y videos
imagen_hsv_1 = cargarImgHsv('im1_tp2.jpg')
imagen_hsv_2 = cargarImgHsv('im2_tp2.jpg')
video_hsv = cargarVidHsv('video_tp2.mp4')

#Creacion de la mascara amarilla en las imagenes
mascara_amarilla_1 = crearMascara(imagen_hsv_1)
mascara_amarilla_2 = crearMascara(imagen_hsv_2)

#Aplicacion de la mascara al canal de saturacion
mascara_saturacion_1 = aplicarMascaraEnSaturacion(imagen_hsv_1, mascara_amarilla_1)
mascara_saturacion_2 = aplicarMascaraEnSaturacion(imagen_hsv_2, mascara_amarilla_2)

#Deteccion de bordes
bordes_1 = detectarBordes(mascara_saturacion_1)
bordes_2 = detectarBordes(mascara_saturacion_2)

#Dilatacion y Clausura de bordes
bordes_mod_1 = dilatacionClausura(bordes_1)
bordes_mod_2 = dilatacionClausura(bordes_2)

#Visualizacion de imagenes
cv2.imshow('imagenOriginal', cv2.cvtColor(imagen_hsv_1, cv2.COLOR_HSV2BGR))
cv2.imshow('imagenHsv', imagen_hsv_1)
cv2.imshow('mascaraAmarilla', mascara_amarilla_1)
cv2.imshow('mascaraSaturacion', mascara_saturacion_1)
cv2.imshow('bordesSaturacion', bordes_1)
cv2.imshow('bordesDilatadosClausurados', bordes_mod_1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('imagenOriginal', cv2.cvtColor(imagen_hsv_2, cv2.COLOR_HSV2BGR))
cv2.imshow('imagenHsv', imagen_hsv_2)
cv2.imshow('mascaraAmarilla', mascara_amarilla_2)
cv2.imshow('mascaraSaturacion', mascara_saturacion_2)
cv2.imshow('bordesSaturacion', bordes_2)
cv2.imshow('bordesDilatadosClausurados', bordes_mod_2)
cv2.waitKey(0)
cv2.destroyAllWindows()