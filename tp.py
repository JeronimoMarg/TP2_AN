import cv2
import numpy as np
import os

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
def seleccionarFramesPorSegundo(frames_hsv, fps, duracion):
    selected_frames = []
    total_frames = len(frames_hsv)
    for i in range(min(duracion, total_frames)):
        selected_frames.append(frames_hsv[int(i * fps)])
    return np.array(selected_frames)

def crearMascara(imagen):
    amarillo_minimo = np.array([20, 70  , 70], np.uint8)
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
    kernel = np.ones((5,5), np.uint8)
    bordes_dilatacion = cv2.dilate(bordes, kernel, iterations=1)
    bordes_clausura = cv2.morphologyEx(bordes_dilatacion, cv2.MORPH_CLOSE, kernel, iterations=1)
    return bordes_clausura

'''
Lo que hace esta funcion es primero crear el elemento estructurante (una matriz kernel de 3x3).

Despues se aplica una dilatacion con ese kernel y seguido una clausura o cierre.

Se retorna el resultado de esa operacion
'''

def areaAmarillo(bordes):
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mascara_llena = np.zeros_like(bordes)
    cv2.drawContours(mascara_llena, contornos, -1, (255), thickness=cv2.FILLED)

    area_amarillo = cv2.countNonZero(mascara_llena)
    area_total = mascara_llena.shape[0] * mascara_llena.shape[1]
    porcentaje_area_amarillo = (area_amarillo / area_total) * 100

    print("Porcentaje del área amarilla respecto al área total de la imagen:", porcentaje_area_amarillo)

    return mascara_llena

'''
Lo que hace esta funcion es buscar el contorno de la figura delimitada por los bordes
Despues la llena de color blanco.

Se cuentan los colores blanco y negro y se saca un porcentaje.
El porcentaje simboliza cuanto del total de la imagen es de color amarillo
'''

def aplicarIntensidad (imagen,mascara):
    intensidad = imagen[:,:,2]
    mascaraCanalIntensidad = cv2.bitwise_and(intensidad, intensidad, mask=mascara)
    return mascaraCanalIntensidad

'''
Esta funcion devuelve los valores de intensidad de la imagen original con la mascara de los contornos llenadas anteriormente.

Es decir, devuelve los valores de intensidad de la imagen original para los valores 1 de la mascara.
'''

def calculoMayorIntensidad (imagen,imagen2,mascara,mascara2):
    mascaraCanalIntensidad = aplicarIntensidad(imagen, mascara)
    mascaraCanalIntensidad2 = aplicarIntensidad(imagen2, mascara2)
    prom1 = np.sum(mascaraCanalIntensidad)
    prom2 = np.sum(mascaraCanalIntensidad2)
    if prom1 == prom2 : 
        print("Las muestras son equivalentes")
    elif prom1 > prom2 :
        nombreImg = "im1_tp2.jpg"
    else:
        nombreImg = "im2_tp2.jpg"
    print("La imagen que posee más cantidad de muestra es : ", nombreImg)

'''
El objetivo de esta funcion es tomar los valores de intensidad y sumarlos.

Compara la cantidad sumada, si es mayor significa que tiene mayor cantidad de muestra.
'''
    
#Cambiar el directorio al actual donde se encuentra el archivo python
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
#print("New Working Directory:", os.getcwd())

'''-------------PROCESAMIENTO DE IMAGENES---------------------'''

#Cargar info de imagenes y videos
imagen_hsv_1 = cargarImgHsv('im1_tp2.jpg')
imagen_hsv_2 = cargarImgHsv('im2_tp2.jpg')

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

#Contornos y porcentaje de area amarilla
contorno_1 = areaAmarillo(bordes_mod_1)
contorno_2 = areaAmarillo(bordes_mod_2)

#Area mas intensa
calculoMayorIntensidad(imagen_hsv_1,imagen_hsv_2,contorno_1,contorno_2)
mascara_intensidad_1 = aplicarIntensidad(imagen_hsv_1,contorno_1) 
mascara_intensidad_2 = aplicarIntensidad(imagen_hsv_2,contorno_2) 

#Visualizacion de imagenes
cv2.imshow('imagenOriginal', cv2.cvtColor(imagen_hsv_1, cv2.COLOR_HSV2BGR))
cv2.imshow('imagenHsv', imagen_hsv_1)
cv2.imshow('mascaraAmarilla', mascara_amarilla_1)
cv2.imshow('mascaraSaturacion', mascara_saturacion_1)
cv2.imshow('bordesSaturacion', bordes_1)
cv2.imshow('bordesDilatadosClausurados', bordes_mod_1)
cv2.imshow('areaAmarilla', contorno_1)
cv2.imshow('mascaraIntensidad',mascara_intensidad_1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('imagenOriginal', cv2.cvtColor(imagen_hsv_2, cv2.COLOR_HSV2BGR))
cv2.imshow('imagenHsv', imagen_hsv_2)
cv2.imshow('mascaraAmarilla', mascara_amarilla_2)
cv2.imshow('mascaraSaturacion', mascara_saturacion_2)
cv2.imshow('bordesSaturacion', bordes_2)
cv2.imshow('bordesDilatadosClausurados', bordes_mod_2)
cv2.imshow('areaAmarilla', contorno_2)
cv2.imshow('mascaraIntensidad',mascara_intensidad_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''-------------PROCESAMIENTO DE VIDEO---------------------'''

video_hsv = cargarVidHsv('video_tp2.mp4')

cap = cv2.VideoCapture('video_tp2.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
duracion = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
cap.release()
video_hsv_acortado = seleccionarFramesPorSegundo(video_hsv, fps, duracion)

for i in range(len(video_hsv_acortado)):
    frame = video_hsv_acortado[i]
    mascara_amarilla_vid = crearMascara(frame)
    mascara_saturacion_vid = aplicarMascaraEnSaturacion(frame, mascara_amarilla_vid)
    bordes_vid = detectarBordes(mascara_saturacion_vid)
    bordes_mod_vid = dilatacionClausura(bordes_vid)
    contorno_vid = areaAmarillo(bordes_mod_vid)
    mascara_intensidad_vid = aplicarIntensidad(frame,contorno_vid)
    cv2.imshow('imagenOriginal', cv2.cvtColor(frame, cv2.COLOR_HSV2BGR))
    cv2.imshow('imagenHsv', frame)
    cv2.imshow('mascaraAmarilla', mascara_amarilla_vid)
    cv2.imshow('mascaraSaturacion', mascara_saturacion_vid)
    cv2.imshow('bordesSaturacion', bordes_vid)
    cv2.imshow('bordesDilatadosClausurados', bordes_mod_vid)
    cv2.imshow('areaAmarilla', contorno_vid)
    cv2.imshow('mascaraIntensidad',mascara_intensidad_vid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
