# Práctica 2 de VC (Visión por Computador)

## Autores

Leslie Liu Romero Martín
María Cabrera Vérgez

## Tareas realizadas

Para la segunda práctica de la asignatura, se hará uso de OpenCV junto a algunas de sus funciones básicas para acceder al valor de un píxel y modificarlo. Se procesará una imagen de diferentes formas para ver como esta se ve afectada. Las tareas en cuestión son:

1. Realiza la cuenta de píxeles blancos por filas (en lugar de por columnas). Determina el valor máximo de píxeles blancos para filas, maxfil, mostrando el número de filas y sus respectivas posiciones, con un número de píxeles blancos mayor o igual que 0.90*maxfil.

2. Aplica umbralizado a la imagen resultante de Sobel (convertida a 8 bits), y posteriormente realiza el conteo por filas y columnas similar al realizado en el ejemplo con la salida de Canny de píxeles no nulos. Calcula el valor máximo de la cuenta por filas y columnas, y determina las filas y columnas por encima del 0.90*máximo. Remarca con alguna primitiva gráfica dichas filas y columnas sobre la imagen del mandril. ¿Cómo se comparan los resultados obtenidos a partir de Sobel y Canny?

3.  Proponer un demostrador que capture las imágenes de la cámara, y les permita exhibir lo aprendido en estas dos prácticas ante quienes no cursen la asignatura. Es por ello que además de poder mostrar la imagen original de la webcam, permita cambiar de modo, incluyendo al menos dos procesamientos diferentes como resultado de aplicar las funciones de OpenCV trabajadas hasta ahora.

4. ras ver los vídeos My little piece of privacy, Messa di voce y Virtual air guitar proponer un demostrador reinterpretando la parte de procesamiento de la imagen, tomando como punto de partida alguna de dichas instalaciones.

## Instalación

Será necesario tener instalados los paquetes cv2, numpy, matplotlib.pylot, Image, ImageDraw, ImageFont, ImageFilter.

```
import cv2  
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter
```

## Tareas

## Tarea 1

Primero se debe leer la imagen que vamos a estar usando, mandril.jpg. Se convierte a escala de grises y se le aplica Canny, un detector de bordes. Los píxeles de lor bordes valen 255 y el resto 0.

```
img = cv2.imread('mandril.jpg') 
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ggris = cv2.GaussianBlur(gris, (3, 3), 0)
```

Posteriormente, se suma el valor de los píxeles por fila. A diferencia del cv2.reduce() hecho como ejemplo para las columnas, se le debe indicar como segundo parámetro un 1 para que el proceso se haga correctamente. En esta y otras funciones, esta es la diferencia que indica con qué se trabaja: columnas o filas. Para poder seguir trabajando, se debe hacer un reshape. Esto se debe a que necesitamos convertir el vector en un array 1D. Una vez hecho, se obtienen las filas.

```
row_counts = cv2.reduce(canny, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
row_counts = row_counts.reshape((len(row_counts)))
# rows
rows = row_counts / (255 * canny.shape[0])
```

Usaremos la variable maxfil para guardar el valor máximo de píxeles blancos en las filas haciendo uso de la función .max(). Con este valor, iremos comparando los elementos de rows para guardarlos si son mayores o iguales al 90 % de maxfil. Por otro lado, también iremos añadiendo a un array donde se encuentran, como pide la tarea.

```
maxfil = rows.max()
max_rows_values = []
max_rows_indexes = []
for i in range(len(rows)):
    if rows[i] >= 0.9*maxfil:
        max_rows_values.append(float(rows[i]))
        max_rows_indexes.append(i)
print(max_rows_values, max_rows_indexes)
```

Con todo hecho, solo queda mostrar la imagen con Canny, mostrar el número de píxeles blancos por filas y aquellos que cumplen la condición de ser mayores o iguales al 90 % de maxfil.

<img alt="example1" src="/Ejemplos/example1.png">

## Tarea 3

Como es necesario captar la imagen por medio de la webcam, se hará uso de cv2.VideoCapture(0). El propósito de estarea tarea es poder cambiar entre diferentes modos de procesamiento de imagen, recordando lo aprendido durante la práctica 1. Una vez captamos fotograma a fotograma, convertiremos la imagen a escala de grises para poder trabajar.

Los modos se irán regulando con ifs sobre el valor de una variable llamada precisamente "modo". Si estamos en el primero caso, la salida de la imagen será normal. Si pasamos al modo 1, se aplicará la umbralización aprendida durante la segunda tarea. Todo píxel que sea mayor a 130, pasa a blanco, de ahí el 255. En caso contrario, se queda en negro, 0.

```
elif modo == 1:
        # Umbralización
        _, salida = cv2.threshold(gris, 130, 255, cv2.THRESH_BINARY)
```

<img alt="example1" src="/Ejemplos/example3-1.png">

Para el modo 2 se aplica Sobel. Primero se suaviza la imagen y se eliminan altas frecuencias. Posteriormente, se calcula sober  tanto en horizontal como en vertical y se combinan los resultados para tener la imagen final. En este modo, esa será la salida que se va conseguirá.

```
elif modo == 2:
        # Sobel
        ggris = cv2.GaussianBlur(gris, (3, 3), 0)
        sobelx = cv2.Sobel(ggris, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(ggris, cv2.CV_64F, 0, 1)
        sobel = cv2.convertScaleAbs(cv2.add(sobelx, sobely))
        salida = sobel
```

<img alt="example1" src="/Ejemplos/example3-2.png">

Una vez aplicados los conocimientos de la práctica 2, el modo 3 recuerda la última tarea de la primera práctica. En este modo, vemos uno de los resultados de pop art obtenidos. Para ello, se dividen los 3 canales para ir jugando con ellos, invirtiendo el valor de los canales r y g, mientras el b se deja tal como está.

```
elif modo == 3:
        # Pop art
        r = frame[:,:,2]
        g = frame[:,:,1]
        b = frame[:,:,0]

        frame[:,:,0] = b
        frame[:,:,1] = 255 - r
        frame[:,:,2] = 255 - g

        salida = frame
```

<img alt="example1" src="/Ejemplos/example3-3.png">

Para no provocar problemas entre el cambio de modos o la finalización del programa, se recoge una vez el valor de la tecla pulsado y se va comparando en cada iteración del while. Si pulsamos ESC (27), detenemos todo. Si, en su lugar, pulsamos d, avanzaremos entre los diferentes modos de imagen. Por lo contrario, a nos hace volver hacia atrás. Se usa ord() para 'a' o 'd' porque es una función que transforma caracteres en valores numéricos (como el caso del 27). Cuando se avanza o retrocede, hacemos % 4 para que el modo nunca se salga de los valores establecidos. Solo debe ir entre 0 y 3, por lo que esto nos permite tener unos límites.

```
elif key == ord('d'):
        modo = (modo + 1) % 4 
    elif key == ord('a'):
        modo = (modo - 1) % 4
```












## Referencias
- https://medium.com/@siromermer/detecting-and-tracking-moving-objects-with-background-subtractors-using-opencv-f2ff7f94586f
- https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
- https://www.geeksforgeeks.org/python/python-opencv-waitkey-function/
- https://www.w3schools.com/python/matplotlib_plotting.asp
- https://docs.opencv.org/4.x/de/de1/group__video__motion.html#ga818a6d66b725549d3709aa4cfda3f301
- https://medium.com/@happynehra/exploring-unique-features-of-opencv-haar-cascade-classifiers-and-background-subtraction-deff4eec7a51
- https://programarfacil.com/blog/vision-artificial/detector-de-bordes-canny-opencv
- https://omes-va.com/rostros-borrosos-uso-del-trackbar-opencv-python
- https://www.datacamp.com/es/tutorial/face-detection-python-opencv
- https://keepcoding.io/blog/que-es-ord-en-python-y-como-usarlo

