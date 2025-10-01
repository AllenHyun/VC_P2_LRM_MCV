# Práctica 2 de VC (Visión por Computador)

## Autores

Leslie Liu Romero Martín
<br>
María Cabrera Vérgez

## Tareas realizadas

Para la segunda práctica de la asignatura, se hará uso de OpenCV junto a algunas de sus funciones básicas para acceder al valor de un píxel y modificarlo. Se procesará una imagen de diferentes formas para ver como esta se ve afectada. Las tareas en cuestión son:

1. Realiza la cuenta de píxeles blancos por filas (en lugar de por columnas). Determina el valor máximo de píxeles blancos para filas, maxfil, mostrando el número de filas y sus respectivas posiciones, con un número de píxeles blancos mayor o igual que 0.90*maxfil.

2. Aplica umbralizado a la imagen resultante de Sobel (convertida a 8 bits), y posteriormente realiza el conteo por filas y columnas similar al realizado en el ejemplo con la salida de Canny de píxeles no nulos. Calcula el valor máximo de la cuenta por filas y columnas, y determina las filas y columnas por encima del 0.90*máximo. Remarca con alguna primitiva gráfica dichas filas y columnas sobre la imagen del mandril. ¿Cómo se comparan los resultados obtenidos a partir de Sobel y Canny?

3.  Proponer un demostrador que capture las imágenes de la cámara, y les permita exhibir lo aprendido en estas dos prácticas ante quienes no cursen la asignatura. Es por ello que además de poder mostrar la imagen original de la webcam, permita cambiar de modo, incluyendo al menos dos procesamientos diferentes como resultado de aplicar las funciones de OpenCV trabajadas hasta ahora.

4. Tras ver los vídeos My little piece of privacy, Messa di voce y Virtual air guitar proponer un demostrador reinterpretando la parte de procesamiento de la imagen, tomando como punto de partida alguna de dichas instalaciones.

## Instalación

Será necesario tener instalados los paquetes cv2, numpy, matplotlib.pylot, Image, ImageDraw, ImageFont, ImageFilter.

```
import cv2  
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter
```

## Tareas

### Tarea 1

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

### Tarea 2

Para la tarea 2 hemos reutilizado ciertas partes de la tarea 1 para el conteo de píxeles tanto en filas como en columnas. En este caso, partimos de la imagen resultante de Sobel combinada en X e Y:

```
# Gaussiana para suavizar la imagen original, eliminando altas frecuencias
img = cv2.imread('mandril.jpg') 
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ggris = cv2.GaussianBlur(gris, (3, 3), 0)

# Calcula en ambas direcciones (horizontal y vertical)
sobelx = cv2.Sobel(ggris, cv2.CV_64F, 1, 0)  # x
sobely = cv2.Sobel(ggris, cv2.CV_64F, 0, 1)  # y
# Combina ambos resultados
sobel = cv2.add(sobelx, sobely)

valor_umbral = 130

_, imagenUmbralizada = cv2.threshold(cv2.convertScaleAbs(sobel), valor_umbral, 255, cv2.THRESH_BINARY)

# Mostramos Sobel
fig, (ax1, ax2, ax3)= plt.subplots(1, 3, figsize=(15, 4))
ax1.set_axis_off()
ax1.set_title("Sobel")
ax1.imshow(imagenUmbralizada, cmap='gray')

col_counts = cv2.reduce(imagenUmbralizada, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
cols = col_counts / (255 * imagenUmbralizada.shape[1])
cols = cols[0]

row_counts = cv2.reduce(imagenUmbralizada, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
row_counts = row_counts.reshape((len(row_counts)))
rows = row_counts / (255 * imagenUmbralizada.shape[0])
```

Tras el conteo, procedemos a mostrar la primera de dos figuras distintas de matplotlib (también imitando el método de la Tarea 1). Esta figura, con 3 elementos distintos, contendrá la imagen resultante de Sobel y los gráficos de la cuenta de píxeles para las columnas y las filas.

<img alt="example2_1" src="/Ejemplos/example2-1.png">

Para el cálculo de las filas y columnas por encima del 90% usamos también el mismo método que en la tarea anterior.

```
max_cols_values = []
max_cols_indexes = []
for i in range(len(cols)):
    if cols[i] >= 0.9*maxcol:
        max_cols_values.append(float(cols[i]))
        max_cols_indexes.append(i)

max_rows_values = []
max_rows_indexes = []
for i in range(len(rows)):
    if rows[i] >= 0.9*maxfil:
        max_rows_values.append(float(rows[i]))
        max_rows_indexes.append(i)
```

Finalmente, para la segunda figura, mostramos nuevamente el resultado de Sobel, pero con líneas de distintos colores resaltando las columnas (verde) y filas (azul) que superan los valores del 90% para maxcol (columnas) y maxfil (filas) junto con sus respectivas gráficas.

<img alt="example2_2" src="/Ejemplos/example2-2.png">

Como último punto, queda comentar la diferencia entre Canny y Sobel. Si observamos la respuesta de Canny en las filas (+90%) y la de Sobel, se ve la misma cantidad de filas que superan el 90% en la cuenta de píxeles, sin embargo, en Canny se muestran más juntos los puntos, concentrados en los extremos, mientras que en Sobel se encuentran repartidos más equitativamente en todo el espacio muestral de las filas.

<img alt="example2_3" src="/Ejemplos/example2-3.png">
<img alt="example2_4" src="/Ejemplos/example2-4.png">

### Tarea 3

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

### Tarea 4

Para la tarea 4, hemos tenido la idea de seguir con la línea de 'My Little Piece of Privacy', que utiliza el procesamiento de imágenes para detectar posiciones y lo hemos aplicado a un caso común también conectado a la privacidad: el desenfoque de las caras cuando nos encontramos en algún tipo de videollamada o cuando hay algún procesamiento de vídeo y se quiere conservar la privacidad de las personas que no han accedido a que su imagen sea tomada.

Empezamos con la implementación base, procesamos la entrada de la webcam y nos basamos del modelo que nos aporta OpenCV para la detección facial:

```
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)
```

Primeramente, nos aseguramos de que se detecten las caras y que podamos poner una elipse que añada desenfoque sobre ellas. El proceso para convertir una elipse común en una elipse desenfocada fue un poco complejo, ya que hay que detectar el área de la cara, aplicar el desenfoque (Gaussian Blur), crear una máscara binaria con la elipse, transformar dicha máscara de un canal (blanco y negro) a tres canales (color) y utilizar la máscara para recortar el desenfoque en forma de elipse.

```
# Sacamos la región donde está la cara detectada
roi = vid.copy()

# Aplicamos Gaussian Blur
blurred_roi = cv2.GaussianBlur(vid, (51, 51), 30)

# Creamos elipse blanca sobre fondo negro para la máscara
mask = np.zeros((vid.shape[0], vid.shape[1]), dtype=np.uint8)
center = (x + (w//2), y + (h//2))
cv2.ellipse(mask, center, (w//2, h//2), 0, 0, 360, 255, -1)

# Hacemos máscara de 3 canales (de grises a color)
mask3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
mask_normalized = mask3ch.astype(float) / 255

# Reemplazamos la región original por la nueva
vid[:] = (1 - mask_normalized) * vid + mask_normalized * blurred_roi
```

Sin embargo, lo anterior es el primer paso. Para poder centrarnos en desenfocar solo a las personas que se encuentran en el fondo, hemos decidido combinar este desenfoque con la separación del fondo que vimos en clase: 

```
back_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
```

Para lograr el efecto que queremos, hemos utilizado la sustracción de fondo de OpenCV para dividir la entrada de vídeo entre 'background' y 'foreground', dándole 30 frames para que el algoritmo se calibre.

```
# Sacamos el fondo y nos quedamos con la 'Foreground mask'
fg_mask = back_sub.apply(frame)
if frame_count > 30:  # Calibramos el modelo
    faces = detectar_caras(frame, fg_mask)
```

Con la llamada a la función detectar_caras(), le pasamos la detección del fondo y la usamos para comprobar qué caras se pueden considerar como parte del fondo y cuáles son las que estan en primer plano. Para detectar cuáles están en el fondo, tomamos una sub-región cerca del centro de la región detectada por el modelo como una posible cara y comprobamos si sus píxeles pertenecen o no al fondo. Con esto generamos un background_ratio que será el que usaremos para decidir si debemos o no desenfocar la cara.

```
for (x, y, w, h) in faces:
    # Extraemos parte de la región de la cara detectada de fg_mask
    x0 = max(0, x + w//4)
    y0 = max(0, y + h//4)
    x1 = min(fg_mask.shape[1], x + 3*w//4)
    y1 = min(fg_mask.shape[0], y + 3*h//4)
    face_region = fg_mask[y0:y1, x0:x1]
    # Sacamos el ratio de píxeles de la cara que pertenecerían al fondo
    background_ratio = np.sum(face_region == 0) / face_region.size
    if face_region.size > 0 and background_ratio > 0.87:
```

El resultado deseado es que el desenfoque solo afecte a las personas en el fondo y que la(s) persona(s) en primer plano se mantengan sin desenfocar. La clave es que las personas en el fondo se mantengan estáticas para que se registren como fondo y que la(s) persona(s) en primer plano muestren algún movimiento. Sin embargo, los resultados son variables y no suficientemente consistentes para afirmar un funcionamiento óptimo.

<img alt="example4" src="/Ejemplos/example4.png">


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

