import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar el modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Parámetros para el algoritmo de Lucas-Kanade
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Variables para almacenar la selección del objeto y su seguimiento
seleccionando_objeto = False
objeto_seleccionado = False
roi = (0, 0, 0, 0)
puntos = []
fps = cap.get(cv2.CAP_PROP_FPS)

# Callback para el evento de mouse
def seleccionar_objeto(event, x, y, flags, param):
    global seleccionando_objeto, roi, objeto_seleccionado
    if event == cv2.EVENT_LBUTTONDOWN:
        seleccionando_objeto = True
        roi = (x, y, x, y)
    elif event == cv2.EVENT_MOUSEMOVE and seleccionando_objeto:
        roi = (roi[0], roi[1], x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        seleccionando_objeto = False
        objeto_seleccionado = True
        roi = (roi[0], roi[1], x, y)

# Ventana de video y callback del mouse
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', seleccionar_objeto)

# Procesar cada cuadro del video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if objeto_seleccionado:
        x1, y1, x2, y2 = roi
        frame_roi = frame[y1:y2, x1:x2]
        results = model(frame_roi)

        if len(results.xyxy[0]) > 0:
            x1_obj, y1_obj, x2_obj, y2_obj, confidence, cls = results.xyxy[0][0]
            if confidence > 0.5:  # Umbral de confianza
                cx = int(x1 + (x1_obj + x2_obj) / 2)
                cy = int(y1 + (y1_obj + y2_obj) / 2)
                puntos.append((cx, cy))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                for i in range(1, len(puntos)):
                    cv2.line(frame, puntos[i - 1], puntos[i], (0, 255, 0), 2)

    elif seleccionando_objeto:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if puntos:
    def calcular_velocidad(puntos, fps):
        velocidades = []
        for i in range(1, len(puntos)):
            distancia = np.linalg.norm(np.array(puntos[i]) - np.array(puntos[i - 1]))
            tiempo = 1 / fps
            velocidad = distancia / tiempo
            velocidades.append(velocidad)
        return velocidades

    velocidades = calcular_velocidad(puntos, fps)

    x = [p[0] for p in puntos]
    y = [p[1] for p in puntos]

    coeficientes = np.polyfit(x, y, 2)
    polinomio = np.poly1d(coeficientes)

    punto_cero = polinomio.r
    caida = y[-1] - y[0]

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x, y, label='Trayectoria')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(velocidades, label='Velocidad')
    plt.xlabel('Tiempo (frames)')
    plt.ylabel('Velocidad (pixeles/segundo)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f'Punto Cero: {punto_cero}')
    print(f'Caída: {caida}')
