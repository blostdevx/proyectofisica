import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

# Variables para aprendizaje
historial_predicciones = []
errores = []

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

# Función para calcular velocidad y aceleración
def calcular_mrua(puntos, fps):
    velocidades = []
    aceleraciones = []
    for i in range(1, len(puntos)):
        distancia = np.linalg.norm(np.array(puntos[i]) - np.array(puntos[i - 1]))
        tiempo = 1 / fps
        velocidad = distancia / tiempo
        velocidades.append(velocidad)
        if i > 1:
            aceleracion = (velocidades[-1] - velocidades[-2]) / tiempo
            aceleraciones.append(aceleracion)
    return velocidades, aceleraciones

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

                # Predecir punto de caída
                if len(puntos) > 5:
                    x = np.array([p[0] for p in puntos]).reshape(-1, 1)
                    y = np.array([p[1] for p in puntos])
                    modelo = LinearRegression().fit(x, y)
                    x_pred = np.array([frame.shape[1]]).reshape(-1, 1)
                    y_pred = modelo.predict(x_pred)
                    cv2.circle(frame, (frame.shape[1], int(y_pred)), 10, (0, 0, 255), -1)

                    # Guardar predicción en historial
                    historial_predicciones.append((frame.shape[1], int(y_pred)))

    elif seleccionando_objeto:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if puntos:
    velocidades, aceleraciones = calcular_mrua(puntos, fps)

    x = [p[0] for p in puntos]
    y = [p[1] for p in puntos]

    coeficientes = np.polyfit(x, y, 2)
    polinomio = np.poly1d(coeficientes)

    punto_cero = polinomio.r
    caida = y[-1] - y[0]

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(x, y, label='Trayectoria')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(velocidades, label='Velocidad')
    plt.xlabel('Tiempo (frames)')
    plt.ylabel('Velocidad (pixeles/segundo)')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(aceleraciones, label='Aceleración')
    plt.xlabel('Tiempo (frames)')
    plt.ylabel('Aceleración (pixeles/segundo^2)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f'Punto Cero: {punto_cero}')
    print(f'Caída: {caida}')

    # Aprendizaje basado en errores
    respuesta_correcta = input("¿El cálculo fue correcto? (s/n): ")
    if respuesta_correcta.lower() == 'n':
        x_real = int(input("Ingrese la coordenada X real de la caída: "))
        y_real = int(input("Ingrese la coordenada Y real de la caída: "))
        errores.append((historial_predicciones[-1], (x_real, y_real)))
        print("Error registrado para aprendizaje futuro.")
