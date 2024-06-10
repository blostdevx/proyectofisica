import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar el modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Leer el video
cap = cv2.VideoCapture('ruta/a/tu/video.mp4')

# Parámetros para el algoritmo de Lucas-Kanade
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Definir listas para almacenar las coordenadas del objeto
puntos = []

# Procesar cada cuadro del video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detección de objetos
    results = model(frame)
    
    # Suponiendo que detectamos un solo objeto de interés
    if len(results.xyxy[0]) > 0:
        x1, y1, x2, y2, confidence, cls = results.xyxy[0][0]
        if confidence > 0.5:  # Umbral de confianza
            # Obtener el centro del objeto
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            puntos.append((cx, cy))
            
            # Dibujar rectángulo y centro
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calcular velocidad
def calcular_velocidad(puntos, fps):
    velocidades = []
    for i in range(1, len(puntos)):
        distancia = np.linalg.norm(np.array(puntos[i]) - np.array(puntos[i-1]))
        tiempo = 1 / fps
        velocidad = distancia / tiempo
        velocidades.append(velocidad)
    return velocidades

# Supongamos que fps es 30
fps = 30
velocidades = calcular_velocidad(puntos, fps)

# Extraer coordenadas
x = [p[0] for p in puntos]
y = [p[1] for p in puntos]

# Ajustar una parábola (segunda orden)
coeficientes = np.polyfit(x, y, 2)
polinomio = np.poly1d(coeficientes)

# Punto cero
punto_cero = polinomio.r

# Calcular la caída en el último punto
caida = y[-1] - y[0]

# Visualización
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
