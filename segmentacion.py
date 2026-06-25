import cv2
import numpy as np


def nothing(x):
    pass


# 1. Configurar el video local
video_path = "test2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: No se pudo abrir el video '{video_path}'.")
    exit()

# 2. Crear ventana ajustable para que no rompa la pantalla
window_name = "Aislamiento de Luz por Canal L"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 480)

# 3. Crear Trackbars para seleccionar el rango de luz a aislar
# Umbral Mínimo: Píxeles más oscuros que queremos capturar
cv2.createTrackbar("Luz Minima", window_name, 0, 255, nothing)
# Umbral Máximo: Píxeles más claros que queremos capturar
cv2.createTrackbar("Luz Maxima", window_name, 255, 255, nothing)

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video.")
        break

    # 4. Leer los límites de luz desde los Trackbars
    luz_min = cv2.getTrackbarPos("Luz Minima", window_name)
    luz_max = cv2.getTrackbarPos("Luz Maxima", window_name)

    # 5. Transformar al espacio de color LAB y extraer solo la luz (L)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 6. CREAR LA MÁSCARA DE LUZ (Segmentación)
    # cv2.inRange creará una imagen binaria (blanco y negro pura):
    # Todo lo que esté en el rango de luz seleccionado se vuelve BLANCO.
    # Lo que quede fuera se vuelve NEGRO.
    mascara_luz = cv2.inRange(l, luz_min, luz_max)

    # Opcional: Aplicar un pequeño filtro morfológico para limpiar el ruido
    kernel = np.ones((3, 3), np.uint8)
    mascara_luz = cv2.morphologyEx(mascara_luz, cv2.MORPH_OPEN, kernel)

    # 7. Preparar la visualización
    # Como 'mascara_luz' es de 1 solo canal (escala de grises),
    # debemos convertirla a BGR (3 canales) para poder pegarla al lado del video original.
    mascara_bgr = cv2.cvtColor(mascara_luz, cv2.COLOR_GRAY2BGR)

    # 8. Concatenar Original vs Máscara de Luz
    resultado_comparativo = cv2.hconcat([frame, mascara_bgr])
    cv2.imshow(window_name, resultado_comparativo)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()