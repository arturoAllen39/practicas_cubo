import cv2
import numpy as np


def nothing(x):
    pass


# 1. Configurar la captura de video usando tu archivo local
# Al estar en la misma carpeta, solo ponemos el nombre del archivo
video_path = "test2.mp4"
cap = cv2.VideoCapture(video_path)

# Verificación de seguridad por si el video no se encuentra
if not cap.isOpened():
    print(
        f"Error: No se pudo abrir el video '{video_path}'. "
        f"Asegúrate de que el nombre esté bien escrito y en la misma carpeta."
    )
    exit()

# 2. Crear la ventana y los trackbars
cv2.namedWindow("Ajuste CLAHE en Tiempo Real", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Ajuste CLAHE en Tiempo Real", 400, 200)
cv2.createTrackbar("Clip Limit", "Ajuste CLAHE en Tiempo Real", 20, 100, nothing)
cv2.createTrackbar("Grid Size", "Ajuste CLAHE en Tiempo Real", 8, 32, nothing)

print("Presiona la tecla 'q' para salir antes de que termine el video.")

while True:
    ret, frame = cap.read()

    # Si 'ret' es False, significa que el video terminó
    if not ret:
        print("Fin del video.")
        break

    # 3. Leer los valores actuales de los trackbars
    clip_limit_raw = cv2.getTrackbarPos("Clip Limit", "Ajuste CLAHE en Tiempo Real")
    grid_size = cv2.getTrackbarPos("Grid Size", "Ajuste CLAHE en Tiempo Real")

    # Conversiones para evitar errores en OpenCV
    clip_limit = clip_limit_raw / 10.0
    if grid_size < 1:
        grid_size = 1
    if clip_limit < 0.1:
        clip_limit = 0.1

    # 4. Procesamiento de la imagen (Espacio de color LAB)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 5. Aplicar CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(grid_size, grid_size)
    )
    cl = clahe.apply(l)

    # Fusionar y regresar a BGR
    limg = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 6. Redimensionar temporalmente si el video es muy grande (Opcional)
    # Si notas que las dos pantallas juntas no caben en tu monitor,
    # puedes descomentar las siguientes líneas para encoger el resultado:
    # frame = cv2.resize(frame, (640, 360))
    # enhanced_frame = cv2.resize(enhanced_frame, (640, 360))

    # Mostrar el resultado comparativo
    resultado_comparativo = cv2.hconcat([frame, enhanced_frame])
    cv2.imshow("Ajuste CLAHE en Tiempo Real", resultado_comparativo)

    # Ajustamos el delay a 25ms para que el video corra a una velocidad natural (~40 FPS)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()