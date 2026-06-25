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

# 2. Crear ventana ajustable
cv2.namedWindow("Filtro de Iluminacion Homogenea", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Filtro de Iluminacion Homogenea", 1280, 480)

# 3. Crear Trackbars
cv2.createTrackbar(
    "Suavizado Luz", "Filtro de Iluminacion Homogenea", 101, 255, nothing
)
cv2.createTrackbar(
    "Brillo Base", "Filtro de Iluminacion Homogenea", 120, 255, nothing
)

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video.")
        break

    # 4. Leer parámetros de los Trackbars
    ksize = cv2.getTrackbarPos("Suavizado Luz", "Filtro de Iluminacion Homogenea")
    brillo_base = cv2.getTrackbarPos(
        "Brillo Base", "Filtro de Iluminacion Homogenea"
    )

    # 5. Cambiar a espacio LAB para procesar solo el brillo (L)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # --- 6. ESTIMAR EL MAPA DE LUZ (¡ZONA OPTIMIZADA!) ---
    # Reducimos el canal L al 10% (0.1) de su tamaño para que la CPU trabaje menos
    l_mini = cv2.resize(l, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)

    # Escalamos el tamaño del kernel proporcionalmente al tamaño miniatura
    ksize_mini = int(ksize * 0.1)
    if ksize_mini % 2 == 0:
        ksize_mini += 1
    if ksize_mini < 3:
        ksize_mini = 3

    # Desenfocamos la miniatura (esto ahora toma microsegundos)
    mapa_luz_mini = cv2.GaussianBlur(l_mini, (ksize_mini, ksize_mini), 0)

    # Estiramos el mapa desenfocado de vuelta al tamaño original del video
    mapa_luz = cv2.resize(
        mapa_luz_mini, (l.shape[1], l.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    # -----------------------------------------------------

    # 7. DIVISIÓN ARITMÉTICA
    l_plana = cv2.divide(l, mapa_luz, scale=brillo_base)

    # 8. Reconstruir la imagen
    lab_plano = cv2.merge((l_plana, a, b))
    frame_plano = cv2.cvtColor(lab_plano, cv2.COLOR_LAB2BGR)

    # 9. Mostrar la comparativa lado a lado
    resultado_comparativo = cv2.hconcat([frame, frame_plano])
    cv2.imshow("Filtro de Iluminacion Homogenea", resultado_comparativo)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()