import cv2 as cv
import numpy as np

mostrar_filtros = True

def callback(val):
    pass

def cannyEdge():
    global mostrar_filtros

    cap = cv.VideoCapture('video.mov')

    if not cap.isOpened():
        print("No se pudo abrir el video")
        return

    win_video    = 'Video'
    win_controls = 'Controles'

    cv.namedWindow(win_video, cv.WINDOW_NORMAL)
    cv.namedWindow(win_controls, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_controls, 400, 200)

    cv.createTrackbar('Blur',       win_controls, 7,  31,  callback)
    cv.createTrackbar('Min_thresh', win_controls, 0,  255, callback)
    cv.createTrackbar('Max_thresh', win_controls, 60, 255, callback)
    cv.createTrackbar('Close',      win_controls, 5,  20,  callback)
    cv.createTrackbar('Open',       win_controls, 3,  20,  callback)

    panel = np.zeros((100, 400), dtype=np.uint8)
    cv.imshow(win_controls, panel)

    print("Presiona [F] para alternar entre vista original y con filtros")
    print("Presiona [Q] para salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue

        blur_k    = cv.getTrackbarPos('Blur',       win_controls)
        min_val   = cv.getTrackbarPos('Min_thresh', win_controls)
        max_val   = cv.getTrackbarPos('Max_thresh', win_controls)
        close_k   = cv.getTrackbarPos('Close',      win_controls)
        open_k    = cv.getTrackbarPos('Open',        win_controls)

        if blur_k < 1: blur_k = 1
        if blur_k % 2 == 0: blur_k += 1
        if close_k < 1: close_k = 1
        if open_k < 1: open_k = 1
        # Asegurar que min no supere a max
        if min_val >= max_val: min_val = max(0, max_val - 1)

        if mostrar_filtros:
            # 1. Escala de grises
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # 2. Blur para suavizar ruido
            blur = cv.GaussianBlur(gray, (blur_k, blur_k), 0)

            # 3. Rango de threshold (personas dentro del rango min-max)
            mask = cv.inRange(blur, min_val, max_val)

            # 4. Cerrar huecos
            kernel_close = np.ones((close_k, close_k), np.uint8)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_close)

            # 5. Eliminar ruido pequeño
            kernel_open = np.ones((open_k, open_k), np.uint8)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open)

            display = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            cv.putText(display, 'MODO: SEGMENTACION [F para cambiar]', (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            display = frame.copy()
            cv.putText(display, 'MODO: ORIGINAL [F para cambiar]', (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv.imshow(win_video, display)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            mostrar_filtros = not mostrar_filtros

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    cannyEdge()