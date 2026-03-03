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

    cv.namedWindow(win_video)
    cv.namedWindow(win_controls)

    # Todos los trackbars van en la ventana de controles
    cv.createTrackbar('Blur',        win_controls, 5,   31,  callback)
    cv.createTrackbar('minThres',    win_controls, 50,  255, callback)
    cv.createTrackbar('maxThres',    win_controls, 150, 255, callback)
    cv.createTrackbar('Dilate',      win_controls, 1,   10,  callback)
    cv.createTrackbar('Erode',       win_controls, 0,   10,  callback)
    cv.createTrackbar('Bilateral_d', win_controls, 5,   20,  callback)
    cv.createTrackbar('Bilateral_s', win_controls, 50,  150, callback)

    # Imagen de relleno para que la ventana de controles tenga cuerpo visible
    panel = np.zeros((100, 400), dtype=np.uint8)
    cv.imshow(win_controls, panel)

    print("Presiona [F] para alternar entre vista original y con filtros")
    print("Presiona [Q] para salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue

        blur_k      = cv.getTrackbarPos('Blur',        win_controls)
        minThres    = cv.getTrackbarPos('minThres',     win_controls)
        maxThres    = cv.getTrackbarPos('maxThres',     win_controls)
        dilate_iter = cv.getTrackbarPos('Dilate',       win_controls)
        erode_iter  = cv.getTrackbarPos('Erode',        win_controls)
        bil_d       = cv.getTrackbarPos('Bilateral_d',  win_controls)
        bil_s       = cv.getTrackbarPos('Bilateral_s',  win_controls)

        if blur_k < 1: blur_k = 1
        if blur_k % 2 == 0: blur_k += 1
        if bil_d < 1: bil_d = 1

        if mostrar_filtros:
            gray      = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            bilateral = cv.bilateralFilter(gray, bil_d, bil_s, bil_s)
            blur      = cv.GaussianBlur(bilateral, (blur_k, blur_k), 0)
            edges     = cv.Canny(blur, minThres, maxThres)

            kernel = np.ones((3, 3), np.uint8)
            if dilate_iter > 0:
                edges = cv.dilate(edges, kernel, iterations=dilate_iter)
            if erode_iter > 0:
                edges = cv.erode(edges, kernel, iterations=erode_iter)

            display = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
            cv.putText(display, 'MODO: FILTROS [F para cambiar]', (10, 30),
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