import cv2 as cv
import numpy as npexit
from sort import Sort
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

    cv.resizeWindow(win_video, 700, 600)
    cv.resizeWindow(win_controls, 400, 200)

    cv.moveWindow(win_video, 0, 0)
    cv.moveWindow(win_controls, 710, 0)

    cv.createTrackbar('Blur',       win_controls, 7,  31,  callback)
    cv.createTrackbar('Min_thresh', win_controls, 0,  255, callback)
    cv.createTrackbar('Max_thresh', win_controls, 60, 255, callback)
    cv.createTrackbar('Close',      win_controls, 5,  20,  callback)
    cv.createTrackbar('Open',       win_controls, 3,  20,  callback)

    panel = np.zeros((100, 400), dtype=np.uint8)
    cv.imshow(win_controls, panel)

    # Inicializar SORT
    # max_age     = cuántos frames mantiene una ID sin detectarse
    # min_hits    = cuántas detecciones seguidas antes de asignar ID
    # iou_threshold = qué tan parecidas deben ser las cajas para asociarlas
    tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

    # Colores únicos por ID
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

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
        if min_val >= max_val: min_val = max(0, max_val - 1)

        # Pipeline de segmentación
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (blur_k, blur_k), 0)
        mask = cv.inRange(blur, min_val, max_val)

        kernel_close = np.ones((close_k, close_k), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_close)
        kernel_open = np.ones((open_k, open_k), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open)

        # Extraer bounding boxes desde contornos
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < 500:  # Ignorar manchas muy pequeñas (ruido)
                continue
            x, y, w, h = cv.boundingRect(cnt)
            detections.append([x, y, x + w, y + h, 1.0])  # [x1,y1,x2,y2, score]

        detections = np.array(detections) if detections else np.empty((0, 5))

        # Actualizar SORT
        tracked = tracker.update(detections)

        if mostrar_filtros:
            display = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        else:
            display = frame.copy()

        # Dibujar resultados del tracker
        for obj in tracked:
            x1, y1, x2, y2, obj_id = map(int, obj)
            color = colors[obj_id % 100].tolist()

            cv.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv.putText(display, f'ID {obj_id}', (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Centroide
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv.circle(display, (cx, cy), 5, color, -1)

        modo = 'SEGMENTACION' if mostrar_filtros else 'ORIGINAL'
        cv.putText(display, f'MODO: {modo} [F para cambiar]', (10, 30),
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