import cv2 as cv
import numpy as np
from sort import Sort
from collections import defaultdict

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
    cv.moveWindow(win_video, 0, 0)
    cv.moveWindow(win_controls, 710, 0)

    cv.createTrackbar('Blur',       win_controls, 5,  31,  callback)
    cv.createTrackbar('Min_thresh', win_controls, 0,  255, callback)
    cv.createTrackbar('Max_thresh', win_controls, 45, 255, callback)
    cv.createTrackbar('Close',      win_controls, 17,  20,  callback)
    cv.createTrackbar('Open',       win_controls, 3,  20,  callback)

    panel = np.zeros((100, 400), dtype=np.uint8)
    cv.imshow(win_controls, panel)

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(200, 3), dtype=np.uint8)

    # Historial por ID
    historial_centroides = defaultdict(list)  # ID → [(cx, cy), ...]
    historial_areas      = defaultdict(list)  # ID → [area, ...]
    MAX_HISTORIAL        = 30                 # cuántos frames guardamos

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

        # Extraer bounding boxes
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < 500:
                continue
            x, y, w, h = cv.boundingRect(cnt)
            detections.append([x, y, x + w, y + h, 1.0])

        detections = np.array(detections) if detections else np.empty((0, 5))

        tracked = tracker.update(detections)

        if mostrar_filtros:
            display = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        else:
            display = frame.copy()

        for obj in tracked:
            x1, y1, x2, y2, obj_id = map(int, obj)
            color = colors[obj_id % 200].tolist()

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            area_actual = (x2 - x1) * (y2 - y1)

            # Guardar historial
            historial_centroides[obj_id].append((cx, cy))
            historial_areas[obj_id].append(area_actual)

            # Limitar tamaño del historial
            if len(historial_centroides[obj_id]) > MAX_HISTORIAL:
                historial_centroides[obj_id].pop(0)
            if len(historial_areas[obj_id]) > MAX_HISTORIAL:
                historial_areas[obj_id].pop(0)

            # Detectar posible fusión: área mucho mayor al promedio histórico
            fusion_detectada = False
            if len(historial_areas[obj_id]) > 5:
                area_promedio = np.mean(historial_areas[obj_id][:-1])
                if area_actual > area_promedio * 1.7:
                    fusion_detectada = True

            # Dibujar bounding box
            grosor = 3 if fusion_detectada else 2
            cv.rectangle(display, (x1, y1), (x2, y2), color, grosor)

            # Etiqueta con aviso de fusión
            label = f'ID {obj_id}'
            if fusion_detectada:
                label += ' [FUSION]'
            cv.putText(display, label, (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Dibujar centroide
            cv.circle(display, (cx, cy), 5, color, -1)

            # Dibujar trayectoria
            puntos = historial_centroides[obj_id]
            for i in range(1, len(puntos)):
                cv.line(display, puntos[i - 1], puntos[i], color, 1)

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