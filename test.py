import cv2 as cv
import numpy as np
from sort import Sort
from collections import defaultdict

mostrar_filtros = True
puntos_roi = []
calibrando = True

def callback(val):
    pass

def mouse_callback(event, x, y, flags, param):
    global puntos_roi
    if event == cv.EVENT_LBUTTONDOWN:
        puntos_roi.append((x, y))
        print(f"Punto agregado: ({x}, {y}) — total: {len(puntos_roi)}")

def calibrar_roi(cap):
    global puntos_roi, calibrando

    ret, frame = cap.read()
    if not ret:
        return None

    cap.set(cv.CAP_PROP_POS_FRAMES, 0)  # Rebobinar

    win_cal = 'Calibracion - Haz clic en las esquinas del area jugable'
    cv.namedWindow(win_cal, cv.WINDOW_NORMAL)
    cv.setMouseCallback(win_cal, mouse_callback)

    print("=== MODO CALIBRACION ===")
    print("Haz clic en las esquinas del area jugable en orden")
    print("Presiona [Enter] para confirmar")
    print("Presiona [Z] para deshacer el ultimo punto")
    print("Presiona [R] para reiniciar")

    while True:
        display = frame.copy()

        # Dibujar puntos colocados
        for i, punto in enumerate(puntos_roi):
            cv.circle(display, punto, 6, (0, 255, 0), -1)
            cv.putText(display, str(i + 1), (punto[0] + 8, punto[1] - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Dibujar líneas entre puntos
        if len(puntos_roi) > 1:
            for i in range(1, len(puntos_roi)):
                cv.line(display, puntos_roi[i - 1], puntos_roi[i], (0, 255, 0), 2)

        # Cerrar el polígono si hay 3+ puntos
        if len(puntos_roi) > 2:
            cv.line(display, puntos_roi[-1], puntos_roi[0], (0, 255, 0), 2)
            overlay = display.copy()
            cv.fillPoly(overlay, [np.array(puntos_roi, dtype=np.int32)], (0, 100, 255))
            cv.addWeighted(overlay, 0.3, display, 0.7, 0, display)

        # Instrucciones en pantalla
        cv.putText(display, f'Puntos: {len(puntos_roi)} | Enter=confirmar Z=deshacer R=reiniciar',
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv.imshow(win_cal, display)

        key = cv.waitKey(1) & 0xFF
        if key == 13 and len(puntos_roi) >= 3:  # Enter
            print(f"ROI confirmado con {len(puntos_roi)} puntos")
            cv.destroyWindow(win_cal)
            return np.array(puntos_roi, dtype=np.int32)
        elif key == ord('z') and puntos_roi:    # Deshacer
            puntos_roi.pop()
            print(f"Punto eliminado — total: {len(puntos_roi)}")
        elif key == ord('r'):                   # Reiniciar
            puntos_roi = []
            print("Puntos reiniciados")

def cannyEdge():
    global mostrar_filtros

    cap = cv.VideoCapture('video.mp4')

    if not cap.isOpened():
        print("No se pudo abrir el video")
        return

    # Calibración antes del tracking
    roi_poly = calibrar_roi(cap)
    if roi_poly is None:
        print("Error en calibración")
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

    # Max_age: Cuántos frames esperar antes de eliminar un track perdido
    # Min_hits: Cuántos frames esperar antes de marcar un track perdido
    # IoU_threshold: Umbral de IOU para considerar un track perdido
    tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(200, 3), dtype=np.uint8)

    historial_centroides = defaultdict(list)
    historial_areas      = defaultdict(list)
    MAX_HISTORIAL        = 50

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

        # Aplicar ROI calibrado
        roi_mask = np.zeros_like(mask)
        cv.fillPoly(roi_mask, [roi_poly], 255)
        mask = cv.bitwise_and(mask, roi_mask)

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

        # Dibujar ROI en el display
        cv.polylines(display, [roi_poly], isClosed=True, color=(0, 255, 255), thickness=2)

        for obj in tracked:
            x1, y1, x2, y2, obj_id = map(int, obj)
            color = colors[obj_id % 200].tolist()

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            area_actual = (x2 - x1) * (y2 - y1)

            historial_centroides[obj_id].append((cx, cy))
            historial_areas[obj_id].append(area_actual)

            if len(historial_centroides[obj_id]) > MAX_HISTORIAL:
                historial_centroides[obj_id].pop(0)
            if len(historial_areas[obj_id]) > MAX_HISTORIAL:
                historial_areas[obj_id].pop(0)

            fusion_detectada = False
            if len(historial_areas[obj_id]) > 5:
                area_promedio = np.mean(historial_areas[obj_id][:-1])
                if area_actual > area_promedio * 1.4:  # Si el área actual es mucho mayor que el promedio reciente
                    fusion_detectada = True

            grosor = 3 if fusion_detectada else 2
            cv.rectangle(display, (x1, y1), (x2, y2), color, grosor)

            label = f'ID {obj_id}'
            if fusion_detectada:
                label += ' [FUSION]'
            cv.putText(display, label, (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv.circle(display, (cx, cy), 5, color, -1)

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