import cv2 as cv
import numpy as np
from collections import defaultdict

mostrar_filtros = True
puntos_roi = []

def callback(val):
    pass

def mouse_callback(event, x, y, flags, param):
    global puntos_roi
    if event == cv.EVENT_LBUTTONDOWN:
        puntos_roi.append((x, y))
        print(f"Punto agregado: ({x}, {y}) — total: {len(puntos_roi)}")

def calibrar_roi(cap):
    global puntos_roi

    ret, frame = cap.read()
    if not ret:
        return None
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

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

        for i, punto in enumerate(puntos_roi):
            cv.circle(display, punto, 6, (0, 255, 0), -1)
            cv.putText(display, str(i + 1), (punto[0] + 8, punto[1] - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if len(puntos_roi) > 1:
            for i in range(1, len(puntos_roi)):
                cv.line(display, puntos_roi[i - 1], puntos_roi[i], (0, 255, 0), 2)

        if len(puntos_roi) > 2:
            cv.line(display, puntos_roi[-1], puntos_roi[0], (0, 255, 0), 2)
            overlay = display.copy()
            cv.fillPoly(overlay, [np.array(puntos_roi, dtype=np.int32)], (0, 100, 255))
            cv.addWeighted(overlay, 0.3, display, 0.7, 0, display)

        cv.putText(display, f'Puntos: {len(puntos_roi)} | Enter=confirmar Z=deshacer R=reiniciar',
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv.imshow(win_cal, display)

        key = cv.waitKey(1) & 0xFF
        if key == 13 and len(puntos_roi) >= 3:
            print(f"ROI confirmado con {len(puntos_roi)} puntos")
            cv.destroyWindow(win_cal)
            return np.array(puntos_roi, dtype=np.int32)
        elif key == ord('z') and puntos_roi:
            puntos_roi.pop()
        elif key == ord('r'):
            puntos_roi = []


class BlobTracker:
    def __init__(self, max_edad_invisible=150, vel_history=10,
                 max_dist=150, peso_dist=0.7, peso_angulo=0.3):
        self.tracks      = {}
        self.next_id     = 1
        self.max_edad    = max_edad_invisible
        self.vel_history = vel_history
        self.max_dist    = max_dist
        self.peso_dist   = peso_dist
        self.peso_angulo = peso_angulo

    def _angulo(self, vx, vy):
        return np.degrees(np.arctan2(vy, vx))

    def _diff_angulo(self, a1, a2):
        diff = abs(a1 - a2) % 360
        return diff if diff <= 180 else 360 - diff

    def _predecir(self):
        for t in self.tracks.values():
            t['cx_pred'] = int(t['cx'] + t['vx'])
            t['cy_pred'] = int(t['cy'] + t['vy'])

    def _calcular_velocidad(self, historial):
        if len(historial) < 2:
            return 0.0, 0.0
        vxs = [historial[i][0] - historial[i-1][0] for i in range(1, len(historial))]
        vys = [historial[i][1] - historial[i-1][1] for i in range(1, len(historial))]
        return float(np.mean(vxs)), float(np.mean(vys))

    def actualizar(self, centroides):
        self._predecir()

        ids_activos     = list(self.tracks.keys())
        asignados_track = set()
        asignados_det   = set()

        if ids_activos and centroides:
            scores = np.full((len(centroides), len(ids_activos)), np.inf)

            for di, (cx, cy) in enumerate(centroides):
                for ti, tid in enumerate(ids_activos):
                    t = self.tracks[tid]
                    dist = np.sqrt((cx - t['cx_pred'])**2 + (cy - t['cy_pred'])**2)
                    if dist > self.max_dist:
                        continue
                    vel_mag = np.sqrt(t['vx']**2 + t['vy']**2)
                    if vel_mag > 1.0:
                        angulo_track = self._angulo(t['vx'], t['vy'])
                        angulo_det   = self._angulo(cx - t['cx'], cy - t['cy'])
                        ang_score    = self._diff_angulo(angulo_track, angulo_det) / 180.0
                    else:
                        ang_score = 0.0
                    scores[di, ti] = (self.peso_dist   * (dist / self.max_dist) +
                                      self.peso_angulo * ang_score)

            while True:
                if np.all(np.isinf(scores)):
                    break
                di, ti = np.unravel_index(np.argmin(scores), scores.shape)
                if scores[di, ti] == np.inf:
                    break
                tid = ids_activos[ti]
                asignados_det.add(di)
                asignados_track.add(tid)
                scores[di, :] = np.inf
                scores[:, ti] = np.inf

                cx, cy = centroides[di]
                t = self.tracks[tid]
                t['historial'].append((cx, cy))
                if len(t['historial']) > self.vel_history:
                    t['historial'].pop(0)
                t['vx'], t['vy']    = self._calcular_velocidad(t['historial'])
                t['cx']             = cx
                t['cy']             = cy
                t['visible']        = True
                t['edad_invisible'] = 0

        for di, (cx, cy) in enumerate(centroides):
            if di not in asignados_det:
                self.tracks[self.next_id] = {
                    'cx': cx, 'cy': cy,
                    'cx_pred': cx, 'cy_pred': cy,
                    'vx': 0.0, 'vy': 0.0,
                    'historial': [(cx, cy)],
                    'visible': True,
                    'edad_invisible': 0
                }
                self.next_id += 1

        for tid in ids_activos:
            if tid not in asignados_track:
                t = self.tracks[tid]
                t['visible']        = False
                t['edad_invisible'] += 1
                t['cx'] = t['cx_pred']
                t['cy'] = t['cy_pred']

        muertos = [tid for tid, t in self.tracks.items()
                   if t['edad_invisible'] > self.max_edad]
        for tid in muertos:
            del self.tracks[tid]

        return [(t['cx'], t['cy'], tid, t['vx'], t['vy'])
                for tid, t in self.tracks.items() if t['visible']]


def cannyEdge():
    global mostrar_filtros

    cap = cv.VideoCapture('video.mp4')
    if not cap.isOpened():
        print("No se pudo abrir el video")
        return

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
    cv.createTrackbar('Close',      win_controls, 17, 20,  callback)
    cv.createTrackbar('Open',       win_controls, 3,  20,  callback)

    panel = np.zeros((100, 400), dtype=np.uint8)
    cv.imshow(win_controls, panel)

    tracker = BlobTracker(
        max_edad_invisible = 50,
        vel_history        = 10,
        max_dist           = 50,
        peso_dist          = 0.6,
        peso_angulo        = 0.4
    )

    np.random.seed(42)
    colors        = np.random.randint(0, 255, size=(500, 3), dtype=np.uint8)
    historial_areas = defaultdict(list)
    MAX_HISTORIAL   = 50

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

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (blur_k, blur_k), 0)
        mask = cv.inRange(blur, min_val, max_val)

        kernel_close = np.ones((close_k, close_k), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_close)
        kernel_open  = np.ones((open_k, open_k), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open)

        roi_mask = np.zeros_like(mask)
        cv.fillPoly(roi_mask, [roi_poly], 255)
        mask = cv.bitwise_and(mask, roi_mask)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        centroides = []
        bboxes     = {}

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < 400:
                continue
            x, y, w, h = cv.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            centroides.append((cx, cy))
            bboxes[(cx, cy)] = (x, y, x + w, y + h, area)

        resultado = tracker.actualizar(centroides)

        if mostrar_filtros:
            display = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        else:
            display = frame.copy()

        cv.polylines(display, [roi_poly], isClosed=True,
                     color=(0, 255, 255), thickness=2)


        # Tracks visibles
        for (cx, cy, tid, vx, vy) in resultado:
            color = colors[tid % 500].tolist()

            bbox_key = min(bboxes.keys(),
                           key=lambda k: abs(k[0]-cx) + abs(k[1]-cy),
                           default=None)

            if bbox_key:
                x1, y1, x2, y2, area_actual = bboxes[bbox_key]

                historial_areas[tid].append(area_actual)
                if len(historial_areas[tid]) > MAX_HISTORIAL:
                    historial_areas[tid].pop(0)

                fusion_detectada = False
                if len(historial_areas[tid]) > 5:
                    area_prom = np.mean(historial_areas[tid][:-1])
                    if area_actual > area_prom * 2.0:
                        fusion_detectada = True

                grosor = 3 if fusion_detectada else 2
                cv.rectangle(display, (x1, y1), (x2, y2), color, grosor)

                label = f'ID {tid}'
                if fusion_detectada:
                    label += ' [FUSION]'
                cv.putText(display, label, (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Centroide
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