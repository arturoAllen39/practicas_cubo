# 12 de Marzo del 2026
import cv2 as cv
import numpy as np
from collections import defaultdict, deque

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


# ─────────────────────────────────────────────
#  CLASIFICADOR DE BORDES DEL ROI
# ─────────────────────────────────────────────
def clasificar_bordes(roi_poly):
    
    """
    Dado un polígono de 4 puntos, clasifica cada segmento
    como 'superior', 'inferior', 'izquierdo' o 'derecho'.
    Devuelve un dict con el segmento (p1, p2) de cada borde.
    """
    puntos = roi_poly.tolist()
    n      = len(puntos)

    # Calcular el centro del polígono
    cx = np.mean([p[0] for p in puntos])
    cy = np.mean([p[1] for p in puntos])

    segmentos = []
    for i in range(n):
        p1  = puntos[i]
        p2  = puntos[(i + 1) % n]
        mx  = (p1[0] + p2[0]) / 2  # punto medio del segmento
        my  = (p1[1] + p2[1]) / 2
        segmentos.append({'p1': p1, 'p2': p2, 'mx': mx, 'my': my})

    # Clasificar por posición relativa al centro
    # El segmento cuyo punto medio esté más arriba  → superior
    # El segmento cuyo punto medio esté más abajo   → inferior
    # El segmento cuyo punto medio esté más a la izq → izquierdo
    # El segmento cuyo punto medio esté más a la der → derecho
    bordes = {}
    bordes['superior']   = min(segmentos, key=lambda s: s['my'])
    bordes['inferior']   = max(segmentos, key=lambda s: s['my'])
    bordes['izquierdo']  = min(segmentos, key=lambda s: s['mx'])
    bordes['derecho']    = max(segmentos, key=lambda s: s['mx'])

    print("=== BORDES DETECTADOS ===")
    for nombre, seg in bordes.items():
        print(f"  {nombre}: {seg['p1']} → {seg['p2']}")

    return bordes


def punto_cerca_segmento(px, py, p1, p2, umbral=50):
    """
    Devuelve True si el punto (px, py) está a menos de
    'umbral' píxeles del segmento p1→p2.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    largo2 = dx*dx + dy*dy

    if largo2 == 0:
        dist = np.sqrt((px - x1)**2 + (py - y1)**2)
    else:
        t      = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / largo2))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        dist   = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    return dist <= umbral


def detectar_borde(cx, cy, bordes, umbral=50):
    """
    Devuelve el nombre del borde ('superior', 'inferior',
    'izquierdo', 'derecho') si el punto está cerca de él,
    o None si está lejos de todos los bordes.
    """
    for nombre, seg in bordes.items():
        if punto_cerca_segmento(cx, cy, seg['p1'], seg['p2'], umbral):
            return nombre
    return None


# ─────────────────────────────────────────────
#  SISTEMA DE RESERVAS FIFO
# ─────────────────────────────────────────────
class ReservaBordes:
    def __init__(self, max_ids=21):
        self.max_ids  = max_ids
        self.reservas = {
            'superior':  deque(),
            'inferior':  deque(),
            'izquierdo': deque(),
            'derecho':   deque()
        }
        self.ids_en_reserva = set()   # IDs que están en alguna reserva
        self.ids_activos    = set()   # IDs dentro del área jugable
        self.total_ids      = 0       # contador global de IDs creadas

    def reset(self):
        """Reinicia todas las reservas y contadores."""
        for cola in self.reservas.values():
            cola.clear()
        self.ids_en_reserva.clear()
        self.ids_activos.clear()
        self.total_ids = 0
        print("=== RESERVAS REINICIADAS ===")

    def registrar_salida(self, track_id, borde):
        """Cuando un blob sale por un borde, guarda su ID en la reserva."""
        if track_id not in self.ids_en_reserva:
            self.reservas[borde].append(track_id)
            self.ids_en_reserva.add(track_id)
            self.ids_activos.discard(track_id)
            print(f"[SALIDA] ID {track_id} → reserva {borde} "
                  f"(reserva {borde}: {list(self.reservas[borde])})")

    def registrar_entrada(self, borde):
        
        # Cuando un blob entra por un borde, devuelve el ID
        # que le corresponde. Si la reserva está vacía y no
        # se superó el límite, crea un ID nuevo.
        
        if self.reservas[borde]:
            # Recuperar el primer ID de la reserva (FIFO)
            track_id = self.reservas[borde].popleft()
            self.ids_en_reserva.discard(track_id)
            self.ids_activos.add(track_id)
            print(f"[ENTRADA] borde {borde} → recupera ID {track_id}")
            return track_id

        # Reserva vacía → crear ID nuevo si no superamos el límite
        total_en_uso = len(self.ids_activos) + len(self.ids_en_reserva)
        if total_en_uso < self.max_ids:
            self.total_ids += 1
            nuevo_id = self.total_ids
            self.ids_activos.add(nuevo_id)
            print(f"[ENTRADA] borde {borde} → ID nueva {nuevo_id}")
            return nuevo_id

        print(f"[ENTRADA] borde {borde} → límite de {self.max_ids} IDs alcanzado")
        return None

    def confirmar_activo(self, track_id):
        # Marca un ID como activo dentro del área jugable
        self.ids_activos.add(track_id)
        self.ids_en_reserva.discard(track_id)

    def estado(self):
        # Imprime el estado actual de las reserva
        for borde, cola in self.reservas.items():
            if cola:
                print(f"  Reserva {borde}: {list(cola)}")


# ─────────────────────────────────────────────
#  BLOB TRACKER
# ─────────────────────────────────────────────
class BlobTracker:
    def __init__(
                self, 
                max_edad_invisible = 350,
                vel_history        = 200,
                max_dist           = 30,
                peso_dist          = 0.4,
                peso_angulo        = 0.6
                 ):
        self.tracks      = {}
        self.next_id     = 1
        self.max_edad    = max_edad_invisible
        self.vel_history = vel_history
        self.max_dist    = max_dist
        self.peso_dist   = peso_dist
        self.peso_angulo = peso_angulo
        self.fusiones_activas = {}  # blob_id → [lista de IDs absorbidas con su ultima pos]


    def reset(self):
        # Reinicia todos los tracks.
        self.tracks           = {}
        self.next_id          = 1
        self.fusiones_activas = {}
        print("=== TRACKER REINICIADO ===")

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
        vx = float(np.mean(vxs))
        vy = float(np.mean(vys))
        # Limitar velocidad maxima a 5px por frame
        MAX_VEL = 3  
        vx = max(-MAX_VEL, min(MAX_VEL, vx))
        vy = max(-MAX_VEL, min(MAX_VEL, vy))
        return vx, vy

    def agregar_track(self, track_id, cx, cy):
        """Agrega un track con ID específica (para recuperaciones de reserva)."""
        self.tracks[track_id] = {
            'cx': cx, 'cy': cy,
            'cx_pred': cx, 'cy_pred': cy,
            'vx': 0.0, 'vy': 0.0,
            'historial': [(cx, cy)],
            'visible': True,
            'edad_invisible': 0
        }

    def actualizar(self, centroides, reserva, bordes, umbral_borde=10):
        self._predecir()

        ids_activos     = list(self.tracks.keys())
        asignados_track = set()
        asignados_det   = set()

        if ids_activos and centroides:
            scores = np.full((len(centroides), len(ids_activos)), np.inf)

            for di, (cx, cy) in enumerate(centroides):
                for ti, tid in enumerate(ids_activos):
                    t    = self.tracks[tid]
                    dist = np.sqrt((cx - t['cx_pred'])**2 +
                                   (cy - t['cy_pred'])**2)
                    if dist > self.max_dist:
                        continue
                    vel_mag = np.sqrt(t['vx']**2 + t['vy']**2)
                    if vel_mag > 1.0:
                        ang_track = self._angulo(t['vx'], t['vy'])
                        ang_det   = self._angulo(cx - t['cx'], cy - t['cy'])
                        ang_score = self._diff_angulo(ang_track, ang_det) / 180.0
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

        # Detecciones sin asignar → revisar si vienen de fusión o entran por borde
        for di, (cx, cy) in enumerate(centroides):
            if di not in asignados_det:

                # Buscar si este blob nuevo está cerca de alguna fusión activa
                id_recuperado = None
                fusion_origen = None
                mejor_dist    = self.max_dist * 2  # radio más amplio para fusiones

                for blob_id, ids_absorbidas in list(self.fusiones_activas.items()):
                    for entrada in ids_absorbidas:
                        tid_abs  = entrada['id']
                        ux, uy   = entrada['ultima_pos']
                        dist     = np.sqrt((cx - ux)**2 + (cy - uy)**2)
                        if dist < mejor_dist:
                            mejor_dist    = dist
                            id_recuperado = tid_abs
                            fusion_origen = blob_id

                if id_recuperado is not None:
                    # Viene de una fusión → recuperar ID absorbida
                    ids_absorbidas = self.fusiones_activas[fusion_origen]
                    self.fusiones_activas[fusion_origen] = [
                        e for e in ids_absorbidas if e['id'] != id_recuperado
                    ]
                    if not self.fusiones_activas[fusion_origen]:
                        del self.fusiones_activas[fusion_origen]
                    print(f"[FUSION SEPARACION] blob en ({cx},{cy}) → recupera ID {id_recuperado}")
                    self.agregar_track(id_recuperado, cx, cy)
                    reserva.confirmar_activo(id_recuperado)
                    self.next_id = max(self.next_id, id_recuperado + 1)
                    continue

                # No viene de fusión → revisar si entra por un borde
                borde_entrada = detectar_borde(cx, cy, bordes, umbral_borde)

                if borde_entrada:
                    nuevo_id = reserva.registrar_entrada(borde_entrada)
                else:
                    nuevo_id = reserva.registrar_entrada('izquierdo') \
                               if not any(reserva.reservas.values()) else None

                if nuevo_id is None:
                    total_en_uso = len(reserva.ids_activos) + len(reserva.ids_en_reserva)
                    if total_en_uso < reserva.max_ids:
                        nuevo_id = self.next_id
                    else:
                        continue

                self.agregar_track(nuevo_id, cx, cy)
                reserva.confirmar_activo(nuevo_id)
                self.next_id = max(self.next_id, nuevo_id + 1)

        # Tracks sin asignar → fantasma, fusión o salida
        for tid in ids_activos:
            if tid not in asignados_track:
                t = self.tracks[tid]
                t['visible']        = False
                t['edad_invisible'] += 1

                # Guardar ultima posicion real antes de mover a prediccion
                ultima_cx = t['cx']
                ultima_cy = t['cy']

                # Reducir velocidad gradualmente cuando esta invisible
                t['vx'] *= 0.50
                t['vy'] *= 0.50

                t['cx'] = t['cx_pred']
                t['cy'] = t['cy_pred']

                # Verificar si este track desapareció cerca de un blob fusionado
                if t['edad_invisible'] == 1:
                    for blob_id_asignado in asignados_track:
                        bt = self.tracks[blob_id_asignado]
                        dist_al_blob = np.sqrt((ultima_cx - bt['cx'])**2 +
                                               (ultima_cy - bt['cy'])**2)
                        if dist_al_blob < self.max_dist * 2:
                            # Este track fue absorbido por blob_id_asignado
                            if blob_id_asignado not in self.fusiones_activas:
                                self.fusiones_activas[blob_id_asignado] = []
                            # Evitar duplicados
                            ids_ya_guardadas = [e['id'] for e in self.fusiones_activas[blob_id_asignado]]
                            if tid not in ids_ya_guardadas:
                                self.fusiones_activas[blob_id_asignado].append({
                                    'id':        tid,
                                    'ultima_pos': (ultima_cx, ultima_cy)
                                })
                                print(f"[FUSION] ID {tid} absorbida por blob {blob_id_asignado} "
                                      f"en ({ultima_cx}, {ultima_cy})")
                            break

                # Revisar borde con posicion real, no con prediccion
                borde_salida = detectar_borde(ultima_cx, ultima_cy,
                                              bordes, umbral_borde)
                if borde_salida and t['edad_invisible'] == 5:
                    apunta_afuera = False
                    if borde_salida == 'inferior'  and t['vy'] > 0:
                        apunta_afuera = True
                    elif borde_salida == 'superior' and t['vy'] < 0:
                        apunta_afuera = True
                    elif borde_salida == 'derecho'  and t['vx'] > 0:
                        apunta_afuera = True
                    elif borde_salida == 'izquierdo' and t['vx'] < 0:
                        apunta_afuera = True

                    if apunta_afuera:
                        reserva.registrar_salida(tid, borde_salida)
        # Limpiar fantasmas viejos
        muertos = [tid for tid, t in self.tracks.items()
                   if t['edad_invisible'] > self.max_edad]
        for tid in muertos:
            del self.tracks[tid]

        return [(t['cx'], t['cy'], tid, t['vx'], t['vy'])
                for tid, t in self.tracks.items() if t['visible']]


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def cannyEdge():
    global mostrar_filtros

    cap = cv.VideoCapture('video3.mp4')
    fps = cap.get(cv.CAP_PROP_FPS)
    delay = int(500 / fps) if fps > 0 else 30
    if not cap.isOpened():
        print("No se pudo abrir el video")
        return

    roi_poly = calibrar_roi(cap)
    if roi_poly is None:
        print("Error en calibración")
        return

    # Clasificar bordes automáticamente
    bordes  = clasificar_bordes(roi_poly)
    bordes = clasificar_bordes(roi_poly)
    for nombre, seg in bordes.items():
        print(f"Borde {nombre}: {seg['p1']} → {seg['p2']}")
    reserva = ReservaBordes(max_ids=21)

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
        max_edad_invisible = 350,
        vel_history        = 200,
        max_dist           = 30,
        peso_dist          = 0.4,
        peso_angulo        = 0.6
    )

    np.random.seed(42)
    colors          = np.random.randint(0, 255, size=(500, 3), dtype=np.uint8)
    historial_areas = defaultdict(list)
    MAX_HISTORIAL   = 50
    UMBRAL_BORDE    = 30
    video_terminado = False

    print("Presiona [F] para alternar entre vista original y con filtros")
    print("Presiona [Q] para salir")

    while True:
        ret, frame = cap.read()

        if not ret:
            if not video_terminado:
                video_terminado = True
                print("=== VIDEO TERMINADO — REINICIANDO ===")
                # Reiniciar todo
                cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                tracker.reset()
                reserva.reset()
                historial_areas.clear()
                video_terminado = False
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

        resultado = tracker.actualizar(centroides, reserva, bordes, UMBRAL_BORDE)

        if mostrar_filtros:
            display = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        else:
            display = frame.copy()

        cv.polylines(display, [roi_poly], isClosed=True,
                     color=(0, 255, 255), thickness=2)

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

            cv.circle(display, (cx, cy), 5, color, -1)

        modo = 'SEGMENTACION' if mostrar_filtros else 'ORIGINAL'
        cv.putText(display, f'MODO: {modo} [F para cambiar]', (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv.imshow(win_video, display)

        key = cv.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            mostrar_filtros = not mostrar_filtros

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    cannyEdge()