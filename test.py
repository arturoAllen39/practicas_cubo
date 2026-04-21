# Codigo Nuevo
import cv2 as cv
import numpy as np
from collections import defaultdict, deque
from config import *

mostrar_filtros = True
puntos_roi = []

def callback(val):
    pass

def mouse_callback(event, x, y, flags, param):
    global puntos_roi
    if event == cv.EVENT_LBUTTONDOWN:
        puntos_roi.append((x, y))

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
    print("Presiona [Enter] para confirmar | [Z] deshacer | [R] reiniciar")

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
    """Clasifica cada segmento del polígono como superior/inferior/izquierdo/derecho."""
    puntos = roi_poly.tolist()
    n      = len(puntos)

    segmentos = []
    for i in range(n):
        p1 = puntos[i]
        p2 = puntos[(i + 1) % n]
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2
        segmentos.append({'p1': p1, 'p2': p2, 'mx': mx, 'my': my})

    bordes = {}
    bordes['superior']  = min(segmentos, key=lambda s: s['my'])
    bordes['inferior']  = max(segmentos, key=lambda s: s['my'])
    bordes['izquierdo'] = min(segmentos, key=lambda s: s['mx'])
    bordes['derecho']   = max(segmentos, key=lambda s: s['mx'])

    return bordes


def punto_cerca_segmento(px, py, p1, p2, umbral=50):
    """Devuelve True si el punto está a menos de 'umbral' píxeles del segmento."""
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
    """Devuelve el nombre del borde más cercano al punto, o None si está lejos de todos."""
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
        self.ids_en_reserva = set()  # IDs esperando reentrada
        self.ids_activos    = set()  # IDs dentro del área jugable
        self.total_ids      = 0      # contador global

    def reset(self):
        for cola in self.reservas.values():
            cola.clear()
        self.ids_en_reserva.clear()
        self.ids_activos.clear()
        self.total_ids = 0

    def registrar_salida(self, track_id, borde):
        """Guarda el ID en la reserva del borde por el que salió."""
        if track_id not in self.ids_en_reserva:
            self.reservas[borde].append(track_id)
            self.ids_en_reserva.add(track_id)
            self.ids_activos.discard(track_id)

    def registrar_entrada(self, borde):
        """Recupera el ID de la reserva FIFO, o crea uno nuevo si hay cupo."""
        if self.reservas[borde]:
            track_id = self.reservas[borde].popleft()
            self.ids_en_reserva.discard(track_id)
            self.ids_activos.add(track_id)
            return track_id

        total_en_uso = len(self.ids_activos) + len(self.ids_en_reserva)
        if total_en_uso < self.max_ids:
            self.total_ids += 1
            nuevo_id = self.total_ids
            self.ids_activos.add(nuevo_id)
            return nuevo_id

        return None

    def confirmar_activo(self, track_id):
        self.ids_activos.add(track_id)
        self.ids_en_reserva.discard(track_id)


# ─────────────────────────────────────────────
#  BLOB TRACKER
# ─────────────────────────────────────────────
class BlobTracker:
    def __init__(
                self,
                max_edad_invisible = 350,
                vel_history        = 10,
                max_dist           = 30,
                peso_dist          = 0.4,
                peso_angulo        = 0.6
                 ):
        self.tracks           = {}
        self.next_id          = 1
        self.max_edad         = max_edad_invisible
        self.vel_history      = vel_history
        self.max_dist         = max_dist
        self.peso_dist        = peso_dist
        self.peso_angulo      = peso_angulo
        self.fusiones_activas = {}           # blob_id → [lista de IDs absorbidas]
        self.frames_juntos    = defaultdict(int)
        self.pares_fusionados = set()        # pares (id_menor, id_mayor) fusionados

    def reset(self):
        self.tracks           = {}
        self.next_id          = 1
        self.fusiones_activas = {}
        self.frames_juntos    = defaultdict(int)
        self.pares_fusionados = set()

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
        
        vx = max(-MAX_VEL, min(MAX_VEL, vx))
        vy = max(-MAX_VEL, min(MAX_VEL, vy))
        return vx, vy

    def agregar_track(self, track_id, cx, cy):
        self.tracks[track_id] = {
            'cx': cx, 'cy': cy,
            'cx_pred': cx, 'cy_pred': cy,
            'vx': 0.0, 'vy': 0.0,
            'ultima_dir_x': 0.0,    
            'ultima_dir_y': 0.0,    
            'historial': [(cx, cy)],
            'visible': True,
            'edad_invisible': 0,
            'area_hist': deque(maxlen=20)   
        }

    def _actualizar_proximidad(self, umbral_fusion=40):
        """Detecta pares de tracks visibles que llevan 3+ frames consecutivos muy juntos."""
        tracks_visibles  = [(tid, t) for tid, t in self.tracks.items() if t['visible']]
        pares_cerca_ahora = set()

        for i in range(len(tracks_visibles)):
            tid_a, t_a = tracks_visibles[i]
            for j in range(i + 1, len(tracks_visibles)):
                tid_b, t_b = tracks_visibles[j]
                dist = np.sqrt((t_a['cx'] - t_b['cx'])**2 +
                               (t_a['cy'] - t_b['cy'])**2)
                if dist < umbral_fusion:
                    par = (min(tid_a, tid_b), max(tid_a, tid_b))
                    pares_cerca_ahora.add(par)
                    self.frames_juntos[par] += 1

        # Pares que ya no están cerca → limpiar
        for par in list(self.frames_juntos.keys()):
            if par not in pares_cerca_ahora:
                self.frames_juntos[par] = 0
                self.pares_fusionados.discard(par)

        # Pares estables → marcar como fusionados
        for par, frames in self.frames_juntos.items():
            if frames >= FRAMES_PARA_FUSION:
                self.pares_fusionados.add(par)
                
        

    def actualizar(self, centroides, reserva, bordes, bboxes, umbral_borde=10):
        self._predecir()

        ids_activos     = list(self.tracks.keys())
        asignados_track = set()
        asignados_det   = set()

        if ids_activos and centroides:
            scores = np.full((len(centroides), len(ids_activos)), np.inf)

            for di, (cx, cy) in enumerate(centroides):
                for ti, tid in enumerate(ids_activos):
                    t    = self.tracks[tid]
                    px, py = self._posicion_esperada(tid, pasos=3)
                    dist   = np.sqrt((cx - px)**2 + (cy - py)**2)

                    # IDs absorbidos en fusión no participan en el greedy
                    ids_absorbidos_global = set()
                    for abs_list in self.fusiones_activas.values():
                        for e in abs_list:
                            ids_absorbidos_global.add(e['id'])
                    if tid in ids_absorbidos_global:
                        continue

                    if dist > self.max_dist:
                        continue

                    vel_mag = np.sqrt(t['vx']**2 + t['vy']**2)
                    if vel_mag > 1.0:
                        ang_track = self._angulo(t['vx'], t['vy'])
                        ang_det   = self._angulo(cx - t['cx'], cy - t['cy'])
                        ang_score = self._diff_angulo(ang_track, ang_det) / 180.0
                    else:
                        ang_score = 0.0


                    area_score = 0.0
                    bbox_k = min(bboxes.keys(),
                                 key=lambda k: abs(k[0]-cx) + abs(k[1]-cy),
                                 default=None)
                    if bbox_k and t['area_hist']:
                        area_det  = bboxes[bbox_k][4]
                        area_prom = np.mean(t['area_hist'])
                        area_score = min(abs(area_det - area_prom) / (area_prom + 1e-5), 1.0)

                    dir_score = 0.0
                    # Solo penalizar si el track NO está en una fusión activa
                    tid_en_fusion = tid in self.fusiones_activas or any(
                        any(e['id'] == tid for e in lista)
                        for lista in self.fusiones_activas.values()
                    )
                    if not tid_en_fusion:
                        bt_dir_x = t['ultima_dir_x']
                        bt_dir_y = t['ultima_dir_y']
                        if bt_dir_x != 0.0 or bt_dir_y != 0.0:
                            mov_x = cx - t['cx']
                            mov_y = cy - t['cy']
                            mov_mag = np.sqrt(mov_x**2 + mov_y**2)
                            if mov_mag > 1.0:
                                mov_x /= mov_mag
                                mov_y /= mov_mag
                                producto = bt_dir_x * mov_x + bt_dir_y * mov_y
                                if producto < UMBRAL_DIR_OPUESTA:
                                    dir_score = PENALIZACION_DIR

                    scores[di, ti] = (0.35 * (dist / self.max_dist) +
                                      0.35 * ang_score          +
                                      0.15 * area_score         +
                                      0.15 * dir_score)

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
                vel_mag = np.sqrt(t['vx']**2 + t['vy']**2)
                if vel_mag > 0.5:                          # ── NUEVO ──
                    t['ultima_dir_x'] = t['vx'] / vel_mag  # ── NUEVO ──
                    t['ultima_dir_y'] = t['vy'] / vel_mag  # ── NUEVO ──
                t['cx']             = cx
                t['cy']             = cy
                t['visible']        = True
                t['edad_invisible'] = 0
                # Guardar área actual en el historial del track
                bbox_k = min(bboxes.keys(),
                             key=lambda k: abs(k[0]-cx) + abs(k[1]-cy),
                             default=None)
                if bbox_k and bbox_k in bboxes:
                    t['area_hist'].append(bboxes[bbox_k][4])

        # Detecciones sin asignar → revisar si vienen de fusión o entran por borde
        for di, (cx, cy) in enumerate(centroides):
            if di not in asignados_det:
                id_recuperado = None
                fusion_origen = None
                mejor_dist    = self.max_dist * 2

                for blob_id, ids_absorbidas in list(self.fusiones_activas.items()):
                    if blob_id not in self.tracks:
                        continue
                    ux = self.tracks[blob_id]['cx']
                    uy = self.tracks[blob_id]['cy']
                    for entrada in ids_absorbidas:
                        dist = np.sqrt((cx - ux)**2 + (cy - uy)**2)
                        if dist < mejor_dist:
                            mejor_dist    = dist
                            id_recuperado = entrada['id']
                            fusion_origen = blob_id

                if id_recuperado is not None:
                    candidatos = []
                    for blob_id, ids_absorbidas in list(self.fusiones_activas.items()):
                        for entrada in ids_absorbidas:
                            tid_abs = entrada['id']
                            if tid_abs in self.tracks:
                                ux, uy = entrada['ultima_pos']
                                # Usar dirección guardada al momento de absorción
                                dx = entrada.get('dir_x', 0.0)
                                dy = entrada.get('dir_y', 0.0)
                                px = ux + dx * self.max_dist
                                py = uy + dy * self.max_dist
                            else:
                                px, py = entrada['ultima_pos']
                            dist_proyectada = np.sqrt((cx - px)**2 + (cy - py)**2)
                            candidatos.append((dist_proyectada, tid_abs, blob_id))

                    if candidatos:
                        candidatos.sort(key=lambda x: x[0])
                        _, id_recuperado, fusion_origen = candidatos[0]

                    for fid in list(self.fusiones_activas.keys()):
                        self.fusiones_activas[fid] = [
                            e for e in self.fusiones_activas[fid] if e['id'] != id_recuperado
                        ]
                        if not self.fusiones_activas[fid]:
                            del self.fusiones_activas[fid]
                    self.agregar_track(id_recuperado, cx, cy)
                    reserva.confirmar_activo(id_recuperado)
                    self.next_id = max(self.next_id, id_recuperado + 1)
                    continue

                borde_entrada = detectar_borde(cx, cy, bordes, umbral_borde)

                if not borde_entrada:
                    # Blob lejos de bordes → intentar rescate de track cercano
                    mejor_rescue      = None
                    mejor_dist_rescue = self.max_dist * MULT_DIST_RESCATE
                    for tid_r, t_r in self.tracks.items():
                        if tid_r in asignados_track:
                            continue
                        if t_r['edad_invisible'] > MAX_EDAD_RESCATE:
                            continue
                        dist_r = np.sqrt((cx - t_r['cx'])**2 + (cy - t_r['cy'])**2)
                        if dist_r < mejor_dist_rescue:
                            mejor_dist_rescue = dist_r
                            mejor_rescue      = tid_r

                    if mejor_rescue is not None:
                        t_r = self.tracks[mejor_rescue]
                        t_r['historial'].append((cx, cy))
                        if len(t_r['historial']) > self.vel_history:
                            t_r['historial'].pop(0)
                        t_r['vx'], t_r['vy'] = self._calcular_velocidad(t_r['historial'])
                        t_r['cx']             = cx
                        t_r['cy']             = cy
                        t_r['visible']        = True
                        t_r['edad_invisible'] = 0
                        asignados_track.add(mejor_rescue)
                        asignados_det.add(di)
                        continue

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

        # Tracks sin asignar → fantasma, fusión o salida por borde
        for tid in ids_activos:
            if tid not in asignados_track:
                t = self.tracks[tid]
                t['visible']        = False
                t['edad_invisible'] += 1

                ultima_cx = t['cx']
                ultima_cy = t['cy']

                t['vx'] *= FACTOR_AMORTIGUACION
                t['vy'] *= FACTOR_AMORTIGUACION
                t['cx']  = t['cx_pred']
                t['cy']  = t['cy_pred']

                # Detectar absorción por blob cercano
                if t['edad_invisible'] <= 3:
                    for blob_id_asignado in asignados_track:
                        bt           = self.tracks[blob_id_asignado]
                        dist_al_blob = np.sqrt((ultima_cx - bt['cx'])**2 +
                                               (ultima_cy - bt['cy'])**2)
                        if dist_al_blob < self.max_dist * MULT_DIST_ABSORCION:
                            dir_x          = bt['cx'] - ultima_cx
                            dir_y          = bt['cy'] - ultima_cy
                            dist_inmediata = np.sqrt(dir_x**2 + dir_y**2)
                            if dist_inmediata > self.max_dist * MULT_DIST_ABSORCION:
                                dirigido = (t['vx'] * dir_x + t['vy'] * dir_y) >= 0
                                if not dirigido:
                                    continue

                            bt_dir_x = bt['ultima_dir_x']
                            bt_dir_y = bt['ultima_dir_y']
                            t_dir_x  = t['ultima_dir_x']
                            t_dir_y  = t['ultima_dir_y']
                            producto_punto = bt_dir_x * t_dir_x + bt_dir_y * t_dir_y
                            direcciones_opuestas = producto_punto < -0.5
                            if direcciones_opuestas and t['edad_invisible'] > 1:
                                continue

                            
                            if blob_id_asignado not in self.fusiones_activas:
                                self.fusiones_activas[blob_id_asignado] = []
                            ya_absorbido     = any(
                                any(e['id'] == tid for e in lista)
                                for lista in self.fusiones_activas.values()
                            )
                            ids_ya_guardadas = [e['id'] for e in self.fusiones_activas[blob_id_asignado]]
                            if tid not in ids_ya_guardadas and not ya_absorbido:
                                self.fusiones_activas[blob_id_asignado].append({
                                    'id':         tid,
                                    'ultima_pos': (ultima_cx, ultima_cy),
                                    'dir_x':      t['ultima_dir_x'],
                                    'dir_y':      t['ultima_dir_y']
                                })
                            break

                # Registrar salida si el track apunta hacia afuera del borde
                borde_salida = detectar_borde(ultima_cx, ultima_cy, bordes, umbral_borde)
                if borde_salida and t['edad_invisible'] == 5:
                    apunta_afuera = False
                    if borde_salida == 'inferior'   and t['vy'] > 0: apunta_afuera = True
                    elif borde_salida == 'superior'  and t['vy'] < 0: apunta_afuera = True
                    elif borde_salida == 'derecho'   and t['vx'] > 0: apunta_afuera = True
                    elif borde_salida == 'izquierdo' and t['vx'] < 0: apunta_afuera = True
                    if apunta_afuera:
                        reserva.registrar_salida(tid, borde_salida)

        # IDs absorbidos en fusión no acumulan edad_invisible
        ids_en_fusion = set()
        for ids_absorbidas in self.fusiones_activas.values():
            for entrada in ids_absorbidas:
                ids_en_fusion.add(entrada['id'])
        for tid in ids_en_fusion:
            if tid in self.tracks:
                self.tracks[tid]['edad_invisible'] = 0

        # Eliminar tracks viejos sin actividad, excepto los absorbidos
        muertos = [tid for tid, t in self.tracks.items()
                   if t['edad_invisible'] > self.max_edad and tid not in ids_en_fusion]
        for tid in muertos:
            del self.tracks[tid]

        self._actualizar_proximidad(umbral_fusion=UMBRAL_FUSION)

        if self.fusiones_activas:
            print(f"[FUSIONES] {dict({k: [e['id'] for e in v] for k, v in self.fusiones_activas.items()})}")

        return [(t['cx'], t['cy'], tid, t['vx'], t['vy'])
                for tid, t in self.tracks.items() if t['visible']]
    # Al final de actualizar(), antes del return
  
    def _posicion_esperada(self, track_id, pasos=3):
        """Proyecta la posición esperada usando los últimos N desplazamientos."""
        h = self.tracks[track_id]['historial']
        if len(h) < 2:
            return self.tracks[track_id]['cx'], self.tracks[track_id]['cy']
        # Promedio de los últimos 'pasos' desplazamientos
        n = min(pasos, len(h) - 1)
        dx = np.mean([h[-(i+1)][0] - h[-(i+2)][0] for i in range(n)])
        dy = np.mean([h[-(i+1)][1] - h[-(i+2)][1] for i in range(n)])
        return int(self.tracks[track_id]['cx'] + dx), int(self.tracks[track_id]['cy'] + dy)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def cannyEdge():
    global mostrar_filtros

    cap = cv.VideoCapture(VIDEO_PATH)
    fps   = cap.get(cv.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30
    if not cap.isOpened():
        print("No se pudo abrir el video")
        return

    roi_poly = calibrar_roi(cap)
    if roi_poly is None:
        print("Error en calibración")
        return

    bordes  = clasificar_bordes(roi_poly)
    reserva = ReservaBordes(max_ids=MAX_IDS)

    win_video    = 'Video'
    win_controls = 'Controles'

    cv.namedWindow(win_video,    cv.WINDOW_NORMAL)
    cv.namedWindow(win_controls, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_controls, 400, 200)
    cv.moveWindow(win_video,    0,   0)
    cv.moveWindow(win_controls, 710, 0)

    '''
    cv.createTrackbar('Blur',       win_controls, 5,  31,  callback)
    cv.createTrackbar('Min_thresh', win_controls, 0,  255, callback)
    cv.createTrackbar('Max_thresh', win_controls, 45, 255, callback)
    cv.createTrackbar('Close',      win_controls, 17, 20,  callback)
    cv.createTrackbar('Open',       win_controls, 3,  20,  callback)

    
    '''


    cv.createTrackbar('Blur',       win_controls, 10,  31,  callback)
    cv.createTrackbar('Min_thresh', win_controls, 0,  255, callback)
    cv.createTrackbar('Max_thresh', win_controls, 45, 255, callback)
    cv.createTrackbar('Close',      win_controls, 17, 20,  callback)
    cv.createTrackbar('Open',       win_controls, 6,  20,  callback)

    cv.imshow(win_controls, np.zeros((100, 400), dtype=np.uint8))

    tracker = BlobTracker(
        max_edad_invisible = MAX_EDAD_INVISIBLE,
        vel_history        = VEL_HISTORY,
        max_dist           = MAX_DIST,
        peso_dist          = PESO_DIST,
        peso_angulo        = PESO_ANGULO
    )

    np.random.seed(42)
    colors          = np.random.randint(0, 255, size=(500, 3), dtype=np.uint8)
    historial_areas = defaultdict(list)
    MAX_HISTORIAL   = MAX_HISTORIAL_AREAS
    # UMBRAL_BORDE    = 10  
    video_terminado = False

    print("Presiona [F] para alternar entre vista original y con filtros")
    print("Presiona [Q] para salir")

    while True:
        ret, frame = cap.read()

        if not ret:
            if not video_terminado:
                video_terminado = True
                cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                tracker.reset()
                reserva.reset()
                historial_areas.clear()
                video_terminado = False
            continue

        blur_k  = cv.getTrackbarPos('Blur',       win_controls)
        min_val = cv.getTrackbarPos('Min_thresh',  win_controls)
        max_val = cv.getTrackbarPos('Max_thresh',  win_controls)
        close_k = cv.getTrackbarPos('Close',       win_controls)
        open_k  = cv.getTrackbarPos('Open',        win_controls)

        if blur_k < 1:        blur_k  = 1
        if blur_k % 2 == 0:   blur_k += 1
        if close_k < 1:       close_k = 1
        if open_k < 1:        open_k  = 1
        if min_val >= max_val: min_val = max(0, max_val - 1)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (blur_k, blur_k), 0)
        mask = cv.inRange(blur, min_val, max_val)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  np.ones((open_k,  open_k),  np.uint8))

        roi_mask = np.zeros_like(mask)
        cv.fillPoly(roi_mask, [roi_poly], 255)
        mask = cv.bitwise_and(mask, roi_mask)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        centroides = []
        bboxes     = {}
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < AREA_MIN_CONTORNO:
                continue
            x, y, w, h = cv.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            centroides.append((cx, cy))
            bboxes[(cx, cy)] = (x, y, x + w, y + h, area)

        resultado = tracker.actualizar(centroides, reserva, bordes, bboxes, UMBRAL_BORDE)

        display = cv.cvtColor(mask, cv.COLOR_GRAY2BGR) if mostrar_filtros else frame.copy()
        cv.polylines(display, [roi_poly], isClosed=True, color=(0, 255, 255), thickness=2)

        for (cx, cy, tid, vx, vy) in resultado:
            color    = colors[tid % 500].tolist()
            bbox_key = min(bboxes.keys(),
                           key=lambda k: abs(k[0]-cx) + abs(k[1]-cy),
                           default=None)

            if bbox_key:
                x1, y1, x2, y2, area_actual = bboxes[bbox_key]

                historial_areas[tid].append(area_actual)
                if len(historial_areas[tid]) > MAX_HISTORIAL:
                    historial_areas[tid].pop(0)

                # Etiqueta: ID simple o ID vector si está fusionado
                ids_fusionados_con_tid = []
                for (id_a, id_b) in tracker.pares_fusionados:
                    if id_a == tid:
                        ids_fusionados_con_tid.append(str(id_b))
                    elif id_b == tid:
                        ids_fusionados_con_tid.append(str(id_a))

                grosor = 3 if ids_fusionados_con_tid else 2
                cv.rectangle(display, (x1, y1), (x2, y2), color, grosor)

                if ids_fusionados_con_tid:
                    label = f'ID: [{tid}, {", ".join(ids_fusionados_con_tid)}]'
                else:
                    label = f'ID {tid}'

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