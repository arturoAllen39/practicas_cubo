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
        self.ids_en_reserva = set()
        self.ids_activos    = set()
        self.total_ids      = 0

    def reset(self):
        for cola in self.reservas.values():
            cola.clear()
        self.ids_en_reserva.clear()
        self.ids_activos.clear()
        self.total_ids = 0

    def registrar_salida(self, track_id, borde):
        if track_id not in self.ids_en_reserva:
            self.reservas[borde].append(track_id)
            self.ids_en_reserva.add(track_id)
            self.ids_activos.discard(track_id)

    def registrar_entrada(self, borde):
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
        self.pares_fusionados = set()
        self.contactos_activos = {}

    def reset(self):
        self.tracks            = {}
        self.next_id           = 1
        self.pares_fusionados  = set()
        self.contactos_activos = {}

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
        vx  = float(np.mean(vxs))
        vy  = float(np.mean(vys))
        vx  = max(-MAX_VEL, min(MAX_VEL, vx))
        vy  = max(-MAX_VEL, min(MAX_VEL, vy))
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

    # ── NUEVO ──────────────────────────────────────────────────────────────────
    def _congelar_capsula(self, tid):
        """
        Devuelve un dict con todos los datos relevantes de un track en este
        instante. Se llama en el frame exacto del contacto físico, mientras
        el blob todavía es visible, para tener la mejor información posible
        antes de que los blobs se mezclen visualmente.
        """
        t  = self.tracks[tid]
        vx = t['vx']
        vy = t['vy']
        speed = float(np.sqrt(vx**2 + vy**2))
        return {
            'id':          tid,
            'cx':          t['cx'],
            'cy':          t['cy'],
            'vx':          vx,
            'vy':          vy,
            'speed':       speed,
            'dir_x':       t['ultima_dir_x'],
            'dir_y':       t['ultima_dir_y'],
            # Proyección hacia dónde debería ir si nadie lo hubiese tocado.
            # Se actualiza en cada frame que el contacto sigue activo para
            # mantener la estimación fresca.
            'px_pred':     t['cx'] + t['vx'],
            'py_pred':     t['cy'] + t['vy'],
            'cx_inicio':   t['cx'],
            'cy_inicio':   t['cy'],
            'frames_dentro': 0,   # cuántos frames lleva dentro de la fusión
        }

    def _actualizar_prediccion_capsula(self, capsula):
        """
        Avanza la posición predicha de la cápsula un frame más.

        Usa FACTOR_AMORTIGUACION_CAPSULA (config) en lugar de
        FACTOR_AMORTIGUACION porque las fusiones pueden durar decenas de
        frames. Con FACTOR_AMORTIGUACION=0.50 la velocidad llega a cero en
        ~10 frames; con un factor de 0.97 la predicción sigue siendo útil
        incluso después de 100 frames de fusión.
        """
        capsula['px_pred'] += capsula['vx']
        capsula['py_pred'] += capsula['vy']
        capsula['vx']      *= FACTOR_AMORTIGUACION_CAPSULA
        capsula['vy']      *= FACTOR_AMORTIGUACION_CAPSULA
        capsula['frames_dentro'] += 1

    def _bboxes_en_contacto(self, bbox_a, bbox_b, margen=UMBRAL_CONTACTO_BBOX):
        """
        Devuelve True si dos bboxes se solapan o están a menos de 'margen'
        píxeles de distancia entre sí.

        bbox formato: (x1, y1, x2, y2, area)

        El margen permite capturar el momento justo antes del solapamiento
        completo, cuando los bordes de las personas se empiezan a tocar.
        Un margen de 0 requiere solapamiento real; un margen positivo actúa
        como un radio de contacto adicional alrededor de cada bbox.
        """
        ax1, ay1, ax2, ay2, _ = bbox_a
        bx1, by1, bx2, by2, _ = bbox_b

        # Expandir cada bbox por el margen antes de comprobar solapamiento
        ax1 -= margen;  ay1 -= margen
        ax2 += margen;  ay2 += margen
        bx1 -= margen;  by1 -= margen
        bx2 += margen;  by2 += margen

        # Dos rectángulos se solapan si ninguno queda completamente fuera del otro
        sin_solape = (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)
        return not sin_solape

    def _detectar_contactos(self, bboxes, bbox_margin=5):
        """
        Recorre todos los pares de tracks visibles y comprueba si sus bboxes
        están en contacto físico en este frame.

        Reglas:
        - Un ID solo puede participar en UN contacto a la vez. Si un blob
          fusionado toca a un tercero, se ignora ese nuevo contacto hasta
          que el actual se resuelva. Esto evita que el ID del blob combinado
          cambie cuando un tercer jugador se acerca.
        - El contacto debe persistir al menos FRAMES_MIN_CONTACTO frames
          consecutivos antes de marcarse como fusión real y antes de que
          _resolver_separacion pueda recuperar IDs de él. Esto elimina los
          falsos positivos de contacto+separación en el mismo frame.
        """
        tracks_visibles = [
            (tid, t) for tid, t in self.tracks.items() if t['visible']
        ]
        pares_en_contacto_ahora = set()

        # IDs ya involucrados en algún contacto activo (como host o guest).
        # Ninguno de ellos puede formar un NUEVO par con un tercero.
        ids_ocupados = set()
        for par in self.contactos_activos:
            ids_ocupados.add(par[0])
            ids_ocupados.add(par[1])

        for i in range(len(tracks_visibles)):
            tid_a, t_a = tracks_visibles[i]
            for j in range(i + 1, len(tracks_visibles)):
                tid_b, t_b = tracks_visibles[j]

                par = (min(tid_a, tid_b), max(tid_a, tid_b))

                # Si es un par NUEVO y alguno de los IDs ya está ocupado
                # en otro contacto, no crear uno nuevo.
                if par not in self.contactos_activos:
                    if tid_a in ids_ocupados or tid_b in ids_ocupados:
                        continue

                # Buscar el bbox de cada track en el dict del frame actual
                bbox_a = self._buscar_bbox(t_a['cx'], t_a['cy'], bboxes)
                bbox_b = self._buscar_bbox(t_b['cx'], t_b['cy'], bboxes)

                if bbox_a is None or bbox_b is None:
                    continue

                if not self._bboxes_en_contacto(bbox_a, bbox_b, margen=bbox_margin):
                    continue

                pares_en_contacto_ahora.add(par)

                if par not in self.contactos_activos:
                    # Contacto nuevo: congelar cápsulas en este frame
                    self.contactos_activos[par] = {
                        'capsula_a':       self._congelar_capsula(tid_a),
                        'capsula_b':       self._congelar_capsula(tid_b),
                        'frames_contacto': 1,
                    }
                    # Aún no se marca como fusión real hasta FRAMES_MIN_CONTACTO
                    print(f"[CONTACTO DETECTADO] Par {par} frame 1/{FRAMES_MIN_CONTACTO}")
                else:
                    # Contacto continúa: avanzar predicciones
                    contacto = self.contactos_activos[par]
                    contacto['frames_contacto'] += 1
                    self._actualizar_prediccion_capsula(contacto['capsula_a'])
                    self._actualizar_prediccion_capsula(contacto['capsula_b'])

                    # Solo marcar como fusión real una vez que se confirma
                    if contacto['frames_contacto'] == FRAMES_MIN_CONTACTO:
                        self.pares_fusionados.add(par)
                        print(f"[CONTACTO CONFIRMADO] Par {par} — fusión real")
                    elif contacto['frames_contacto'] > FRAMES_MIN_CONTACTO:
                        self.pares_fusionados.add(par)

        # Pares que ya no están en contacto → distinguir fusión de separación
        pares_a_eliminar = set(self.contactos_activos.keys()) - pares_en_contacto_ahora
        for par in pares_a_eliminar:
            contacto  = self.contactos_activos[par]
            tid_a, tid_b = par

            if contacto['frames_contacto'] < FRAMES_MIN_CONTACTO:
                # No confirmado → falso positivo, eliminar sin más
                del self.contactos_activos[par]
                self.pares_fusionados.discard(par)
                print(f"[CONTACTO CANCELADO] Par {par} — no llegó al mínimo "
                      f"({contacto['frames_contacto']}/{FRAMES_MIN_CONTACTO})")
                continue

            # Confirmado → revisar si es fusión real o separación real
            t_a = self.tracks.get(tid_a)
            t_b = self.tracks.get(tid_b)
            a_invisible = (t_a is not None and not t_a['visible'])
            b_invisible = (t_b is not None and not t_b['visible'])

            if a_invisible or b_invisible:
                # Uno desapareció → fusión real. Mantener el contacto vivo.
                # Actualizar la predicción de la cápsula invisible para que
                # siga al blob visible mientras se mueven juntos.
                # Sin esto, la predicción congela en el frame de la fusión
                # y cuando se separan no se encuentra al blob en el lugar correcto.
                t_visible  = (t_b if a_invisible else t_a)
                cap_invis  = (contacto['capsula_a'] if a_invisible else contacto['capsula_b'])
                if t_visible is not None:
                    dx = cap_invis['dir_x']
                    dy = cap_invis['dir_y']
                    # Usar una distancia fija razonable (MAX_DIST) en vez de
                    # speed*FACTOR_SEPARACION, que colapsa cuando la velocidad
                    # se amortigua durante fusiones largas
                    offset = max(cap_invis['speed'] * FACTOR_SEPARACION,
                                 self.max_dist * 0.8)
                    cap_invis['px_pred'] = t_visible['cx'] + dx * offset
                    cap_invis['py_pred'] = t_visible['cy'] + dy * offset
                    cap_invis['frames_dentro'] = cap_invis.get('frames_dentro', 0) + 1
                self.pares_fusionados.add(par)
            else:
                # Ambos visibles pero ya no en contacto → separación real,
                # o el contacto duró muy poco antes de que ambos se alejasen.
                del self.contactos_activos[par]
                self.pares_fusionados.discard(par)
                print(f"[SEPARACION] Par {par} — separación real "
                      f"({contacto['frames_contacto']} frames)")

    def _buscar_bbox(self, cx, cy, bboxes):
        """
        Devuelve el bbox del frame actual cuyo centroide esté más cerca de
        (cx, cy). Devuelve None si bboxes está vacío.
        """
        if not bboxes:
            return None
        clave = min(bboxes.keys(), key=lambda k: abs(k[0]-cx) + abs(k[1]-cy))
        return bboxes[clave]

    def _resolver_separacion(self, cx, cy):
        """
        Cuando aparece un blob nuevo sin track asignado, busca en
        contactos_activos cuál de las cápsulas congeladas predice una
        posición más cercana a (cx, cy).

        Solo considera contactos que hayan superado FRAMES_MIN_CONTACTO.
        Esto evita que un contacto falso (dos blobs rozándose un frame)
        provoque una recuperación de ID inmediata errónea.

        Devuelve (id_recuperado, par) si encuentra un candidato,
        o (None, None) si no hay ningún contacto confirmado cerca.
        """
        mejor_dist = self.max_dist * MULT_DIST_RESCATE * 2
        mejor_id   = None
        mejor_par  = None

        for par, contacto in self.contactos_activos.items():
            # Ignorar contactos que aún no están confirmados
            if contacto['frames_contacto'] < FRAMES_MIN_CONTACTO:
                continue
            for clave_cap in ('capsula_a', 'capsula_b'):
                cap = contacto[clave_cap]
                if cap.get('liberado', False):
                    continue
                dist = np.sqrt((cx - cap['px_pred'])**2 + (cy - cap['py_pred'])**2)
                if dist < mejor_dist:
                    mejor_dist = dist
                    mejor_id   = cap['id']
                    mejor_par  = par

        return mejor_id, mejor_par

    def _cerrar_contacto_por_separacion(self, par, id_liberado):
        """
        Cuando uno de los dos IDs de un par se separa, quita su cápsula del
        contacto. Si ya no quedan cápsulas, cierra el contacto completamente.
        """
        if par not in self.contactos_activos:
            return
        contacto = self.contactos_activos[par]
        for clave_cap in ('capsula_a', 'capsula_b'):
            if contacto[clave_cap]['id'] == id_liberado:
                # Marcar cápsula como liberada sin borrar el par todavía,
                # por si el otro ID también necesita separarse en el mismo frame
                contacto[clave_cap]['liberado'] = True
                break

        # Si ambas cápsulas están liberadas, cerrar el contacto
        a_liberada = contacto['capsula_a'].get('liberado', False)
        b_liberada = contacto['capsula_b'].get('liberado', False)
        if a_liberada and b_liberada:
            del self.contactos_activos[par]
            self.pares_fusionados.discard(par)
    # ── FIN NUEVO ──────────────────────────────────────────────────────────────


    def actualizar(self, centroides, reserva, bordes, bboxes, umbral_borde=10, bbox_margin=5):
        self._predecir()

        # ── NUEVO: detectar contactos físicos ANTES de la asignación ──────────
        # Esto ocurre mientras todos los blobs son aún visibles, garantizando
        # que las cápsulas se congelan con la mejor información posible.
        self._detectar_contactos(bboxes, bbox_margin=bbox_margin)
        ids_en_contacto      = set()
        ids_excluidos_greedy = set()
        for par, contacto in self.contactos_activos.items():
            confirmado = contacto['frames_contacto'] >= FRAMES_MIN_CONTACTO
            for clave_cap in ('capsula_a', 'capsula_b'):
                cap = contacto[clave_cap]
                if cap.get('liberado', False):
                    continue
                cap_id = cap['id']
                ids_en_contacto.add(cap_id)
                # Solo excluir del greedy si está confirmado Y es invisible
                if confirmado and cap_id in self.tracks and not self.tracks[cap_id]['visible']:
                    ids_excluidos_greedy.add(cap_id)
        # ──────────────────────────────────────────────────────────────────────

        ids_activos     = list(self.tracks.keys())
        asignados_track = set()
        asignados_det   = set()

        if ids_activos and centroides:
            scores = np.full((len(centroides), len(ids_activos)), np.inf)

            for di, (cx, cy) in enumerate(centroides):
                for ti, tid in enumerate(ids_activos):
                    t      = self.tracks[tid]
                    px, py = self._posicion_esperada(tid, pasos=3)
                    dist   = np.sqrt((cx - px)**2 + (cy - py)**2)

                    # Tracks invisibles dentro de una fusión confirmada no
                    # compiten: su blob ya desapareció en el detector.
                    if tid in ids_excluidos_greedy:
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
                        area_det   = bboxes[bbox_k][4]
                        area_prom  = np.mean(t['area_hist'])
                        area_score = min(abs(area_det - area_prom) / (area_prom + 1e-5), 1.0)

                    # dir_score solo se aplica a tracks libres (no fusionados)
                    dir_score     = 0.0
                    tid_en_fusion = tid in ids_excluidos_greedy
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
                                      0.35 * ang_score              +
                                      0.15 * area_score             +
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
                t['vx'], t['vy'] = self._calcular_velocidad(t['historial'])
                vel_mag = np.sqrt(t['vx']**2 + t['vy']**2)
                if vel_mag > 0.5:
                    t['ultima_dir_x'] = t['vx'] / vel_mag
                    t['ultima_dir_y'] = t['vy'] / vel_mag
                t['cx']             = cx
                t['cy']             = cy
                t['visible']        = True
                t['edad_invisible'] = 0
                bbox_k = min(bboxes.keys(),
                             key=lambda k: abs(k[0]-cx) + abs(k[1]-cy),
                             default=None)
                if bbox_k and bbox_k in bboxes:
                    t['area_hist'].append(bboxes[bbox_k][4])

        # ── Detecciones sin asignar ───────────────────────────────────────────
        for di, (cx, cy) in enumerate(centroides):
            if di not in asignados_det:

                # ── Intentar resolver como separación de contacto ─────────────
                id_recuperado, par_origen = self._resolver_separacion(cx, cy)

                if id_recuperado is not None:
                    self._cerrar_contacto_por_separacion(par_origen, id_recuperado)
                    self.agregar_track(id_recuperado, cx, cy)
                    reserva.confirmar_activo(id_recuperado)
                    self.next_id = max(self.next_id, id_recuperado + 1)
                    print(f"[SEPARACION] ID {id_recuperado} recuperado del par {par_origen}")
                    continue

                # ── Guardia: si hay un contacto confirmado cuyo blob visible
                # está cerca de esta detección, este blob ES la cápsula invisible
                # de ese contacto aunque la predicción no cuadre exactamente.
                # Esto impide crear IDs nuevos en separaciones donde la predicción
                # se desvió por movimiento errático durante la fusión.
                id_forzado = None
                dist_forzado = self.max_dist * MULT_DIST_ABSORCION
                for par, contacto in self.contactos_activos.items():
                    if contacto['frames_contacto'] < FRAMES_MIN_CONTACTO:
                        continue
                    tid_a_par, tid_b_par = par
                    t_a_par = self.tracks.get(tid_a_par)
                    t_b_par = self.tracks.get(tid_b_par)
                    a_vis = t_a_par is not None and t_a_par['visible']
                    b_vis = t_b_par is not None and t_b_par['visible']

                    # Solo actuar si exactamente uno es visible (fusión activa)
                    if not (a_vis ^ b_vis):
                        continue

                    t_visible_par  = t_a_par if a_vis else t_b_par
                    cap_invis_par  = (contacto['capsula_b']
                                      if a_vis else contacto['capsula_a'])

                    if cap_invis_par.get('liberado', False):
                        continue

                    # Medir distancia entre esta detección y el blob visible
                    dist_al_visible = np.sqrt(
                        (cx - t_visible_par['cx'])**2 +
                        (cy - t_visible_par['cy'])**2
                    )
                    if dist_al_visible < dist_forzado:
                        dist_forzado = dist_al_visible
                        id_forzado   = cap_invis_par['id']
                        par_forzado  = par

                if id_forzado is not None:
                    self._cerrar_contacto_por_separacion(par_forzado, id_forzado)
                    self.agregar_track(id_forzado, cx, cy)
                    reserva.confirmar_activo(id_forzado)
                    self.next_id = max(self.next_id, id_forzado + 1)
                    print(f"[SEPARACION FORZADA] ID {id_forzado} del par {par_forzado}")
                    continue
                # ─────────────────────────────────────────────────────────────

                borde_entrada = detectar_borde(cx, cy, bordes, umbral_borde)

                if not borde_entrada:
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

        # ── Tracks sin asignar ────────────────────────────────────────────────

        for tid in ids_activos:
            if tid not in asignados_track:
                t = self.tracks[tid]
                t['visible']         = False
                t['edad_invisible'] += 1

                ultima_cx = t['cx']
                ultima_cy = t['cy']

                t['vx'] *= FACTOR_AMORTIGUACION
                t['vy'] *= FACTOR_AMORTIGUACION
                t['cx']  = t['cx_pred']
                t['cy']  = t['cy_pred']

                # ── Si este ID ya está registrado en un contacto activo,
                # su cápsula congelada tiene toda la información necesaria.
                # No buscar absorción: el sistema nuevo ya lo cubre.
                if tid in ids_en_contacto:
                    continue

                borde_salida = detectar_borde(ultima_cx, ultima_cy, bordes, umbral_borde)
                if borde_salida and t['edad_invisible'] == 5:
                    apunta_afuera = False
                    if borde_salida == 'inferior'   and t['vy'] > 0: apunta_afuera = True
                    elif borde_salida == 'superior'  and t['vy'] < 0: apunta_afuera = True
                    elif borde_salida == 'derecho'   and t['vx'] > 0: apunta_afuera = True
                    elif borde_salida == 'izquierdo' and t['vx'] < 0: apunta_afuera = True
                    if apunta_afuera:
                        reserva.registrar_salida(tid, borde_salida)

        # ids_en_contacto ya fue construido al inicio de "tracks sin asignar".
        # Es el unico conjunto de proteccion ahora: cualquier ID con capsula
        # congelada en contactos_activos no acumula edad_invisible y no muere.
        ids_protegidos = ids_en_contacto

        for tid in ids_protegidos:
            if tid in self.tracks:
                self.tracks[tid]['edad_invisible'] = 0

        muertos = [tid for tid, t in self.tracks.items()
                   if t['edad_invisible'] > self.max_edad and tid not in ids_protegidos]
        for tid in muertos:
            del self.tracks[tid]

        if self.contactos_activos:
            print(f"[CONTACTOS] {list(self.contactos_activos.keys())}")

        return [(t['cx'], t['cy'], tid, t['vx'], t['vy'])
                for tid, t in self.tracks.items() if t['visible']]

    def _posicion_esperada(self, track_id, pasos=3):
        """Proyecta la posición esperada usando los últimos N desplazamientos."""
        h = self.tracks[track_id]['historial']
        if len(h) < 2:
            return self.tracks[track_id]['cx'], self.tracks[track_id]['cy']
        n  = min(pasos, len(h) - 1)
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

    cv.createTrackbar('Blur',       win_controls, 10,  31,  callback)
    cv.createTrackbar('Min_thresh', win_controls, 0,  255, callback)
    cv.createTrackbar('Max_thresh', win_controls, 45, 255, callback)
    cv.createTrackbar('Close',      win_controls, 17, 20,  callback)
    cv.createTrackbar('Open',       win_controls, 6,  20,  callback)
    cv.createTrackbar('Dilate',     win_controls, 1,  15,  callback)
    cv.createTrackbar('Erode',      win_controls, 1,  15,  callback)
    cv.createTrackbar('BBox_margin', win_controls, 20, 40, callback)

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
        dilate_k = cv.getTrackbarPos('Dilate',      win_controls)
        erode_k  = cv.getTrackbarPos('Erode',       win_controls)

        if blur_k < 1:        blur_k  = 1
        if blur_k % 2 == 0:   blur_k += 1
        if close_k < 1:       close_k = 1
        if open_k < 1:        open_k  = 1
        if dilate_k < 1:     dilate_k = 1
        if erode_k < 1:      erode_k  = 1
        if min_val >= max_val: min_val = max(0, max_val - 1)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (blur_k, blur_k), 0)
        mask = cv.inRange(blur, min_val, max_val)
        mask = cv.morphologyEx(
            mask,
            cv.MORPH_CLOSE,
            np.ones((close_k, close_k), np.uint8)
        )

        mask = cv.morphologyEx(
            mask,
            cv.MORPH_OPEN,
            np.ones((open_k, open_k), np.uint8)
        )

        # NUEVO: HACER BLOBS MÁS GORDOS
        mask = cv.dilate(
            mask,
            np.ones((dilate_k, dilate_k), np.uint8),
            iterations=1
        )

        mask = cv.erode(
            mask,
            np.ones((erode_k, erode_k), np.uint8),
            iterations=1
        )

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

        bbox_margin = cv.getTrackbarPos('BBox_margin', win_controls)
        bbox_margin = bbox_margin - 20
        resultado   = tracker.actualizar(centroides, reserva, bordes, bboxes,
                                         UMBRAL_BORDE, bbox_margin=bbox_margin)

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

                # Solo mostrar array fusionado cuando hay un contacto CONFIRMADO
                # Y al menos uno de los dos IDs es invisible (fusión real en detector)
                ids_fusionados_con_tid = []
                for (id_a, id_b) in tracker.pares_fusionados:
                    par = (min(id_a, id_b), max(id_a, id_b))
                    if par not in tracker.contactos_activos:
                        continue
                    contacto = tracker.contactos_activos[par]

                    # Verificar que al menos uno de los dos es invisible
                    t_a_check = tracker.tracks.get(id_a)
                    t_b_check = tracker.tracks.get(id_b)
                    a_invisible = t_a_check is not None and not t_a_check['visible']
                    b_invisible = t_b_check is not None and not t_b_check['visible']
                    fusion_real = a_invisible or b_invisible

                    if not fusion_real:
                        continue

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



    # Fusion antes de tiempo
    