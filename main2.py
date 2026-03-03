import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque

# ===== MÓDULOS DEL PROYECTO =====
from controls2 import create_controls_window, read_controls, save_config, update_help_display, render_controls
from coverage_map import CoverageMap
from detectors2 import create_detector


# ==============================================================
#  KALMAN TRACKER
# ==============================================================
class KalmanTracker:
    """Tracker individual con filtro de Kalman para predicción de movimiento"""

    def __init__(self, initial_position, process_noise, track_id):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        self.kf.statePre  = np.array([[initial_position[0]],
                                       [initial_position[1]],
                                       [0], [0]], np.float32)
        self.kf.statePost = np.array([[initial_position[0]],
                                       [initial_position[1]],
                                       [0], [0]], np.float32)

        self.last_position = initial_position
        self.lost_frames   = 0
        self.track_id      = track_id

        # Historial de trayectoria (últimas 500 posiciones)
        self.trajectory = deque(maxlen=500)
        self.trajectory.append(initial_position)

    def predict(self):
        prediction = self.kf.predict()
        return (int(prediction[0][0]), int(prediction[1][0]))

    def update(self, measurement):
        self.kf.correct(np.array([[np.float32(measurement[0])],
                                   [np.float32(measurement[1])]]))
        self.last_position = measurement
        self.lost_frames   = 0
        self.trajectory.append(measurement)


# ==============================================================
#  PLAY AREA  (calibración y normalización de coordenadas)
# ==============================================================
class PlayArea:
    """Gestiona el área de juego calibrada y la normalización a 0-1"""

    def __init__(self):
        self.corners          = None
        self.is_calibrated    = False
        self.transform_matrix = None

    def set_corners(self, corners):
        self.corners       = corners
        self.is_calibrated = True
        self._calculate_transform()

    def _calculate_transform(self):
        dst_points = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        src_points = np.float32(self.corners)
        self.transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    def is_inside_area(self, point):
        """Devuelve True si el punto (x,y) está DENTRO del polígono calibrado.
           Usa pointPolygonTest de OpenCV: resultado > 0 = dentro, 0 = borde, < 0 = fuera."""
        if not self.is_calibrated:
            return True  # Sin calibración aceptamos todo

        pts = np.array(self.corners, dtype=np.float32)
        result = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
        return result >= 0  # >= 0 incluye el borde

    def normalize_point(self, point):
        """Convierte píxeles → coordenadas normalizadas (0-1).
           (0,0) = esquina inferior izquierda."""
        if not self.is_calibrated:
            return None

        pt         = np.array([[[point[0], point[1]]]], dtype=np.float32)
        normalized = cv2.perspectiveTransform(pt, self.transform_matrix)

        x_norm = float(normalized[0][0][0])
        y_norm = 1.0 - float(normalized[0][0][1])   # Invertir Y

        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))

        return (x_norm, y_norm)

    def draw(self, frame):
        """Oscurece la zona fuera del area calibrada y dibuja el contorno"""
        if not self.is_calibrated:
            return

        h, w = frame.shape[:2]
        pts  = np.array(self.corners, dtype=np.int32)

        # Mascara: blanco DENTRO del area, negro fuera
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # Capa oscura sobre toda la imagen
        dark = (frame * 0.35).astype(np.uint8)

        # Combinar: interior = original, exterior = oscuro
        frame[:] = np.where(mask[:, :, np.newaxis] == 255, frame, dark)

        # Borde del area
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=3)

        labels = ["TL(0,0)", "TR(1,0)", "BR(1,1)", "BL(0,1)"]
        for corner, label in zip(self.corners, labels):
            cv2.circle(frame, corner, 8, (0, 255, 255), -1)
            cv2.putText(frame, label, (corner[0] + 10, corner[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


# ==============================================================
#  MULTI OBJECT TRACKER
# ==============================================================
class MultiObjectTracker:
    """Detecta círculos y mantiene IDs persistentes con Hungarian + Kalman"""

    PLAYER_COLORS = [
        (255,   0,   0),   # Azul
        (  0, 255,   0),   # Verde
        (  0,   0, 255),   # Rojo
        (255, 255,   0),   # Cyan
        (255,   0, 255),   # Magenta
        (  0, 255, 255),   # Amarillo
        (128,   0, 255),   # Púrpura
        (255, 128,   0),   # Naranja
        (  0, 128, 255),   # Azul claro
        (128, 255,   0),   # Verde lima
    ]

    def __init__(self, params, play_area):
        self.trackers    = {}
        self.next_id     = 1
        self.params      = params
        self.play_area   = play_area
        self.debug_images = {}
        
        # Crear detector según el modo actual
        self.current_mode = params.get('detection_mode', 0)
        self.detector = create_detector(self.current_mode, params.get('bg_buffer_size', 30))

    def get_player_color(self, track_id):
        return self.PLAYER_COLORS[(track_id - 1) % len(self.PLAYER_COLORS)]
    
    def change_detection_mode(self, new_mode):
        """Cambia el modo de detección en tiempo real"""
        if new_mode != self.current_mode:
            self.current_mode = new_mode
            self.detector = create_detector(new_mode, self.params.get('bg_buffer_size', 30))
            mode_names = ['CIRCLES', 'SHADOWS', 'ADAPTIVE_BG', 'HYBRID', 'OPTICAL_FLOW', 'BLOB_TRACKING']
            print(f"  Modo cambiado a: {mode_names[new_mode]}")

    # ----------------------------------------------------------
    def detect_objects(self, frame):
        """Detecta objetos usando el detector seleccionado"""
        # Verificar si cambió el modo
        new_mode = self.params.get('detection_mode', 0)
        if new_mode != self.current_mode:
            self.change_detection_mode(new_mode)
        
        # Detectar con el método actual
        detections = self.detector.detect(frame, self.params)
        
        # Copiar imágenes de debug
        self.debug_images = self.detector.debug_images.copy()
        
        # Filtrar: ignorar detecciones fuera del area calibrada
        detections = [d for d in detections if self.play_area.is_inside_area(d)]

        return detections

    # ----------------------------------------------------------
    def update(self, detections):
        """Actualiza trackers con Hungarian Algorithm"""
        predictions = {tid: t.predict() for tid, t in self.trackers.items()}
        max_dist    = self.params['max_distance']

        if predictions and detections:
            track_ids = list(predictions.keys())

            cost_matrix = np.array([
                [np.sqrt((predictions[tid][0] - dx)**2 + (predictions[tid][1] - dy)**2)
                 for dx, dy in detections]
                for tid in track_ids
            ])

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_det   = set()
            matched_track = set()

            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < max_dist:
                    tid = track_ids[i]
                    self.trackers[tid].update(detections[j])
                    matched_det.add(j)
                    matched_track.add(tid)

            for tid in track_ids:
                if tid not in matched_track:
                    self.trackers[tid].lost_frames += 1

            for j, det in enumerate(detections):
                if j not in matched_det:
                    self._new_tracker(det)

        elif detections:
            for det in detections:
                self._new_tracker(det)
        else:
            for t in self.trackers.values():
                t.lost_frames += 1

        self.trackers = {
            tid: t for tid, t in self.trackers.items()
            if t.lost_frames < self.params['max_lost_frames']
        }
        return self.trackers

    def _new_tracker(self, position):
        noise = self.params['process_noise'] / 100.0
        self.trackers[self.next_id] = KalmanTracker(position, noise, self.next_id)
        self.next_id += 1

    # ----------------------------------------------------------
    def get_normalized_positions(self):
        """Devuelve posiciones normalizadas (0-1) de todos los jugadores activos"""
        result = {}
        for tid, t in self.trackers.items():
            norm = self.play_area.normalize_point(t.last_position)
            if norm:
                result[tid] = {
                    'pixel_pos':      t.last_position,
                    'normalized_pos': norm,
                    'color':          self.get_player_color(tid),
                    'lost_frames':    t.lost_frames,
                }
        return result


# ==============================================================
#  HERRAMIENTA DE CALIBRACIÓN
# ==============================================================
class CalibrationTool:
    LABELS = [
        "Top-Left  (0,0)",
        "Top-Right (1,0)",
        "Bottom-Right (1,1)",
        "Bottom-Left  (0,1)"
    ]

    def __init__(self):
        self.corners        = []
        self.current_corner = 0

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current_corner < 4:
            self.corners.append((x, y))
            self.current_corner += 1
            print(f"  ✓ Esquina {self.current_corner}/4  {self.LABELS[self.current_corner-1]}  →  ({x}, {y})")
            if self.current_corner == 4:
                print("  ✓ ¡Listo! Presiona ESPACIO para confirmar  |  R para reiniciar")

    def draw(self, frame):
        for i, corner in enumerate(self.corners):
            cv2.circle(frame, corner, 8, (0, 255, 0), -1)
            cv2.putText(frame, self.LABELS[i], (corner[0] + 10, corner[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if len(self.corners) > 1:
            pts = np.array(self.corners, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

        if len(self.corners) == 4:
            pts = np.array(self.corners, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

        if self.current_corner < 4:
            cv2.putText(frame, f"Clic en: {self.LABELS[self.current_corner]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return len(self.corners) == 4

    def reset(self):
        self.corners        = []
        self.current_corner = 0


# ==============================================================
#  MAIN
# ==============================================================
def main():
    # Parámetros por defecto
    params = {
        # Modo de detección (0=Circles, 1=Shadows, 2=Adaptive, 3=Hybrid)
        'detection_mode':        0,
        
        # Circle detection (modo 0)
        'min_dist':              30,
        'canny_threshold':       50,
        'accumulator_threshold': 30,
        'min_radius':            10,
        'max_radius':            50,
        
        # Preprocesamiento general
        'blur_kernel':            9,
        'threshold_value':      100,
        'morph_kernel':           5,
        
        # Área de detección
        'min_area':             200,
        'max_area':            5000,
        
        # Shadow detection (modo 1)
        'shadow_threshold':      30,
        'shadow_blur':           21,
        'local_window':          51,
        
        # Adaptive background (modo 2/3)
        'bg_threshold':          25,
        'bg_buffer_size':        30,
        'bg_min_frames':         10,
        
        # Hybrid (modo 3)
        'merge_distance':        30,
        
        # Optical Flow (modo 4)
        'flow_threshold':       2.0,  # Umbral de magnitud de movimiento
        
        # Blob Tracking (modo 5)
        'blob_similarity_threshold': 0.3,  # Umbral de similitud (0.0-1.0)
        
        # Tracking
        'max_distance':         100,
        'max_lost_frames':       15,
        'process_noise':          3,
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara")
        return

    play_area        = PlayArea()
    calibration_tool = CalibrationTool()

    # ----------------------------------------------------------
    # FASE 1 — CALIBRACIÓN
    # ----------------------------------------------------------
    cv2.namedWindow('Calibracion Area de Juego')
    cv2.setMouseCallback('Calibracion Area de Juego', calibration_tool.mouse_callback)

    print("=" * 60)
    print("  CUBO NEGRO — FASE 1: CALIBRACIÓN DEL ÁREA")
    print("=" * 60)
    print("  Haz clic en las 4 esquinas del suelo:")
    print("  1. Top-Left     → (0,0)")
    print("  2. Top-Right    → (1,0)")
    print("  3. Bottom-Right → (1,1)")
    print("  4. Bottom-Left  → (0,1)")
    print("  ESPACIO = confirmar  |  R = reiniciar  |  Q = salir")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display      = frame.copy()
        is_complete  = calibration_tool.draw(display)
        cv2.imshow('Calibracion Area de Juego', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord('r'):
            calibration_tool.reset()
            print("  Calibración reiniciada")
        elif key == ord(' ') and is_complete:
            play_area.set_corners(calibration_tool.corners)
            print("  ✓ ¡Área calibrada correctamente!")
            break

    cv2.destroyWindow('Calibracion Area de Juego')

    # ----------------------------------------------------------
    # FASE 2 — TRACKING
    # ----------------------------------------------------------
    tracker      = MultiObjectTracker(params, play_area)
    coverage     = CoverageMap(size=600)
    create_controls_window(params)  # Crea ventanas Controles + Ayuda

    show_debug   = True
    show_visual_debug = True  # NUEVO: mostrar procesamiento interno

    print("\n" + "=" * 60)
    print("  CUBO NEGRO — FASE 2: TRACKING MULTI-MODE")
    print("=" * 60)
    print("  Ventanas abiertas:")
    print("    · Tracking Cubo Negro  — video con IDs y trayectorias")
    print("    · Controles            — sliders + modo de detección")
    print("    · Mapa de Cobertura    — territorio pintado (0-1)")
    print("    · Debug Visual         — procesamiento interno (paso a paso)")
    print("  Modos de detección:")
    print("    0: CIRCLES     (tablet/simulación)")
    print("    1: SHADOWS     (cubo negro - sombras)")
    print("    2: ADAPTIVE_BG (cubo negro - fondo adaptativo)")
    print("    3: HYBRID      (combinado)")
    print("    4: OPTICAL_FLOW (vectores de movimiento)")
    print("    5: BLOB_TRACKING (del paper - área+color+centroide)")
    print("  Teclas:")
    print("    Q = salir  |  M = cambiar modo  |  D = debug  |  R = reset IDs")
    print("    V = toggle debug visual  |  C = limpiar mapa")
    print("    P = posiciones  |  S = guardar config")
    print("=" * 60)

    cv2.namedWindow('Tracking Cubo Negro')
    cv2.namedWindow('Debug Visual', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Debug Visual', 1400, 450)
    
    def show_debug_visual_window(debug_images, current_mode):
        """Muestra el procesamiento interno paso a paso según el modo"""
        if not debug_images:
            return
        
        mode_names = ['CIRCLES', 'SHADOWS', 'ADAPTIVE_BG', 'HYBRID', 'OPTICAL_FLOW', 'BLOB_TRACKING']
        panels = []
        
        # Según el modo, mostrar imágenes relevantes
        if current_mode == 0:  # CIRCLES
            if 'blurred' in debug_images:
                blur_img = cv2.cvtColor(debug_images['blurred'], cv2.COLOR_GRAY2BGR)
                cv2.putText(blur_img, "1. BLUR", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(blur_img)
            
            if 'binary' in debug_images:
                bin_img = cv2.cvtColor(debug_images['binary'], cv2.COLOR_GRAY2BGR)
                cv2.putText(bin_img, "2. BINARY", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(bin_img)
            
            if 'morphed' in debug_images:
                morph_img = cv2.cvtColor(debug_images['morphed'], cv2.COLOR_GRAY2BGR)
                cv2.putText(morph_img, "3. MORPHED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(morph_img)
            
            if 'contours' in debug_images:
                cont_img = debug_images['contours'].copy()
                cv2.putText(cont_img, "4. CONTOURS", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(cont_img)
        
        elif current_mode == 1:  # SHADOWS
            if 'blurred' in debug_images:
                blur_img = cv2.cvtColor(debug_images['blurred'], cv2.COLOR_GRAY2BGR)
                cv2.putText(blur_img, "1. BLUR", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(blur_img)
            
            if 'mean_local' in debug_images:
                mean_img = cv2.cvtColor(debug_images['mean_local'], cv2.COLOR_GRAY2BGR)
                cv2.putText(mean_img, "2. LOCAL AVG", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(mean_img)
            
            if 'shadow_mask' in debug_images:
                shadow_img = cv2.cvtColor(debug_images['shadow_mask'], cv2.COLOR_GRAY2BGR)
                cv2.putText(shadow_img, "3. SHADOW MASK", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(shadow_img)
            
            if 'morphed' in debug_images:
                morph_img = cv2.cvtColor(debug_images['morphed'], cv2.COLOR_GRAY2BGR)
                cv2.putText(morph_img, "4. MORPHED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(morph_img)
            
            if 'contours' in debug_images:
                cont_img = debug_images['contours'].copy()
                cv2.putText(cont_img, "5. CONTOURS", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(cont_img)
        
        elif current_mode == 2:  # ADAPTIVE_BG
            if 'background' in debug_images:
                bg_img = cv2.cvtColor(debug_images['background'], cv2.COLOR_GRAY2BGR)
                cv2.putText(bg_img, "1. BACKGROUND", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(bg_img)
            
            if 'diff' in debug_images:
                diff_img = cv2.cvtColor(debug_images['diff'], cv2.COLOR_GRAY2BGR)
                cv2.putText(diff_img, "2. DIFF", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(diff_img)
            
            if 'fg_mask' in debug_images:
                fg_img = cv2.cvtColor(debug_images['fg_mask'], cv2.COLOR_GRAY2BGR)
                cv2.putText(fg_img, "3. FG MASK", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(fg_img)
            
            if 'morphed' in debug_images:
                morph_img = cv2.cvtColor(debug_images['morphed'], cv2.COLOR_GRAY2BGR)
                cv2.putText(morph_img, "4. MORPHED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(morph_img)
            
            if 'contours' in debug_images:
                cont_img = debug_images['contours'].copy()
                cv2.putText(cont_img, "5. CONTOURS", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(cont_img)
        
        elif current_mode == 3:  # HYBRID
            # Mostrar imágenes clave de ambos detectores
            if 'shadow_mask' in debug_images:
                shadow_img = cv2.cvtColor(debug_images['shadow_mask'], cv2.COLOR_GRAY2BGR)
                cv2.putText(shadow_img, "1. SHADOW", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(shadow_img)
            
            if 'adaptive_fg_mask' in debug_images:
                adapt_img = cv2.cvtColor(debug_images['adaptive_fg_mask'], cv2.COLOR_GRAY2BGR)
                cv2.putText(adapt_img, "2. ADAPTIVE", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(adapt_img)
            
            if 'contours' in debug_images:
                cont_img = debug_images['contours'].copy()
                cv2.putText(cont_img, "3. MERGED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(cont_img)
        
        elif current_mode == 4:  # OPTICAL_FLOW
            if 'flow_magnitude' in debug_images:
                mag_img = cv2.cvtColor(debug_images['flow_magnitude'], cv2.COLOR_GRAY2BGR)
                cv2.putText(mag_img, "1. FLOW MAGNITUDE", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(mag_img)
            
            if 'flow_visualization' in debug_images:
                flow_vis = debug_images['flow_visualization'].copy()
                cv2.putText(flow_vis, "2. FLOW VECTORS", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(flow_vis, "(Color=direccion)", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                panels.append(flow_vis)
            
            if 'motion_mask' in debug_images:
                motion_img = cv2.cvtColor(debug_images['motion_mask'], cv2.COLOR_GRAY2BGR)
                cv2.putText(motion_img, "3. MOTION MASK", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(motion_img)
            
            if 'morphed' in debug_images:
                morph_img = cv2.cvtColor(debug_images['morphed'], cv2.COLOR_GRAY2BGR)
                cv2.putText(morph_img, "4. MORPHED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(morph_img)
            
            if 'contours' in debug_images:
                cont_img = debug_images['contours'].copy()
                cv2.putText(cont_img, "5. DETECTIONS", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(cont_img)
        
        elif current_mode == 5:  # BLOB_TRACKING
            if 'fg_mask_raw' in debug_images:
                fg_raw = cv2.cvtColor(debug_images['fg_mask_raw'], cv2.COLOR_GRAY2BGR)
                cv2.putText(fg_raw, "1. FG MASK RAW", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(fg_raw, "(gray=shadow)", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                panels.append(fg_raw)
            
            if 'fg_no_shadow' in debug_images:
                fg_clean = cv2.cvtColor(debug_images['fg_no_shadow'], cv2.COLOR_GRAY2BGR)
                cv2.putText(fg_clean, "2. NO SHADOW", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(fg_clean)
            
            if 'eroded' in debug_images:
                erode_img = cv2.cvtColor(debug_images['eroded'], cv2.COLOR_GRAY2BGR)
                cv2.putText(erode_img, "3. ERODED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(erode_img)
            
            if 'morphed' in debug_images:
                morph_img = cv2.cvtColor(debug_images['morphed'], cv2.COLOR_GRAY2BGR)
                cv2.putText(morph_img, "4. DILATED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(morph_img)
            
            if 'contours' in debug_images:
                cont_img = debug_images['contours'].copy()
                cv2.putText(cont_img, "5. BLOB TRACKING", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                panels.append(cont_img)
        
        # Combinar paneles horizontalmente
        if panels:
            # Redimensionar todos al mismo alto
            target_height = 400
            resized = []
            for panel in panels:
                h, w = panel.shape[:2]
                new_w = int(w * target_height / h)
                resized_panel = cv2.resize(panel, (new_w, target_height))
                resized.append(resized_panel)
            
            # Concatenar
            combined = np.hstack(resized)
            
            # Agregar título
            header = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
            title = f"MODO: {mode_names[current_mode]} - Procesamiento Interno"
            cv2.putText(header, title, (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            final = np.vstack([header, combined])
            cv2.imshow('Debug Visual', final)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Leer sliders
        params = read_controls(params)
        
        # Obtener modo actual
        current_mode = params.get('detection_mode', 0)
        
        # Actualizar help dinámicamente si cambió el modo (comparar con el anterior)
        if 'prev_mode' not in locals():
            prev_mode = current_mode
        if current_mode != prev_mode:
            update_help_display(current_mode)  # Actualiza ventana AYUDA
            prev_mode = current_mode

        # Detección y tracking
        detections = tracker.detect_objects(frame)
        tracked    = tracker.update(detections)
        positions  = tracker.get_normalized_positions()

        # Dibujar área calibrada
        play_area.draw(frame)

        # Dibujar trayectorias y posiciones
        for tid, t in tracker.trackers.items():
            color = tracker.get_player_color(tid)

            if len(t.trajectory) > 1:
                pts = np.array(list(t.trajectory), dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)

            thickness = 3 if t.lost_frames == 0 else 2
            cv2.circle(frame, t.last_position, 15, color, thickness)
            cv2.putText(frame, f"ID:{tid}",
                        (t.last_position[0] - 30, t.last_position[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if t.lost_frames > 0:
                cv2.putText(frame, f"Lost:{t.lost_frames}",
                            (t.last_position[0] - 30, t.last_position[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Detecciones brutas en amarillo
        if show_debug:
            for dx, dy in detections:
                cv2.circle(frame, (dx, dy), 5, (255, 255, 0), 2)

        # Info
        mode_names = ['CIRCLES', 'SHADOWS',      'OPTICAL_FLOW', 'BLOB_TRACKING']
        mode_text = f"Modo: {mode_names[current_mode]}"
        
        cv2.putText(frame, f"Jugadores: {len(tracked)}  Detectados: {len(detections)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, mode_text,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Actualizar mapa de cobertura con leyenda
        coverage.update(positions)
        player_colors = {tid: tracker.get_player_color(tid) for tid in tracker.trackers}
        coverage.draw_legend(player_colors)

        cv2.imshow('Tracking Cubo Negro', frame)
        render_controls()  # Redibujar panel con scroll
        
        # Mostrar debug visual si está activado
        if show_visual_debug:
            show_debug_visual_window(tracker.debug_images, current_mode)

        # Teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"  Debug: {'ON' if show_debug else 'OFF'}")
        elif key == ord('v'):
            show_visual_debug = not show_visual_debug
            print(f"  Debug Visual: {'ON' if show_visual_debug else 'OFF'}")
            if not show_visual_debug:
                cv2.destroyWindow('Debug Visual')
        elif key == ord('m'):
            mode_names = ['CIRCLES', 'SHADOWS', 'ADAPTIVE_BG', 'HYBRID', 'OPTICAL_FLOW', 'BLOB_TRACKING']
            from controls2 import _st
            new_mode = (_st["vals"].get("detection_mode", 0) + 1) % 6
            _st["vals"]["detection_mode"] = new_mode
            params['detection_mode'] = new_mode
            print(f"  Modo cambiado: {mode_names[new_mode]}")
        elif key == ord('r'):
            tracker.trackers  = {}
            tracker.next_id   = 1
            print("  IDs reseteados")
        elif key == ord('c'):
            coverage.clear()
        elif key == ord('p'):
            print("\n" + "=" * 60)
            print("  POSICIONES NORMALIZADAS ACTUALES:")
            for tid, data in positions.items():
                print(f"    ID {tid}: norm={data['normalized_pos']}  px={data['pixel_pos']}")
            print("=" * 60)
        elif key == ord('s'):
            save_config(params)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()