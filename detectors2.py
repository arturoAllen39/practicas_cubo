"""
Módulo de detectores para diferentes escenarios del Cubo Negro.

Modos disponibles:
- CIRCLES: Detección de círculos oscuros en fondo claro (simulación con tablet)
- SHADOWS: Detección de sombras/siluetas oscuras sobre fondo brillante (cubo negro real)
- ADAPTIVE_BG: Background subtraction con modelo temporal adaptativo
- HYBRID: Combinación de Shadow + Adaptive BG
"""

import cv2
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment


class CircleDetector:
    """Detector original: círculos negros en fondo claro (para simulación)"""
    
    def __init__(self):
        self.debug_images = {}
    
    def detect(self, frame, params):
        """Detecta círculos usando HoughCircles + contornos como fallback"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        kernel_size = params['blur_kernel']
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 2)
        self.debug_images['blurred'] = blurred
        
        # MÉTODO 1: HoughCircles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist    = params['min_dist'],
            param1     = params['canny_threshold'],
            param2     = params['accumulator_threshold'],
            minRadius  = params['min_radius'],
            maxRadius  = params['max_radius']
        )
        
        detections = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for cx, cy, _ in circles[0]:
                detections.append((int(cx), int(cy)))
        
        # MÉTODO 2: Contornos (fallback)
        if not detections:
            _, binary = cv2.threshold(
                gray, params['threshold_value'], 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            self.debug_images['binary'] = binary.copy()
            
            k = params['morph_kernel']
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            self.debug_images['morphed'] = binary.copy()
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_img = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if params['min_area'] < area < params['max_area']:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        detections.append((cx, cy))
                        cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)
                        cv2.circle(contour_img, (cx, cy), 5, (0, 0, 255), -1)
                else:
                    cv2.drawContours(contour_img, [cnt], -1, (0, 0, 255), 1)
            
            self.debug_images['contours'] = contour_img
        
        return detections


class ShadowDetector:
    """Detector de sombras: personas como blobs oscuros sobre fondo brillante"""
    
    def __init__(self):
        self.debug_images = {}
    
    def detect(self, frame, params):
        """Detecta regiones significativamente más oscuras que el promedio local"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Blur para reducir ruido de animaciones
        blur_size = params.get('shadow_blur', 21)
        if blur_size % 2 == 0:
            blur_size += 1
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        self.debug_images['blurred'] = blurred
        
        # Calcular promedio local (ventana grande para capturar contexto)
        local_avg_size = params.get('local_window', 51)
        if local_avg_size % 2 == 0:
            local_avg_size += 1
        mean_local = cv2.boxFilter(blurred, -1, (local_avg_size, local_avg_size))
        self.debug_images['mean_local'] = mean_local
        
        # Diferencia: zonas más oscuras que el promedio local
        diff = mean_local.astype(np.int16) - blurred.astype(np.int16)
        diff_clipped = np.clip(diff, 0, 255).astype(np.uint8)
        
        # Umbralizar para detectar sombras
        shadow_threshold = params.get('shadow_threshold', 30)
        _, shadow_mask = cv2.threshold(diff_clipped, shadow_threshold, 255, cv2.THRESH_BINARY)
        self.debug_images['shadow_mask'] = shadow_mask
        
        # Limpieza morfológica
        k = params.get('morph_kernel', 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN,  kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        self.debug_images['morphed'] = shadow_mask.copy()
        
        # Encontrar blobs de sombra
        contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        contour_img = cv2.cvtColor(shadow_mask.copy(), cv2.COLOR_GRAY2BGR)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if params['min_area'] < area < params['max_area']:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    detections.append((cx, cy))
                    cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)
                    cv2.circle(contour_img, (cx, cy), 5, (0, 0, 255), -1)
            else:
                cv2.drawContours(contour_img, [cnt], -1, (0, 0, 255), 1)
        
        self.debug_images['contours'] = contour_img
        
        return detections


class AdaptiveBackgroundDetector:
    """Background subtraction con modelo temporal adaptativo"""
    
    def __init__(self, buffer_size=30):
        self.frame_buffer = deque(maxlen=buffer_size)
        self.debug_images = {}
        self.background = None
    
    def detect(self, frame, params):
        """Detecta objetos usando median temporal como background"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Actualizar buffer
        self.frame_buffer.append(gray.copy())
        
        # Necesitamos al menos algunos frames para construir el background
        if len(self.frame_buffer) < params.get('bg_min_frames', 10):
            return []
        
        # Calcular background como median temporal
        self.background = np.median(self.frame_buffer, axis=0).astype(np.uint8)
        self.debug_images['background'] = self.background
        
        # Diferencia absoluta
        diff = cv2.absdiff(gray, self.background)
        self.debug_images['diff'] = diff
        
        # Umbralizar
        bg_threshold = params.get('bg_threshold', 25)
        _, fg_mask = cv2.threshold(diff, bg_threshold, 255, cv2.THRESH_BINARY)
        self.debug_images['fg_mask'] = fg_mask
        
        # Limpieza morfológica
        k = params.get('morph_kernel', 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        self.debug_images['morphed'] = fg_mask.copy()
        
        # Encontrar contornos
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        contour_img = cv2.cvtColor(fg_mask.copy(), cv2.COLOR_GRAY2BGR)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if params['min_area'] < area < params['max_area']:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    detections.append((cx, cy))
                    cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)
                    cv2.circle(contour_img, (cx, cy), 5, (0, 0, 255), -1)
            else:
                cv2.drawContours(contour_img, [cnt], -1, (0, 0, 255), 1)
        
        self.debug_images['contours'] = contour_img
        
        return detections
    
    def reset(self):
        """Reinicia el buffer de frames"""
        self.frame_buffer.clear()
        self.background = None


class HybridDetector:
    """Detector híbrido: combina Shadow + Adaptive Background"""
    
    def __init__(self, buffer_size=30):
        self.shadow_detector = ShadowDetector()
        self.adaptive_detector = AdaptiveBackgroundDetector(buffer_size)
        self.debug_images = {}
    
    def detect(self, frame, params):
        """Combina detecciones de ambos métodos"""
        # Detectar con ambos métodos
        shadow_detections = self.shadow_detector.detect(frame, params)
        bg_detections = self.adaptive_detector.detect(frame, params)
        
        # Combinar imágenes de debug
        self.debug_images.update(self.shadow_detector.debug_images)
        self.debug_images.update({
            f'adaptive_{k}': v for k, v in self.adaptive_detector.debug_images.items()
        })
        
        # Fusionar detecciones (eliminar duplicados cercanos)
        all_detections = shadow_detections + bg_detections
        
        if not all_detections:
            return []
        
        # Clustering simple: agrupar detecciones cercanas
        merge_distance = params.get('merge_distance', 30)
        merged = []
        used = set()
        
        for i, det1 in enumerate(all_detections):
            if i in used:
                continue
            
            cluster = [det1]
            for j, det2 in enumerate(all_detections[i+1:], start=i+1):
                if j in used:
                    continue
                
                dist = np.sqrt((det1[0] - det2[0])**2 + (det1[1] - det2[1])**2)
                if dist < merge_distance:
                    cluster.append(det2)
                    used.add(j)
            
            # Promediar el cluster
            avg_x = int(np.mean([d[0] for d in cluster]))
            avg_y = int(np.mean([d[1] for d in cluster]))
            merged.append((avg_x, avg_y))
            used.add(i)
        
        return merged
    
    def reset(self):
        """Reinicia el detector adaptativo"""
        self.adaptive_detector.reset()


class OpticalFlowDetector:
    """Detector basado en flujo óptico: detecta movimiento sin necesidad de IDs persistentes"""
    
    def __init__(self):
        self.prev_gray = None
        self.debug_images = {}
        
        # Parámetros de flujo óptico Farneback
        self.flow_params = dict(
            pyr_scale=0.5,    # Escala de pirámide
            levels=3,         # Niveles de pirámide
            winsize=15,       # Tamaño de ventana
            iterations=3,     # Iteraciones
            poly_n=5,         # Tamaño vecindario para aproximación polinomial
            poly_sigma=1.2,   # Desviación estándar gaussiana
            flags=0
        )
    
    def detect(self, frame, params):
        """Detecta movimiento usando flujo óptico y genera detecciones de zonas activas"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Necesitamos frame previo para calcular flujo
        if self.prev_gray is None:
            self.prev_gray = gray
            return []
        
        # Calcular flujo óptico denso
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, **self.flow_params
        )
        
        # Calcular magnitud del movimiento
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        angle = np.arctan2(flow[..., 1], flow[..., 0])
        
        # Guardar para debug
        self.debug_images['flow_magnitude'] = (magnitude * 10).clip(0, 255).astype(np.uint8)
        
        # Crear visualización HSV del flujo
        hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)  # Hue = dirección
        hsv[..., 1] = 255  # Saturación máxima
        hsv[..., 2] = np.minimum(magnitude * 10, 255).astype(np.uint8)  # Value = magnitud
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.debug_images['flow_visualization'] = flow_vis
        
        # Umbralizar movimiento
        motion_threshold = params.get('flow_threshold', 2.0)
        motion_mask = (magnitude > motion_threshold).astype(np.uint8) * 255
        self.debug_images['motion_mask'] = motion_mask
        
        # Limpieza morfológica
        k = params.get('morph_kernel', 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        # Dilatar para unir regiones cercanas
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        motion_mask = cv2.dilate(motion_mask, kernel_dilate, iterations=2)
        self.debug_images['morphed'] = motion_mask.copy()
        
        # Encontrar contornos de zonas de movimiento
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        contour_img = cv2.cvtColor(motion_mask.copy(), cv2.COLOR_GRAY2BGR)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if params['min_area'] < area < params['max_area']:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    detections.append((cx, cy))
                    cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)
                    cv2.circle(contour_img, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # Mostrar vector promedio de flujo en esta región
                    mask_region = np.zeros_like(motion_mask)
                    cv2.drawContours(mask_region, [cnt], -1, 255, -1)
                    flow_region = flow[mask_region > 0]
                    if len(flow_region) > 0:
                        avg_flow = np.mean(flow_region, axis=0)
                        # Dibujar flecha de dirección
                        arrow_end = (int(cx + avg_flow[0] * 5), int(cy + avg_flow[1] * 5))
                        cv2.arrowedLine(contour_img, (cx, cy), arrow_end, (255, 0, 255), 2, tipLength=0.3)
            else:
                cv2.drawContours(contour_img, [cnt], -1, (0, 0, 255), 1)
        
        self.debug_images['contours'] = contour_img
        
        self.prev_gray = gray
        return detections
    
    def reset(self):
        """Reinicia el detector"""
        self.prev_gray = None


class BlobTrackingDetector:
    """Detector basado en blob tracking del paper - tracking por área, color y centroide"""
    
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=True
        )
        self.debug_images = {}
        # Almacena info de blobs del frame anterior para tracking
        self.prev_blobs = []
        self.next_blob_id = 1
        self.blob_history = {}  # {blob_id: [(x,y,area,color), ...]}
        
    def detect(self, frame, params):
        """Detecta y trackea blobs usando BackgroundSubtractorMOG2 + características"""
        
        # 1. Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Guardar para debug
        self.debug_images['fg_mask_raw'] = fg_mask.copy()
        
        # 2. Eliminar sombras (valor 127 en MOG2)
        # Shadows = 127, Foreground = 255
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        self.debug_images['fg_no_shadow'] = fg_mask.copy()
        
        # 3. Morphological operations (del paper: erosion + dilation)
        k = params.get('morph_kernel', 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        
        # Erosion (elimina ruido pequeño)
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        self.debug_images['eroded'] = fg_mask.copy()
        
        # Dilation (recupera tamaño original)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        self.debug_images['morphed'] = fg_mask.copy()
        
        # 4. Blob detection (findContours)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_blobs = []
        contour_img = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if params['min_area'] < area < params['max_area']:
                # Calcular características del blob
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Extraer color promedio del blob en la imagen original
                mask_blob = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask_blob, [cnt], -1, 255, -1)
                mean_color = cv2.mean(frame, mask=mask_blob)[:3]  # BGR
                
                # Bounding box para visualización
                x, y, w, h = cv2.boundingRect(cnt)
                
                blob = {
                    'centroid': (cx, cy),
                    'area': area,
                    'color': mean_color,
                    'contour': cnt,
                    'bbox': (x, y, w, h),
                    'id': None  # Se asignará en tracking
                }
                current_blobs.append(blob)
                
                # Dibujar en debug
                cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)
                cv2.circle(contour_img, (cx, cy), 5, (0, 0, 255), -1)
                cv2.rectangle(contour_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            else:
                cv2.drawContours(contour_img, [cnt], -1, (0, 0, 255), 1)
        
        self.debug_images['contours'] = contour_img
        
        # 5. Blob tracking (asociar blobs actuales con anteriores)
        self._track_blobs(current_blobs, params)
        
        # 6. Actualizar historial
        self.prev_blobs = current_blobs
        
        # 7. Retornar solo centroides para compatibilidad con el sistema existente
        detections = [blob['centroid'] for blob in current_blobs if blob['id'] is not None]
        
        return detections
    
    def _track_blobs(self, current_blobs, params):
        """Asocia blobs actuales con blobs anteriores usando similitud"""
        
        if not self.prev_blobs:
            # Primer frame: asignar IDs nuevos a todos
            for blob in current_blobs:
                blob['id'] = self.next_blob_id
                self.blob_history[self.next_blob_id] = [blob]
                self.next_blob_id += 1
            return
        
        # Crear matriz de similitud (más alto = más similar)
        similarity_matrix = np.zeros((len(current_blobs), len(self.prev_blobs)))
        
        for i, curr in enumerate(current_blobs):
            for j, prev in enumerate(self.prev_blobs):
                similarity_matrix[i, j] = self._compute_similarity(curr, prev, params)
        
        # Asociar usando Hungarian algorithm (queremos MAXIMIZAR similitud)
        # linear_sum_assignment minimiza, así que invertimos
        cost_matrix = -similarity_matrix
        
        if len(similarity_matrix) > 0 and len(similarity_matrix[0]) > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_current = set()
            matched_prev = set()
            
            # Umbral mínimo de similitud para considerar match válido
            min_similarity = params.get('blob_similarity_threshold', 0.3)
            
            for i, j in zip(row_ind, col_ind):
                if similarity_matrix[i, j] > min_similarity:
                    # Match válido: asignar mismo ID
                    current_blobs[i]['id'] = self.prev_blobs[j]['id']
                    
                    # Actualizar historial
                    if current_blobs[i]['id'] in self.blob_history:
                        self.blob_history[current_blobs[i]['id']].append(current_blobs[i])
                        # Mantener solo últimos 30 frames
                        if len(self.blob_history[current_blobs[i]['id']]) > 30:
                            self.blob_history[current_blobs[i]['id']].pop(0)
                    
                    matched_current.add(i)
                    matched_prev.add(j)
        else:
            matched_current = set()
        
        # Asignar IDs nuevos a blobs no matched
        for i, blob in enumerate(current_blobs):
            if i not in matched_current:
                blob['id'] = self.next_blob_id
                self.blob_history[self.next_blob_id] = [blob]
                self.next_blob_id += 1
    
    def _compute_similarity(self, blob1, blob2, params):
        """Calcula similitud entre dos blobs (0=diferente, 1=idéntico)"""
        
        # 1. Similitud de posición (distancia euclidiana normalizada)
        cx1, cy1 = blob1['centroid']
        cx2, cy2 = blob2['centroid']
        distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        max_distance = params.get('max_distance', 100)
        pos_similarity = max(0, 1 - distance / max_distance)
        
        # 2. Similitud de área
        area1, area2 = blob1['area'], blob2['area']
        area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        
        # 3. Similitud de color (diferencia de canales BGR)
        color1 = np.array(blob1['color'])
        color2 = np.array(blob2['color'])
        color_distance = np.linalg.norm(color1 - color2)
        # Normalizar (máximo = sqrt(255^2 * 3) ≈ 441)
        color_similarity = max(0, 1 - color_distance / 441.0)
        
        # Pesos (del paper: posición es más importante)
        w_pos = 0.5
        w_area = 0.25
        w_color = 0.25
        
        similarity = (w_pos * pos_similarity + 
                     w_area * area_ratio + 
                     w_color * color_similarity)
        
        return similarity
    
    def reset(self):
        """Reinicia el detector"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        self.prev_blobs = []
        self.next_blob_id = 1
        self.blob_history = {}


# Factory para crear detectores
def create_detector(mode, buffer_size=30):
    """
    Crea un detector según el modo especificado.
    
    Modos:
    - 0: CIRCLES (círculos en tablet)
    - 1: SHADOWS (sombras en cubo negro)
    - 2: OPTICAL_FLOW (flujo óptico)
    - 3: BLOB_TRACKING (tracking por blobs del paper)
    """
    if mode == 0:
        return CircleDetector()
    elif mode == 1:
        return ShadowDetector()
    elif mode == 2:
        return OpticalFlowDetector()
    elif mode == 3:
        return BlobTrackingDetector()
    else:
        raise ValueError(f"Modo de detección inválido: {mode}")