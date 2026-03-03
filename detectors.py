<<<<<<< HEAD
"""
Módulo de detectores para diferentes escenarios del Cubo Negro.

Modos disponibles:
- CIRCLES: Detección de círculos oscuros en fondo claro (simulación con tablet)
- SHADOWS: Detección de sombras/siluetas oscuras sobre fondo brillante (cubo negro real)
- OPTICAL_FLOW: Detección por vectores de movimiento
- BLOB_TRACKING: Tracking por área+color+centroide (del paper)

Preprocesamiento opcional:
- Background Subtraction (KNN o MOG2) para eliminar fondo
"""

=======
>>>>>>> d1c69e2ed2717a018c58a372c588256fb29e2b7b
import cv2
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment


<<<<<<< HEAD
class BackgroundSubtractor:
    """Preprocesador avanzado con múltiples filtros para Cubo Negro"""
    
    def __init__(self, method='MOG2', enabled=False):
        """
        Args:
            method: 'MOG2' o 'KNN'
            enabled: Si True, aplica background subtraction
        """
        self.method = method
        self.enabled = enabled
        
        if enabled:
            if method == 'MOG2':
                self.subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=500,
                    varThreshold=16,
                    detectShadows=True
                )
            elif method == 'KNN':
                self.subtractor = cv2.createBackgroundSubtractorKNN(
                    history=500,
                    dist2Threshold=400.0,
                    detectShadows=True
                )
            else:
                raise ValueError(f"Método '{method}' no válido. Usa 'MOG2' o 'KNN'")
        else:
            self.subtractor = None
    
    def process(self, frame, params=None, remove_shadows=True):
        """
        Procesa el frame con filtros configurables.
        
        Args:
            frame: Frame original
            params: Diccionario con configuración de filtros (nuevo método)
                    Si es None, usa valores por defecto
            remove_shadows: (deprecated) Solo para compatibilidad, usar params['remove_shadows']
        
        Returns:
            Tuple (processed_frame, debug_images_dict)
        """
        debug_imgs = {}
        
        # Si background subtraction está desactivado, retornar original
        if not self.enabled or self.subtractor is None:
            return frame, debug_imgs
        
        # Si params es None, usar valores por defecto para compatibilidad
        if params is None:
            params = {
                'bg_threshold': 200,
                'remove_shadows': remove_shadows,
                'erode_kernel': 3,
                'erode_iterations': 1,
                'dilate_kernel': 5,
                'dilate_iterations': 2,
                'closing_kernel': 7,
                'opening_kernel': 5,
                'use_closing': True,
                'use_opening': True,
            }
        
        # 1. Background Subtraction con learning rate bajo (para fondos dinámicos)
        learning_rate = params.get('bg_learning_rate', 0.001)  # Bajo = no absorbe animaciones rápido
        fg_mask = self.subtractor.apply(frame, learningRate=learning_rate)
        debug_imgs['01_fg_mask_raw'] = fg_mask.copy()
        
        # 2. Eliminar sombras (valor 127)
        if params.get('remove_shadows', True):
            fg_mask[fg_mask == 127] = 0
        debug_imgs['02_no_shadows'] = fg_mask.copy()
        
        # 3. Threshold adicional para limpiar ruido
        threshold_val = params.get('bg_threshold', 200)
        _, fg_mask = cv2.threshold(fg_mask, threshold_val, 255, cv2.THRESH_BINARY)
        debug_imgs['03_threshold'] = fg_mask.copy()
        
        # 4. Erosión (elimina píxeles aislados - ruido pequeño)
        erode_size = params.get('erode_kernel', 3)
        if erode_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
            fg_mask = cv2.erode(fg_mask, kernel, iterations=params.get('erode_iterations', 1))
            debug_imgs['04_eroded'] = fg_mask.copy()
        
        # 5. Dilatación (recupera tamaño de objetos)
        dilate_size = params.get('dilate_kernel', 5)
        if dilate_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=params.get('dilate_iterations', 2))
            debug_imgs['05_dilated'] = fg_mask.copy()
        
        # 6. Closing (cierra huecos dentro de objetos)
        if params.get('use_closing', True):
            close_size = params.get('closing_kernel', 7)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            debug_imgs['06_closed'] = fg_mask.copy()
        
        # 7. Opening (elimina pequeños objetos que sobrevivieron)
        if params.get('use_opening', True):
            open_size = params.get('opening_kernel', 5)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            debug_imgs['07_opened'] = fg_mask.copy()
        
        # 8. Aplicar máscara al frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
        debug_imgs['08_final_masked'] = masked_frame.copy()
        
        return masked_frame, debug_imgs


=======
>>>>>>> d1c69e2ed2717a018c58a372c588256fb29e2b7b
class CircleDetector:
    """Detector de círculos para simulación en tablet"""
    
    def __init__(self):
        self.debug_images = {}
    
    def detect(self, frame, params):
        """Detecta círculos oscuros en fondo claro usando HoughCircles"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Aplicar blur
        blur_k = params.get('blur_kernel', 9)
        if blur_k % 2 == 0:
            blur_k += 1
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        self.debug_images['blurred'] = blurred
        
        # Umbralizar
        _, binary = cv2.threshold(blurred, params['threshold_value'], 255, cv2.THRESH_BINARY)
        self.debug_images['binary'] = binary
        
        # Morfología
        k = params.get('morph_kernel', 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        self.debug_images['morphed'] = morphed
        
        # Detección de círculos
        circles = cv2.HoughCircles(
            morphed,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=params['min_dist'],
            param1=params['canny_threshold'],
            param2=params['accumulator_threshold'],
            minRadius=params['min_radius'],
            maxRadius=params['max_radius']
        )
        
        detections = []
        contour_img = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                cx, cy, r = circle
                detections.append((int(cx), int(cy)))
                cv2.circle(contour_img, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(contour_img, (cx, cy), 2, (0, 0, 255), 3)
        
        self.debug_images['contours'] = contour_img
        return detections
    
    def reset(self):
        """No necesita reset"""
        pass


class ShadowDetector:
    """Detector de sombras/siluetas para Cubo Negro"""
    
    def __init__(self):
        self.debug_images = {}
    
    def detect(self, frame, params):
        """Detecta siluetas oscuras mediante contraste local"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Blur para reducir ruido
        blur_k = params.get('shadow_blur', 21)
        if blur_k % 2 == 0:
            blur_k += 1
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        self.debug_images['blurred'] = blurred
        
        # Promedio local
        local_k = params.get('local_window', 51)
        if local_k % 2 == 0:
            local_k += 1
        mean_local = cv2.blur(blurred, (local_k, local_k))
        self.debug_images['mean_local'] = mean_local
        
        # Diferencia: píxeles más oscuros que su entorno
        diff = mean_local.astype(np.int16) - blurred.astype(np.int16)
        
        # Umbralizar sombras
        shadow_threshold = params.get('shadow_threshold', 30)
        shadow_mask = (diff > shadow_threshold).astype(np.uint8) * 255
        self.debug_images['shadow_mask'] = shadow_mask
        
        # Morfología
        k = params.get('morph_kernel', 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        self.debug_images['morphed'] = shadow_mask.copy()
        
        # Encontrar contornos
        contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        contour_img = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR)
        
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
        """No necesita reset"""
        pass


class OpticalFlowDetector:
    """Detector basado en flujo óptico: detecta movimiento sin necesidad de IDs persistentes"""
    
    def __init__(self):
        self.prev_gray = None
        self.debug_images = {}
        
        # Parámetros de flujo óptico Farneback
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    
    def detect(self, frame, params):
        """Detecta movimiento usando flujo óptico"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return []
        
        # Calcular flujo óptico
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, **self.flow_params
        )
        
        # Magnitud y ángulo
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        angle = np.arctan2(flow[..., 1], flow[..., 0])
        
        self.debug_images['flow_magnitude'] = (magnitude * 10).clip(0, 255).astype(np.uint8)
        
        # Visualización HSV
        hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = np.minimum(magnitude * 10, 255).astype(np.uint8)
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
        
        # Dilatar
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        motion_mask = cv2.dilate(motion_mask, kernel_dilate, iterations=2)
        self.debug_images['morphed'] = motion_mask.copy()
        
        # Contornos
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
                    
                    # Vector de flujo promedio
                    mask_region = np.zeros_like(motion_mask)
                    cv2.drawContours(mask_region, [cnt], -1, 255, -1)
                    flow_region = flow[mask_region > 0]
                    if len(flow_region) > 0:
                        avg_flow = np.mean(flow_region, axis=0)
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
        self.prev_blobs = []
        self.next_blob_id = 1
        self.blob_history = {}
        
    def detect(self, frame, params):
        """Detecta y trackea blobs usando BackgroundSubtractorMOG2 + características"""
        
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        self.debug_images['fg_mask_raw'] = fg_mask.copy()
        
        # Eliminar sombras
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        self.debug_images['fg_no_shadow'] = fg_mask.copy()
        
        # Morfología
        k = params.get('morph_kernel', 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        self.debug_images['eroded'] = fg_mask.copy()
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        self.debug_images['morphed'] = fg_mask.copy()
        
        # Blob detection
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_blobs = []
        contour_img = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if params['min_area'] < area < params['max_area']:
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Color promedio
                mask_blob = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask_blob, [cnt], -1, 255, -1)
                mean_color = cv2.mean(frame, mask=mask_blob)[:3]
                
                x, y, w, h = cv2.boundingRect(cnt)
                
                blob = {
                    'centroid': (cx, cy),
                    'area': area,
                    'color': mean_color,
                    'contour': cnt,
                    'bbox': (x, y, w, h),
                    'id': None
                }
                current_blobs.append(blob)
                
                cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)
                cv2.circle(contour_img, (cx, cy), 5, (0, 0, 255), -1)
                cv2.rectangle(contour_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            else:
                cv2.drawContours(contour_img, [cnt], -1, (0, 0, 255), 1)
        
        self.debug_images['contours'] = contour_img
        
        # Tracking
        self._track_blobs(current_blobs, params)
        self.prev_blobs = current_blobs
        
        # Retornar centroides
        detections = [blob['centroid'] for blob in current_blobs if blob['id'] is not None]
        return detections
    
    def _track_blobs(self, current_blobs, params):
        """Asocia blobs actuales con blobs anteriores usando similitud"""
        
        if not self.prev_blobs:
            for blob in current_blobs:
                blob['id'] = self.next_blob_id
                self.blob_history[self.next_blob_id] = [blob]
                self.next_blob_id += 1
            return
        
        # Matriz de similitud
        similarity_matrix = np.zeros((len(current_blobs), len(self.prev_blobs)))
        
        for i, curr in enumerate(current_blobs):
            for j, prev in enumerate(self.prev_blobs):
                similarity_matrix[i, j] = self._compute_similarity(curr, prev, params)
        
        # Hungarian algorithm
        cost_matrix = -similarity_matrix
        
        if len(similarity_matrix) > 0 and len(similarity_matrix[0]) > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_current = set()
            min_similarity = params.get('blob_similarity_threshold', 0.3)
            
            for i, j in zip(row_ind, col_ind):
                if similarity_matrix[i, j] > min_similarity:
                    current_blobs[i]['id'] = self.prev_blobs[j]['id']
                    
                    if current_blobs[i]['id'] in self.blob_history:
                        self.blob_history[current_blobs[i]['id']].append(current_blobs[i])
                        if len(self.blob_history[current_blobs[i]['id']]) > 30:
                            self.blob_history[current_blobs[i]['id']].pop(0)
                    
                    matched_current.add(i)
        else:
            matched_current = set()
        
        # Nuevos IDs
        for i, blob in enumerate(current_blobs):
            if i not in matched_current:
                blob['id'] = self.next_blob_id
                self.blob_history[self.next_blob_id] = [blob]
                self.next_blob_id += 1
    
    def _compute_similarity(self, blob1, blob2, params):
        """Calcula similitud entre dos blobs (0=diferente, 1=idéntico)"""
        
        # Posición
        cx1, cy1 = blob1['centroid']
        cx2, cy2 = blob2['centroid']
        distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        max_distance = params.get('max_distance', 100)
        pos_similarity = max(0, 1 - distance / max_distance)
        
        # Área
        area1, area2 = blob1['area'], blob2['area']
        area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        
        # Color
        color1 = np.array(blob1['color'])
        color2 = np.array(blob2['color'])
        color_distance = np.linalg.norm(color1 - color2)
        color_similarity = max(0, 1 - color_distance / 441.0)
        
        # Pesos
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


<<<<<<< HEAD
class DarkRegionDetector:
    """Detector especializado para Cubo Negro: detecta personas oscuras vs fondo brillante"""
    
    def __init__(self):
        self.debug_images = {}
        self.persistent_blobs = {}  # {blob_id: frames_count}
        self.next_blob_id = 1
        
    def detect(self, frame, params):
        """Detecta regiones oscuras (personas) ignorando círculos brillantes"""
        
        # 1. Convertir a HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        self.debug_images['01_value_channel'] = v.copy()
        
        # 2. Filtrar por BAJO brillo (personas oscuras)
        # Las personas son oscuras, los círculos son brillantes
        max_brightness = params.get('dark_max_value', 80)  # Ajustable
        _, dark_mask = cv2.threshold(v, max_brightness, 255, cv2.THRESH_BINARY_INV)
        self.debug_images['02_dark_mask'] = dark_mask.copy()
        
        # 3. Morfología AGRESIVA (elimina círculos pequeños)
        # Erosión fuerte elimina objetos pequeños (círculos)
        erode_size = params.get('dark_erode', 7)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
        dark_mask = cv2.erode(dark_mask, kernel_erode, iterations=2)
        self.debug_images['03_eroded'] = dark_mask.copy()
        
        # Dilatación recupera tamaño de personas
        dilate_size = params.get('dark_dilate', 11)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        dark_mask = cv2.dilate(dark_mask, kernel_dilate, iterations=3)
        self.debug_images['04_dilated'] = dark_mask.copy()
        
        # Closing (cierra huecos en siluetas)
        closing_size = params.get('dark_closing', 15)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel_close)
        self.debug_images['05_closed'] = dark_mask.copy()
        
        # 4. Encontrar contornos
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. Filtrar por características
        candidates = []
        contour_img = cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2BGR)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Filtro 1: Área (personas son grandes, círculos pequeños)
            if not (params['min_area'] < area < params['max_area']):
                cv2.drawContours(contour_img, [cnt], -1, (0, 0, 255), 1)  # Rojo = rechazado
                continue
            
            # Filtro 2: Circularidad (rechazar círculos perfectos)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            max_circularity = params.get('max_circularity', 0.85)
            
            if circularity > max_circularity:
                # Muy circular = probablemente círculo proyectado
                cv2.drawContours(contour_img, [cnt], -1, (255, 0, 255), 1)  # Magenta = muy circular
                continue
            
            # Filtro 3: Relación aspecto (personas vistas desde arriba)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Personas vistas desde arriba no son muy alargadas
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                cv2.drawContours(contour_img, [cnt], -1, (0, 255, 255), 1)  # Cyan = aspecto raro
                continue
            
            # Candidato válido
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                candidates.append({
                    'centroid': (cx, cy),
                    'area': area,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'contour': cnt,
                    'bbox': (x, y, w, h)
                })
                
                # Dibujar
                cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)  # Verde = aceptado
                cv2.circle(contour_img, (cx, cy), 5, (0, 0, 255), -1)
                cv2.rectangle(contour_img, (x, y), (x+w, y+h), (255, 255, 0), 1)
                
                # Mostrar métricas
                cv2.putText(contour_img, f"C:{circularity:.2f}", 
                           (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        self.debug_images['06_filtered'] = contour_img
        
        # 6. Filtro temporal (persistencia)
        min_persistence = params.get('min_persistence_frames', 3)
        detections = self._apply_temporal_filter(candidates, min_persistence)
        
        return detections
    
    def _apply_temporal_filter(self, candidates, min_frames):
        """Solo acepta blobs que persisten N frames consecutivos"""
        
        # Actualizar persistencia
        current_ids = set()
        
        for candidate in candidates:
            cx, cy = candidate['centroid']
            
            # Buscar si es blob conocido
            matched = False
            for blob_id, data in self.persistent_blobs.items():
                prev_cx, prev_cy = data['pos']
                dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                
                if dist < 50:  # Mismo blob
                    self.persistent_blobs[blob_id]['count'] += 1
                    self.persistent_blobs[blob_id]['pos'] = (cx, cy)
                    current_ids.add(blob_id)
                    matched = True
                    break
            
            if not matched:
                # Nuevo blob
                self.persistent_blobs[self.next_blob_id] = {
                    'pos': (cx, cy),
                    'count': 1
                }
                current_ids.add(self.next_blob_id)
                self.next_blob_id += 1
        
        # Eliminar blobs que desaparecieron
        to_remove = [bid for bid in self.persistent_blobs if bid not in current_ids]
        for bid in to_remove:
            del self.persistent_blobs[bid]
        
        # Solo retornar blobs con suficiente persistencia
        detections = []
        for candidate in candidates:
            cx, cy = candidate['centroid']
            for blob_id, data in self.persistent_blobs.items():
                pcx, pcy = data['pos']
                dist = np.sqrt((cx - pcx)**2 + (cy - pcy)**2)
                if dist < 50 and data['count'] >= min_frames:
                    detections.append((cx, cy))
                    break
        
        return detections
    
    def reset(self):
        """Reinicia el detector"""
        self.persistent_blobs = {}
        self.next_blob_id = 1
    """Detector basado en blob tracking del paper - tracking por área, color y centroide"""
    
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=True
        )
        self.debug_images = {}
        self.prev_blobs = []
        self.next_blob_id = 1
        self.blob_history = {}
        
    def detect(self, frame, params):
        """Detecta y trackea blobs usando BackgroundSubtractorMOG2 + características"""
        
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        self.debug_images['fg_mask_raw'] = fg_mask.copy()
        
        # Eliminar sombras
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        self.debug_images['fg_no_shadow'] = fg_mask.copy()
        
        # Morfología
        k = params.get('morph_kernel', 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        self.debug_images['eroded'] = fg_mask.copy()
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        self.debug_images['morphed'] = fg_mask.copy()
        
        # Blob detection
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_blobs = []
        contour_img = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if params['min_area'] < area < params['max_area']:
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Color promedio
                mask_blob = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask_blob, [cnt], -1, 255, -1)
                mean_color = cv2.mean(frame, mask=mask_blob)[:3]
                
                x, y, w, h = cv2.boundingRect(cnt)
                
                blob = {
                    'centroid': (cx, cy),
                    'area': area,
                    'color': mean_color,
                    'contour': cnt,
                    'bbox': (x, y, w, h),
                    'id': None
                }
                current_blobs.append(blob)
                
                cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)
                cv2.circle(contour_img, (cx, cy), 5, (0, 0, 255), -1)
                cv2.rectangle(contour_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            else:
                cv2.drawContours(contour_img, [cnt], -1, (0, 0, 255), 1)
        
        self.debug_images['contours'] = contour_img
        
        # Tracking
        self._track_blobs(current_blobs, params)
        self.prev_blobs = current_blobs
        
        # Retornar centroides
        detections = [blob['centroid'] for blob in current_blobs if blob['id'] is not None]
        return detections
    
    def _track_blobs(self, current_blobs, params):
        """Asocia blobs actuales con blobs anteriores usando similitud"""
        
        if not self.prev_blobs:
            for blob in current_blobs:
                blob['id'] = self.next_blob_id
                self.blob_history[self.next_blob_id] = [blob]
                self.next_blob_id += 1
            return
        
        # Matriz de similitud
        similarity_matrix = np.zeros((len(current_blobs), len(self.prev_blobs)))
        
        for i, curr in enumerate(current_blobs):
            for j, prev in enumerate(self.prev_blobs):
                similarity_matrix[i, j] = self._compute_similarity(curr, prev, params)
        
        # Hungarian algorithm
        cost_matrix = -similarity_matrix
        
        if len(similarity_matrix) > 0 and len(similarity_matrix[0]) > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_current = set()
            min_similarity = params.get('blob_similarity_threshold', 0.3)
            
            for i, j in zip(row_ind, col_ind):
                if similarity_matrix[i, j] > min_similarity:
                    current_blobs[i]['id'] = self.prev_blobs[j]['id']
                    
                    if current_blobs[i]['id'] in self.blob_history:
                        self.blob_history[current_blobs[i]['id']].append(current_blobs[i])
                        if len(self.blob_history[current_blobs[i]['id']]) > 30:
                            self.blob_history[current_blobs[i]['id']].pop(0)
                    
                    matched_current.add(i)
        else:
            matched_current = set()
        
        # Nuevos IDs
        for i, blob in enumerate(current_blobs):
            if i not in matched_current:
                blob['id'] = self.next_blob_id
                self.blob_history[self.next_blob_id] = [blob]
                self.next_blob_id += 1
    
    def _compute_similarity(self, blob1, blob2, params):
        """Calcula similitud entre dos blobs (0=diferente, 1=idéntico)"""
        
        # Posición
        cx1, cy1 = blob1['centroid']
        cx2, cy2 = blob2['centroid']
        distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        max_distance = params.get('max_distance', 100)
        pos_similarity = max(0, 1 - distance / max_distance)
        
        # Área
        area1, area2 = blob1['area'], blob2['area']
        area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        
        # Color
        color1 = np.array(blob1['color'])
        color2 = np.array(blob2['color'])
        color_distance = np.linalg.norm(color1 - color2)
        color_similarity = max(0, 1 - color_distance / 441.0)
        
        # Pesos
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


=======
>>>>>>> d1c69e2ed2717a018c58a372c588256fb29e2b7b
# Factory para crear detectores
def create_detector(mode, buffer_size=30):
    """
    Crea un detector según el modo especificado.
    
    Modos:
    - 0: CIRCLES (círculos en tablet)
    - 1: SHADOWS (sombras en cubo negro)
    - 2: OPTICAL_FLOW (flujo óptico)
    - 3: BLOB_TRACKING (tracking por blobs del paper)
<<<<<<< HEAD
    - 4: DARK_REGION (detector por oscuridad - RECOMENDADO para Cubo Negro)
=======
>>>>>>> d1c69e2ed2717a018c58a372c588256fb29e2b7b
    """
    if mode == 0:
        return CircleDetector()
    elif mode == 1:
        return ShadowDetector()
    elif mode == 2:
        return OpticalFlowDetector()
    elif mode == 3:
        return BlobTrackingDetector()
<<<<<<< HEAD
    elif mode == 4:
        return DarkRegionDetector()
=======
>>>>>>> d1c69e2ed2717a018c58a372c588256fb29e2b7b
    else:
        raise ValueError(f"Modo de detección inválido: {mode}")