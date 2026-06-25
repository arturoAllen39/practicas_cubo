# ─────────────────────────────────────────────
#  CONFIGURACIÓN GENERAL — CUBO NEGRO
#  Modifica aquí sin tocar el código principal
# ─────────────────────────────────────────────


# ── VIDEO ─────────────────────────────────────
VIDEO_PATH = 'test2.mp4'       # Ruta del ideo a procesar


# ── ÁREA DE JUEGO ─────────────────────────────
MAX_IDS = 21                    # Máximo de jugadores simultáneos rastreados
UMBRAL_BORDE = 10               # Distancia en px para considerar que un blob
                                # está cerca de un borde (entrada/salida)


# ── BLOB TRACKER ──────────────────────────────
MAX_EDAD_INVISIBLE = 350        # Frames que un track "fantasma" sobrevive
                                # sin ser detectado antes de eliminarse
VEL_HISTORY = 10                # Frames usados para promediar la velocidad
                                # Valores bajos = reacción rápida pero nerviosa
                                # Valores altos = suavizado pero lento
MAX_DIST = 25                   # Distancia máxima en px para asociar un blob
                                # a un track existente en el greedy
PESO_DIST = 0.4                 # Peso de la distancia en el score de asociación
PESO_ANGULO = 0.6               # Peso de la dirección en el score de asociación
                                # PESO_DIST + PESO_ANGULO deben sumar 1.0



# ── DETECCIÓN DE FUSIÓN POR PROXIMIDAD ────────
UMBRAL_FUSION = 1              # Distancia en px entre dos centroides para
                                # considerarlos "fusionados"
FRAMES_PARA_FUSION = 3          # Frames consecutivos juntos para confirmar fusión


# ── DETECCIÓN DE ABSORCIÓN (fusiones_activas) ─
MULT_DIST_ABSORCION = 2.5         # Multiplicador de MAX_DIST para detectar si
                                # un blob fue absorbido por otro cercano
                                # Radio efectivo = MAX_DIST * MULT_DIST_ABSORCION


# ── RESCATE DE TRACKS ─────────────────────────
MULT_DIST_RESCATE = 4           # Multiplicador de MAX_DIST para el radio de
                                # búsqueda de rescate de tracks sin asignar
MAX_EDAD_RESCATE = 5            # Edad máxima invisible de un track para
                                # ser elegible como candidato de rescate


# ── FÍSICA DEL FANTASMA ───────────────────────
FACTOR_AMORTIGUACION = 0.50     # Factor por el que se reduce la velocidad
                                # cada frame que el track está invisible
                                # 0.0 = se detiene inmediatamente
                                # 1.0 = mantiene velocidad indefinidamente
MAX_VEL = 3                     # Velocidad máxima en px/frame


# ── FILTROS DE IMAGEN ─────────────────────────
BLUR_INIT = 5                   # Valor inicial del kernel de desenfoque (impar)
MIN_THRESH_INIT = 0             # Umbral mínimo de intensidad para segmentación
MAX_THRESH_INIT = 45            # Umbral máximo de intensidad para segmentación
CLOSE_INIT = 17                 # Tamaño inicial del kernel de cierre morfológico
                                # Kernels grandes = une blobs separados
                                # Kernels pequeños = más rápido pero menos unión
OPEN_INIT = 3                   # Tamaño inicial del kernel de apertura morfológica


# ── DETECCIÓN DE CONTORNOS ────────────────────
AREA_MIN_CONTORNO = 350         # Área mínima en px² para considerar un contorno
                                # como blob válido. Subir para filtrar más ruido.


# ── HISTORIAL DE ÁREAS ────────────────────────
MAX_HISTORIAL_AREAS = 50        # Frames de historial de área por blob
                                # Usado para detectar cambios bruscos de tamaño


# ── VENTANAS ──────────────────────────────────
WIN_CONTROLS_W = 400            # Ancho de la ventana de controles
WIN_CONTROLS_H = 200            # Alto de la ventana de controles
WIN_CONTROLS_X = 710            # Posición X de la ventana de controles
WIN_VIDEO_X    = 0              # Posición X de la ventana de video
WIN_VIDEO_Y    = 0              # Posición Y de la ventana de video


# ── GREEDY ────────────────────────────────────
UMBRAL_DIR_OPUESTA = -0.0       # Producto punto mínimo para considerar
                                # direcciones opuestas en el score
                                # -1.0 = exactamente opuestos
                                # -0.3 = más de 107 grados de diferencia
PENALIZACION_DIR   = 0.5        # Penalización aplicada al score cuando
                                # el movimiento va en dirección contraria


# ── CONTACTO FÍSICO DE BBOXES ─────────────────
UMBRAL_CONTACTO_BBOX = 5   # Píxeles de margen alrededor de cada bbox
                            # para considerar que dos blobs "se tocan".
                            # 0 = requiere solapamiento real.
                            # 8-15 = captura el momento justo antes de
                            # que los blobs se mezclen visualmente.


# ── CONTACTO FÍSICO DE BBOXES ─────────────────
UMBRAL_CONTACTO_BBOX       = 2      # Reducido: solo detecta contacto real,
                                    # no proximidad. 0 = solapamiento puro.
FRAMES_MIN_CONTACTO        = 1      # Frames consecutivos de contacto antes
                                    # de tratarlo como fusión real. Evita
                                    # falsos positivos de un solo frame.
FACTOR_AMORTIGUACION_CAPSULA = 0.97 # Factor de desaceleración de la predicción
                                    # de cápsulas. Mucho más suave que
                                    # FACTOR_AMORTIGUACION (0.50) porque las
                                    # fusiones pueden durar 100+ frames.


FRAMES_COOLDOWN_CONTACTO = 6   # Frames de gracia antes de descartar un contacto
                                # no confirmado. Si el par reaparece dentro de
                                # este margen, retoma el contador acumulado.
                                # Aumentar si los blobs oscilan mucho en el umbral.


# ── CONTACTO FÍSICO DE BBOXES ─────────────────
FACTOR_SEPARACION = 2.0    # Distancia en "speeds" a la que se predice la
                            # cápsula invisible respecto al blob visible.
                            # Con speed=3px/frame y factor=3: predice 9px
                            # en la dirección original de la cápsula.
                            # Subir si las personas se separan abruptamente.


# ── FUSIÓN DE FRAGMENTOS ──────────────────────
MERGE_DIST = 60    # Distancia máxima en px entre dos centroides detectados
                   # para considerarlos fragmentos de la misma persona.
                   # Ajustar según el tamaño típico del blob a esa altura.