import cv2
import numpy as np

_GROUPS = [
    {
        "titulo": "GENERAL  (todos los modos)",
        "color":  (100, 100, 100),
        "items": [
            ("detection_mode",        "MODO          (0-5)",   5),
            ("min_area",              "Min Area",          10000),
            ("max_area",              "Max Area",          20000),
            ("morph_kernel",          "Morph Kernel",         21),
            ("max_distance",          "Max Distance",        300),
            ("max_lost_frames",       "Max Lost Frames",      60),
            ("process_noise",         "Process Noise",        10),
        ],
    },
    {
        "titulo": "MODO 0 — CIRCLES",
        "color":  (100, 200, 255),
        "items": [
            ("accumulator_threshold", "[0] Accum Thresh",    100),
            ("min_dist",              "[0] Min Distance",    200),
            ("min_radius",            "[0] Min Radius",      100),
            ("max_radius",            "[0] Max Radius",      200),
            ("blur_kernel",           "[0] Blur Kernel",      31),
            ("threshold_value",       "[0] Threshold Val",   255),
            ("canny_threshold",       "[0] Canny Thresh",    200),
        ],
    },
    {
        "titulo": "MODO 1 — SHADOWS",
        "color":  (180, 100, 255),
        "items": [
            ("shadow_threshold",      "[1] Shadow Thresh",   100),
            ("shadow_blur",           "[1] Shadow Blur",      51),
            ("local_window",          "[1] Local Window",    101),
        ],
    },
    {
        "titulo": "MODO 2 — ADAPTIVE BG",
        "color":  (100, 255, 150),
        "items": [
            ("bg_threshold",          "[2] BG Threshold",    100),
            ("bg_buffer_size",        "[2] Buffer Size",      60),
            ("bg_min_frames",         "[2] Min Frames",       30),
        ],
    },
    {
        "titulo": "MODO 3 — HYBRID",
        "color":  (255, 200, 100),
        "items": [
            ("merge_distance",        "[3] Merge Distance",  100),
        ],
    },
    {
        "titulo": "MODO 4 — OPTICAL FLOW",
        "color":  (255, 100, 255),
        "items": [
            ("flow_threshold_x10",    "[4] Flow Threshold",  100),
        ],
    },
    {
        "titulo": "MODO 5 — BLOB TRACKING",
        "color":  (100, 255, 255),
        "items": [
            ("blob_similarity_threshold_x100", "[5] Similarity Thresh", 100),
        ],
    },
]

_WIN_W   = 460
_WIN_H   = 680
_ROW_H   = 30
_LABEL_W = 200

_st = {"scroll": 0, "drag": None, "vals": {}}

def _build_rows():
    rows = []
    for g, group in enumerate(_GROUPS):
        rows.append(("sep", g))
        for i in range(len(group["items"])):
            rows.append(("slider", g, i))
    return rows

_ROWS    = _build_rows()
_TOTAL_H = len(_ROWS) * _ROW_H + 20


def _mouse(event, x, y, flags, param):
    max_scroll = max(0, _TOTAL_H - _WIN_H)

    if event == cv2.EVENT_MOUSEWHEEL:
        direction = -1 if flags > 0 else 1
        _st["scroll"] = max(0, min(max_scroll, _st["scroll"] + direction * 40))
        return

    abs_y = y + _st["scroll"]
    row_i = abs_y // _ROW_H
    if row_i < 0 or row_i >= len(_ROWS):
        return
    row = _ROWS[row_i]
    if row[0] != "slider":
        return

    _, g, i = row
    key, label, maxv = _GROUPS[g]["items"][i]
    bx0, bx1 = _LABEL_W, _WIN_W - 52
    bw = bx1 - bx0

    if event == cv2.EVENT_LBUTTONDOWN and bx0 <= x <= bx1:
        _st["drag"] = (g, i)
        _st["vals"][key] = max(0, min(maxv, int((x - bx0) / bw * maxv)))
    elif event == cv2.EVENT_MOUSEMOVE and _st["drag"] == (g, i) and bx0 <= x <= bx1:
        _st["vals"][key] = max(0, min(maxv, int((x - bx0) / bw * maxv)))
    elif event == cv2.EVENT_LBUTTONUP:
        _st["drag"] = None


def _draw():
    canvas_h = _TOTAL_H + 20
    img = np.zeros((canvas_h, _WIN_W, 3), dtype=np.uint8)

    for row_i, row in enumerate(_ROWS):
        y0 = row_i * _ROW_H
        yc = y0 + _ROW_H // 2 + 5

        if row[0] == "sep":
            g   = row[1]
            col = _GROUPS[g]["color"]
            cv2.rectangle(img, (0, y0), (_WIN_W, y0 + _ROW_H), (28, 28, 28), -1)
            cv2.rectangle(img, (0, y0), (5, y0 + _ROW_H), col, -1)
            cv2.putText(img, _GROUPS[g]["titulo"], (12, y0 + 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)
        else:
            _, g, i = row
            key, label, maxv = _GROUPS[g]["items"][i]
            val = _st["vals"].get(key, 0)
            col = _GROUPS[g]["color"]
            bg  = (22, 22, 22) if i % 2 == 0 else (16, 16, 16)

            cv2.rectangle(img, (0, y0), (_WIN_W, y0 + _ROW_H), bg, -1)
            cv2.putText(img, label, (8, yc),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, col, 1)

            bx0, bx1 = _LABEL_W, _WIN_W - 52
            bw  = bx1 - bx0
            by  = y0 + _ROW_H // 2
            fill = int(val / maxv * bw) if maxv > 0 else 0

            cv2.rectangle(img, (bx0, by - 3), (bx1, by + 3), (50, 50, 50), -1)
            if fill > 0:
                cv2.rectangle(img, (bx0, by - 3), (bx0 + fill, by + 3), col, -1)
            cv2.circle(img, (bx0 + fill, by), 7, (240, 240, 240), -1)
            cv2.circle(img, (bx0 + fill, by), 7, col, 2)
            cv2.putText(img, str(val), (bx1 + 4, yc),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)

    s   = _st["scroll"]
    vis = img[s: s + _WIN_H].copy()

    # Scrollbar
    if _TOTAL_H > _WIN_H:
        sbx = _WIN_W - 7
        sy0 = int(s / _TOTAL_H * _WIN_H)
        sy1 = int((s + _WIN_H) / _TOTAL_H * _WIN_H)
        cv2.rectangle(vis, (sbx, 0), (_WIN_W - 1, _WIN_H), (35, 35, 35), -1)
        cv2.rectangle(vis, (sbx, sy0), (_WIN_W - 1, sy1), (140, 140, 140), -1)

    return vis


# ─────────────────────────────────────────────────────────────
#  API PÚBLICA
# ─────────────────────────────────────────────────────────────
def create_controls_window(params):
    for group in _GROUPS:
        for key, _, _ in group["items"]:
            _st["vals"][key] = int(params.get(key, 0))
    _st["vals"]["flow_threshold_x10"] = int(params.get("flow_threshold", 2.0) * 10)

    cv2.namedWindow("Controles", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controles", _WIN_W, _WIN_H)
    cv2.setMouseCallback("Controles", _mouse)
    create_help_window(params.get("detection_mode", 0))
    return None


def render_controls():
    cv2.imshow("Controles", _draw())


def read_controls(params):
    v = _st["vals"]
    params["detection_mode"]        = v.get("detection_mode", 0)
    params["min_area"]              = v.get("min_area", 200)
    params["max_area"]              = v.get("max_area", 5000)
    params["morph_kernel"]          = max(1, v.get("morph_kernel", 5))
    params["max_distance"]          = v.get("max_distance", 100)
    params["max_lost_frames"]       = v.get("max_lost_frames", 15)
    params["process_noise"]         = v.get("process_noise", 3)
    params["accumulator_threshold"] = v.get("accumulator_threshold", 30)
    params["min_dist"]              = v.get("min_dist", 30)
    params["min_radius"]            = v.get("min_radius", 10)
    params["max_radius"]            = v.get("max_radius", 50)
    params["blur_kernel"]           = max(1, v.get("blur_kernel", 9))
    params["threshold_value"]       = v.get("threshold_value", 100)
    params["canny_threshold"]       = v.get("canny_threshold", 50)
    params["shadow_threshold"]      = v.get("shadow_threshold", 30)
    params["shadow_blur"]           = max(3, v.get("shadow_blur", 21))
    params["local_window"]          = max(3, v.get("local_window", 51))
    params["bg_threshold"]          = v.get("bg_threshold", 25)
    params["bg_buffer_size"]        = v.get("bg_buffer_size", 30)
    params["bg_min_frames"]         = v.get("bg_min_frames", 10)
    params["merge_distance"]        = v.get("merge_distance", 30)
    params["flow_threshold"]        = v.get("flow_threshold_x10", 20) / 10.0
    params["blob_similarity_threshold"] = v.get("blob_similarity_threshold_x100", 30) / 100.0

    if params["shadow_blur"] % 2 == 0:    params["shadow_blur"] += 1
    if params["local_window"] % 2 == 0:   params["local_window"] += 1
    if params["min_radius"] >= params["max_radius"]:  params["max_radius"] = params["min_radius"] + 1
    if params["min_area"]   >= params["max_area"]:    params["max_area"]   = params["min_area"] + 100
    return params


def update_help_display(mode):
    create_help_window(mode)


def create_help_window(mode):
    cv2.namedWindow("AYUDA", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AYUDA", 1000, 560)
    img   = np.zeros((560, 1000, 3), dtype=np.uint8)
    cols  = [(100,200,255),(180,100,255),(100,255,150),(255,200,100),(255,100,255),(100,255,255)]
    names = ["CIRCLES","SHADOWS","ADAPTIVE","HYBRID","OPTFLOW","BLOB"]

    cv2.putText(img, f"MODO {mode}: {names[mode]}",
                (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, cols[mode], 2)
    cv2.line(img, (15, 44), (485, 44), cols[mode], 3)

    guides = {
        0: [("CIRCLES — Simulacion / Tablet",(255,255,255),0.55),("",None,0),
            ("[0] Accum Thresh  CLAVE (15-40)",(255,200,100),0.55),
            ("  Bajo = detecta mas circulos",(180,180,180),0.50),
            ("  Alto = menos falsos positivos",(180,180,180),0.50),
            ("[0] Min/Max Radius = tamano px",(255,200,100),0.55),
            ("[0] Min Distance  = separacion",(255,200,100),0.55),
            ("[0] Blur Kernel   = suavizado",(255,200,100),0.55)],
        1: [("SHADOWS — Cubo Negro",(255,255,255),0.55),("",None,0),
            ("[1] Shadow Thresh  CLAVE (15-50)",(255,200,100),0.55),
            ("  Bajo = sombras suaves",(180,180,180),0.50),
            ("  Alto = solo muy oscuras",(180,180,180),0.50),
            ("[1] Shadow Blur = ignora animac.",(255,200,100),0.55),
            ("  Sube si proyector da ruido",(180,180,180),0.50),
            ("[1] Local Window = contexto",(255,200,100),0.55)],
        2: [("ADAPTIVE BG — Cubo Negro",(255,255,255),0.55),("",None,0),
            ("[2] BG Threshold = sensibilidad",(255,200,100),0.55),
            ("[2] Buffer Size  = frames mem.",(255,200,100),0.55),
            ("  Bajo=rapido  Alto=estable",(180,180,180),0.50),
            ("[2] Min Frames   = inicio",(255,200,100),0.55),("",None,0),
            ("! Esperar sin personas al inicio",(255,100,100),0.52)],
        3: [("HYBRID — Shadows + Adaptive",(255,255,255),0.55),("",None,0),
            ("[3] Merge Distance = fusion",(255,200,100),0.55),("",None,0),
            ("Configura tambien:",(255,255,255),0.55),
            ("  Params [1] Shadows",(180,180,180),0.50),
            ("  Params [2] Adaptive",(180,180,180),0.50),("",None,0),
            ("+ Mas precision",(100,255,100),0.50),("- Mas lento",(100,100,255),0.50)],
        4: [("OPTICAL FLOW — RECOMENDADO!",(100,255,100),0.60),("",None,0),
            ("[4] Flow Thresh  CLAVE (10-40)",(255,200,100),0.55),
            ("  10-15 = movimiento lento",(180,180,180),0.50),
            ("  20-30 = normal",(180,180,180),0.50),
            ("  30-40 = solo rapido",(180,180,180),0.50),("",None,0),
            ("+ Ignora animaciones estaticas",(100,255,100),0.50),
            ("+ Detecta velocidad y direccion",(100,255,100),0.50),
            ("+ No necesita IDs fijos",(100,255,100),0.50),
            ("+ Robusto a cambios de luz",(100,255,100),0.50),("",None,0),
            ("Presiona V para ver el flujo!",(255,255,0),0.55)],
        5: [("BLOB TRACKING — Del Paper",(100,255,255),0.60),("",None,0),
            ("[5] Similarity Thresh (20-50)",(255,200,100),0.55),
            ("  Bajo = acepta matches lejanos",(180,180,180),0.50),
            ("  Alto = solo matches exactos",(180,180,180),0.50),("",None,0),
            ("Tracking por Area+Color+Centroide",(255,255,255),0.52),("",None,0),
            ("+ Mas simple que Kalman",(100,255,100),0.50),
            ("+ No pierde IDs facilmente",(100,255,100),0.50),
            ("+ Usa BackgroundSubtractorMOG2",(100,255,100),0.50),
            ("+ Elimina sombras automaticamente",(100,255,100),0.50),("",None,0),
            ("Ideal para Cubo Negro!",(255,255,0),0.55)],
    }

    y = 60
    for text, color, size in guides[mode]:
        if text:
            cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1)
        y += 26

    cv2.putText(img, "PARAMETROS GENERALES",
                (510, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.line(img, (510, 44), (985, 44), (0,255,255), 3)

    right = [
        ("Sin prefijo = TODOS los modos",(255,255,255),0.55),("",None,0),
        ("Min/Max Area   tamano deteccion",(220,220,220),0.50),
        ("Morph Kernel   limpieza",(220,220,220),0.50),
        ("Max Distance   asociacion IDs",(220,220,220),0.50),
        ("Max Lost Fr.   tolerancia",(220,220,220),0.50),
        ("Process Noise  suavidad",(220,220,220),0.50),("",None,0),
        ("PREFIJOS:",(0,255,255),0.58),
        ("[0] = CIRCLES",cols[0],0.52),("[1] = SHADOWS",cols[1],0.52),
        ("[2] = ADAPTIVE",cols[2],0.52),("[3] = HYBRID",cols[3],0.52),
        ("[4] = OPTFLOW",cols[4],0.52),("[5] = BLOB",cols[5],0.52),("",None,0),
        ("TECLAS:",(0,255,255),0.58),
        ("M  Cambiar modo  (0-1-2-3-4-0)",(255,255,255),0.50),
        ("V  Debug visual procesamiento",(255,255,255),0.50),
        ("D  Puntos debug amarillos",(255,255,255),0.50),
        ("R  Reset IDs",(255,255,255),0.50),
        ("C  Limpiar mapa cobertura",(255,255,255),0.50),
        ("P  Imprimir posiciones",(255,255,255),0.50),
        ("S  Guardar config",(255,255,255),0.50),
        ("Q  Salir",(255,255,255),0.50),
    ]
    y = 60
    for text, color, size in right:
        if text:
            cv2.putText(img, text, (510, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1)
        y += 26

    cv2.line(img, (15, 470), (985, 470), (70,70,70), 2)
    y = 490
    for text, color, size in [
        ("GUIA:",(0,255,255),0.60),
        ("Tablet->MODO=0  |  Cubo Negro->MODO=4  |  Ajusta param CLAVE  |  Presiona S para guardar",(200,200,200),0.46),
    ]:
        cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1)
        y += 28

    cv2.imshow("AYUDA", img)


def save_config(params):
    with open("tracking_config.txt", "w") as f:
        for k, v in params.items():
            f.write(f"{k} = {v}\n")
    print("CONFIG GUARDADA en tracking_config.txt")