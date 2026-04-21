import cv2 as cv
import numpy as np

# Segmento (borde)
p1 = (100, 300)
p2 = (500, 300)

# Blob (puedes moverlo con el mouse)
blob = [300, 150]

def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_MOUSEMOVE:
        blob[0] = x
        blob[1] = y

def calcular_proyeccion(px, py, p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1

    largo2 = dx*dx + dy*dy

    if largo2 == 0:
        return x1, y1

    t = ((px - x1)*dx + (py - y1)*dy) / largo2

    # limitar entre 0 y 1
    t = max(0, min(1, t))

    proj_x = int(x1 + t * dx)
    proj_y = int(y1 + t * dy)

    return proj_x, proj_y

# Ventana
cv.namedWindow("Demo")
cv.setMouseCallback("Demo", mouse_callback)

while True:
    img = np.zeros((500, 600, 3), dtype=np.uint8)

    px, py = blob

    # calcular proyección
    proj_x, proj_y = calcular_proyeccion(px, py, p1, p2)

    # dibujar segmento
    cv.line(img, p1, p2, (255, 255, 255), 2)

    # dibujar blob
    cv.circle(img, (px, py), 6, (0, 255, 0), -1)

    # dibujar punto proyectado
    cv.circle(img, (proj_x, proj_y), 6, (0, 0, 255), -1)

    # línea de distancia
    cv.line(img, (px, py), (proj_x, proj_y), (255, 0, 0), 1)

    # texto
    cv.putText(img, f"Blob: ({px},{py})", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv.putText(img, f"Proy: ({proj_x},{proj_y})", (10, 50),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    cv.imshow("Demo", img)

    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()