<<<<<<< HEAD
"""
Mapa de cobertura para visualizar el territorio pintado por cada jugador.
"""

=======
>>>>>>> d1c69e2ed2717a018c58a372c588256fb29e2b7b
import cv2
import numpy as np


<<<<<<< HEAD
class CoverageMap:
    """Visualiza qué áreas del espacio normalizado ha visitado cada jugador"""
    
    def __init__(self, size=600):
        self.size = size
        self.map = np.zeros((size, size, 3), dtype=np.uint8)
        self.display = None
        
    def update(self, positions):
        """
        Actualiza el mapa con las posiciones normalizadas actuales.
        
        Args:
            positions: Dict de {player_id: {'normalized_pos': (x, y), 'color': (b,g,r), ...}}
        """
        for player_id, data in positions.items():
            norm_x, norm_y = data['normalized_pos']
            color = data['color']
            
            # Convertir de coordenadas normalizadas a píxeles del mapa
            # norm_x, norm_y están en [0, 1] con origen abajo-izquierda
            px = int(norm_x * self.size)
            py = int((1 - norm_y) * self.size)  # Invertir Y para que coincida con imagen
            
            # Dibujar círculo en el mapa
            if 0 <= px < self.size and 0 <= py < self.size:
                cv2.circle(self.map, (px, py), 3, color, -1)
    
    def draw_legend(self, player_colors):
        """
        Dibuja el mapa con leyenda de colores.
        
        Args:
            player_colors: Dict de {player_id: (b,g,r)}
        """
        # Crear display con espacio para leyenda
        legend_width = 180
        legend_height = max(len(player_colors) * 35 + 40, self.size)  # Ajustar altura
        
        # Crear canvas más grande
        self.display = np.zeros((legend_height, self.size + legend_width, 3), dtype=np.uint8)
        
        # Copiar mapa (ajustar tamaño si es necesario)
        map_h = min(self.size, legend_height)
        self.display[:map_h, :self.size] = self.map[:map_h]
        
        # Dibujar borde del mapa
        cv2.rectangle(self.display, (0, 0), (self.size-1, map_h-1), (100, 100, 100), 2)
        
        # Título de la leyenda
        cv2.putText(self.display, "JUGADORES:", 
                    (self.size + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Dibujar cada jugador en la leyenda
        y_offset = 60
        for player_id in sorted(player_colors.keys()):
            color = player_colors[player_id]
            
            # Círculo de color
            center_x = self.size + 30
            center_y = y_offset
            cv2.circle(self.display, (center_x, center_y), 12, color, -1)
            cv2.circle(self.display, (center_x, center_y), 12, (255, 255, 255), 2)
            
            # Texto del ID
            text = f"ID {player_id}"
            cv2.putText(self.display, text,
                       (center_x + 25, center_y + 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            y_offset += 35
        
        # Mostrar el mapa
        cv2.imshow('Mapa de Cobertura', self.display)
    
    def clear(self):
        """Limpia el mapa de cobertura"""
        self.map = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        print("  Mapa de cobertura limpiado")
=======
# Tamaño del canvas del mapa de cobertura
MAP_SIZE = 600


class CoverageMap:
    """Gestiona el mapa de cobertura donde se pintan las trayectorias normalizadas"""

    def __init__(self, size=MAP_SIZE):
        self.size = size
        self.canvas = np.zeros((size, size, 3), dtype=np.uint8)
        self._create_window()

    def _create_window(self):
        """Crea la ventana del mapa"""
        cv2.namedWindow('Mapa de Cobertura', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Mapa de Cobertura', self.size, self.size)

    def update(self, positions):
        """
        Pinta los puntos de cada jugador en el mapa.

        positions: dict {track_id: {'normalized_pos': (x,y), 'color': (b,g,r)}}
        """
        for track_id, data in positions.items():
            norm_x, norm_y = data['normalized_pos']
            color = data['color']

            # Convertir coordenadas normalizadas (0-1) a píxeles del mapa
            # norm_y se invierte porque en imagen Y crece hacia abajo
            map_x = int(norm_x * (self.size - 1))
            map_y = int((1.0 - norm_y) * (self.size - 1))

            # Clamp por seguridad
            map_x = max(0, min(self.size - 1, map_x))
            map_y = max(0, min(self.size - 1, map_y))

            cv2.circle(self.canvas, (map_x, map_y), 4, color, -1)

    def draw_grid(self):
        """Dibuja una cuadrícula de referencia (0.25, 0.5, 0.75)"""
        overlay = self.canvas.copy()
        step = self.size // 4

        for i in range(1, 4):
            pos = i * step
            cv2.line(overlay, (pos, 0), (pos, self.size), (40, 40, 40), 1)
            cv2.line(overlay, (0, pos), (self.size, pos), (40, 40, 40), 1)

        # Etiquetas de esquinas
        labels = {
            (5, self.size - 10): "(0,0)",
            (self.size - 50, self.size - 10): "(1,0)",
            (5, 15): "(0,1)",
            (self.size - 50, 15): "(1,1)",
        }
        for (lx, ly), text in labels.items():
            cv2.putText(overlay, text, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

        return overlay

    def show(self):
        """Muestra el mapa con la cuadrícula de referencia"""
        display = self.draw_grid()
        cv2.imshow('Mapa de Cobertura', display)

    def clear(self):
        """Limpia el mapa de cobertura"""
        self.canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        print("Mapa de cobertura limpiado")

    def draw_legend(self, player_colors):
        """
        Dibuja leyenda de colores por jugador.
        player_colors: dict {track_id: (b,g,r)}
        """
        legend_height = max(len(player_colors) * 25 + 20, 60)
        legend = np.zeros((legend_height, 160, 3), dtype=np.uint8)

        cv2.putText(legend, "Jugadores:", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        for i, (track_id, color) in enumerate(player_colors.items()):
            y = 35 + i * 22
            cv2.circle(legend, (15, y - 4), 7, color, -1)
            cv2.putText(legend, f"ID {track_id}", (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Pegar leyenda en esquina superior derecha del canvas
        display = self.draw_grid()
        lh, lw = legend.shape[:2]
        x_off = self.size - lw - 5
        y_off = 5
        display[y_off:y_off + lh, x_off:x_off + lw] = legend

        cv2.imshow('Mapa de Cobertura', display)
>>>>>>> d1c69e2ed2717a018c58a372c588256fb29e2b7b
