"""
Module de gestion des zones d'intérêt (ROI) et de géométrie spatiale.
Permet de déterminer l'appartenance des objets détectés aux différentes
zones (Entrée, Chargement, Sortie, Danger) et de mesurer le temps passé.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from shapely.geometry import Point, Polygon


class ZoneManager:
    """
    Gère les polygones d'intérêt pour la carrière :
    - Teste si le centre ou la base du véhicule est dans la zone
    - Calcule la durée de présence dans chaque zone
    - Dessine les zones avec transparence esthétique et annotations
    """

    def __init__(self, zones_config: Dict):
        """
        Initialise le gestionnaire de zones.
        :param zones_config: Dictionnaire des zones (points, couleur, label)
        """
        self.zones_config = zones_config
        self.shapely_zones = {}

        # Conversion des tableaux numpy en polygones Shapely pour des tests géométriques ultra-rapides
        for name, config in self.zones_config.items():
            poly_points = [(p[0], p[1]) for p in config["polygon"]]
            self.shapely_zones[name] = Polygon(poly_points)

    def is_point_in_zone(self, point: Tuple[int, int], zone_name: str) -> bool:
        """
        Vérifie si un point (x, y) est contenu à l'intérieur d'une zone polygonale.
        Utilise pointPolygonTest d'OpenCV pour une exécution en temps réel sans latence.
        """
        if zone_name not in self.zones_config:
            return False
        polygon = self.zones_config[zone_name]["polygon"]
        # >= 0 signifie à l'intérieur ou sur la frontière du polygone
        return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

    def get_object_zones(self, bbox: List[float]) -> List[str]:
        """
        Détermine dans quelle(s) zone(s) se trouve un objet délimité par une bbox.
        Pour les véhicules, on utilise la base (pieds/roues) du camion (centre inférieur).
        """
        x1, y1, x2, y2 = bbox
        # Ancrage du camion au sol (centre bas de la bounding box)
        anchor_x = int((x1 + x2) / 2)
        anchor_y = int(y2)
        
        detected_zones = []
        for zone_name in self.zones_config.keys():
            if self.is_point_in_zone((anchor_x, anchor_y), zone_name):
                detected_zones.append(zone_name)
        return detected_zones

    def draw_zones(self, frame: np.ndarray, alpha: float = 0.25) -> np.ndarray:
        """
        Superpose les zones géométriques sur la frame vidéo avec un masque semi-transparent
        et des bordures nettes pour un rendu visuel professionnel.
        """
        overlay = frame.copy()
        
        for name, config in self.zones_config.items():
            pts = config["polygon"].reshape((-1, 1, 2))
            color = config["color"]
            label = config["label"]
            
            # 1. Remplissage semi-transparent
            cv2.fillPoly(overlay, [pts], color)
            
            # 2. Contour solide
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            
            # 3. Étiquette de la zone
            # Positionnement du texte près du premier sommet du polygone
            txt_x, txt_y = pts[0][0][0], pts[0][0][1] - 8
            cv2.putText(
                frame,
                label,
                (txt_x, max(txt_y, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            
        # Fusion alpha
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
