"""
Module d'inspection et de classification visuelle du chargement de la benne.
Effectue le recadrage (crop) de la benne du camion et analyse le contenu
pour determiner l'etat : empty, partially_loaded, loaded ou overloaded (debordement).
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from src.config import BENNE_CROP_RATIO, LOAD_STATES


class LoadInspector:
    """
    Analyseur visuel de la benne des camions :
    1. Recadrage de la zone de cargaison du camion
    2. Extraction des caracteristiques de remplissage (gradient, densite de granulats, profil de relief)
    3. Classification dans les 4 categories :
       - 'empty' : Benne vide (fond metallique plat uniforme, contraste faible)
       - 'partially_loaded' : Chargement incomplet (materiau present mais en deça des rebords)
       - 'loaded' : Chargement optimal (remplissage homogene a niveau)
       - 'overloaded' : Surcharge / debordement potentiel (matiere depassant les parois laterales)
    """

    def __init__(self, classifier_model_path: Optional[str] = None):
        """
        :param classifier_model_path: Chemin optionnel vers des poids YOLO-cls fine-tunes
        """
        self.classifier_model_path = classifier_model_path
        self.ml_classifier = None
        
        # Si un modele de deep learning personnalise est fourni, on le charge
        if self.classifier_model_path:
            try:
                from ultralytics import YOLO
                self.ml_classifier = YOLO(self.classifier_model_path)
            except Exception as e:
                print(f"[LoadInspector] Avertissement: Impossible de charger le modele {classifier_model_path}: {e}")

    def crop_benne(self, frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
        """
        Recadre la region d'interet correspondant a la benne du camion a partir de la bbox du vehicule.
        """
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Coordonnees relatives de la benne dans le gabarit du camion
        bw = x2 - x1
        bh = y2 - y1
        
        crop_x1 = max(0, int(x1 + bw * BENNE_CROP_RATIO["left"]))
        crop_x2 = min(w_frame, int(x1 + bw * BENNE_CROP_RATIO["right"]))
        crop_y1 = max(0, int(y1 + bh * BENNE_CROP_RATIO["top"]))
        crop_y2 = min(h_frame, int(y1 + bh * BENNE_CROP_RATIO["bottom"]))
        
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None
            
        return frame[crop_y1:crop_y2, crop_x1:crop_x2]

    def estimate_load_state(self, crop: np.ndarray) -> Tuple[str, float, bool]:
        """
        Estime l'etat du chargement a partir du crop de la benne.
        Retourne : (load_state, confidence, is_overflowing)
        """
        if crop is None or crop.size == 0:
            return "empty", 0.5, False

        # Si un modele YOLO-cls est instancie, on l'utilise en priorite
        if self.ml_classifier is not None:
            try:
                res = self.ml_classifier(crop, verbose=False)[0]
                cls_id = int(res.probs.top1)
                conf = float(res.probs.top1conf)
                predicted_state = LOAD_STATES[min(cls_id, len(LOAD_STATES)-1)]
                is_overflow = (predicted_state == "overloaded")
                return predicted_state, conf, is_overflow
            except Exception:
                pass

        # Algorithme robuste par vision par ordinateur industrielle (texture et contraste de remplissage)
        h, w = crop.shape[:2]
        if h < 5 or w < 5:
            return "empty", 0.5, False

        # Analyse de la texture et du relief de matière dans la benne
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = float(laplacian.var())

        # Taux de remplissage : détection de la présence de matière par rapport au fond métallique uniforme [80, 95, 110]
        metal_diff = np.mean(np.abs(crop.astype(float) - [80, 95, 110]), axis=2)
        fill_ratio = float(np.mean(metal_diff > 16))

        # Classification multi-critère industrielle :
        # - 'overloaded' : débordement de roche et fortes projections de matière (variance extrême > 1550)
        # - 'loaded' : chargement plein et uniforme (fill_ratio >= 0.70 avec texture de roche)
        # - 'partially_loaded' : chargement en cours ou partiel
        # - 'empty' : benne vide, surface métallique lisse
        is_overflowing = False
        if texture_variance > 1550:
            load_state = "overloaded"
            is_overflowing = True
            confidence = min(0.97, 0.85 + (texture_variance / 4000.0) * 0.12)
        elif fill_ratio >= 0.70 and texture_variance > 600:
            load_state = "loaded"
            confidence = 0.93
        elif fill_ratio >= 0.22 or texture_variance > 350:
            load_state = "partially_loaded"
            confidence = 0.86
        else:
            load_state = "empty"
            confidence = 0.92

        return load_state, confidence, is_overflowing
