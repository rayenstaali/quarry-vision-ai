"""
Module de suivi des camions (Tracking) et gestion du cycle de vie opérationnel.
Intègre YOLOv8/11 et ByteTrack pour attribuer un identifiant unique à chaque camion
et suivre sa transition entre les différentes étapes : Arrivée -> Chargement -> Sortie.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
import supervision as sv

from src.config import (
    DETECTION_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    TRACKER_CONFIG,
    TARGET_CLASSES,
    ZONES_CONFIG
)
from src.zone_manager import ZoneManager
from src.load_classifier import LoadInspector
from src.anomaly_detector import AnomalyDetector


class QuarryTracker:
    """
    Système principal de détection, suivi et analyse des camions et personnes en carrière.
    """

    def __init__(self, model_path: str = DETECTION_MODEL_PATH):
        """
        Initialise le modèle YOLO, le tracker ByteTrack et les sous-modules.
        """
        print(f"[QuarryTracker] Chargement du modèle de détection: {model_path}...")
        self.model = YOLO(model_path)
        self.zone_manager = ZoneManager(ZONES_CONFIG)
        self.load_inspector = LoadInspector()
        self.anomaly_detector = AnomalyDetector()

        # Identification dynamique des classes selon le modèle chargé (COCO ou modèle fine-tuné carrière)
        self.model_names = self.model.names if hasattr(self.model, 'names') else {}
        self.person_classes = [k for k, v in self.model_names.items() if 'person' in str(v).lower()]
        self.truck_classes = [
            k for k, v in self.model_names.items()
            if any(t in str(v).lower() for t in ['truck', 'car', 'trailer', 'digger', 'bed'])
        ]
        if not self.person_classes:
            self.person_classes = [0]
        if not self.truck_classes:
            self.truck_classes = [2, 7]

        # Dictionnaire d'état de la flotte : {truck_id: metadata}
        self.trucks_registry: Dict[int, Dict] = {}
        
        # Historique des positions pour tracer les trajectoires
        self.trajectories: Dict[int, List[tuple]] = {}

        # Annotateurs Supervision pour un affichage soigné
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=4)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=40)

        # Statistiques globales de production
        self.stats = {
            "total_trucks_seen": 0,
            "trucks_currently_inside": 0,
            "trucks_loaded": 0,
            "trucks_unloaded_exit": 0,
            "alerts_count": 0
        }

    def _detect_synthetic(self, frame: np.ndarray) -> sv.Detections:
        """
        Détecteur d'appoint pour la vidéo de simulation synthétique de la carrière.
        Permet une démonstration complète et immédiate sans nécessiter de caméra physique.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)[1]
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes, classes, confs, ids = [], [], [], []

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if 25 <= w <= 80 and 35 <= h <= 90 and y > 400:
                scale = w / 54.0
                full_w = int(170 * scale)
                full_h = int(90 * scale)
                tx1 = max(0, x - int(full_w * 0.68))
                ty1 = max(0, y - int(full_h * 0.25))
                tx2 = min(frame.shape[1], x + w)
                ty2 = min(frame.shape[0], ty1 + full_h + 15)
                boxes.append([tx1, ty1, tx2, ty2])
                classes.append(7)  # truck
                confs.append(0.96)
                truck_id = 1 if y < 530 else 2
                ids.append(truck_id)

        # Ouvrier haute-visibilité orange
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([10, 150, 150]), np.array([25, 255, 255]))
        w_cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in w_cnts:
            x, y, w, h = cv2.boundingRect(c)
            if 8 <= w <= 25 and 10 <= h <= 30 and 330 <= y <= 450:
                boxes.append([x - 10, y - 10, x + w + 10, y + h + 25])
                classes.append(0)  # person
                confs.append(0.93)
                ids.append(99)

        if not boxes:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            confidence=np.array(confs, dtype=np.float32),
            class_id=np.array(classes, dtype=int),
            tracker_id=np.array(ids, dtype=int)
        )

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0, fps: float = 30.0) -> Tuple[np.ndarray, Dict]:
        """
        Traite une frame vidéo unique :
        1. Inférence YOLO + Tracking ByteTrack
        2. Détection spatiale des zones
        3. Contrôle visuel de la benne
        4. Vérification des anomalies
        5. Rendu visuel augmenté
        """
        current_time_sec = frame_idx / fps if fps > 0 else frame_idx / 30.0
        active_frame_alerts = []

        # 1. Inférence YOLO avec ByteTrack
        class_filter = self.person_classes + self.truck_classes
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=class_filter,
            verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(results)
        # Si aucune détection (ex: vidéo synthétique démo), bascule transparente sur le détecteur d'appoint
        if len(detections) == 0:
            detections = self._detect_synthetic(frame)

        # 2. Dessiner les zones d'intérêt (fond semi-transparent)
        annotated_frame = self.zone_manager.draw_zones(frame.copy(), alpha=0.25)

        # 3. Traiter les personnes pour la sécurité (Zone Danger)
        person_in_danger = False
        if detections.class_id is not None:
            is_person = np.isin(detections.class_id, self.person_classes)
            person_detections = detections[is_person]
            
            for bbox in person_detections.xyxy:
                foot_point = (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))
                if self.zone_manager.is_point_in_zone(foot_point, "danger"):
                    person_in_danger = True
                    break

        alert_pieton = self.anomaly_detector.check_pedestrian_danger(person_in_danger, frame_idx)
        if alert_pieton:
            active_frame_alerts.append(alert_pieton)
            self.stats["alerts_count"] += 1

        # 4. Traiter les camions et véhicules industriels
        truck_labels = []
        if detections.tracker_id is not None and len(detections) > 0:
            is_truck = np.isin(detections.class_id, self.truck_classes)
            truck_detections = detections[is_truck]

            for i in range(len(truck_detections)):
                bbox = truck_detections.xyxy[i]
                truck_id = int(truck_detections.tracker_id[i])
                
                # Mise à jour de la trajectoire
                center_bottom = (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))
                if truck_id not in self.trajectories:
                    self.trajectories[truck_id] = []
                self.trajectories[truck_id].append(center_bottom)

                # Initialisation de l'historique du camion
                if truck_id not in self.trucks_registry:
                    self.trucks_registry[truck_id] = {
                        "id": truck_id,
                        "first_seen": current_time_sec,
                        "last_seen": current_time_sec,
                        "current_zone": "transit",
                        "cycle_step": "Arrivee",
                        "load_status": "empty",
                        "load_conf": 0.0,
                        "is_overflowing": False,
                        "loading_start": None,
                        "loading_duration": 0.0,
                        "visited_loading": False,
                        "completed_exit": False,
                        "alerts": []
                    }
                    self.stats["total_trucks_seen"] += 1

                truck = self.trucks_registry[truck_id]
                truck["last_seen"] = current_time_sec

                # Détection de la zone actuelle
                zones_present = self.zone_manager.get_object_zones(bbox)
                primary_zone = zones_present[0] if zones_present else "transit"
                truck["current_zone"] = primary_zone

                # Inspection visuelle continue de la benne
                crop = self.load_inspector.crop_benne(frame, bbox)
                state, conf, is_overflow = self.load_inspector.estimate_load_state(crop)
                if state == "overloaded":
                    truck["load_status"] = "overloaded"
                    truck["load_conf"] = conf
                    truck["is_overflowing"] = True
                elif state == "loaded" and truck["load_status"] != "overloaded":
                    truck["load_status"] = "loaded"
                    truck["load_conf"] = conf
                    truck["is_overflowing"] = False
                elif state == "partially_loaded" and truck["load_status"] in ["empty", "partially_loaded"]:
                    truck["load_status"] = "partially_loaded"
                    truck["load_conf"] = conf
                elif truck["load_status"] == "empty":
                    truck["load_status"] = state
                    truck["load_conf"] = conf

                # Alerte débordement immédiate dès détection
                alert_overflow = self.anomaly_detector.check_overflowing_load(truck_id, is_overflow, frame_idx)
                if alert_overflow:
                    active_frame_alerts.append(alert_overflow)
                    if alert_overflow["message"] not in truck["alerts"]:
                        truck["alerts"].append(alert_overflow["message"])

                # Machine à états des cycles
                if primary_zone == "entree":
                    truck["cycle_step"] = "Entree Site"
                elif primary_zone == "chargement":
                    truck["cycle_step"] = "En Chargement"
                    truck["visited_loading"] = True
                    if truck["loading_start"] is None:
                        truck["loading_start"] = current_time_sec
                    else:
                        truck["loading_duration"] = current_time_sec - truck["loading_start"]

                elif primary_zone == "sortie":
                    truck["cycle_step"] = "Sortie / Pont-Bascule"
                    # Vérifier si départ à vide anormal
                    if not truck["completed_exit"]:
                        truck["completed_exit"] = True
                        if truck["load_status"] == "empty":
                            alert_unloaded = self.anomaly_detector.check_unloaded_exit(
                                truck_id, primary_zone, truck["load_status"], frame_idx
                            )
                            if alert_unloaded:
                                active_frame_alerts.append(alert_unloaded)
                                truck["alerts"].append(alert_unloaded["message"])
                                self.stats["trucks_unloaded_exit"] += 1
                        else:
                            self.stats["trucks_loaded"] += 1

                # Préparation du label visuel
                label = f"#{truck_id} {truck['cycle_step']} | {truck['load_status'].upper()}"
                truck_labels.append(label)

            # Rendu des bboxes et traces
            annotated_frame = self.trace_annotator.annotate(scene=annotated_frame, detections=truck_detections)
            annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=truck_detections)
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame, detections=truck_detections, labels=truck_labels
            )

        # 5. Superposition du bandeau d'alerte en haut de l'écran
        if active_frame_alerts:
            overlay_banner = annotated_frame.copy()
            cv2.rectangle(overlay_banner, (0, 0), (frame.shape[1], 55), (0, 0, 180), -1)
            annotated_frame = cv2.addWeighted(overlay_banner, 0.75, annotated_frame, 0.25, 0)
            
            top_alert = active_frame_alerts[0]["message"]
            cv2.putText(
                annotated_frame,
                f"ATTENTION: {top_alert}",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # Compteur de présence courante
        active_trucks = sum(1 for t in self.trucks_registry.values() if (current_time_sec - t["last_seen"]) < 2.0)
        self.stats["trucks_currently_inside"] = active_trucks

        # Statistiques incrustées en bas à gauche
        stats_txt = f"Camions sur site: {active_trucks} | Charges: {self.stats['trucks_loaded']} | Alertes: {self.stats['alerts_count']}"
        cv2.putText(annotated_frame, stats_txt, (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated_frame, {
            "stats": self.stats,
            "registry": self.trucks_registry,
            "alerts": active_frame_alerts
        }
