"""
Module de détection des situations anormales et des alertes de sécurité en carrière.
Vérifie en continu les règles métier :
1. Présence d'un piéton dans une zone de manœuvre dangereuse
2. Camion quittant la carrière à vide (anomalie de rotation/chargement)
3. Chargement débordant de la benne (risque de sécurité routière)
4. Durée de chargement excessive (blocage du poste de chargement)
"""

from typing import Dict, List, Optional
import time


class AnomalyDetector:
    """
    Moteur de règles d'anomalies opérationnelles et de sécurité.
    """

    def __init__(self):
        # Historique des alertes levées (journal des incidents uniques)
        self.active_alerts: List[Dict] = []
        self.alert_history: List[Dict] = []
        # Suivi d'état pour éviter les alertes dupliquées sur le même incident (hystérésis)
        self.pedestrian_in_danger: bool = False
        self._last_pedestrian_frame: int = -9999
        self.logged_overflow_trucks: set = set()

    def check_pedestrian_danger(self, person_in_danger_zone: bool, frame_idx: int) -> Optional[Dict]:
        """
        Règle 1 : Présence d'une personne dans la zone d'évolution des engins lourds.
        Déclenche l'affichage d'alerte en continu et consigne l'incident une fois au journal
        avec filtre anti-rebond (hystérésis).
        """
        if person_in_danger_zone:
            alert = {
                "type": "SECURITE_PIETON",
                "severity": "CRITICAL",
                "message": "PIÉTON DÉTECTÉ DANS LA ZONE D'ENGINS !",
                "frame": frame_idx,
                "timestamp": time.strftime("%H:%M:%S")
            }
            # Enregistrement unique de l'incident (hystérésis de 45 frames pour éviter le flicker sur la frontière de zone)
            if not self.pedestrian_in_danger and (frame_idx - self._last_pedestrian_frame > 45):
                self.alert_history.append(alert)
            self.pedestrian_in_danger = True
            self._last_pedestrian_frame = frame_idx
            return alert
        else:
            if frame_idx - self._last_pedestrian_frame > 45:
                self.pedestrian_in_danger = False
            return None

    def check_unloaded_exit(self, truck_id: int, current_zone: str, last_known_load: str, frame_idx: int) -> Optional[Dict]:
        """
        Règle 2 : Camion franchissant la zone de sortie alors que sa benne est vide.
        """
        if current_zone == "sortie" and last_known_load == "empty":
            alert = {
                "type": "ANOMALIE_ROTATION",
                "severity": "WARNING",
                "truck_id": truck_id,
                "message": f"Camion #{truck_id} quitte le site sans chargement !",
                "frame": frame_idx,
                "timestamp": time.strftime("%H:%M:%S")
            }
            self.alert_history.append(alert)
            return alert
        return None

    def check_overflowing_load(self, truck_id: int, is_overflowing: bool, frame_idx: int) -> Optional[Dict]:
        """
        Règle 3 : Chargement débordant au-dessus des ridelles ou sur les côtés.
        Alerte visuelle active tant que le débordement dure, consigné une fois par camion.
        """
        if is_overflowing:
            alert = {
                "type": "SECURITE_CHARGEMENT",
                "severity": "WARNING",
                "truck_id": truck_id,
                "message": f"Camion #{truck_id} : chargement débordant (risque chute matière) !",
                "frame": frame_idx,
                "timestamp": time.strftime("%H:%M:%S")
            }
            # Consigner l'incident au journal une seule fois par camion
            if truck_id not in self.logged_overflow_trucks:
                self.alert_history.append(alert)
                self.logged_overflow_trucks.add(truck_id)
            return alert
        return None

    def check_dwell_time(self, truck_id: int, current_zone: str, duration_seconds: float, max_allowed_seconds: float = 900.0) -> Optional[Dict]:
        """
        Règle 4 : Temps de présence excessif dans la zone de chargement (> 15-20 minutes).
        """
        if current_zone == "chargement" and duration_seconds > max_allowed_seconds:
            alert = {
                "type": "ANOMALIE_DUREE",
                "severity": "INFO",
                "truck_id": truck_id,
                "message": f"Camion #{truck_id} : temps de chargement anormalement long ({duration_seconds/60:.1f} min)",
                "timestamp": time.strftime("%H:%M:%S")
            }
            self.alert_history.append(alert)
            return alert
        return None
