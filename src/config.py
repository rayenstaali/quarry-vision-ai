"""
Module de configuration générale du système de surveillance en carrière.
Définit les paramètres du modèle, les zones géométriques d'intérêt (ROI),
les classes cibles et les seuils d'alertes.
"""

from pathlib import Path
import numpy as np

# ==============================================================================
# 1. CHEMINS DU PROJET
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Création automatique des répertoires si non existants
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, DATA_DIR / "raw_videos"]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. CONFIGURATION DES MODÈLES DE VISION
# ==============================================================================
# Modèle de détection principal (COCO ou modèle fine-tuné spécifique carrière)
DETECTION_MODEL_PATH = "yolov8n.pt"
FINE_TUNED_MODEL_PATH = MODELS_DIR / "quarry_load_best.pt"
CONFIDENCE_THRESHOLD = 0.35          # Seuil minimal de confiance pour les détections
IOU_THRESHOLD = 0.45                 # Seuil Non-Max Suppression (IoU)

# Classes COCO standards :
TARGET_CLASSES = {
    0: "person",
    2: "car",
    7: "truck"
}

# Classes spécifiques carrière (Dataset Roboflow quarry-load-verification) :
QUARRY_LOAD_CLASSES = {
    0: "BUCKET_CARRY",
    1: "BUCKET_DIG",
    2: "BUCKET_DUMP",
    3: "digger",
    4: "person",
    5: "trailer open bed",
    6: "truck open bed"
}

# ==============================================================================
# 3. CONFIGURATION DU TRACKING (ByteTrack / BoT-SORT)
# ==============================================================================
TRACKER_CONFIG = "bytetrack.yaml"    # Fichier de config Ultralytics pour ByteTrack
TRACK_BUFFER = 30                    # Nombre de frames pour maintenir l'ID en cas d'occlusion

# ==============================================================================
# 4. ZONES GÉOMÉTRIQUES D'INTÉRÊT (ROI - Coordonnées en pixels sur résolution 1280x720)
# ==============================================================================
# Zone 1 : Entrée de la carrière (Détection de l'arrivée du camion)
ZONE_ENTRY = np.array([
    [50, 400],
    [300, 400],
    [350, 680],
    [50, 680]
], dtype=np.int32)

# Zone 2 : Zone de chargement (Poste des pelles/chargeuses)
ZONE_LOADING = np.array([
    [450, 300],
    [850, 300],
    [900, 600],
    [400, 600]
], dtype=np.int32)

# Zone 3 : Zone de sortie / pont-bascule
ZONE_EXIT = np.array([
    [950, 380],
    [1250, 380],
    [1250, 680],
    [920, 680]
], dtype=np.int32)

# Zone 4 : Zone de sécurité / exclusion piétons (Rayon d'action des pelles mécaniques)
ZONE_DANGER = np.array([
    [420, 260],
    [880, 260],
    [920, 500],
    [380, 500]
], dtype=np.int32)

# Dictionnaire regroupant les zones avec leurs couleurs d'affichage (BGR)
ZONES_CONFIG = {
    "entree": {"polygon": ZONE_ENTRY, "color": (255, 165, 0), "label": "Zone Entree"},
    "chargement": {"polygon": ZONE_LOADING, "color": (255, 255, 0), "label": "Zone Chargement"},
    "sortie": {"polygon": ZONE_EXIT, "color": (0, 200, 0), "label": "Zone Sortie"},
    "danger": {"polygon": ZONE_DANGER, "color": (0, 0, 255), "label": "Zone Danger Pietons"}
}

# ==============================================================================
# 5. PARAMÈTRES DU CONTRÔLE DE CHARGEMENT DE LA BENNE
# ==============================================================================
LOAD_STATES = ["empty", "partially_loaded", "loaded", "overloaded"]

BENNE_CROP_RATIO = {
    "top": 0.22,      # 22% pour démarrer au niveau de la ridelle supérieure
    "bottom": 0.82,   # 82% pour capturer l'intérieur de la benne sans le châssis
    "left": 0.04,     # 4% côté arrière
    "right": 0.65     # 65% (jusqu'à la cabine avant)
}

# ==============================================================================
# 6. CONFIGURATION DES ALERTES
# ==============================================================================
ALERT_DEPART_A_VIDE = "Camion quittant le site sans chargement"
ALERT_PIETON_DANGER = "Presence pieton en zone dangereuse de manoeuvre"
ALERT_SURCHARGE     = "Suspicion de debordement / surcharge de la benne"
ALERT_TEMPS_ANORMAL = "Duree de chargement excessive (> 20 min)"
