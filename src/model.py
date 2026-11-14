"""
Module d'instanciation, d'inférence et d'export du modèle de vision QuarryVision AI.
Fournit une interface uniforme pour les modèles YOLOv8 (baseline COCO ou fine-tuné carrière).
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


class QuarryVisionModel:
    """
    Wrapper pour charger, exécuter et exporter les modèles de détection YOLOv8.
    """

    def __init__(self, weights_path: Optional[str] = None):
        if weights_path is None:
            default_ft = MODELS_DIR / "quarry_load_best.pt"
            self.weights_path = str(default_ft) if default_ft.exists() else "yolov8n.pt"
        else:
            self.weights_path = weights_path

        print(f"[Model] Chargement des poids : {self.weights_path}")
        self.model = YOLO(self.weights_path)
        self.class_names: Dict[int, str] = self.model.names if hasattr(self.model, "names") else {}

    def predict(self, source, conf: float = 0.35, iou: float = 0.45, imgsz: int = 640):
        """
        Inférence standard sur image unique, dossier ou frame NumPy.
        """
        return self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False
        )

    def track(self, source, tracker_config: str = "bytetrack.yaml", conf: float = 0.35, persist: bool = True):
        """
        Inférence avec suivi multi-objets (ByteTrack / BoT-SORT).
        """
        return self.model.track(
            source=source,
            tracker=tracker_config,
            conf=conf,
            persist=persist,
            verbose=False
        )

    def export(self, format: str = "onnx"):
        """
        Exporte le modèle vers un format de production (ex: ONNX, TensorRT).
        """
        print(f"[Model] Export vers le format {format.upper()}...")
        return self.model.export(format=format)

    def get_summary(self) -> Dict[str, Any]:
        """
        Retourne les métadonnées techniques du modèle.
        """
        return {
            "weights": self.weights_path,
            "classes_count": len(self.class_names),
            "classes": self.class_names
        }


if __name__ == "__main__":
    qvm = QuarryVisionModel()
    print("Résumé du modèle :", qvm.get_summary())
