"""
Module d'évaluation expérimentale pour QuarryVision AI.
Calcule les métriques académiques standards (mAP@50, mAP@50-95, Précision, Rappel, Matrices de confusion)
sur le jeu de validation ou de test du dataset quarry-load-verification.
"""

import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATASET_YAML = BASE_DIR / "data" / "quarry_dataset" / "data.yaml"


def evaluate_model(weights_path: str, data_yaml: str = str(DATASET_YAML), split: str = "test", imgsz: int = 640):
    """
    Lance l'évaluation complète d'un checkpoint YOLOv8 sur un split donné.
    """
    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(f"Fichier de poids introuvable : {weights_path}")

    yaml_p = Path(data_yaml)
    if not yaml_p.exists():
        raise FileNotFoundError(f"Configuration du dataset introuvable : {data_yaml}")

    print("=" * 70)
    print(f"📊 ÉVALUATION DU MODÈLE : {weights.name}")
    print(f"📁 Dataset config : {yaml_p}")
    print(f"🔍 Jeu de données évalué : split = '{split}'")
    print("=" * 70)

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(yaml_p),
        split=split,
        imgsz=imgsz,
        batch=16,
        plots=True,
        save_json=True
    )

    print("\n" + "=" * 70)
    print("📈 RÉSULTATS D'ÉVALUATION :")
    print(f"  • mAP@50     : {metrics.box.map50:.4f} ({metrics.box.map50 * 100:.2f}%)")
    print(f"  • mAP@50-95  : {metrics.box.map:.4f} ({metrics.box.map * 100:.2f}%)")
    print(f"  • Précision  : {metrics.box.mp:.4f} ({metrics.box.mp * 100:.2f}%)")
    print(f"  • Rappel     : {metrics.box.mr:.4f} ({metrics.box.mr * 100:.2f}%)")
    print("=" * 70)

    print("\n📋 Détails par classe (mAP@50) :")
    for cls_id, cls_name in model.names.items():
        if cls_id < len(metrics.box.maps):
            print(f"  - [{cls_id}] {cls_name:20s} : mAP@50 = {metrics.box.maps[cls_id]:.4f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation expérimentale du modèle YOLOv8")
    parser.add_argument("--weights", type=str, default=str(MODELS_DIR / "quarry_load_best.pt"), help="Chemin des poids .pt")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test", "train"], help="Split à évaluer")
    parser.add_argument("--imgsz", type=int, default=640, help="Résolution de l'image")
    args = parser.parse_args()

    evaluate_model(args.weights, split=args.split, imgsz=args.imgsz)
