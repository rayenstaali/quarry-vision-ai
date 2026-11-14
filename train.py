"""
Script d'entraînement et de fine-tuning YOLOv8 pour la vérification du chargement en carrière.
Utilise le dataset Roboflow 'quarry-load-verification' (7 classes industrielles) :
- BUCKET_CARRY, BUCKET_DIG, BUCKET_DUMP
- digger, person
- trailer open bed, truck open bed
"""

import sys
from pathlib import Path
import shutil
import argparse
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATASET_YAML = BASE_DIR / "data" / "quarry_dataset" / "data.yaml"
PRETRAINED_MODEL = BASE_DIR / "yolov8n.pt"
OUTPUT_MODELS_DIR = BASE_DIR / "models"
OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_quarry_model(epochs: int = 15, batch_size: int = 16, imgsz: int = 512):
    print("=" * 70)
    print("🚀 DÉMARRAGE DU FINE-TUNING YOLOv8 - QUARRY LOAD VERIFICATION")
    print("=" * 70)
    print(f"Dataset YAML   : {DATASET_YAML}")
    print(f"Modèle source  : {PRETRAINED_MODEL}")
    print(f"Époques        : {epochs}")
    print(f"Taille images  : {imgsz}")
    print(f"Taille batch   : {batch_size}")
    
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Accélération   : {'GPU (CUDA)' if device != 'cpu' else 'CPU'}")
    print("=" * 70)

    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"Fichier data.yaml introuvable : {DATASET_YAML}")

    # Chargement du modèle pré-entraîné
    model = YOLO(str(PRETRAINED_MODEL))

    # Entraînement
    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=str(BASE_DIR / "runs" / "quarry_train"),
        name="quarry_v8n",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        patience=8,
        save=True,
        plots=True,
        verbose=True
    )

    # Récupération et archivage des meilleurs poids
    best_weight = BASE_DIR / "runs" / "quarry_train" / "quarry_v8n" / "weights" / "best.pt"
    target_weight = OUTPUT_MODELS_DIR / "quarry_load_best.pt"
    
    if best_weight.exists():
        shutil.copy2(best_weight, target_weight)
        print(f"\n✅ Modèle optimisé sauvegardé : {target_weight}")
    else:
        last_weight = BASE_DIR / "runs" / "quarry_train" / "quarry_v8n" / "weights" / "last.pt"
        if last_weight.exists():
            shutil.copy2(last_weight, target_weight)
            print(f"\n✅ Dernier modèle sauvegardé : {target_weight}")

    # Évaluation sur le split de validation
    print("\n🔍 Évaluation finale sur le jeu de validation...")
    val_model = YOLO(str(target_weight))
    metrics = val_model.val(data=str(DATASET_YAML), split="test")

    print("\n" + "=" * 70)
    print("📊 RÉSULTATS DE L'ENTRAÎNEMENT :")
    print(f"   mAP50     : {metrics.box.map50:.3f}")
    print(f"   mAP50-95  : {metrics.box.map:.3f}")
    print("=" * 70)
    return results, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning YOLOv8 Quarry Load Verification")
    parser.add_argument("--epochs", type=int, default=10, help="Nombre d'époques d'entraînement")
    parser.add_argument("--batch", type=int, default=16, help="Taille du batch")
    parser.add_argument("--imgsz", type=int, default=512, help="Résolution des images")
    args = parser.parse_args()

    train_quarry_model(epochs=args.epochs, batch_size=args.batch, imgsz=args.imgsz)

