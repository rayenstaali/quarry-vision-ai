# 🧠 Modèles et Poids Entraînés

Ce répertoire héberge les points de contrôle (checkpoints) et poids de détection pour le projet **QuarryVision AI**.

---

## 📌 Modèles Disponibles

### 1. `quarry_load_best.pt` (Modèle Spécifique Carrière - Fine-tuné)
* **Architecture de base :** YOLOv8 Nano (`yolov8n.pt`)
* **Nombre de paramètres :** 3,006,233 (3.0M)
* **FLOPs :** 8.2 GFLOPs
* **Entraînement :** Fine-tuning supervisé sur les 7 classes du dataset `quarry-load-verification`
* **Optimiseur :** AdamW (learning rate initial $0.002$, momentum $0.9$, weight decay $0.0005$)
* **Performances globales (Test set) :**
  * **mAP@50 :** **0.901** ($90.1\%$)
  * **mAP@50-95 :** **0.752** ($75.2\%$)
  * **Précision :** $86.7\%$
  * **Rappel :** $88.3\%$

### 2. `yolov8n.pt` (Modèle Pré-entraîné Baseline COCO)
* **Classes cibles :** `truck` (ID 7), `car` (ID 2), `person` (ID 0)
* Sert de modèle de référence (baseline) pour mesurer le gain apporté par le fine-tuning métier sur site carrier.

---

## 🚀 Ré-entraîner ou Évaluer les Modèles

```bash
# Entraîner le modèle
python src/train.py --epochs 20 --batch 16 --imgsz 640

# Évaluer les performances sur le jeu de test
python src/evaluate.py --weights models/quarry_load_best.pt
```

