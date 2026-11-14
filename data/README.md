# 📊 Données et Datasets du Projet

Ce répertoire documente la gestion des données pour le projet **QuarryVision AI**.
Conformément aux bonnes pratiques de versioning de code, les fichiers volumineux (vidéos brutes, images, datasets complets) **ne sont pas hébergés directement sur le dépôt GitHub**.

---

## 📂 Source du Dataset

* **Nom officiel :** `quarry-load-verification` (v3)
* **Auteur / Workspace :** Smarty-pants
* **Source :** Roboflow Universe
* **Lien d'accès :** [Roboflow Universe - Quarry Load Verification](https://universe.roboflow.com/smarty-pants/quarry-load-verification/dataset/3)
* **Licence :** CC BY 4.0 (Utilisation académique et recherche)

---

## 🔍 Caractéristiques du Dataset

* **Nombre total d'images :** 502 images (extraites de flux vidéo réels de carrière - séquence `video3`)
* **Répartition des données :**
  * `train/` : 442 images (avec labels YOLOv8)
  * `valid/` : 40 images
  * `test/` : 20 images
* **Résolution d'image :** $640 \times 640$ pixels (auto-orientation et normalisation EXIF)
* **Format des annotations :** Format Ultralytics YOLO (`<class_id> <x_center> <y_center> <width> <height>` normalisé entre 0 et 1)
* **Augmentation de données appliquée :**
  * Flip horizontal (50%)
  * Variation de luminosité ($\pm 15\%$)
  * Flou gaussien (0 à 1 px)

### 🏷️ Classes annotées (7 classes métier) :

| ID | Label | Description métier |
|:--:|:------|:-------------------|
| **0** | `BUCKET_CARRY` | Godet de l'excavatrice transportant de la matière |
| **1** | `BUCKET_DIG` | Godet en phase de creusement / extraction |
| **2** | `BUCKET_DUMP` | Godet en phase de déversement dans la benne |
| **3** | `digger` | Pelleteuse mécanique / excavatrice industrielle |
| **4** | `person` | Ouvrier / piéton sur site carrier |
| **5** | `trailer open bed` | Benne ouverte de la remorque du camion |
| **6** | `truck open bed` | Benne ouverte du camion benne porteur |

---

## 📥 Procédure de Téléchargement Automatique

Pour télécharger le dataset localement via l'API Roboflow :

1. Créez un fichier `.env` à la racine du projet (en vous basant sur `.env.example`) :
```ini
ROBOFLOW_API_KEY=votre_cle_api_roboflow
ROBOFLOW_WORKSPACE=smarty-pants
ROBOFLOW_PROJECT=quarry-load-verification
ROBOFLOW_VERSION=3
```

2. Exécutez le script de prétraitement et de téléchargement :
```bash
python src/data_preprocessing.py --download
```

Le script télécharge l'archive dans `data/quarry_dataset/` et configure automatiquement les chemins absolus dans `data.yaml`.

