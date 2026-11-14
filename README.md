# 🏭 QuarryVision AI : Système Intelligent de Suivi des Camions & Contrôle du Chargement en Carrière

> **Projet de Computer Vision & Deep Learning appliqué à l'Industrie et aux Chantiers**  
> *Supervision automatisée des flux de transport, inspection de l'état des bennes et détection des anomalies de sécurité.*

---

## 👥 Membres du Groupe

| Nom & Prénom | Compte GitHub | Rôle / Contribution |
|:-------------|:-------------:|:--------------------|
| **Rayen STAALI** | [@rayenstaali](https://github.com/rayenstaali) | Architecture pipeline, Inférence temps réel & Dashboard |
| **Mohamed Zied ACHOURI** | [@Ziedash](https://github.com/Ziedash) | Préparation des données, Fine-tuning YOLO & Évaluation |
| **Haythem GHAZOUANI** | [@haythemghz](https://github.com/haythemghz) | Moteur de tracking (ByteTrack), Règles métier & Alertes |

---

## 📌 1. Contexte & Problématique

### Contexte
Dans les carrières d'extraction de granulats et les chantiers de terrassement lourd, la rotation continue des camions bennes et des engins d'extraction (pelleteuses, chargeuses) constitue le poumon économique de l'exploitation. Chaque jour, des centaines de rotations sont effectuées entre les fronts de taille, les postes de concassage et les ponts-bascules de sortie.

### Problématique à résoudre
Cette cadence intensive génère des défis majeurs non résolus par les méthodes manuelles traditionnelles :
1. **Risques élevés de sécurité (accidents graves) :** Coactivité permanente entre engins lourds en mouvement aveugle et piétons/opérateurs au sol dans les zones de chargement et de manœuvre.
2. **Pertes économiques et litiges :** Départs de camions à vide non détectés, rotations fantômes ou erreurs de facturation à la pesée.
3. **Risques de sécurité routière et infractions :** Chargements asymétriques ou débordant au-dessus des ridelles de la benne, provoquant des chutes de roches sur les routes publiques et des détériorations d'essieux.
4. **Goulots d'étranglement opérationnels :** Temps de chargement excessifs immobilisant les pelles mécaniques et générant des files d'attente coûteuses.

---

## 🎯 2. Objectifs du Projet

L'objectif principal est de concevoir et déployer un **système de vision par ordinateur de bout en bout** capable d'analyser en temps réel les flux vidéo de caméras fixes sur site carrier pour :
* **Détecter et classifier** avec haute précision les camions bennes, remorques, pelleteuses, godets en action et piétons.
* **Assurer un suivi multi-objets persistant (MOT)** attribuant un identifiant unique à chaque camion tout au long de son cycle opérationnel (*Arrivée $\rightarrow$ Chargement $\rightarrow$ Sortie*).
* **Superviser l'occupation spatiale** par zones géométriques polygonales (Entrée, Chargement, Sortie, Zone d'exclusion piéton).
* **Inspecter visuellement l'état de la benne** (*vide*, *partiellement chargée*, *chargée conforme*, *débordante/surchargée*).
* **Déclencher des alertes de sécurité prioritaires** (présence humaine en zone d'engins, surcharge de matière).
* **Fournir un tableau de bord industriel interactif** pour le suivi des KPIs de production et l'export des rapports réglementaires.

---

## 💡 3. Solution Envisagée

La solution **QuarryVision AI** s'articule autour d'une chaîne de traitement modulaire en 4 niveaux :
```
[ Flux Vidéo Caméra / Fichier MP4 ]
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Perception & Détection Multi-classes (YOLOv8 Fine-tuné)  │
│    Classes : truck open bed, trailer, digger, bucket, person│
└──────────────────────────────┬──────────────────────────────┘
                               │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Suivi Temporel & Trajectoire (ByteTrack / Kalman Filter) │
│    Maintien des identifiants (IDs stables) malgré poussière │
└──────────────────────────────┬──────────────────────────────┘
                               │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Analyse Spatiale & Contrôle d'État (Zones ROI & Benne)   │
│    - Test d'inclusion point-in-polygon (Shapely / OpenCV)   │
│    - Recadrage ROI benne & estimation de saturation         │
│    - Moteur d'anomalies avec hystérésis anti-faux positifs   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Restitution & Supervision (Streamlit Dashboard & CSV)    │
│    Vidéo H.264 web, KPIs, Timeline d'alertes & Rapports CSV │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 4. Approches Techniques & Architectures Envisagées

### Détection d'Objets : YOLOv8 (You Only Look Once v8)
* **Pourquoi YOLOv8 ?** Architecture State-of-the-Art offrant un compromis idéal entre vitesse d'inférence (30+ FPS sur CPU/Edge) et précision géométrique.
* **Architecture :** Backbone CSPDarknet53 modifié avec modules C2f, Neck PANet multi-échelles et tête de détection découplée (*Anchor-Free*).
* **Entraînement :** Fine-tuning supervisé avec fonction de perte combinée (CIoU loss + Distribution Focal Loss pour les boîtes, BCE loss pour les classes).

### Suivi Multi-Objets (MOT) : ByteTrack
* Utilisation de l'algorithme **ByteTrack** via filtre de Kalman et association par algorithme hongrois. Contrairement aux approches SORT traditionnelles qui ignorent les détections à faible score, ByteTrack associe d'abord les détections haute confiance puis récupère les détections de faible confiance (occlusions partielles, poussière de carrière), garantissant la continuité des identifiants camions.

### Analyse Géométrique & Classification d'État :
* **Polygones ROI :** Modélisation vectorielle 2D des zones par `Shapely` et tests `pointPolygonTest` OpenCV en temps réel ($O(1)$).
* **Contrôle de benne :** Découpage adaptatif de la région de la benne, analyse de texture, calcul du ratio de remplissage et détection des excédents rocheux au-dessus du plan des ridelles.

---

## 📊 5. Dataset Utilisé

* **Nom :** `quarry-load-verification` (Version 3)
* **Source & Lien :** [Roboflow Universe - Quarry Load Verification](https://universe.roboflow.com/smarty-pants/quarry-load-verification/dataset/3)
* **Volume :** 502 images annotées issues d'enregistrements vidéo réels de carrière industrielle.
* **Classes cibles (7) :** `truck open bed`, `trailer open bed`, `digger`, `BUCKET_DUMP`, `BUCKET_CARRY`, `BUCKET_DIG`, `person`.
* **Répartition :** 442 train, 40 valid, 20 test ($640 \times 640$ px).
* **Résultats obtenus :** **mAP@50 = 0.901** ($90.1\%$) et **mAP@50-95 = 0.752** ($75.2\%$) sur le jeu de test.

---

## 🗂️ 6. Organisation du Projet

Le dépôt respecte rigoureusement la structure modulaire recommandée :

```text
quarry-vision-ai/
│
├── README.md                      # Présentation complète et documentation du projet
├── requirements.txt               # Dépendances Python du projet
├── .gitignore                     # Exclusion des fichiers lourds, datasets et caches
├── .env.example                   # Modèle de configuration des variables d'environnement
│
├── data/
│   ├── README.md                  # Documentation du dataset, source et procédure de téléchargement
│   └── raw_videos/                # Répertoire local des vidéos sources (non poussé sur git)
│
├── notebooks/
│   └── exploratory_analysis.ipynb # Analyse exploratoire des données (EDA) & inférences de test
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Paramètres généraux, ROI géométriques et seuils
│   ├── data_preprocessing.py      # Téléchargement Roboflow, formatage YAML & extraction vidéo
│   ├── model.py                   # Wrapper de chargement et d'inférence YOLOv8
│   ├── tracker.py                 # Suivi ByteTrack, gestion des trajectoires et registre de flotte
│   ├── zone_manager.py            # Moteur géométrique vectoriel des zones ROI
│   ├── load_classifier.py         # Recadrage automatique et inspection de la benne
│   ├── anomaly_detector.py        # Moteur de règles de sécurité et alertes avec hystérésis
│   ├── pipeline.py                # Orchestrateur vidéo complet et exports de données
│   ├── train.py                   # Script d'entraînement / fine-tuning YOLOv8
│   ├── evaluate.py                # Évaluation quantitative et métriques (mAP, Précision, Rappel)
│   └── synthetic_generator.py     # Générateur de séquence de simulation pour validation
│
├── models/
│   ├── README.md                  # Description des checkpoints et métriques du modèle
│   └── quarry_load_best.pt        # Poids fine-tunés sur les 7 classes carrière (6.5 Mo)
│
├── results/
│   └── README.md                  # Tableaux comparatifs des performances et exports d'essais
│
└── docs/
    ├── references.md              # Références bibliographiques, état de l'art et licences
    └── cahier_des_charges.md      # Cahier des charges exhaustif des spécifications système
```

---

## 🛠️ 7. Technologies & Bibliothèques

* **Langage :** Python 3.10+ / 3.11 / 3.12 / 3.13
* **Deep Learning & Vision :** `ultralytics` (YOLOv8), `torch`, `torchvision`, `supervision` (Roboflow)
* **Traitement d'Image & Vidéo :** `opencv-python` (cv2), `imageio-ffmpeg` (encodage H.264 natif)
* **Calcul Géométrique & Données :** `shapely`, `numpy`, `pandas`
* **Visualisation & Dashboard :** `streamlit`, `plotly`, `matplotlib`
* **Gestion d'environnement :** `python-dotenv`, `pyyaml`

---

## 🚀 8. Instructions d'Installation & d'Exécution

### Étape 1 : Cloner le dépôt GitHub
```bash
git clone https://github.com/rayenstaali/quarry-vision-ai.git
cd quarry-vision-ai
```

### Étape 2 : Créer un environnement virtuel & installer les dépendances
```bash
python -m venv .venv
# Sur Windows :
.venv\Scripts\activate
# Sur Linux/macOS :
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Étape 3 : Configurer les accès API (Optionnel pour téléchargement dataset)
Copiez le fichier `.env.example` en `.env` :
```bash
cp .env.example .env
```
Renseignez votre clé API Roboflow si vous souhaitez ré-entraîner le modèle sur le dataset complet.

### Étape 4 : Lancer le Tableau de Bord Interactif Streamlit
```bash
python -m streamlit run dashboard.py
```
Accédez ensuite à l'application dans votre navigateur à l'adresse **http://localhost:8501** (ou port indiqué).

### Étape 5 : Évaluer le modèle sur le jeu de test
```bash
python src/evaluate.py --weights models/quarry_load_best.pt --split test
```

---

## 📅 9. Principales Étapes de Réalisation

1. **Phase 1 : Exploration & Curation des Données (Terminée)**
   * Analyse du dataset `quarry-load-verification` v3 (502 images).
   * Validation des annotations des 7 classes métier.
2. **Phase 2 : Apprentissage & Évaluation du Modèle (Terminée)**
   * Entraînement supervisé de YOLOv8 Nano sur GPU/CPU avec régularisation AdamW.
   * Atteinte d'un **mAP@50 de 90.1%** sur le jeu de test.
3. **Phase 3 : Intégration du Tracking & Moteur Métier (Terminée)**
   * Couplage de YOLOv8 avec ByteTrack pour la persistance des identifiants.
   * Définition spatiale des 4 zones ROI et moteur d'alertes de sécurité avec hystérésis.
4. **Phase 4 : Interface Utilisateur & Dashboard (Terminée)**
   * Développement du tableau de bord Streamlit avec lecteur vidéo H.264 intégré, KPIs, registre de flotte et téléchargement des rapports CSV.
5. **Phase 5 : Perspectives & Déploiement Industriel (En cours)**
   * Intégration de flux RTSP directs depuis les caméras IP sur site.
   * Estimation volumétrique 3D par caméra stéréo ou estimation de profondeur.
