# Cahier des Charges Fonctionnel et Technique (CDCF)
## Projet : Système de Vision par Ordinateur pour le Suivi des Camions et le Contrôle de Chargement en Carrière

---

### Informations Générales
- **Intitulé du Projet** : Système automatisé de surveillance des flux de transport et contrôle d'état de chargement en carrière minérale par vision artificielle.
- **Porteurs du Projet** : Rayen Staali & Mohamed Zied Achouri
- **Établissement** : Polytech International
- **Technologies Clés** : Python, YOLOv8 / YOLO11, ByteTrack, OpenCV, Supervision, Streamlit.

---

### 1. Contexte et Enjeux Métier

Dans les carrières d'extraction et de négoce de matériaux (sable, gravier, calcaire concassé, granulats), le flux logistique repose sur des rotations continues de camions bennes venant charger des matériaux sous des pelles mécaniques ou des chargeuses sur pneus.

Actuellement, la gestion de ces flux présente plusieurs limites majeures :
1. **Comptage manuel ou imprécis des rotations** : Risque d'erreurs humaines, omissions de passages ou absence de traçabilité en temps réel.
2. **Contrôle visuel du chargement fastidieux et subjectif** : Difficulté pour les opérateurs au sol de vérifier si la benne est vide, correctement chargée ou dangereusement surchargée.
3. **Risques majeurs de sécurité des personnes** : Présence impromptue de personnel à pied dans l'angle mort des engins lourds en manœuvre (pelles et camions).
4. **Pertes économiques** : Camions quittant le site à vide suite à un dysfonctionnement de logistique ou temps de chargement excessifs bloquant les postes de production.

Le présent projet vise à implémenter une solution logicielle non invasive, basée exclusivement sur l'analyse de flux vidéo par caméras existantes, sans installation de capteurs embarqués coûteux sur les camions.

---

### 2. Objectifs du Projet

#### 2.1. Objectifs Généraux
- Automatiser la surveillance des mouvements de camions sur l'ensemble du cycle de rotation.
- Assurer un contrôle qualité visuel automatisé de l'état de remplissage de la benne.
- Détecter en temps réel les infractions de sécurité et les anomalies d'exploitation.
- Fournir un tableau de bord d'aide à la décision pour le chef de carrière.

#### 2.2. Objectifs Opérationnels et Chiffrés
- **Précision de détection (mAP@50)** : $\ge 85\%$ sur les classes `truck` et `person`.
- **Fiabilité du suivi (IDF1 score)** : $\ge 80\%$ pour conserver l'identifiant camion d'un bout à l'autre de la scène.
- **Précision de classification de chargement** : $\ge 85\%$ de distinction correcte entre `empty`, `partially_loaded` et `loaded`.
- **Temps de traitement (FPS)** : $\ge 20$ images par seconde sur GPU grand public (ex: RTX 3060 / 4060) ou $\ge 10$ FPS sur CPU récent.

---

### 3. Périmètre du Projet (Phasage)

#### Phase 1 : MVP (Périmètre prioritaire réalisé)
1. **Détection d'objets** : Détection robuste des camions et personnes dans les conditions d'éclairage de la carrière avec YOLOv8.
2. **Tracking multi-objets (MOT)** : Attribution d'un identifiant numérique persistant (`truck_id`) avec ByteTrack.
3. **Définition de zones polygonales (ROI)** : Entrée, Zone de chargement, Zone de sortie/pont-bascule, Zone dangereuse.
4. **Machine à états du cycle de vie** : `Arrivée` $\rightarrow$ `En Chargement` $\rightarrow$ `Sortie`.
5. **Contrôle de benne** : Recadrage de la zone utile et classification en 3 états (`empty`, `partially_loaded`, `loaded`).
6. **Moteur d'alertes** : Alerte piéton en zone rouge et camion sortant à vide.
7. **Tableau de bord Streamlit** : Interface graphique de consultation, KPIs et export CSV.

#### Phase 2 : Extensions Avancées (V2)
1. **Segmentation d'instance de la cargaison (YOLOv8-seg)** : Calcul surfacique et détection fine des débordements latéraux.
2. **Reconnaissance optique des plaques d'immatriculation (LPR / ANPR)** : Rapprochement de l'ID visuel avec l'immatriculation officielle.
3. **Interfaçage avec les données du pont-bascule** : Corrélation entre l'estimation visuelle volumique et la masse réelle pesée.

---

### 4. Spécifications Fonctionnelles Détaillées

| Réf. | Module | Description Fonctionnelle | Critère d'Acceptation |
| :--- | :--- | :--- | :--- |
| **SF-01** | Détection d'objets | Détection en temps réel des camions (`truck`), pelles (`excavator`), chargeuses (`loader`) et ouvriers (`person`). | Bounding boxes fiables même sous poussière modérée. |
| **SF-02** | Tracking persistant | Attribution et conservation d'un ID de suivi unique pour chaque véhicule traversant le champ de vision. | Maintien de l'ID après occlusion brève (poteau, nuage de poussière). |
| **SF-03** | Zones ROI polygonales | Surveillance du franchissement de polygones configurables (Entrée, Chargement, Sortie, Danger). | Calcul d'appartenance par point d'ancrage (base du véhicule). |
| **SF-04** | Chronométrage de cycle | Calcul automatique de la durée de séjour dans chaque zone, notamment le temps d'opération sous la pelle. | Horodatage à la seconde près et enregistrement en base. |
| **SF-05** | Recadrage de benne | Extraction automatisée de la région d'intérêt contenant la matière transportée. | Recadrage adaptatif en fonction de la taille et de la perspective du camion. |
| **SF-06** | Classification chargement | Catégorisation du remplissage : `empty`, `partially_loaded`, `loaded`, `overloaded`. | Sortie d'un label avec score de confiance associé. |
| **SF-07** | Alerte sécurité piéton | Déclenchement immédiat d'une alerte visuelle si une personne entre dans la zone de manœuvre d'un engin. | Notification à l'écran sous moins de 500 ms. |
| **SF-08** | Alerte départ à vide | Signalement d'un camion atteignant la zone de sortie sans avoir été chargé. | Alerte consignée dans le journal d'incidents. |
| **SF-09** | Reporting & Export | Génération de statistiques (rotations/jour, temps moyen, alertes) et export direct au format CSV. | Fichier CSV exploitable sous Excel ou PowerBI. |

---

### 5. Spécifications Techniques et Architecture

#### 5.1. Stack Technologique
- **Langage de développement** : Python 3.10 à 3.13
- **Framework de Deep Learning** : PyTorch $\ge 2.0$, Ultralytics (YOLOv8 / YOLO11)
- **Tracking & Vision** : ByteTrack (intégré via Ultralytics), OpenCV 4.x, Supervision
- **Traitement géométrique** : Shapely, NumPy
- **Interface Utilisateur** : Streamlit, Plotly, Pandas

#### 5.2. Architecture des Composants
1. `src/config.py` : Centralisation des hyperparamètres, coordonnées géométriques et seuils.
2. `src/zone_manager.py` : Moteur vectoriel de détection polygonale et rendu graphique des zones.
3. `src/load_classifier.py` : Extraction de texture, analyse de contraste et modèle de classification d'état de cargaison.
4. `src/anomaly_detector.py` : Moteur de règles déterministes évaluant les situations à risque.
5. `src/tracker.py` : Cœur algorithmique intégrant détection, tracking et mise à jour du registre des camions.
6. `src/pipeline.py` : Orchestrateur vidéo en mode batch ou continu avec sérialisation CSV.
7. `dashboard.py` : Interface web réactive avec graphes et alertes.

---

### 6. Données, Datasets et Stratégie d'Entraînement

#### 6.1. Datasets Recommandés
- **Roboflow Quarry Load Verification** : Dataset spécialisé contenant des classes spécifiques de carrières (`truck`, `trailer`, `bucket`, `person`).
- **Roboflow Construction Vehicles Detection** : Engins de chantier lourds (`Excavator`, `Crane`, `Loader`, `Truck`).
- **Captures locales in-situ** : Recommandé d'extraire des frames à intervalle régulier (1 image / seconde) pour annoter les camions et bennes spécifiques à la carrière cible.

#### 6.2. Procédure de Fine-Tuning YOLO
```bash
# Entraînement sur GPU / Google Colab
yolo detect train \
    data=data/carriere_dataset.yaml \
    model=yolov8n.pt \
    epochs=60 \
    imgsz=640 \
    batch=16 \
    optimizer=AdamW \
    lr0=0.001
```

---

### 7. Contraintes Réglementaires et Limites Physiques

1. **Différence entre Contrôle Visuel et Masse Réglementaire** :
   - *Avertissement important* : L'analyse visuelle par caméra ne remplace en aucun cas un pont-bascule certifié métrologie légale. La densité des granulats varie selon l'humidité et la granulométrie ; une estimation volumique ne permet pas d'attester de la masse exacte en tonnes.
2. **Respect de la Vie Privée (RGPD)** :
   - Le système est conçu pour détecter la silhouette humaine afin de prévenir les accidents. Les visages et plaques d'immatriculation ne doivent pas être conservés en clair sans déclaration appropriée.
3. **Contraintes d'Environnement Sévère** :
   - Poussière dense, pluie, contre-jour et phares de nuit. Le modèle doit intégrer des techniques d'augmentation de données (Data Augmentation : flou, contraste, bruit gaussien) pour rester robuste.

---

### 8. Métriques d'Évaluation et Recette du Système

| Métrique | Définition | Cible Requise |
| :--- | :--- | :--- |
| **mAP@0.5** | Moyenne de la précision moyenne à IoU=0.5 | $\ge 85\%$ |
| **mAP@0.5:0.95** | Précision moyenne sur des seuils IoU stricts | $\ge 60\%$ |
| **IDF1 Score** | Mesure de consistance de l'identité du tracker | $\ge 80\%$ |
| **Taux de faux positifs sécurité** | Fausse alerte piéton en zone rouge | $< 5\%$ |
| **Précision classification benne** | Pourcentage de bennes classées correctement | $\ge 85\%$ |

---

### 9. Livrables Attendus
1. **Code source complet et commenté** du pipeline de détection, suivi et classification.
2. **Générateur de données de test** permettant de valider l'application sans équipement physique.
3. **Tableau de bord Streamlit fonctionnel** avec relecture vidéo, graphes et export des données.
4. **Modèles de vision entraînés et configurés** (poids `.pt`).
5. **Cahier des charges officiel validé** et rapport de projet.
