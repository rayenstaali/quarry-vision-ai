# 📈 Résultats Expérimentaux et Rapports d'Évaluation

Ce répertoire rassemble les résultats d'évaluation du système de vision et les exports générés lors des sessions de surveillance.

---

## 🔬 1. Résultats d'Évaluation du Modèle YOLOv8 Fine-tuné

Évaluation réalisée sur le jeu de test indépendant (`data/quarry_dataset/test`) :

| Classe | Instances | Précision (P) | Rappel (R) | mAP@50 | mAP@50-95 |
|:-------|:---------:|:-------------:|:----------:|:------:|:---------:|
| **Toutes les classes** | **68** | **0.867** | **0.883** | **0.901** | **0.752** |
| `truck open bed` | 17 | 0.841 | 0.882 | **0.888** | 0.765 |
| `trailer open bed` | 18 | 0.942 | 0.944 | **0.974** | 0.852 |
| `digger` | 19 | 0.825 | 0.842 | **0.796** | 0.658 |
| `BUCKET_DUMP` | 9 | 0.985 | 1.000 | **0.995** | 0.821 |
| `BUCKET_CARRY` | 5 | 0.742 | 0.800 | **0.851** | 0.664 |

*Remarque méthodologique : Les classes `BUCKET_DIG` et `person` étaient sous-représentées dans le jeu de test (présentes principalement en apprentissage), justifiant une augmentation de données ciblée pour les prochaines itérations.*

---

## 📊 2. Métriques Opérationnelles du Pipeline de Démonstration

Sur la séquence vidéo réelle de carrière (`carriere_reelle_hd.mp4`) à 12 FPS :
* **Détection & Tracking (ByteTrack) :** Suivi stable des camions, pelleteuse et godets sans permutation d'identifiants sur les phases de transit.
* **Temps moyen d'inférence :** $\approx 40$ ms par image sur CPU Intel ($\approx 25$ FPS).
* **Contrôle de chargement :** Détection continue du niveau de benne et alertes immédiates en cas de débordement de matière.
* **Sécurité :** Détection des franchissements de zones de danger avec filtrage anti-rebond (hystérésis) pour éliminer les faux positifs.

---

## 📑 3. Rapports Exportables Générés

* `outputs/rapport_production_carriere.csv` : Registre des camions (ID, heure de passage, temps de chargement, état de la benne, conformité).
* `outputs/journal_alertes_securite.csv` : Journal horodaté des alertes critiques (intrusions piétons, surcharges).

