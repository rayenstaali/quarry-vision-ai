# 📚 Références Bibliographiques & État de l'Art

Ce document recense les articles scientifiques, publications, frameworks et documentations techniques exploités dans le cadre du projet **QuarryVision AI**.

---

## 📄 1. Détection d'Objets & Modèles en Temps Réel

1. **YOLOv8 & YOLO Architecture :**
   * Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8* (Version 8.0.0). [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
   * Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 779-788.
   * Wang, C. Y., Bochkovskiy, A., & Liao, H. Y. M. (2023). *YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors*. CVPR 2023.

2. **Vision Transformers & Backbones Avancés :**
   * Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). *End-to-End Object Detection with Transformers (DETR)*. European Conference on Computer Vision (ECCV).

---

## 🎯 2. Suivi Multi-Objets (Multi-Object Tracking - MOT)

1. **ByteTrack (Algorithme retenu) :**
   * Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. European Conference on Computer Vision (ECCV).
   * *Apport pour le projet :* Association systématique des détections à forte et faible confiance via filtre de Kalman et algorithme hongrois, réduisant drastiquement les pertes de trajectoire en présence de poussière ou de masquage partiel.

2. **BoT-SORT :**
   * Aharon, N., Orfaig, R., & Bobrovsky, B. Z. (2022). *BoT-SORT: Robust Associations Multi-Pedestrian Tracker*. arXiv:2206.14651.

---

## 🚚 3. Vision par Ordinateur Appliquée aux BTP et Carrières

1. **Surveillance des Chantiers & Sécurité :**
   * Seo, J., Han, S., Lee, S., & Kim, H. (2015). *Computer vision techniques for construction safety management: A critical review*. Automation in Construction, 58, 169-185.
   * Fang, W., Ding, L., Love, P. E., Shen, Q., & Li, H. (2020). *Computer vision applications in construction safety assurance*. Archives of Computational Methods in Engineering, 27(4), 989-1004.

2. **Estimation du Chargement & Volumétrie :**
   * Chen, J., Fang, D., & O'Brien, W. J. (2021). *Automated payload estimation of earthmoving equipment based on computer vision and depth sensors*. Journal of Construction Engineering and Management, 147(6).

---

## 🛠️ 4. Outils, Bibliothèques et Datasets

* **Roboflow Universe :** Dataset *Quarry Load Verification* par Smarty-pants ([Lien](https://universe.roboflow.com/smarty-pants/quarry-load-verification/dataset/3)).
* **Supervision (Roboflow) :** Bibliothèque d'annotation et de post-traitement de vision par ordinateur ([Lien](https://github.com/roboflow/supervision)).
* **OpenCV :** Traitement d'image matriciel, calcul des ROI, conversion d'espaces colorimétriques et filtrage.
* **Shapely :** Calculs géométriques vectoriels dans le plan pour la gestion des zones polygonales d'intérêt (ROI).
* **Streamlit :** Framework de visualisation pour applications industrielles et dashboards interactifs.

