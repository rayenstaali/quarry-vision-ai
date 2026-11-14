"""

Pipeline unifié d'exécution vidéo pour le suivi et le contrôle en carrière.
Orchestre l'ensemble du flux :
1. Lecture vidéo / flux caméra (OpenCV)
2. Traitement frame par frame avec QuarryTracker
3. Encodage et sauvegarde de la vidéo annotée
4. Exportation des statistiques et alertes dans un fichier CSV horodaté
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import pandas as pd
import time
from typing import Optional, Callable
from src.config import OUTPUT_DIR, DATA_DIR
from src.tracker import QuarryTracker


def process_video_pipeline(
    input_video_path: str,
    output_video_path: Optional[str] = None,
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
    max_frames: Optional[int] = None,
    model_path: Optional[str] = None
) -> dict:
    """
    Exécute le pipeline complet sur une vidéo source.
    :param input_video_path: Chemin du fichier MP4 source
    :param output_video_path: Chemin de la vidéo annotée de sortie
    :param progress_callback: Fonction de rappel (progression 0..1, frame_actuelle, total_frames)
    :param max_frames: Limite optionnelle du nombre de frames à traiter (pour tests rapides)
    :param model_path: Chemin optionnel vers des poids YOLO personnalisés (ex: quarry_load_best.pt)
    :return: Dictionnaire contenant les statistiques finales et le chemin des fichiers générés
    """
    input_path = Path(input_video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Vidéo introuvable : {input_video_path}")

    if output_video_path is None:
        output_video_path = str(OUTPUT_DIR / f"annotated_{input_path.name}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir le flux vidéo : {input_video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames and max_frames < total_frames:
        total_frames = max_frames

    # Encodeur MP4 (mp4v ou avc1)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    tracker = QuarryTracker(model_path=model_path) if model_path else QuarryTracker()
    frame_idx = 0
    start_time = time.time()

    print(f"[Pipeline] Début du traitement : {total_frames} frames ({width}x{height} @ {fps:.1f} fps)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Traitement complet par QuarryTracker
        annotated_frame, frame_data = tracker.process_frame(frame, frame_idx=frame_idx, fps=fps)
        out.write(annotated_frame)

        frame_idx += 1
        if progress_callback:
            progress = min(1.0, frame_idx / float(total_frames))
            progress_callback(progress, frame_idx, total_frames)

        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    out.release()
    elapsed_time = time.time() - start_time
    fps_processing = frame_idx / elapsed_time if elapsed_time > 0 else 0

    print(f"[Pipeline] Traitement terminé en {elapsed_time:.1f}s ({fps_processing:.1f} FPS)")
    print(f"[Pipeline] Vidéo annotée enregistrée : {output_video_path}")

    # Ré-encodage en H.264 pour compatibilité navigateur (st.video / HTML5)
    output_p = Path(output_video_path)
    web_video_path = str(output_p.parent / f"web_{output_p.name}")
    demo_web_path = str(output_p.parent / "annotated_demo_web.mp4")
    try:
        import imageio_ffmpeg
        import subprocess
        import shutil
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([
            ffmpeg_exe, '-i', str(output_video_path),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            '-y', web_video_path
        ], check=True, capture_output=True)
        if web_video_path != demo_web_path and "demo" in output_p.name.lower():
            shutil.copy2(web_video_path, demo_web_path)
        print(f"[Pipeline] Vidéo web H.264 générée : {web_video_path}")
    except Exception as e:
        print(f"[Pipeline] Avertissement : ré-encodage H.264 impossible ({e})")

    # ==========================================================================
    # EXPORTATION DU RAPPORT DE PRODUCTION ET DU REGISTRE EN CSV
    # ==========================================================================
    rows = []
    for truck_id, data in tracker.trucks_registry.items():
        rows.append({
            "ID_Camion": truck_id,
            "Premier_Passage_Sec": round(data["first_seen"], 1),
            "Dernier_Passage_Sec": round(data["last_seen"], 1),
            "Derniere_Zone": data["current_zone"],
            "Etape_Cycle": data["cycle_step"],
            "Etat_Benne": data["load_status"],
            "Confiance_Benne": round(data["load_conf"], 2),
            "Debordement_Detecte": "OUI" if data["is_overflowing"] else "NON",
            "Duree_Chargement_Sec": round(data["loading_duration"], 1),
            "Alertes_Associees": " | ".join(data["alerts"]) if data["alerts"] else "Aucune"
        })

    df_report = pd.DataFrame(rows)
    csv_report_path = OUTPUT_DIR / "rapport_production_carriere.csv"
    df_report.to_csv(csv_report_path, index=False, encoding="utf-8-sig")
    print(f"[Pipeline] Rapport CSV exporté : {csv_report_path}")

    # Exportation de l'historique détaillé des alertes de sécurité
    df_alerts = pd.DataFrame(tracker.anomaly_detector.alert_history)
    csv_alerts_path = OUTPUT_DIR / "journal_alertes_securite.csv"
    df_alerts.to_csv(csv_alerts_path, index=False, encoding="utf-8-sig")

    return {
        "output_video": str(output_video_path),
        "web_video": str(web_video_path),
        "csv_report": str(csv_report_path),
        "csv_alerts": str(csv_alerts_path),
        "stats": tracker.stats,
        "total_frames": frame_idx,
        "processing_fps": round(fps_processing, 1),
        "registry": tracker.trucks_registry
    }


if __name__ == "__main__":
    from src.synthetic_generator import generate_demo_video
    demo_video = generate_demo_video()
    process_video_pipeline(str(demo_video))
