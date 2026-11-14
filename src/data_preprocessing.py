"""
Module de téléchargement, vérification et prétraitement des données pour QuarryVision AI.
Gère le téléchargement depuis Roboflow, la validation du format des labels YOLO
et la reconstruction de séquences vidéos à partir des images annotées.
"""

import os
import sys
import glob
import re
import argparse
from pathlib import Path
import cv2
import yaml
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "quarry_dataset"


def download_dataset_from_roboflow():
    """
    Télécharge le dataset quarry-load-verification v3 depuis Roboflow via API key.
    """
    api_key = os.getenv("ROBOFLOW_API_KEY")
    workspace = os.getenv("ROBOFLOW_WORKSPACE", "smarty-pants")
    project_name = os.getenv("ROBOFLOW_PROJECT", "quarry-load-verification")
    version = int(os.getenv("ROBOFLOW_VERSION", "3"))

    if not api_key:
        print("[Erreur] Clé ROBOFLOW_API_KEY manquante dans le fichier .env.")
        print("Consultez .env.example pour configurer vos accès.")
        return False

    print(f"[Data] Téléchargement du dataset '{project_name}' v{version}...")
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        dataset = project.version(version).download("yolov8", location=str(DATASET_DIR))
        print(f"[Data] Dataset téléchargé avec succès dans : {dataset.location}")
        fix_data_yaml()
        return True
    except Exception as e:
        print(f"[Data] Échec du téléchargement via roboflow SDK : {e}")
        return False


def fix_data_yaml():
    """
    S'assure que data.yaml contient des chemins absolus valides pour Ultralytics YOLO.
    """
    yaml_path = DATASET_DIR / "data.yaml"
    if not yaml_path.exists():
        print(f"[Avertissement] {yaml_path} introuvable.")
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data["train"] = str((DATASET_DIR / "train" / "images").resolve()).replace("\\", "/")
    data["val"] = str((DATASET_DIR / "valid" / "images").resolve()).replace("\\", "/")
    data["test"] = str((DATASET_DIR / "test" / "images").resolve()).replace("\\", "/")

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"[Data] Fichier data.yaml mis à jour avec les chemins absolus.")


def build_video_from_frames(output_mp4: str = "data/raw_videos/carriere_reelle_hd.mp4", target_fps: float = 12.0):
    """
    Reconstitue une vidéo HD (1280x720) à partir des images séquentielles de quarry_dataset.
    """
    files = glob.glob(str(DATASET_DIR / "**" / "images" / "*.jpg"), recursive=True)
    frame_map = {}
    for f in files:
        m = re.search(r'video3_frame_(\d+)', os.path.basename(f))
        if m:
            frame_num = int(m.group(1))
            if frame_num not in frame_map:
                frame_map[frame_num] = f

    if not frame_map:
        print("[Data] Aucune image séquentielle trouvée dans quarry_dataset.")
        return

    sorted_keys = sorted(frame_map.keys())
    print(f"[Data] Assemblage de {len(sorted_keys)} frames en vidéo HD...")

    out_path = Path(output_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_avi = str(out_path.parent / "temp_assemble.avi")

    first_img = cv2.imread(frame_map[sorted_keys[0]])
    h, w = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(temp_avi, fourcc, target_fps, (w, h))
    for k in sorted_keys:
        writer.write(cv2.imread(frame_map[k]))
    writer.release()

    try:
        import imageio_ffmpeg
        import subprocess
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, '-y', '-i', temp_avi,
            '-vf', 'scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'fast', '-crf', '20',
            str(out_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(temp_avi):
            os.remove(temp_avi)
        print(f"[Data] Vidéo générée avec succès : {out_path} ({os.path.getsize(out_path)/1e6:.1f} Mo)")
    except Exception as e:
        print(f"[Data] Erreur lors de l'encodage H.264 : {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prétraitement et gestion des données QuarryVision AI")
    parser.add_argument("--download", action="store_true", help="Télécharger le dataset depuis Roboflow")
    parser.add_argument("--fix-yaml", action="store_true", help="Mettre à jour les chemins dans data.yaml")
    parser.add_argument("--build-video", action="store_true", help="Reconstituer la vidéo HD à partir des frames")
    args = parser.parse_args()

    if args.download:
        download_dataset_from_roboflow()
    elif args.fix_yaml:
        fix_data_yaml()
    elif args.build_video:
        build_video_from_frames()
    else:
        fix_data_yaml()
        print("Utilisez --help pour voir les options disponibles.")

