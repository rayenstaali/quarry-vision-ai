"""
Générateur de vidéo de simulation pour carrière de granulats.
Permet de créer automatiquement une vidéo de test synthétique en haute résolution (1280x720)
avec camions en mouvement, cycle de chargement progressif, zones d'intérêt
et simulation de scénarios anormaux (piéton en zone rouge, débordement, camion vide).
"""

import cv2
import numpy as np
from pathlib import Path
from src.config import DATA_DIR, ZONE_ENTRY, ZONE_LOADING, ZONE_EXIT, ZONE_DANGER


def draw_quarry_background(width: int = 1280, height: int = 720) -> np.ndarray:
    """
    Génère un décor de carrière réaliste avec pistes de terre, fosses et monticules de gravier.
    """
    # 1. Sol général de la carrière (ton ocre/sable/pierre)
    bg = np.full((height, width, 3), (170, 185, 195), dtype=np.uint8)

    # 2. Piste de circulation principale (terre battue foncée)
    pts_road = np.array([
        [0, 520], [1280, 480], [1280, 680], [0, 680]
    ], dtype=np.int32)
    cv2.fillPoly(bg, [pts_road], (130, 145, 155))

    # 3. Monticules de granulats / gravats en arrière-plan
    # Monticule 1 (Sable fin - jaunâtre)
    cv2.ellipse(bg, (250, 260), (180, 90), 0, 0, 360, (140, 190, 210), -1)
    # Monticule 2 (Gravier gris concassé)
    cv2.ellipse(bg, (650, 220), (220, 110), 0, 0, 360, (120, 130, 140), -1)
    # Monticule 3 (Roches calcaires claires)
    cv2.ellipse(bg, (1050, 250), (190, 95), 0, 0, 360, (180, 195, 205), -1)

    # 4. Dessin statique de la pelle mécanique (Excavator) en zone de chargement
    # Base chenilles
    cv2.rectangle(bg, (680, 270), (790, 310), (40, 40, 40), -1)
    # Cabine jaune
    cv2.rectangle(bg, (700, 230), (770, 280), (30, 180, 230), -1)
    # Flèche / bras articulé
    cv2.line(bg, (720, 240), (620, 340), (20, 160, 210), 8)
    cv2.line(bg, (620, 340), (580, 390), (40, 40, 40), 6)

    # Bruit de texture pour simuler la poussière et les aspérités
    noise = np.random.normal(0, 5, bg.shape).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return bg


def draw_synthetic_truck(frame: np.ndarray, x: int, y: int, load_state: str = "empty", scale: float = 1.0):
    """
    Dessine un camion de chantier réaliste avec cabine et benne.
    :param load_state: 'empty', 'partially_loaded', 'loaded', 'overloaded'
    """
    w = int(170 * scale)
    h = int(90 * scale)

    # 1. Cabine du camion (blanche / industrielle)
    cab_w = int(w * 0.32)
    cab_h = int(h * 0.75)
    cab_x = x + w - cab_w
    cab_y = y + (h - cab_h)
    cv2.rectangle(frame, (cab_x, cab_y), (cab_x + cab_w, y + h), (220, 220, 220), -1)
    cv2.rectangle(frame, (cab_x, cab_y), (cab_x + cab_w, y + h), (50, 50, 50), 2)
    
    # Pare-brise
    cv2.rectangle(frame, (cab_x + int(cab_w*0.2), cab_y + 8), (cab_x + cab_w - 6, cab_y + int(cab_h*0.45)), (180, 140, 100), -1)

    # 2. Châssis & Roues
    cv2.rectangle(frame, (x, y + h - 14), (x + w, y + h), (30, 30, 30), -1)
    # Roues (arrière, milieu, avant)
    wheel_r = int(14 * scale)
    cv2.circle(frame, (x + int(w * 0.18), y + h), wheel_r, (20, 20, 20), -1)
    cv2.circle(frame, (x + int(w * 0.38), y + h), wheel_r, (20, 20, 20), -1)
    cv2.circle(frame, (x + int(w * 0.85), y + h), wheel_r, (20, 20, 20), -1)
    cv2.circle(frame, (x + int(w * 0.18), y + h), wheel_r // 2, (120, 120, 120), -1)
    cv2.circle(frame, (x + int(w * 0.38), y + h), wheel_r // 2, (120, 120, 120), -1)
    cv2.circle(frame, (x + int(w * 0.85), y + h), wheel_r // 2, (120, 120, 120), -1)

    # 3. Benne du camion (arrière)
    benne_w = int(w * 0.65)
    benne_h = int(h * 0.65)
    benne_x = x + 4
    benne_y = y + (h - benne_h - 10)
    
    # Structure métallique de la benne (Gris acier / bleu chantier)
    cv2.rectangle(frame, (benne_x, benne_y), (benne_x + benne_w, benne_y + benne_h), (80, 95, 110), -1)
    cv2.rectangle(frame, (benne_x, benne_y), (benne_x + benne_w, benne_y + benne_h), (35, 45, 55), 2)

    # 4. Rendu de la matière dans la benne selon load_state
    if load_state == "partially_loaded":
        # Fond avec un monticule bas
        fill_h = int(benne_h * 0.45)
        mat_pts = np.array([
            [benne_x + 5, benne_y + benne_h - 4],
            [benne_x + int(benne_w * 0.4), benne_y + benne_h - fill_h],
            [benne_x + benne_w - 5, benne_y + benne_h - 4]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [mat_pts], (110, 130, 145))
    elif load_state == "loaded":
        # Remplissage complet à ras bord avec monticule régulier
        fill_h = int(benne_h * 0.85)
        mat_pts = np.array([
            [benne_x + 3, benne_y + 12],
            [benne_x + int(benne_w * 0.5), benne_y - 8],
            [benne_x + benne_w - 3, benne_y + 12],
            [benne_x + benne_w - 3, benne_y + benne_h - 4],
            [benne_x + 3, benne_y + benne_h - 4]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [mat_pts], (100, 120, 135))
        # Texturation gravier
        for _ in range(25):
            gx = np.random.randint(benne_x + 8, benne_x + benne_w - 8)
            gy = np.random.randint(benne_y - 4, benne_y + benne_h - 8)
            cv2.circle(frame, (gx, gy), 2, (150, 170, 185), -1)
    elif load_state == "overloaded":
        # Débordement au-dessus des ridelles
        fill_h = int(benne_h * 1.25)
        mat_pts = np.array([
            [benne_x - 6, benne_y + 15],
            [benne_x + int(benne_w * 0.45), benne_y - 25],
            [benne_x + benne_w + 6, benne_y + 15],
            [benne_x + benne_w, benne_y + benne_h - 4],
            [benne_x, benne_y + benne_h - 4]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [mat_pts], (90, 115, 130))
        # Particules qui tombent / débordent
        for _ in range(40):
            gx = np.random.randint(benne_x - 4, benne_x + benne_w + 4)
            gy = np.random.randint(benne_y - 20, benne_y + benne_h - 6)
            cv2.circle(frame, (gx, gy), 3, (160, 180, 195), -1)


def draw_synthetic_worker(frame: np.ndarray, x: int, y: int):
    """
    Dessine un ouvrier avec gilet haute visibilité et casque de chantier.
    """
    # Jambes
    cv2.line(frame, (x, y + 22), (x - 4, y + 42), (40, 40, 40), 3)
    cv2.line(frame, (x, y + 22), (x + 4, y + 42), (40, 40, 40), 3)
    # Torse (Gilet orange fluo)
    cv2.rectangle(frame, (x - 7, y + 6), (x + 7, y + 24), (0, 140, 255), -1)
    # Bandes réfléchissantes
    cv2.line(frame, (x - 6, y + 14), (x + 6, y + 14), (240, 240, 240), 2)
    # Tête & Casque jaune
    cv2.circle(frame, (x, y), 6, (200, 220, 230), -1)
    cv2.ellipse(frame, (x, y - 2), (7, 5), 0, 180, 360, (0, 230, 255), -1)


def generate_demo_video(output_filename: str = "demo_carriere.mp4", duration_sec: int = 15, fps: int = 30):
    """
    Génère la vidéo de démonstration complète intégrant les différents scénarios du cahier des charges.
    """
    output_path = DATA_DIR / "raw_videos" / output_filename
    width, height = 1280, 720
    total_frames = duration_sec * fps

    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    base_bg = draw_quarry_background(width, height)

    print(f"[SyntheticGenerator] Génération de la vidéo démo ({total_frames} frames)...")

    # Scénario :
    # Camion 1 : Arrive vide -> se charge dans la zone -> sort chargé normalement
    # Camion 2 (plus tard) : Arrive -> chargé avec surcharge/débordement
    # Ouvrier : Traverse temporairement la zone de danger entre t=5s et t=9s

    for f_idx in range(total_frames):
        frame = base_bg.copy()
        t = f_idx / fps

        # --- Trajectoire Camion 1 ---
        # Avance de l'entrée (x=60) jusqu'à la zone de chargement (x=530), s'arrête entre t=3s et t=8s,
        # puis repart vers la sortie (x=1150)
        if t < 3.0:
            c1_x = int(60 + (t / 3.0) * 470)
            c1_state = "empty"
        elif t <= 8.0:
            c1_x = 530
            # Remplissage progressif
            if t < 5.0:
                c1_state = "partially_loaded"
            else:
                c1_state = "loaded"
        else:
            c1_x = int(530 + ((t - 8.0) / 7.0) * 620)
            c1_state = "loaded"

        if c1_x < 1250:
            draw_synthetic_truck(frame, c1_x, 480, load_state=c1_state, scale=1.0)

        # --- Camion 2 (Cas particulier : Surcharge) ---
        if t > 6.0:
            t2 = t - 6.0
            c2_x = int(40 + t2 * 90)
            c2_state = "overloaded" if t2 > 4.0 else "empty"
            if c2_x < 1250:
                draw_synthetic_truck(frame, c2_x, 560, load_state=c2_state, scale=0.88)

        # --- Piéton dans la zone dangereuse ---
        if 4.5 <= t <= 10.0:
            # L'ouvrier traverse la zone danger
            w_prog = (t - 4.5) / 5.5
            w_x = int(430 + w_prog * 420)
            w_y = int(370 + np.sin(w_prog * np.pi) * 30)
            draw_synthetic_worker(frame, w_x, w_y)

        out.write(frame)

    out.release()
    print(f"[SyntheticGenerator] Vidéo générée avec succès : {output_path}")
    return output_path


if __name__ == "__main__":
    generate_demo_video()
