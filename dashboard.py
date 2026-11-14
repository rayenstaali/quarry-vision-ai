"""
Tableau de bord interactif Streamlit pour la supervision de carrière.
Version améliorée : démonstration complète avec analyses IA,
frames clés extraites de la vidéo annotée, et statistiques avancées.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import time
import cv2
import base64
import os

from src.config import BASE_DIR, DATA_DIR, OUTPUT_DIR, ZONES_CONFIG
from src.synthetic_generator import generate_demo_video
from src.pipeline import process_video_pipeline

# Configuration de la page Streamlit
st.set_page_config(
    page_title="QuarryVision AI | Suivi Camions & Chargement",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé pour une interface moderne et industrielle
st.markdown("""
<style>
    .metric-card {
        background-color: #1e262c;
        border: 1px solid #2f3b43;
        border-radius: 10px;
        padding: 16px 20px;
        color: #f4f1ea;
    }
    .badge-critical {
        background-color: #d03b3b;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warning {
        background-color: #d9822b;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-success {
        background-color: #27ae60;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-info {
        background-color: #2e86de;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .frame-caption {
        text-align: center;
        font-size: 0.85rem;
        color: #aab4ba;
        margin-top: 4px;
    }
    .ai-insight-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #1e2a35 100%);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    .ai-insight-title {
        color: #60a5fa;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .ai-insight-content {
        color: #d1d5db;
        font-size: 0.95rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def get_or_create_web_video(raw_video_path: Path, web_video_path: Path) -> bytes:
    """
    S'assure que la vidéo au format H.264 (compatible navigateur Web HTML5)
    est prête et retourne ses octets sous forme binaire.
    """
    if web_video_path.exists() and web_video_path.stat().st_size > 0:
        with open(web_video_path, "rb") as f:
            return f.read()

    if raw_video_path.exists() and raw_video_path.stat().st_size > 0:
        try:
            import imageio_ffmpeg
            import subprocess
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run([
                ffmpeg_exe, '-i', str(raw_video_path),
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                '-y', str(web_video_path)
            ], check=True, capture_output=True)
            if web_video_path.exists():
                with open(web_video_path, "rb") as f:
                    return f.read()
        except Exception:
            pass

        # Fallback de secours si le ré-encodage échoue
        with open(raw_video_path, "rb") as f:
            return f.read()

    return None


def extract_key_frames(video_path: str, num_frames: int = 8) -> list:
    """Extrait des frames clés uniformément réparties dans la vidéo annotée."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((int(idx), frame_rgb))
    cap.release()
    return frames


def generate_ai_analysis(df_trucks: pd.DataFrame, df_alerts: pd.DataFrame, total_frames: int, fps: float) -> dict:
    """Génère une analyse IA complète à partir des données du pipeline."""
    analysis = {}

    # Durée totale de la vidéo
    duration_sec = total_frames / fps if fps > 0 else 0
    analysis["duration_min"] = duration_sec / 60.0

    # Taux de conformité : camions chargés sans débordement
    if not df_trucks.empty:
        total = len(df_trucks)
        loaded_ok = len(df_trucks[
            (df_trucks["Etat_Benne"].isin(["loaded", "partially_loaded"])) &
            (df_trucks["Debordement_Detecte"] == "NON")
        ])
        overloaded = len(df_trucks[df_trucks["Debordement_Detecte"] == "OUI"])
        empty_exit = len(df_trucks[
            (df_trucks["Etape_Cycle"].str.contains("Sortie", na=False)) &
            (df_trucks["Etat_Benne"] == "empty")
        ])
        analysis["conformity_rate"] = (loaded_ok / max(1, total)) * 100
        analysis["overload_rate"] = (overloaded / max(1, total)) * 100
        analysis["empty_exit_count"] = empty_exit
        analysis["avg_loading_sec"] = df_trucks["Duree_Chargement_Sec"].mean()
        analysis["max_loading_sec"] = df_trucks["Duree_Chargement_Sec"].max()
    else:
        analysis["conformity_rate"] = 0
        analysis["overload_rate"] = 0
        analysis["empty_exit_count"] = 0
        analysis["avg_loading_sec"] = 0
        analysis["max_loading_sec"] = 0

    # Alertes par type
    if not df_alerts.empty and "type" in df_alerts.columns:
        analysis["alerts_by_type"] = df_alerts["type"].value_counts().to_dict()
        analysis["critical_count"] = len(df_alerts[df_alerts.get("severity", pd.Series()) == "CRITICAL"]) if "severity" in df_alerts.columns else 0
        analysis["warning_count"] = len(df_alerts[df_alerts.get("severity", pd.Series()) == "WARNING"]) if "severity" in df_alerts.columns else 0
    else:
        analysis["alerts_by_type"] = {}
        analysis["critical_count"] = 0
        analysis["warning_count"] = 0

    # Score de sécurité global (0-100)
    penalty = analysis["critical_count"] * 15 + analysis["warning_count"] * 5
    analysis["safety_score"] = max(0, 100 - penalty)

    return analysis


# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.image("https://img.icons8.com/color/96/dump-truck.png", width=70)
st.sidebar.title("QuarryVision AI")
st.sidebar.caption("Système intelligent de contrôle de carrière")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Configuration du pipeline")
conf_threshold = st.sidebar.slider("Seuil de confiance YOLO", 0.1, 0.9, 0.35, 0.05)
selected_tracker = st.sidebar.selectbox("Algorithme de tracking", ["ByteTrack (Recommandé)", "BoT-SORT"])

# Sélection du modèle de vision (Base vs Fine-tuné carrière)
ft_model_path = OUTPUT_DIR.parent / "models" / "quarry_load_best.pt"
available_models = {
    "YOLOv8 Fine-tuné (Quarry-Load-Verification)": str(ft_model_path),
    "YOLOv8 Nano (Base COCO)": "yolov8n.pt"
}
# Sélection par défaut du modèle fine-tuné s'il existe
default_idx = 0 if ft_model_path.exists() else 1
selected_model_label = st.sidebar.selectbox("Modèle de détection", list(available_models.keys()), index=default_idx)
selected_model_path = available_models[selected_model_label]
if ft_model_path.exists() and selected_model_label.startswith("YOLOv8 Fine-tuné"):
    st.sidebar.caption("✅ Poids entraînés sur les 7 classes de chargement de carrière.")

# Sélection de la vidéo source
video_files = list((DATA_DIR / "raw_videos").glob("*.mp4"))
# Priorité aux vidéos réelles en haut de la liste
def video_sort_key(p):
    name = p.name.lower()
    if "reelle_hd" in name:
        return 0
    elif "reelle" in name:
        return 1
    elif "demo" in name:
        return 2
    return 3

video_files.sort(key=video_sort_key)
video_options = [f.name for f in video_files]

st.sidebar.markdown("---")
st.sidebar.subheader("🎬 Vidéos sources")
if not video_options:
    st.sidebar.info("Aucune vidéo trouvée.")
    if st.sidebar.button("🏗️ Générer vidéo démo de carrière", use_container_width=True):
        with st.spinner("Génération du scénario vidéo..."):
            generate_demo_video()
            st.rerun()
else:
    selected_video_name = st.sidebar.selectbox("Choisir une vidéo", video_options, index=0)
    if "demo" in selected_video_name.lower():
        if st.sidebar.button("🔄 Régénérer vidéo démo", use_container_width=True):
            with st.spinner("Génération du scénario vidéo..."):
                generate_demo_video()
                st.rerun()

# ==============================================================================
# EN-TÊTE PRINCIPAL
# ==============================================================================
st.title("🏭 Système de Vision IA : Suivi des Camions & Contrôle du Chargement")
st.markdown(
    "Supervision intelligente des flux de transport, analyse du cycle de chargement, "
    "inspection des bennes et détection des anomalies de sécurité sur site carrier."
)

# ==============================================================================
# CHARGEMENT DES DONNÉES
# ==============================================================================
csv_report_file = OUTPUT_DIR / "rapport_production_carriere.csv"
csv_alerts_file = OUTPUT_DIR / "journal_alertes_securite.csv"

# Résolution dynamique de la vidéo annotée selon la vidéo sélectionnée
cand_annotated = OUTPUT_DIR / f"annotated_{selected_video_name}" if 'selected_video_name' in locals() else None
cand_web = OUTPUT_DIR / f"web_annotated_{selected_video_name}" if 'selected_video_name' in locals() else None

if cand_web and cand_web.exists():
    annotated_video_path = cand_annotated if cand_annotated.exists() else cand_web
    active_web_video_path = cand_web
elif cand_annotated and cand_annotated.exists():
    annotated_video_path = cand_annotated
    active_web_video_path = cand_web
else:
    # Recherche du fichier web ou annoté disponible le plus récent
    all_web_vids = list(OUTPUT_DIR.glob("*web*.mp4"))
    if all_web_vids:
        active_web_video_path = sorted(all_web_vids, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        annotated_video_path = active_web_video_path
    else:
        annotated_video_path = OUTPUT_DIR / "annotated_demo_carriere.mp4"
        active_web_video_path = OUTPUT_DIR / "annotated_demo_web.mp4"

if csv_report_file.exists():
    df_trucks = pd.read_csv(csv_report_file)
else:
    df_trucks = pd.DataFrame()

if csv_alerts_file.exists():
    df_alerts = pd.read_csv(csv_alerts_file)
else:
    df_alerts = pd.DataFrame()

# Paramètres vidéo dynamiques extraits directement du fichier
VIDEO_FPS = 12.0
VIDEO_TOTAL_FRAMES = 208
if annotated_video_path.exists():
    try:
        _cap = cv2.VideoCapture(str(annotated_video_path))
        if _cap.isOpened():
            _fc = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            _fps = float(_cap.get(cv2.CAP_PROP_FPS))
            if _fc > 0:
                VIDEO_TOTAL_FRAMES = _fc
            if _fps > 0:
                VIDEO_FPS = _fps
            _cap.release()
    except Exception:
        pass

# ==============================================================================
# SECTION : LANCEMENT DU PIPELINE
# ==============================================================================
if df_trucks.empty:
    st.warning("⚠️ Aucune donnée d'analyse disponible. Lancez le pipeline pour générer la démonstration.")

if video_options:
    col_run_1, col_run_2 = st.columns([3, 1])
    with col_run_1:
        st.info(f"📹 Vidéo sélectionnée : **{selected_video_name}**")
    with col_run_2:
        run_btn = st.button(
            "▶️ Lancer l'analyse complète",
            type="primary",
            use_container_width=True
        )

    if run_btn:
        selected_vid = DATA_DIR / "raw_videos" / selected_video_name
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(p, current, total):
            progress_bar.progress(p)
            status_text.text(f"🔍 Analyse IA en cours... frame {current}/{total} ({int(p*100)}%)")

        with st.spinner(f"Inférence {selected_model_label} + ByteTrack en cours..."):
            res = process_video_pipeline(
                str(selected_vid),
                progress_callback=update_progress,
                model_path=selected_model_path
            )
        st.success("✅ Analyse terminée avec succès !")
        time.sleep(1)
        st.rerun()

st.markdown("---")

# ==============================================================================
# SECTION : KPIs PRINCIPAUX
# ==============================================================================
if not df_trucks.empty:
    ai_analysis = generate_ai_analysis(df_trucks, df_alerts, VIDEO_TOTAL_FRAMES, VIDEO_FPS)

    st.subheader("📊 Indicateurs Clés de Performance (KPIs)")
    col1, col2, col3, col4, col5 = st.columns(5)

    total_trucks = len(df_trucks)
    loaded_count = len(df_trucks[df_trucks["Etat_Benne"].isin(["loaded", "overloaded", "partially_loaded"])])
    alerts_count = len(df_alerts)

    col1.metric("🚛 Camions détectés", f"{total_trucks}", "Suivi continu")
    col2.metric("📦 Camions chargés", f"{loaded_count}", f"{(loaded_count/max(1, total_trucks))*100:.0f}% du flux")
    col3.metric("⏱️ Temps moy. chargement", f"{ai_analysis['avg_loading_sec']:.0f}s", "Cycle analysé")
    col4.metric("🚨 Alertes détectées", f"{alerts_count}", f"{ai_analysis['critical_count']} critiques", delta_color="inverse")
    col5.metric("🛡️ Score Sécurité", f"{ai_analysis['safety_score']}/100",
                "Bon" if ai_analysis["safety_score"] >= 70 else "Attention requise",
                delta_color="normal" if ai_analysis["safety_score"] >= 70 else "inverse")

    st.markdown("---")

# ==============================================================================
# ONGLETS FONCTIONNELS
# ==============================================================================
tab_demo, tab_analysis, tab_registry, tab_alerts, tab_cahier = st.tabs([
    "🎥 Visualisation Vidéo (MP4)",
    "📈 Statistiques de Production",
    "🚛 Registre de la Flotte",
    "🚨 Journal des Alertes",
    "📋 Cahier des Charges"
])

# --- ONGLET 1 : VISUALISATION VIDÉO & DÉTECTION ---
with tab_demo:
    st.subheader("🎥 Visualisation du Module de Détection & Suivi des Camions")
    st.markdown(
        "Résultat du traitement par vision par ordinateur (**YOLOv8** + **ByteTrack**) : "
        "bounding boxes, identifiants des camions, contrôle de l'état de la benne, "
        "zones géométriques et détection des anomalies."
    )

    if (active_web_video_path.exists() or annotated_video_path.exists()) and not df_trucks.empty:
        video_bytes = get_or_create_web_video(annotated_video_path, active_web_video_path)

        if video_bytes and len(video_bytes) > 0:
            v_col_left, v_col_right = st.columns([3, 1])
            with v_col_left:
                st.video(video_bytes, format="video/mp4", start_time=0)
                st.caption("▶️ Lecteur vidéo MP4 interactif (Contrôles : lecture, pause, plein écran, vitesse).")

            with v_col_right:
                st.markdown("##### 📥 Téléchargement & Détails")
                st.markdown(f"""
                - **Format** : MP4 (H.264 / AVC)
                - **Résolution** : 1280 × 720 px
                - **Cadence** : {VIDEO_FPS:.0f} FPS
                - **Durée** : {VIDEO_TOTAL_FRAMES / VIDEO_FPS:.1f} s ({VIDEO_TOTAL_FRAMES} frames)
                - **Taille** : {len(video_bytes) / (1024 * 1024):.2f} Mo
                - **Modèle** : YOLOv8 + ByteTrack
                """)
                st.download_button(
                    label="📥 Télécharger la Vidéo MP4",
                    data=video_bytes,
                    file_name="quarry_truck_detection_annotated.mp4",
                    mime="video/mp4",
                    type="primary",
                    use_container_width=True
                )
                if csv_report_file.exists():
                    with open(csv_report_file, "rb") as rf:
                        st.download_button(
                            label="📊 Télécharger Rapport CSV",
                            data=rf,
                            file_name="rapport_production_carriere.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

            st.markdown("")

            # ---- Frames clés extraites ----
            with st.expander("📸 Voir les captures clés frame par frame", expanded=False):
                key_frames = extract_key_frames(str(annotated_video_path), num_frames=8)
                if key_frames:
                    for row_start in range(0, len(key_frames), 4):
                        cols = st.columns(4)
                        for j, col in enumerate(cols):
                            idx = row_start + j
                            if idx < len(key_frames):
                                frame_num, frame_img = key_frames[idx]
                                with col:
                                    st.image(frame_img, use_container_width=True)
                                    time_sec = frame_num / VIDEO_FPS
                                    st.markdown(
                                        f"<div class='frame-caption'>Frame #{frame_num} — {time_sec:.1f}s</div>",
                                        unsafe_allow_html=True
                                    )
        else:
            st.warning("⚠️ Impossible de charger la vidéo annotée.")

        st.markdown("---")

        # ---------------------------------------------------------------
        # Section 2 : Rapport d'analyse IA
        # ---------------------------------------------------------------
        st.markdown("#### 🧠 Rapport d'Analyse par l'Agent IA")

        ai_col1, ai_col2 = st.columns([1, 1])

        with ai_col1:
            # Synthèse textuelle
            st.markdown(f"""
<div class="ai-insight-box">
    <div class="ai-insight-title">🔎 Synthèse de la Session d'Analyse</div>
    <div class="ai-insight-content">
        <b>Durée analysée :</b> {ai_analysis['duration_min']:.1f} minutes ({VIDEO_TOTAL_FRAMES} frames à {VIDEO_FPS:.0f} FPS)<br><br>
        <b>Véhicules détectés :</b> {total_trucks} camion(s) identifié(s) et suivis avec identifiants stables.<br><br>
        <b>Taux de conformité :</b> {ai_analysis['conformity_rate']:.0f}% — 
        {"✅ Tous les chargements sont conformes." if ai_analysis['overload_rate'] == 0
         else f"⚠️ {ai_analysis['overload_rate']:.0f}% des camions présentent un débordement de benne."}<br><br>
        <b>Score de sécurité :</b> <span style="color: {'#27ae60' if ai_analysis['safety_score'] >= 70 else '#d03b3b'}; font-weight:bold;">
        {ai_analysis['safety_score']}/100</span><br><br>
        <b>Alertes critiques :</b> {ai_analysis['critical_count']} (piétons en zone danger)<br>
        <b>Alertes avertissement :</b> {ai_analysis['warning_count']} (surcharges détectées)
    </div>
</div>
            """, unsafe_allow_html=True)

        with ai_col2:
            # Recommandations IA
            recommendations = []
            if ai_analysis["critical_count"] > 0:
                recommendations.append(
                    "🔴 **Sécurité piétons** : Des piétons ont été détectés dans la zone de manœuvre des engins. "
                    "Recommandation : renforcer la signalisation et les barrières physiques dans la zone de chargement."
                )
            if ai_analysis["overload_rate"] > 0:
                recommendations.append(
                    "🟠 **Surcharges détectées** : Certains camions quittent le site avec un chargement débordant. "
                    "Risque de chute de matière sur la voie publique. Action : calibrer la chargeuse pour respecter "
                    "le volume utile de la benne."
                )
            if ai_analysis["empty_exit_count"] > 0:
                recommendations.append(
                    "🟡 **Rotations à vide** : Des camions sortent sans chargement. "
                    "Vérifier la coordination avec le poste de chargement."
                )
            if not recommendations:
                recommendations.append(
                    "✅ **Aucune anomalie majeure détectée.** Le site fonctionne dans les paramètres normaux."
                )

            st.markdown(f"""
<div class="ai-insight-box">
    <div class="ai-insight-title">💡 Recommandations de l'Agent IA</div>
    <div class="ai-insight-content">
        {"<br><br>".join(recommendations)}
    </div>
</div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ---------------------------------------------------------------
        # Section 3 : Timeline des alertes
        # ---------------------------------------------------------------
        if not df_alerts.empty and "frame" in df_alerts.columns:
            st.markdown("#### ⏱️ Chronologie des Événements Détectés")
            df_timeline = df_alerts.copy()
            df_timeline["Temps (sec)"] = df_timeline["frame"] / VIDEO_FPS
            df_timeline["Type"] = df_timeline["type"].map({
                "SECURITE_PIETON": "🚶 Piéton en zone danger",
                "SECURITE_CHARGEMENT": "📦 Surcharge benne",
                "ANOMALIE_ROTATION": "🔄 Rotation à vide",
                "ANOMALIE_DUREE": "⏰ Chargement long"
            }).fillna(df_timeline["type"])

            severity_colors = {"CRITICAL": "#d03b3b", "WARNING": "#d9822b", "INFO": "#2e86de"}

            fig_timeline = px.scatter(
                df_timeline,
                x="Temps (sec)",
                y="Type",
                color="severity" if "severity" in df_timeline.columns else None,
                color_discrete_map=severity_colors,
                hover_data=["message", "frame"],
                title="Chronologie des alertes de sécurité et anomalies",
                labels={"severity": "Sévérité"},
                height=300
            )
            fig_timeline.update_traces(marker=dict(size=14, symbol="diamond"))
            fig_timeline.update_layout(
                xaxis_title="Temps dans la vidéo (secondes)",
                yaxis_title="",
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font=dict(color="#f4f1ea"),
                showlegend=True
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

        # ---------------------------------------------------------------
        # Section 4 : Carte des zones ROI
        # ---------------------------------------------------------------
        st.markdown("#### 🗺️ Cartographie des Zones d'Intérêt (ROI)")

        zone_desc = {
            "entree": ("Zone d'Entrée", "Détection de l'arrivée des camions sur le site.", "#FFA500"),
            "chargement": ("Zone de Chargement", "Poste des pelles et chargeuses — suivi du temps de chargement.", "#FFFF00"),
            "sortie": ("Zone de Sortie / Pont-Bascule", "Contrôle de sortie, pesée et vérification de la benne.", "#00C800"),
            "danger": ("Zone Danger Piétons", "Périmètre d'exclusion des piétons (rayon d'action des engins).", "#FF0000")
        }
        zone_cols = st.columns(4)
        for i, (zone_key, (zone_name, zone_info, zone_color)) in enumerate(zone_desc.items()):
            with zone_cols[i]:
                st.markdown(f"""
<div style="background:#1b2228; padding:14px; border-radius:8px; border-left:5px solid {zone_color}; min-height:120px;">
    <strong style="color:{zone_color};">{zone_name}</strong><br>
    <span style="color:#aab4ba; font-size:0.85rem;">{zone_info}</span>
</div>
                """, unsafe_allow_html=True)

    else:
        st.info(
            "🔄 Aucune analyse disponible. Cliquez sur **'Lancer l'analyse complète'** "
            "ci-dessus pour exécuter le pipeline YOLOv8 + ByteTrack sur la vidéo de démonstration."
        )

# --- ONGLET 2 : STATISTIQUES DE PRODUCTION ---
with tab_analysis:
    st.subheader("📈 Tableaux de Bord d'Exploitation")

    if not df_trucks.empty:
        g_col1, g_col2 = st.columns(2)

        with g_col1:
            if "Etat_Benne" in df_trucks.columns:
                state_counts = df_trucks["Etat_Benne"].value_counts().reset_index()
                state_counts.columns = ["État", "Nombre"]
                fig_pie = px.pie(
                    state_counts,
                    values="Nombre",
                    names="État",
                    title="Répartition de l'état de chargement des bennes",
                    color="État",
                    color_discrete_map={
                        "empty": "#95a5a6",
                        "partially_loaded": "#f39c12",
                        "loaded": "#27ae60",
                        "overloaded": "#c0392b"
                    },
                    hole=0.4
                )
                fig_pie.update_layout(
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#f4f1ea")
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        with g_col2:
            if "Duree_Chargement_Sec" in df_trucks.columns:
                fig_bar = px.bar(
                    df_trucks,
                    x="ID_Camion",
                    y="Duree_Chargement_Sec",
                    title="Durée de chargement par camion (secondes)",
                    labels={"Duree_Chargement_Sec": "Secondes", "ID_Camion": "Camion"},
                    color="Etat_Benne",
                    color_discrete_map={
                        "empty": "#95a5a6",
                        "partially_loaded": "#f39c12",
                        "loaded": "#27ae60",
                        "overloaded": "#c0392b"
                    }
                )
                fig_bar.update_layout(
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#f4f1ea")
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # Graphique de confiance par camion
        g2_col1, g2_col2 = st.columns(2)

        with g2_col1:
            if "Confiance_Benne" in df_trucks.columns:
                fig_conf = px.bar(
                    df_trucks,
                    x="ID_Camion",
                    y="Confiance_Benne",
                    title="Indice de confiance de la classification de la benne",
                    labels={"Confiance_Benne": "Confiance", "ID_Camion": "Camion"},
                    color="Confiance_Benne",
                    color_continuous_scale=["#d03b3b", "#f39c12", "#27ae60"],
                    range_color=[0, 1]
                )
                fig_conf.update_layout(
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#f4f1ea")
                )
                st.plotly_chart(fig_conf, use_container_width=True)

        with g2_col2:
            if not df_alerts.empty and "type" in df_alerts.columns:
                alert_counts = df_alerts["type"].value_counts().reset_index()
                alert_counts.columns = ["Type d'Alerte", "Nombre"]
                fig_alert_bar = px.bar(
                    alert_counts,
                    x="Type d'Alerte",
                    y="Nombre",
                    title="Distribution des alertes par catégorie",
                    color="Type d'Alerte",
                    color_discrete_map={
                        "SECURITE_PIETON": "#d03b3b",
                        "SECURITE_CHARGEMENT": "#d9822b",
                        "ANOMALIE_ROTATION": "#f39c12",
                        "ANOMALIE_DUREE": "#2e86de"
                    }
                )
                fig_alert_bar.update_layout(
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#f4f1ea"), showlegend=False
                )
                st.plotly_chart(fig_alert_bar, use_container_width=True)
    else:
        st.info("Lancez l'analyse pour voir les statistiques de production.")

# --- ONGLET 3 : REGISTRE DE LA FLOTTE ---
with tab_registry:
    st.subheader("🚛 Historique des camions et suivi du cycle de chargement")

    if not df_trucks.empty:
        st.dataframe(
            df_trucks,
            use_container_width=True,
            column_config={
                "ID_Camion": st.column_config.NumberColumn("ID Camion", format="#%d"),
                "Premier_Passage_Sec": st.column_config.NumberColumn("Entrée (sec)", format="%.1f s"),
                "Dernier_Passage_Sec": st.column_config.NumberColumn("Dernier vu (sec)", format="%.1f s"),
                "Duree_Chargement_Sec": st.column_config.NumberColumn("Durée chargement", format="%.1f s"),
                "Confiance_Benne": st.column_config.ProgressColumn("Confiance État", min_value=0, max_value=1),
                "Debordement_Detecte": st.column_config.TextColumn("Débordement"),
                "Alertes_Associees": st.column_config.TextColumn("Alertes")
            }
        )

        # Bouton de téléchargement
        if csv_report_file.exists():
            with open(csv_report_file, "rb") as f:
                st.download_button(
                    label="📥 Télécharger le rapport de production CSV",
                    data=f,
                    file_name="rapport_production_carriere.csv",
                    mime="text/csv"
                )
    else:
        st.info("Aucun camion enregistré. Lancez l'analyse pour remplir le registre.")

# --- ONGLET 4 : JOURNAL DES ALERTES ---
with tab_alerts:
    st.subheader("🚨 Événements anormaux et infractions de sécurité")

    if not df_alerts.empty:
        # Résumé en haut
        a_col1, a_col2, a_col3 = st.columns(3)
        critical = len(df_alerts[df_alerts.get("severity", pd.Series()) == "CRITICAL"]) if "severity" in df_alerts.columns else 0
        warning = len(df_alerts[df_alerts.get("severity", pd.Series()) == "WARNING"]) if "severity" in df_alerts.columns else 0
        a_col1.metric("🔴 Critiques", critical)
        a_col2.metric("🟠 Avertissements", warning)
        a_col3.metric("📝 Total", len(df_alerts))

        st.markdown("---")

        # Liste détaillée des alertes
        for _, alert in df_alerts.iterrows():
            sev = alert.get("severity", "INFO")
            icon = "🔴" if sev == "CRITICAL" else ("🟠" if sev == "WARNING" else "🔵")
            badge_class = "badge-critical" if sev == "CRITICAL" else ("badge-warning" if sev == "WARNING" else "badge-info")
            border_color = "#d03b3b" if sev == "CRITICAL" else ("#d9822b" if sev == "WARNING" else "#2e86de")

            frame_num = alert.get("frame", "N/A")
            time_sec = f"{float(frame_num) / VIDEO_FPS:.1f}s" if frame_num != "N/A" else "N/A"

            st.markdown(f"""
            <div style="background:#1b2228; padding:12px 18px; border-radius:8px; margin-bottom:8px; border-left:4px solid {border_color};">
                <span class="{badge_class}">{sev}</span>
                <strong style="margin-left:12px; color:#f4f1ea;">{icon} {alert.get('message', '')}</strong>
                <span style="float:right; color:#aab4ba; font-size:0.85rem;">
                    Frame #{frame_num} — {time_sec} | {alert.get('timestamp', 'N/A')}
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        if csv_alerts_file.exists():
            with open(csv_alerts_file, "rb") as f:
                st.download_button(
                    label="📥 Télécharger le journal des alertes CSV",
                    data=f,
                    file_name="journal_alertes_securite.csv",
                    mime="text/csv"
                )
    else:
        st.success("✅ Aucune alerte anormale enregistrée pour le moment.")

# --- ONGLET 5 : CAHIER DES CHARGES ---
with tab_cahier:
    st.subheader("📋 Cahier des Charges Officiel du Projet")
    cdc_file = BASE_DIR / "cahier_des_charges.md"
    if cdc_file.exists():
        with open(cdc_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Le document `cahier_des_charges.md` est en cours de consultation.")
