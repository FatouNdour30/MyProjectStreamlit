# -*- coding: utf-8 -*-
"""
SamaLearn — ULTRA FINAL PRO (VERSION COMPLÈTE CORRIGÉE)
Intelligence Adaptative, Émotionnelle & Longitudinale
Made in Africa • UCAD • L3 Data Science • Ndeye Fatou NDOUR (2025)
"""
import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import plotly.graph_objects as go
import os
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
import sys
from io import StringIO
from datetime import datetime, timedelta
from streamlit_ace import st_ace  # ← AJOUTEZ CETTE LIGNE
from fpdf import FPDF   # <--- C'est la syntaxe standard pour fpdf2
# =====================================================
# CONFIGURATION
# =====================================================

# =====================================================
# 1. CONFIGURATION & CONSTANTES
# =====================================================
st.set_page_config(page_title="SamaLearn Ultra", page_icon="💠", layout="wide")

# Couleurs du thème
THEME = {
    "primary": "#00F2FF",    # Cyan Néon
    "secondary": "#7000FF",  # Violet Profond
    "bg": "#050505",         # Noir quasi total
    "text": "#E0E0E0",
    "success": "#00FF94",
    "glass": "rgba(255, 255, 255, 0.05)"
}

# Initialisation Session
if "page" not in st.session_state: st.session_state.page = "accueil"
if "user_profile" not in st.session_state: st.session_state.user_profile = None
if "notifications" not in st.session_state: 
    st.session_state.notifications = [
        {"msg": "Nouveau module: Deep Learning", "time": "10 min"},
        {"msg": "Fatou a battu votre record !", "time": "2h"},
        {"msg": "Note d'examen disponible", "time": "1j"}
    ]

# =====================================================
# 2. CSS "NEXT-GEN" (ANIMATIONS & EFFETS)
# =====================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* --- GLOBAL --- */
    html, body, [class*="css"] {{
        font-family: 'Outfit', sans-serif;
        color: {THEME['text']};
    }}

    /* Fond Animé */
    .stApp {{
        background-color: {THEME['bg']};
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(112, 0, 255, 0.2) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(0, 242, 255, 0.15) 0%, transparent 40%);
        animation: pulseBackground 10s ease-in-out infinite alternate;
    }}
    
    @keyframes pulseBackground {{
        0% {{ background-size: 100% 100%; }}
        100% {{ background-size: 110% 110%; }}
    }}

    /* --- GLASSMORPHISM CARDS --- */
    .glass-panel {{
        background: {THEME['glass']};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }}
    .glass-panel:hover {{
        border-color: {THEME['primary']};
        transform: translateY(-4px);
        box-shadow: 0 12px 40px 0 rgba(0, 242, 255, 0.1);
    }}

    /* --- TYPOGRAPHY --- */
    h1, h2, h3 {{ color: white !important; font-weight: 800 !important; letter-spacing: -0.5px; }}
    .highlight {{ color: {THEME['primary']}; }}
    .gradient-text {{
        background: linear-gradient(135deg, {THEME['primary']}, {THEME['secondary']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    /* --- CUSTOM BUTTONS --- */
    div.stButton > button {{
        background: linear-gradient(90deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.1);
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }}
    div.stButton > button:hover {{
        border-color: {THEME['primary']};
        color: {THEME['primary']};
        background: rgba(0, 242, 255, 0.05);
    }}
    div.stButton > button:active {{ transform: scale(0.98); }}

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {{
        background-color: rgba(10, 12, 16, 0.9);
        border-right: 1px solid rgba(255,255,255,0.05);
    }}
    
    /* --- PROGRESS BAR --- */
    .stProgress > div > div > div > div {{
        background-image: linear-gradient(90deg, {THEME['primary']}, {THEME['secondary']});
    }}

</style>
""", unsafe_allow_html=True)


# =====================================================
# CONFIGURATION DU STYLE (INSPIRÉ DE L'IMAGE FOURNIE)
# =====================================================
def set_hero_design():
    st.markdown("""
    <!-- Import de la police Poppins -->
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    
    <style>
        /* 1. FOND DE PAGE (Bleu Nuit Profond comme l'image) */
        .stApp {
            background-color: #030511; 
            background-image: radial-gradient(#1a1f35 1px, transparent 1px);
            background-size: 40px 40px; /* Effet de grille de points subtile */
        }

        /* 2. TYPOGRAPHIE GAUCHE */
        h1.hero-title {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 3.5rem;
            line-height: 1.2;
            color: #FFFFFF;
            margin-bottom: 20px;
        }
        
        p.hero-subtitle {
            font-family: 'Poppins', sans-serif;
            font-weight: 300;
            font-size: 1.1rem;
            color: #B0B8C4; /* Gris clair bleuté */
            line-height: 1.6;
            margin-bottom: 30px;
            max-width: 90%;
        }

        /* 3. INPUT FIELD (Arrondi et sombre) */
        div[data-testid="stTextInput"] input {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border-radius: 50px !important; /* Forme "Pill" */
            padding: 15px 25px !important;
            font-family: 'Poppins', sans-serif !important;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #6C63FF !important;
            background-color: rgba(255, 255, 255, 0.1) !important;
        }
        
        /* Cacher le label de l'input pour le look propre */
        div[data-testid="stTextInput"] label {
            display: none;
        }

        /* 4. BOUTON PRINCIPAL (Violet/Bleu comme "Get Started") */
        div[data-testid="stButton"] button {
            background: linear-gradient(90deg, #5956E9 0%, #3a36cc 100%) !important;
            color: white !important;
            border-radius: 50px !important; /* Forme "Pill" */
            padding: 15px 40px !important;
            border: none !important;
            font-weight: 600 !important;
            font-family: 'Poppins', sans-serif !important;
            box-shadow: 0 10px 20px rgba(89, 86, 233, 0.3);
            transition: all 0.3s ease;
            text-transform: none !important;
            font-size: 1rem !important;
        }
        div[data-testid="stButton"] button:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 30px rgba(89, 86, 233, 0.5);
        }

        /* 5. STATS EN BAS (Comme "150 95 24" sur l'image) */
        .stat-number {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 2rem;
            color: #5956E9; /* Couleur accent */
        }
        .stat-label {
            font-family: 'Poppins', sans-serif;
            font-size: 0.9rem;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Ajustement des marges globales Streamlit */
        .block-container {
            padding-top: 3rem;
            max-width: 1200px;
        }
    </style>
    """, unsafe_allow_html=True)

set_hero_design()

# =====================================================
# LOGIQUE & LAYOUT
# =====================================================
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None

if st.session_state.user_profile is None:
    
    # Création des deux colonnes (Texte vs Image)
    # Ratio 5:5 ou 6:4 pour laisser de la place au texte
    col_text, col_image = st.columns([1.1, 1], gap="large")
    
    # --- COLONNE GAUCHE : TEXTE & LOGIN ---
    with col_text:
        st.markdown("<br>", unsafe_allow_html=True)
        # Titre H1
        st.markdown('<h1 class="hero-title">Débloquez Votre<br>Potentiel Adaptatif</h1>', unsafe_allow_html=True)
        
        # Sous-titre
        st.markdown("""
        <p class="hero-subtitle">
            SamaLearn utilise une IA neuronale avancée pour créer un parcours d'apprentissage 
            unique, évoluant en temps réel selon vos performances. Entrez dans la matrice éducative.
        </p>
        """, unsafe_allow_html=True)
        
        # Zone de Login (Input + Bouton)
        st.markdown("<!-- Input masqué visuellement via CSS -->", unsafe_allow_html=True)
        nom_saisi = st.text_input("Username", placeholder="Entrez votre Identifiant Agent...")
        
        # Espace
        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
        
        # Bouton (Action)
        c_btn, c_void = st.columns([1, 1.5]) # Pour ne pas que le bouton prenne toute la largeur
        with c_btn:
            if st.button("Lancer la Session ➜", use_container_width=True):
                if nom_saisi:
                    # Petit effet de loading "IA"
                    with st.spinner("Analyse du profil neuronal..."):
                        time.sleep(1.2)
                    st.session_state.user_profile = {"nom": nom_saisi, "level": 5, "xp": 1200, "rank": "Junior Data Scientist"}
                    st.rerun()
                else:
                    st.warning("Identification requise.")

        # Les Stats en bas (comme sur l'image 150 / 95 / 24)
        st.markdown("<br><br>", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown('<div class="stat-number">8.5k</div><div class="stat-label">Noeuds IA</div>', unsafe_allow_html=True)
        with s2:
            st.markdown('<div class="stat-number">98%</div><div class="stat-label">Précision</div>', unsafe_allow_html=True)
        with s3:
            st.markdown('<div class="stat-number">24/7</div><div class="stat-label">Adaptatif</div>', unsafe_allow_html=True)

    # --- COLONNE DROITE : IMAGE CERVEAU / RESEAU ---
    with col_image:
        # J'utilise une URL d'image libre de droit qui ressemble TRES fort à votre exemple (Cerveau numérique néon)
        # Si vous avez l'image locale, utilisez st.image("votre_image.png")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Image avec un léger flottement CSS
        st.markdown("""
        <style>
        .floating-img {
            animation: float 6s ease-in-out infinite;
            border-radius: 20px;
            box-shadow: 0 0 50px rgba(89, 86, 233, 0.2);
        }
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-20px); }
            100% { transform: translateY(0px); }
        }
        </style>
        <img src="https://cdn.qwenlm.ai/output/4388553a-b6b9-427b-98a8-b6a756363589/t2i/5ce12bcf-b798-405c-94a7-630d4251f6a4/1763824848.png?key=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyZXNvdXJjZV91c2VyX2lkIjoiNDM4ODU1M2EtYjZiOS00MjdiLTk4YTgtYjZhNzU2MzYzNTg5IiwicmVzb3VyY2VfaWQiOiIxNzYzODI0ODQ4IiwicmVzb3VyY2VfY2hhdF9pZCI6IjFmNWZiZjUxLWYwMWItNDA4OC1iMmQxLWQxNzdmM2IxYTVhZSJ9.pML27bwgTv_j2BEDKTyiQcywi9lvDTaSi8nfStvxa-I&x-oss-process=image/resize,m_mfit,w_450,h_450" 
             width="100%" class="floating-img">
        """, unsafe_allow_html=True)
        
        # Note: L'URL ci-dessus est un placeholder de haute qualité.
        # Vous pouvez la remplacer par : "https://i.imgur.com/votre_image.png"

    st.stop()

# Récupération données
user = st.session_state.user_profile
nom = user["nom"]


# =====================================================
# 1. CONFIGURATION & CONSTANTES
# =====================================================
st.set_page_config(page_title="SamaLearn Ultra", page_icon="💠", layout="wide")

# Couleurs du thème
THEME = {
    "primary": "#00F2FF",    # Cyan Néon
    "secondary": "#7000FF",  # Violet Profond
    "bg": "#050505",         # Noir quasi total
    "text": "#E0E0E0",
    "success": "#00FF94",
    "glass": "rgba(255, 255, 255, 0.05)"
}

# =====================================================
# 2. FONCTIONS UTILITAIRES (DATA SCIENCE)
# =====================================================
# Cette fonction est placée ici pour éviter le NameError
@st.cache_data
def get_student_data():
    file_path = "dataset_synthetic_memoire.csv"
    
    # Si le fichier existe déjà, on le charge
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    
    # Sinon, on le génère (Simulation Data Science pour le Mémoire)
    else:
        np.random.seed(42)
        n_students = 1000
        data = {
            'student_id': range(1, n_students + 1),
            # Temps : mélange de rapides (15s) et lents (60s)
            'avg_time_sec': np.concatenate([np.random.normal(15, 5, 500), np.random.normal(60, 15, 500)]),
            # Scores : corrélés inversement au temps
            'score_avg': np.concatenate([np.random.normal(85, 10, 500), np.random.normal(45, 15, 500)]),
            # Nombre de tentatives
            'attempts': np.random.randint(1, 5, n_students)
        }
        df = pd.DataFrame(data)
        # Nettoyage
        df['score_avg'] = df['score_avg'].clip(0, 100)
        df['avg_time_sec'] = df['avg_time_sec'].clip(5, 120)
        
        # Sauvegarde locale pour réutilisation
        df.to_csv(file_path, index=False)
        return df

# =====================================================
# 3. INITIALISATION SESSION
# =====================================================
if "page" not in st.session_state: st.session_state.page = "accueil"
if "user_profile" not in st.session_state: st.session_state.user_profile = None
if "notifications" not in st.session_state: 
    st.session_state.notifications = [
        {"msg": "Nouveau module: Deep Learning", "time": "10 min"},
        {"msg": "Fatou a battu votre record !", "time": "2h"},
        {"msg": "Note d'examen disponible", "time": "1j"}
    ]
# Messages du forum
if "forum_messages" not in st.session_state:
    st.session_state.forum_messages = [
        {"user": "Amina (Prof)", "msg": "Bonjour à tous ! J'ai ajouté le support PDF du cours 2.", "time": "09:00", "role": "prof"},
        {"user": "Moussa", "msg": "Merci Madame ! C'est disponible dans l'onglet Cours ?", "time": "09:15", "role": "student"},
        {"user": "Jean", "msg": "Quelqu'un a réussi l'exercice Python #3 ?", "time": "10:30", "role": "student"}
    ]

# =====================================================
# 4. CSS "NEXT-GEN" (STYLE)
# =====================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* --- GLOBAL --- */
    html, body, [class*="css"] {{
        font-family: 'Outfit', sans-serif;
        color: {THEME['text']};
    }}

    /* Fond Animé */
    .stApp {{
        background-color: {THEME['bg']};
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(112, 0, 255, 0.2) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(0, 242, 255, 0.15) 0%, transparent 40%);
        animation: pulseBackground 10s ease-in-out infinite alternate;
    }}
    
    @keyframes pulseBackground {{
        0% {{ background-size: 100% 100%; }}
        100% {{ background-size: 110% 110%; }}
    }}

    /* --- GLASSMORPHISM CARDS --- */
    .glass-panel {{
        background: {THEME['glass']};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }}
    .glass-panel:hover {{
        border-color: {THEME['primary']};
        transform: translateY(-4px);
        box-shadow: 0 12px 40px 0 rgba(0, 242, 255, 0.1);
    }}

    /* --- TYPOGRAPHY --- */
    h1, h2, h3 {{ color: white !important; font-weight: 800 !important; letter-spacing: -0.5px; }}
    .highlight {{ color: {THEME['primary']}; }}
    .gradient-text {{
        background: linear-gradient(135deg, {THEME['primary']}, {THEME['secondary']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    /* --- CUSTOM BUTTONS --- */
    div.stButton > button {{
        background: linear-gradient(90deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.1);
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }}
    div.stButton > button:hover {{
        border-color: {THEME['primary']};
        color: {THEME['primary']};
        background: rgba(0, 242, 255, 0.05);
    }}
    div.stButton > button:active {{ transform: scale(0.98); }}

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {{
        background-color: rgba(10, 12, 16, 0.9);
        border-right: 1px solid rgba(255,255,255,0.05);
    }}

    /* --- CHAT BUBBLES --- */
    .chat-bubble {{
        padding: 10px 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        width: fit-content;
        max-width: 80%;
    }}
    .chat-self {{
        background: linear-gradient(90deg, {THEME['primary']}22, {THEME['secondary']}22);
        border: 1px solid {THEME['primary']}44;
        margin-left: auto;
        text-align: right;
    }}
    .chat-other {{
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-right: auto;
    }}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 5. LOGIQUE DE LOGIN
# =====================================================
if st.session_state.user_profile is None:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-panel" style="text-align:center; padding: 50px;">
            <h1 style="font-size: 3.5rem; margin-bottom: 0;" class="gradient-text">SamaLearn</h1>
            <p style="color: #888; letter-spacing: 2px; text-transform: uppercase; font-size: 0.8rem; margin-bottom: 40px;">
                Système d'Apprentissage Neuronal v4.0
            </p>
            <div style="text-align: left; margin-bottom: 20px;">
                <label style="color: #ccc; font-size: 0.9rem; margin-left: 5px;">Identifiant Agent</label>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        nom_saisi = st.text_input("Nom", placeholder="Ex: Ndeye Fatou", label_visibility="collapsed")
        
        col_btn, col_skip = st.columns([1,1])
        if st.button("⚡ Initialiser la Connexion", use_container_width=True):
            if nom_saisi:
                with st.spinner("Authentification biométrique..."):
                    time.sleep(1)
                st.session_state.user_profile = {"nom": nom_saisi, "level": 5, "xp": 1200, "rank": "Data Scientist"}
                st.rerun()

    st.stop()

# Récupération données utilisateur
user = st.session_state.user_profile
nom = user["nom"]

# =====================================================
# 6. SIDEBAR: NAVIGATION
# =====================================================
with st.sidebar:
    # Header
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <div class="gradient-text" style="font-weight:900; font-size: 1.8rem;">SamaLearn</div>
        <div style="font-size: 0.7rem; color: #666;">EDITION MEMOIRE</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Profil Carte
    st.markdown(f"""
    <div class="glass-panel" style="padding: 15px; display: flex; align-items: center; margin-bottom: 20px;">
        <img src="https://api.dicebear.com/7.x/avataaars/svg?seed={nom}" style="width: 45px; height: 45px; border-radius: 50%; border: 2px solid {THEME['primary']}; margin-right: 12px;">
        <div>
            <div style="font-weight: bold; font-size: 0.95rem;">{nom}</div>
            <div style="font-size: 0.7rem; color: {THEME['success']};">● En ligne</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation Items
    nav_items = [
        ("accueil", "🏠 Dashboard"),
        ("cours", "⚡ Mes Cours"),
        ("exercices", "💻 IDE Code"),
        ("examen", "📝 Examen"),
      
        ("enseignant", "👨‍🏫 Espace Enseignant")
    ]
    
    st.caption("NAVIGATION")
    for key, label in nav_items:
        if st.button(label, key=f"nav_{key}", use_container_width=True, type="primary" if st.session_state.page == key else "secondary"):
            st.session_state.page = key
            st.rerun()

    # Notifications
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    with st.expander(f"🔔 Notifications ({len(st.session_state.notifications)})"):
        for notif in st.session_state.notifications:
            st.markdown(f"""
            <div style="margin-bottom: 10px; border-left: 2px solid {THEME['secondary']}; padding-left: 10px;">
                <div style="font-size: 0.8rem; font-weight: bold;">{notif['msg']}</div>
                <div style="font-size: 0.6rem; color: #888;">il y a {notif['time']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Déconnexion
    if st.button("Déconnexion", use_container_width=True):
        st.session_state.user_profile = None
        st.rerun()

# =====================================================
# 7. ROUTAGE DES PAGES (CONTENU)
# =====================================================


# --- CONFIGURATION INITIALE ---
if "page" not in st.session_state:
    st.session_state.page = "accueil"

# --- CSS PREMIUM : INTERFACE "TITANIUM" ---
st.markdown("""
<style>
    /* Import Police Premium 'Inter' */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    :root {
        --primary: #00F0FF;     /* Cyan électrique */
        --secondary: #7B61FF;   /* Violet profond */
        --accent: #F72585;      /* Rose néon */
        --bg-dark: #0B0C15;     /* Bleu nuit très sombre */
        --card-bg: rgba(255, 255, 255, 0.02);
        --border-color: rgba(255, 255, 255, 0.08);
    }

    /* Reset Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: var(--bg-dark);
        color: #E0E6ED;
    }

    /* Background Subtil */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #161b2e 0%, #0B0C15 90%);
    }

    /* Titre Héro */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, var(--primary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1.5px;
        margin-bottom: 5px;
    }

    /* Cartes Glassmorphism */
    div[data-testid="stContainer"] {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stContainer"]:hover {
        border-color: rgba(0, 240, 255, 0.2);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        transform: translateY(-2px);
    }

    /* En-têtes de cartes */
    .card-header {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #94A3B8;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        padding-bottom: 10px;
    }

    /* Badges */
    .badge {
        font-size: 0.7rem;
        padding: 4px 8px;
        border-radius: 12px;
        font-weight: 700;
    }
    .badge-live { background: rgba(0, 240, 255, 0.1); color: var(--primary); box-shadow: 0 0 10px rgba(0,240,255,0.2); }
    .badge-alert { background: rgba(247, 37, 133, 0.1); color: var(--accent); }

    /* Sidebar Clean */
    section[data-testid="stSidebar"] {
        background-color: #08090F;
        border-right: 1px solid #1E293B;
    }
</style>
""", unsafe_allow_html=True)

# --- FONCTION DE STYLE GRAPHIQUE ---
def style_fig(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#94A3B8"),
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', zeroline=False),
        showlegend=False
    )
    return fig

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ CONTROL")
    st.markdown("---")
    mode = st.selectbox("Mode Cognitif", ["Apprentissage", "Focus Profond", "Examen"])
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_kpi1, col_kpi2 = st.columns(2)
    with col_kpi1: st.metric("Session", "1h 42", "+5m")
    with col_kpi2: st.metric("XP", "1,240", "+8%")
    
    st.markdown("---")
    st.caption("SamaLearn OS v10.5")

if st.session_state.page == "accueil":

    # ================= HEADER =================
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown('<div class="hero-title">SamaLearn AI</div>', unsafe_allow_html=True)
        st.markdown("<p style='color:#94A3B8; font-size:1.1rem;'>Bienvenue Fatou. Synchronisation neuronale établie.</p>", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="text-align:right; margin-top:10px;">
            <span class="badge badge-live">● SYSTÈME ONLINE</span>
            <div style="font-size:2.5rem; font-weight:800; color:white; line-height:1.2;">98.5%</div>
            <div style="font-size:0.8rem; color:#64748B;">Indice de Performance</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("") # Espace

    # ================= ZONE 1 : INTELLIGENCE (2 GRAPHS) =================
    col_main_1, col_main_2 = st.columns([2, 1])

    with col_main_1:
        st.markdown('<div class="card-header"><span>Cartographie des Connaissances</span><span class="badge badge-live">LIVE</span></div>', unsafe_allow_html=True)
        
        # --- GRAPH 1: NETWORK NEON ---
        N = 35
        x = np.random.rand(N)
        y = np.random.rand(N)
        colors = np.random.rand(N)
        
        fig_net = go.Figure()
        
        # Connexions fines
        for i in range(N):
            for j in range(i+1, N):
                if np.random.rand() > 0.88:
                    fig_net.add_trace(go.Scatter(
                        x=[x[i], x[j]], y=[y[i], y[j]], mode='lines',
                        line=dict(width=1, color='rgba(255, 255, 255, 0.08)'), hoverinfo='none'
                    ))
        
        # Noeuds lumineux
        fig_net.add_trace(go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(size=colors*25 + 10, color=colors, colorscale='Bluered', 
                        line=dict(width=2, color='white'), opacity=0.9),
            hoverinfo='none'
        ))
        
        fig_net = style_fig(fig_net)
        fig_net.update_layout(height=320, xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig_net, use_container_width=True)

    with col_main_2:
        st.markdown('<div class="card-header"><span>Probabilité Réussite</span></div>', unsafe_allow_html=True)
        
        # --- GRAPH 2: ELEGANT GAUGE ---
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = 87,
            number = {'suffix': "%", 'font': {'size': 40, 'color': 'white', 'family': 'Inter'}},
            gauge = {
                'axis': {'range': [None, 100], 'visible': False},
                'bar': {'color': "#00F0FF", 'thickness': 1},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 0,
                'threshold': {'line': {'color': "#F72585", 'width': 4}, 'thickness': 1, 'value': 95}
            }
        ))
        fig_gauge.update_layout(height=200, margin=dict(t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("""
        <div style="background:rgba(247, 37, 133, 0.08); border-left:3px solid #F72585; padding:15px; border-radius:4px; font-size:0.85rem; color:#E2E8F0;">
            <strong style="color:#F72585">IA Alert:</strong><br>Module "Data Cleaning" nécessite une révision immédiate.
        </div>
        """, unsafe_allow_html=True)

    # ================= ZONE 2 : ANALYSE COGNITIVE (3 GRAPHS) =================
    c_rad, c_fun, c_heat = st.columns(3)

    with c_rad:
        st.markdown('<div class="card-header"><span>Radar Compétences</span></div>', unsafe_allow_html=True)
        # --- GRAPH 3: RADAR ---
        fig_rad = go.Figure(go.Scatterpolar(
            r=[80, 65, 95, 70, 85], theta=['Logique', 'Mémoire', 'Vitesse', 'Analyse', 'Création'],
            fill='toself', fillcolor='rgba(123, 97, 255, 0.2)', line=dict(color='#7B61FF', width=2)
        ))
        fig_rad.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], linecolor='rgba(255,255,255,0.1)', tickfont=dict(size=8, color='#666')),
                       angularaxis=dict(tickfont=dict(size=10, color='#94A3B8')), bgcolor='rgba(0,0,0,0)'),
            paper_bgcolor='rgba(0,0,0,0)', height=220, margin=dict(t=20, b=20, l=30, r=30)
        )
        st.plotly_chart(fig_rad, use_container_width=True)

    with c_fun:
        st.markdown('<div class="card-header"><span>Tunnel Acquisition</span></div>', unsafe_allow_html=True)
        # --- GRAPH 4: FUNNEL ---
        fig_fun = go.Figure(go.Funnel(
            y = ["Vue", "Pratique", "Maîtrise"], x = [100, 65, 30], textinfo = "percent initial",
            marker = dict(color = ["rgba(0, 240, 255, 0.2)", "rgba(123, 97, 255, 0.4)", "#F72585"], line=dict(width=0))
        ))
        fig_fun = style_fig(fig_fun)
        fig_fun.update_layout(height=220, margin=dict(t=10, b=10))
        st.plotly_chart(fig_fun, use_container_width=True)

    with c_heat:
        st.markdown('<div class="card-header"><span>Activité Synaptique</span></div>', unsafe_allow_html=True)
        # --- GRAPH 5: HEATMAP ---
        z = np.random.rand(5, 7)
        fig_heat = go.Figure(data=go.Heatmap(
            z=z, x=['L','M','M','J','V','S','D'], y=['Matin','Midi','Soir','Nuit','Deep'],
            colorscale=[[0, '#0B0C15'], [1, '#00F0FF']], showscale=False
        ))
        fig_heat = style_fig(fig_heat)
        fig_heat.update_layout(height=220, xaxis=dict(visible=True, color='#666'), yaxis=dict(visible=True, color='#666', showgrid=False))
        st.plotly_chart(fig_heat, use_container_width=True)

    # ================= ZONE 3 : DYNAMIQUE (3 GRAPHS) =================
    c_area, c_wave, c_scat = st.columns(3)

    with c_area:
        st.markdown('<div class="card-header"><span>Rétention</span></div>', unsafe_allow_html=True)
        # --- GRAPH 6: AREA ---
        x = np.arange(10)
        y = 100 * np.exp(-0.2 * x)
        fig_area = go.Figure(go.Scatter(
            x=x, y=y, mode='lines', fill='tozeroy',
            line=dict(color='#00F0FF', width=3, shape='spline'), fillcolor='rgba(0, 240, 255, 0.1)'
        ))
        fig_area = style_fig(fig_area)
        fig_area.update_layout(height=180, yaxis=dict(range=[0,110]))
        st.plotly_chart(fig_area, use_container_width=True)

    with c_wave:
        st.markdown('<div class="card-header"><span>Flux Mental</span></div>', unsafe_allow_html=True)
        # --- GRAPH 7: WAVE ---
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + np.random.normal(0, 0.1, 50)
        fig_wave = go.Figure(go.Scatter(
            x=x, y=y, mode='lines', line=dict(color='#F72585', width=3, shape='spline')
        ))
        fig_wave = style_fig(fig_wave)
        fig_wave.update_layout(height=180, yaxis=dict(visible=False))
        st.plotly_chart(fig_wave, use_container_width=True)

    with c_scat:
        st.markdown('<div class="card-header"><span>Clusters</span></div>', unsafe_allow_html=True)
        # --- GRAPH 8: BUBBLE ---
        df = pd.DataFrame({'x': np.random.randn(20), 'y': np.random.randn(20), 's': np.random.rand(20)*30})
        fig_bub = go.Figure(go.Scatter(
            x=df['x'], y=df['y'], mode='markers',
            marker=dict(size=df['s'], color=df['s'], colorscale='Sunsetdark', opacity=0.8)
        ))
        fig_bub = style_fig(fig_bub)
        fig_bub.update_layout(height=180, yaxis=dict(visible=False))
        st.plotly_chart(fig_bub, use_container_width=True)

    # ================= ZONE 4 : STATS (2 GRAPHS) =================
    c_bar, c_pie = st.columns([2, 1])

    with c_bar:
        st.markdown('<div class="card-header"><span>Performance Modules</span></div>', unsafe_allow_html=True)
        # --- GRAPH 9: BARRES (CORRECTION ICI) ---
        fig_bar = go.Figure(go.Bar(
            x=[85, 62, 94, 78], y=['Python', 'SQL', 'Machine Learning', 'Statistics'], orientation='h',
            marker=dict(
                color=['#00F0FF', '#333', '#7B61FF', '#333'],
                cornerradius=5 # CORRECTION: Utilisation de cornerradius au lieu de bar_border_radius
            )
        ))
        fig_bar = style_fig(fig_bar)
        fig_bar.update_layout(height=150, yaxis=dict(showgrid=False, color='white'))
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_pie:
        st.markdown('<div class="card-header"><span>Temps Réparti</span></div>', unsafe_allow_html=True)
        # --- GRAPH 10: DONUT ---
        fig_don = go.Figure(go.Pie(
            labels=['A', 'B', 'C'], values=[45, 30, 25], hole=0.7,
            marker=dict(colors=['#7B61FF', '#00F0FF', '#1A1B2E']), textinfo='none'
        ))
        fig_don.update_layout(height=150, margin=dict(t=0, b=0, l=0, r=0), showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
        fig_don.add_annotation(text="45h", showarrow=False, font=dict(size=20, color="white", family="Inter"))
        st.plotly_chart(fig_don, use_container_width=True)
elif st.session_state.page == "cours":
    st.title("⚡ Mes Cours")
    st.markdown('<div class="glass-panel"><h3>Modules d\'apprentissage</h3><p>Contenu personnalisé selon votre profil.</p></div>', unsafe_allow_html=True)

elif st.session_state.page == "exercices":
    st.title("💻 IDE Code")
    st.markdown('<div class="glass-panel"><h3>Éditeur Python en ligne</h3><p>Résolvez des problèmes algorithmiques.</p></div>', unsafe_allow_html=True)

elif st.session_state.page == "examen":

    # --- 1. INITIALISATION & STATE ---
    if "exam_view" not in st.session_state: st.session_state.exam_view = "catalog"
    if "selected_exam_id" not in st.session_state: st.session_state.selected_exam_id = None
    if "ide_code" not in st.session_state: st.session_state.ide_code = ""
    if "boot_sequence" not in st.session_state: st.session_state.boot_sequence = False
    if "terminal_logs" not in st.session_state: st.session_state.terminal_logs = []

    # --- 2. BANQUE DE DONNÉES (12 EXAMENS MAINTENANT) ---
    EXAM_DB = {
        "DS-101": { "title": "Data Science Foundations", "tag": "Python", "level": "Intermédiaire", "duration": "45 min", "xp": 500, "desc": "Nettoyage Pandas & Viz Matplotlib.", "color": "#3776ab", "code": "import pandas as pd\n# Start coding..." },
        "SQL-300": { "title": "Advanced SQL Reporting", "tag": "SQL", "level": "Difficile", "duration": "60 min", "xp": 800, "desc": "Window Functions & CTEs.", "color": "#00758f", "code": "SELECT * FROM sales..." },
        "ML-202": { "title": "Machine Learning Architect", "tag": "Scikit-Learn", "level": "Expert", "duration": "90 min", "xp": 1200, "desc": "Random Forest optimization.", "color": "#F7931E", "code": "from sklearn.ensemble import RandomForestClassifier..." },
        "DL-404": { "title": "Deep Learning Vision", "tag": "PyTorch", "level": "Grand Maître", "duration": "120 min", "xp": 2500, "desc": "CNN for Medical Imaging.", "color": "#ee4c2c", "code": "import torch.nn as nn..." },
        "DE-500": { "title": "Big Data Spark", "tag": "Spark", "level": "Expert", "duration": "100 min", "xp": 1500, "desc": "Distributed Computing RDD.", "color": "#E25A1C", "code": "spark = SparkSession.builder..." },
        "OPS-101": { "title": "Docker & Kubernetes", "tag": "DevOps", "level": "Intermédiaire", "duration": "50 min", "xp": 600, "desc": "Container orchestration.", "color": "#2496ED", "code": "FROM python:3.9-alpine..." },
        "BI-200": { "title": "Business Intelligence", "tag": "Tableau", "level": "Facile", "duration": "30 min", "xp": 300, "desc": "Dashboarding KPI.", "color": "#E97627", "code": "// Calculation field..." },
        "CL-900": { "title": "AWS Solutions", "tag": "Cloud", "level": "Difficile", "duration": "80 min", "xp": 1000, "desc": "Serverless Architecture.", "color": "#FF9900", "code": "import boto3..." },
        "CY-66": { "title": "Ethical Hacking", "tag": "CyberSec", "level": "Expert", "duration": "150 min", "xp": 3000, "desc": "Penetration Testing.", "color": "#00FF00", "code": "nmap -sV 192.168.1.1..." },
        "WEB-33": { "title": "FullStack React", "tag": "ReactJS", "level": "Intermédiaire", "duration": "60 min", "xp": 700, "desc": "Hooks & Components.", "color": "#61DAFB", "code": "const App = () => {..." },
        # --- NOUVEAUX ---
        "BLOCK-101": { "title": "Smart Contract Dev", "tag": "Solidity", "level": "Expert", "duration": "90 min", "xp": 1800, "desc": "DeFi Protocol & Security.", "color": "#363636", "code": "pragma solidity ^0.8.0;\ncontract Token {..." },
        "CI-CD-99": { "title": "DevSecOps Pipeline", "tag": "Jenkins", "level": "Difficile", "duration": "70 min", "xp": 950, "desc": "Automated Security Gates.", "color": "#D33833", "code": "pipeline {\n agent any\n stages {..." }
    }

    # --- 3. CSS ULTRA-PREMIUM ---
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap');

        /* GLOBAL */
        :root { --bg-dark: #0d1117; --card: #161b22; --border: #30363d; --accent: #58a6ff; }
        
        /* GRID CATALOGUE */
        .exam-card {
            background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); position: relative; overflow: hidden; height: 100%;
        }
        .exam-card:hover { transform: translateY(-8px); border-color: var(--accent); box-shadow: 0 15px 30px rgba(0,0,0,0.6); }
        .exam-card::before {
            content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 4px; background: linear-gradient(90deg, transparent, var(--accent), transparent); opacity: 0; transition: opacity 0.3s;
        }
        .exam-card:hover::before { opacity: 1; }
        
        /* LAB INTERFACE (IDE STYLE) */
        .ide-wrapper { background: #1e1e1e; border-radius: 8px; border: 1px solid #333; overflow: hidden; }
        .ide-toolbar { background: #252526; padding: 8px 15px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; }
        .ide-tab { font-family: 'Inter'; font-size: 0.8rem; color: #ccc; background: #1e1e1e; padding: 5px 15px; border-radius: 4px 4px 0 0; border-top: 2px solid #007acc; }
        
        /* TERMINAL OUTPUT */
        .terminal-log { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #00ff00; background: #000; padding: 15px; border-radius: 6px; height: 200px; overflow-y: auto; border: 1px solid #333; }
        
        /* BOOT SEQUENCE */
        .boot-screen { font-family: 'JetBrains Mono', monospace; color: #00ff00; text-align: center; padding-top: 20vh; }
    </style>
    """, unsafe_allow_html=True)

    # =========================================================================
    # VUE 1 : CATALOGUE (HUB)
    # =========================================================================
    if st.session_state.exam_view == "catalog":
        
        # Header High-Tech
        c1, c2 = st.columns([3, 1])
        with c1:
            st.title("SamaLearn Certification Hub")
            st.caption("Plateforme d'évaluation technique v5.0 • Neural Sync Active")
        with c2:
            st.metric("Rang Mondial", "#42", "+3 places")

        # Filtres
        col_search, col_filter = st.columns([3, 1])
        with col_search: search = st.text_input("🔍 Rechercher une compétence...", placeholder="Blockchain, CI/CD, AI...")
        
        # GRILLE 3 COLONNES
        exams = list(EXAM_DB.items())
        if search: exams = [e for e in exams if search.lower() in e[1]['title'].lower() or search.lower() in e[1]['tag'].lower()]
        
        cols = st.columns(3)
        for i, (eid, d) in enumerate(exams):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="exam-card">
                    <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                        <span style="background:{d['color']}22; color:{d['color']}; padding:4px 8px; border-radius:4px; font-size:0.7rem; font-weight:bold; text-transform:uppercase;">{d['tag']}</span>
                        <span style="color:#8b949e; font-size:0.8rem;">{d['duration']}</span>
                    </div>
                    <h3 style="margin:0 0 10px 0; color:white; font-size:1.1rem;">{d['title']}</h3>
                    <p style="color:#8b949e; font-size:0.85rem; height:40px; overflow:hidden;">{d['desc']}</p>
                    <div style="margin-top:15px; border-top:1px solid #30363d; padding-top:10px; display:flex; justify-content:space-between;">
                        <small style="color:#e3b341;">★ {d['level']}</small>
                        <small style="color:#3fb950; font-weight:bold;">+{d['xp']} XP</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Initialiser {d['tag']}", key=eid, use_container_width=True):
                    st.session_state.selected_exam_id = eid
                    st.session_state.ide_code = d['code']
                    st.session_state.exam_view = "boot" # Transition vers l'écran de chargement
                    st.rerun()
                st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # VUE 2 : SÉQUENCE DE BOOT (CINÉMATIQUE)
    # =========================================================================
    elif st.session_state.exam_view == "boot":
        exam = EXAM_DB[st.session_state.selected_exam_id]
        
        placeholder = st.empty()
        logs = [
            "Initializing Secure Environment...",
            f"Loading {exam['tag']} Kernel v4.2...",
            "Allocating Virtual GPU (12GB VRAM)...",
            "Establishing Neural Uplink...",
            "Importing Libraries...",
            "System Ready."
        ]
        
        # Animation des logs
        current_log = ""
        for log in logs:
            current_log += f"> {log}\n"
            placeholder.markdown(f"""
            <div class="boot-screen">
                <h1>SYSTEM BOOT SEQUENCE</h1>
                <div style="width:50%; margin:auto; text-align:left; background:#111; padding:20px; border:1px solid #00ff00; border-radius:8px;">
                    <pre>{current_log}</pre>
                    <div style="animation: blink 1s infinite;">_</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.6) # Vitesse de l'animation
        
        time.sleep(1)
        st.session_state.exam_view = "lab"
        st.rerun()

    # =========================================================================
    # VUE 3 : LE LAB ULTIME (INTERFACE PRO)
    # =========================================================================
    elif st.session_state.exam_view == "lab":
        exam = EXAM_DB[st.session_state.selected_exam_id]
        
        # --- TOP BAR (HUD) ---
        c1, c2, c3 = st.columns([0.5, 4, 1.5])
        with c1:
            if st.button("✖ Exit"): st.session_state.exam_view = "catalog"; st.rerun()
        with c2:
            st.markdown(f"**{exam['title']}** • Environnement Isolé")
        with c3:
            st.markdown("<span style='color:#f85149; font-family:monospace;'>● REC 00:42:15</span> | <span style='color:#58a6ff;'>RAM: 42%</span>", unsafe_allow_html=True)

        # --- MAIN WORKSPACE (SPLIT) ---
        col_left, col_right = st.columns([1, 2])

        # GAUCHE : INSTRUCTIONS & CHAT
        with col_left:
            tab_inst, tab_chat = st.tabs(["📄 Sujet", "💬 IA Mentor"])
            with tab_inst:
                st.markdown(f"""
                <div style="background:#161b22; padding:15px; border-radius:8px; border:1px solid #30363d; height:600px; overflow-y:auto;">
                    <h3 style="color:{exam['color']};">Mission Principale</h3>
                    <p>Vous devez compléter le script pour répondre aux exigences techniques suivantes.</p>
                    <hr style="border-color:#30363d;">
                    <h4>Tâches :</h4>
                    <ul>
                        <li>Initialiser les variables d'environnement.</li>
                        <li>Implémenter la logique métier principale.</li>
                        <li>Gérer les exceptions (Try/Catch).</li>
                    </ul>
                    <br>
                    <div style="background:#0d1117; padding:10px; border-left:3px solid {exam['color']};">
                        <small>HINT</small><br>Utilisez la documentation officielle pour la syntaxe spécifique.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with tab_chat:
                st.info("L'IA Mentor est en ligne. Posez vos questions sur la syntaxe.")
                st.text_input("Message...", placeholder="Comment faire une boucle for ?")
                st.markdown("> *IA:* Je suis prête à analyser votre code.")

        # DROITE : L'ÉDITEUR DE CODE "SUPERBE"
        with col_right:
            # Simulation Barre d'onglets IDE
            st.markdown("""
            <div class="ide-wrapper">
                <div class="ide-toolbar">
                    <div class="ide-tab">main.py</div>
                    <div style="font-size:0.8rem; color:#aaa;">Python 3.10</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Zone de Code
            code = st.text_area("Code Editor", value=st.session_state.ide_code, height=350, label_visibility="collapsed")
            st.session_state.ide_code = code
            
            # Boutons d'action
            c_run, c_submit = st.columns([1, 1])
            with c_run:
                run_btn = st.button("▶ RUN CODE (Compile)", use_container_width=True)
            with c_submit:
                submit_btn = st.button("✅ SUBMIT (Grade)", type="primary", use_container_width=True)

            # --- SIMULATEUR DE TERMINAL TEMPS RÉEL ---
            st.markdown("**Terminal Output**")
            
            terminal_placeholder = st.empty()
            
            if run_btn:
                # Simulation de streaming de logs (C'est ça qui fait "PRO")
                logs = [
                    f"> Connecting to {exam['tag']} Daemon...",
                    "> Compiling main.py...",
                    "> Checking syntax errors... 0 found.",
                    "> Running Unit Tests...",
                    "> [TEST 1] Initialization... PASSED",
                    "> [TEST 2] Data Processing... PASSED",
                    "> [TEST 3] Edge Cases... WARNING (Timeout)",
                    "> Execution finished in 1.42s"
                ]
                live_log = ""
                for line in logs:
                    live_log += line + "\n"
                    terminal_placeholder.markdown(f"""<div class="terminal-log">{live_log}</div>""", unsafe_allow_html=True)
                    time.sleep(0.3) # Délai pour l'effet "Temps réel"
            else:
                terminal_placeholder.markdown("""<div class="terminal-log">> Ready for execution...</div>""", unsafe_allow_html=True)

            if submit_btn:
                st.balloons()
                st.success(f"Certification {exam['title']} VALIDÉE ! Le certificat a été envoyé à votre wallet.")
                time.sleep(3)
                st.session_state.exam_view = "catalog"
                st.rerun()
# -----------------------------------------------------

# =====================================================
# 3. PAGE DE CONNEXION : "THE QUANTUM CORE EDITION"
# =====================================================


# Configuration de la page pour enlever les marges par défaut
st.set_page_config(page_title="SamaLearn Secure", layout="centered")

# Vérification de l'état de connexion
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None

if st.session_state.user_profile is None:
    
    # ---------------------------------------------------------
    # 1. DESIGN SYSTÈME ULTRA-IMMERSIF (CSS AVANCÉ)
    # ---------------------------------------------------------
    st.markdown("""
    <style>
        /* Import police Futuriste 'Rajdhani' */
        @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;600;700&display=swap');

        /* --- 1. FOND D'ÉCRAN ANIMÉ (CYBER GRID) --- */
        [data-testid="stAppViewContainer"] {
            background-color: #000000;
            font-family: 'Rajdhani', sans-serif;
            overflow: hidden;
        }

        /* Grille Cybernétique en mouvement */
        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: absolute;
            width: 200%;
            height: 200%;
            top: -50%;
            left: -50%;
            z-index: -1;
            background: 
                linear-gradient(transparent 0%, rgba(0, 242, 255, 0.05) 2%, transparent 3%),
                linear-gradient(90deg, transparent 0%, rgba(112, 0, 255, 0.04) 2%, transparent 3%);
            background-size: 50px 50px;
            transform: perspective(500px) rotateX(60deg);
            animation: gridMove 20s linear infinite;
        }
        
        /* Vignette sombre pour focus central */
        [data-testid="stAppViewContainer"]::after {
            content: "";
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            background: radial-gradient(circle at center, transparent 0%, #000000 90%);
            pointer-events: none;
            z-index: -1;
        }

        @keyframes gridMove {
            0% { transform: perspective(500px) rotateX(60deg) translateY(0); }
            100% { transform: perspective(500px) rotateX(60deg) translateY(50px); }
        }

        /* --- 2. CARTE HOLOGRAPHIQUE CENTRALE --- */
        div[data-testid="stVerticalBlock"] > div:has(div.login-container) {
            background: rgba(10, 10, 15, 0.6);
            border: 1px solid rgba(0, 242, 255, 0.1);
            border-radius: 16px;
            padding: 50px;
            backdrop-filter: blur(15px);
            box-shadow: 
                0 0 40px rgba(0, 0, 0, 0.8),
                inset 0 0 20px rgba(0, 242, 255, 0.05);
            position: relative;
            overflow: hidden;
        }

        /* Bordure lumineuse animée (Snake effect) */
        div[data-testid="stVerticalBlock"] > div:has(div.login-container)::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, transparent, #00f2ff, transparent);
            animation: borderFlow 3s ease-in-out infinite;
        }

        @keyframes borderFlow {
            0% { transform: translateX(-100%); }
            50% { transform: translateX(100%); }
            100% { transform: translateX(100%); }
        }

        /* --- 3. TYPOGRAPHIE & GLITCH --- */
        .brand-title {
            font-size: 3.5rem;
            font-weight: 700;
            text-align: center;
            color: #ffffff;
            text-transform: uppercase;
            letter-spacing: 4px;
            text-shadow: 0 0 10px rgba(0, 242, 255, 0.7);
            position: relative;
        }
        
        .brand-subtitle {
            color: #00f2ff;
            text-align: center;
            font-size: 0.9rem;
            letter-spacing: 6px;
            text-transform: uppercase;
            margin-bottom: 40px;
            opacity: 0.8;
        }

        /* --- 4. INPUT FIELDS (TERMINAL STYLE) --- */
        div[data-testid="stTextInput"] label { display: none; }
        div[data-testid="stTextInput"] input {
            background-color: rgba(0, 0, 0, 0.5) !important;
            border: 1px solid #333 !important;
            border-left: 3px solid #333 !important;
            color: #00f2ff !important;
            font-family: 'Courier New', monospace !important;
            padding: 15px !important;
            font-size: 1.1rem !important;
            transition: all 0.3s ease;
        }
        
        div[data-testid="stTextInput"] input:focus {
            border-color: #00f2ff !important;
            border-left: 3px solid #00f2ff !important;
            box-shadow: 0 0 20px rgba(0, 242, 255, 0.2) !important;
            background-color: rgba(0, 242, 255, 0.05) !important;
        }

        /* --- 5. BOUTON (REACTOR CORE) --- */
        div.stButton > button {
            width: 100%;
            background: transparent;
            border: 1px solid #00f2ff;
            color: #00f2ff;
            padding: 18px;
            font-family: 'Rajdhani', sans-serif;
            font-weight: 700;
            font-size: 1.2rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            position: relative;
            overflow: hidden;
            transition: all 0.3s;
            box-shadow: 0 0 10px rgba(0, 242, 255, 0.1);
        }

        div.stButton > button:hover {
            background: #00f2ff;
            color: #000;
            box-shadow: 0 0 30px rgba(0, 242, 255, 0.6);
            transform: scale(1.02);
        }
        
        /* Suppression des éléments UI par défaut de Streamlit */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 2rem;}

        /* Animation Pulse pour le logo */
        .logo-pulse {
            animation: pulse-glow 3s infinite;
        }
        @keyframes pulse-glow {
            0% { filter: drop-shadow(0 0 5px rgba(0, 242, 255, 0.4)); }
            50% { filter: drop-shadow(0 0 20px rgba(0, 242, 255, 0.8)); }
            100% { filter: drop-shadow(0 0 5px rgba(0, 242, 255, 0.4)); }
        }

    </style>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 2. STRUCTURE VISUELLE
    # ---------------------------------------------------------
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Marqueur pour le CSS container
        st.markdown('<div class="login-container"></div>', unsafe_allow_html=True)

        # LOGO SVG REVISITÉ (Version Cyber)
        st.markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <svg class="logo-pulse" width="120" height="120" viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <!-- Core Glow -->
                    <circle cx="100" cy="100" r="30" fill="#00f2ff" fill-opacity="0.2"/>
                    
                    <!-- Circuit Path S -->
                    <path d="M140 50 C 190 50, 190 100, 140 100 L 60 100 C 10 100, 10 150, 60 150" 
                          stroke="#ffffff" stroke-width="8" stroke-linecap="square"/>
                    
                    <!-- Tech Rings -->
                    <circle cx="100" cy="100" r="80" stroke="#00f2ff" stroke-width="2" stroke-dasharray="10 20" opacity="0.5">
                        <animateTransform attributeName="transform" type="rotate" from="0 100 100" to="360 100 100" dur="20s" repeatCount="indefinite"/>
                    </circle>
                    <circle cx="100" cy="100" r="65" stroke="#7000ff" stroke-width="1" stroke-dasharray="40 40" opacity="0.7">
                        <animateTransform attributeName="transform" type="rotate" from="360 100 100" to="0 100 100" dur="15s" repeatCount="indefinite"/>
                    </circle>
                    
                    <!-- Data Nodes -->
                    <rect x="135" y="45" width="10" height="10" fill="#00f2ff"/>
                    <rect x="55" y="145" width="10" height="10" fill="#7000ff"/>
                </svg>
                <h1 class="brand-title">SAMA<span style="color:#00f2ff">LEARN</span></h1>
                <div class="brand-subtitle">Neural Interface v4.0</div>
            </div>
        """, unsafe_allow_html=True)

        # FORMULAIRE DE SAISIE
        st.markdown("<div style='margin-bottom: 8px; color: #00f2ff; font-size: 0.8rem; font-family: monospace;'>COMMAND: IDENTIFY_USER</div>", unsafe_allow_html=True)
        
        username = st.text_input("login_dummy", placeholder="> Enter Agent ID...", label_visibility="collapsed")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("INITIALIZE SYSTEM"):
            if username:
                # Simulation d'un chargement "Hacker"
                status_placeholder = st.empty()
                bar = st.progress(0)
                
                # Effet de chargement rapide style terminal
                logs = [
                    "Connecting to Neural Net...", 
                    "Bypassing Firewall...", 
                    "Handshake Verified...", 
                    "Loading User Profile...",
                    "ACCESS GRANTED"
                ]
                
                for i in range(100):
                    time.sleep(0.015) # Plus rapide
                    bar.progress(i + 1)
                    if i % 20 == 0:
                        current_log = logs[int(i/20)]
                        status_placeholder.markdown(f"<div style='color:#00f2ff; font-family:monospace; font-size:0.8rem;'>System: {current_log}</div>", unsafe_allow_html=True)
                
                time.sleep(0.5)
                bar.empty()
                status_placeholder.empty()
                
                st.session_state.user_profile = {"nom": username}
                st.rerun()
            else:
                st.markdown(
                    """<div style='background: rgba(255, 0, 50, 0.2); border: 1px solid #ff0033; color: #ff0033; padding: 10px; border-radius: 5px; text-align: center; font-family: monospace;'>
                    🚫 ERROR: NULL_IDENTITY_DETECTED
                    </div>""", 
                    unsafe_allow_html=True
                )

        # FOOTER TECH
        st.markdown("""
            <div style="text-align: center; margin-top: 30px; color: rgba(255,255,255,0.2); font-size: 0.7rem;">
                ENCRYPTED CONNECTION // SAMA-LABS QUANTUM SERVER<br>
                ID: 8473-XKD-99
            </div>
        """, unsafe_allow_html=True)

    st.stop()
# =====================================================
# 6. PAGE: EXERCICES (IDE SIMULÉ)
# =====================================================
elif st.session_state.page == "exercices":
    st.title("💻 Laboratoire de Code")
    
    col_ide, col_instr = st.columns([2, 1])
    
    with col_instr:
        st.markdown(f"""
        <div class="glass-panel">
            <h3>📜 Instructions</h3>
            <p>Écrivez une fonction Python <code>calcul_moyenne(liste)</code> qui retourne la moyenne des nombres.</p>
            <hr style="border-color: #333;">
            <p style="font-size: 0.8rem; color: #aaa;">Entrée: [10, 20, 30]<br>Sortie Attendue: 20.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("💡 Indice (Cliquer)"):
            st.code("Utilisez sum() et len()", language="python")

    with col_ide:
        # Simulation d'un éditeur de code
        code = st.text_area("main.py", height=300, value="def calcul_moyenne(liste):\n    # Votre code ici\n    pass", help="Écrivez votre code Python ici")
        
        c_run, c_res = st.columns([1, 4])
        with c_run:
            if st.button("▶ Exécuter"):
                with st.spinner("Compilation..."):
                    time.sleep(0.8)
                st.success("Test réussi ! (Simulation)")
                st.balloons()
        
        st.markdown("""
        <div style="background: #1e1e1e; color: #00ff00; font-family: monospace; padding: 15px; border-radius: 8px; border-left: 3px solid #00ff00; margin-top: 10px;">
            > Console Ready...<br>
            > Waiting for execution...
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# 7. PAGE: COMMUNAUTÉ (LEADERBOARD)
# =====================================================
elif st.session_state.page == "community":
    st.title("🏆 Classement Général")
    
    # Podium
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown(f"""
        <div class="glass-panel" style="text-align: center; border-color: gold; transform: scale(1.05);">
            <div style="font-size: 3rem;">👑</div>
            <h2 style="color: gold !important;">1. Fatou</h2>
            <p>2450 XP</p>
        </div>
        """, unsafe_allow_html=True)
    with col1:
        st.markdown(f"""
        <div class="glass-panel" style="text-align: center; margin-top: 30px;">
            <div style="font-size: 2rem;">🥈</div>
            <h3>2. Jean</h3>
            <p>2100 XP</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="glass-panel" style="text-align: center; margin-top: 30px;">
            <div style="font-size: 2rem;">🥉</div>
            <h3>3. Awa</h3>
            <p>1950 XP</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Liste
    fake_users = [
        {"rank": 4, "name": "Moussa (Vous)", "xp": user['xp'], "change": "⬆"},
        {"rank": 5, "name": "Sophie", "xp": 1100, "change": "⬇"},
        {"rank": 6, "name": "Ibrahim", "xp": 950, "change": "="},
    ]
    
    st.markdown("### Votre position")
    for u in fake_users:
        bg = "rgba(0, 242, 255, 0.1)" if "Vous" in u['name'] else "rgba(255,255,255,0.02)"
        border = f"1px solid {THEME['primary']}" if "Vous" in u['name'] else "none"
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; background: {bg}; padding: 15px; border-radius: 10px; margin-bottom: 10px; border: {border};">
            <div style="width: 40px; font-weight: bold; color: #888;">#{u['rank']}</div>
            <div style="flex-grow: 1; font-weight: 600;">{u['name']}</div>
            <div style="margin-right: 20px; font-weight: bold;">{u['xp']} XP</div>
            <div style="color: {THEME['success'] if u['change']=='⬆' else 'white'};">{u['change']}</div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# 8. PAGE: COURS (Placeholder élégant)
# =====================================================
elif st.session_state.page == "cours":
    st.title("⚡ Bibliothèque de Cours")
    
    # Barre de recherche
    st.text_input("🔍 Rechercher un sujet (ex: Python, Data, IA)...")
    
    # Grille de cours
    c1, c2, c3 = st.columns(3)
    
    cours_data = [
        {"title": "Python Masterclass", "level": "Débutant", "img": "🐍"},
        {"title": "Data Science Pro", "level": "Intermédiaire", "img": "📊"},
        {"title": "Deep Learning A-Z", "level": "Avancé", "img": "🧠"}
    ]
    
    for col, course in zip([c1, c2, c3], cours_data):
        with col:
            st.markdown(f"""
            <div class="glass-panel" style="padding: 0; overflow: hidden; height: 250px; display: flex; flex-direction: column;">
                <div style="height: 100px; background: linear-gradient(135deg, #1a1a2e, #16213e); display: flex; align-items: center; justify-content: center; font-size: 3rem;">
                    {course['img']}
                </div>
                <div style="padding: 15px; flex-grow: 1;">
                    <div style="font-size: 0.7rem; color: {THEME['primary']}; text-transform: uppercase; font-weight: bold;">{course['level']}</div>
                    <h3 style="margin: 5px 0; font-size: 1.1rem;">{course['title']}</h3>
                    <button style="margin-top: 10px; width: 100%; background: transparent; border: 1px solid rgba(255,255,255,0.2); color: white; border-radius: 5px; padding: 5px; cursor: pointer;">Voir détails</button>
                </div>
            </div>
            """, unsafe_allow_html=True)
# =====================================================
# INITIALISATION SESSION STATE
# =====================================================
if "page" not in st.session_state:
    st.session_state.page = "accueil"

if "eleve_competences" not in st.session_state:
    st.session_state.eleve_competences = {
        "IA": 0.40, "ML": 0.20, "Proba": 0.60, "Stats": 0.30,
        "Algèbre": 0.05, "Python": 0.10, "Visualisation": 0.12, "Éthique IA": 0.15
    }

if "historique" not in st.session_state:
    st.session_state.historique = []

if "open_modal" not in st.session_state:
    st.session_state.open_modal = None

# =====================================================
# EXAMEN DATA
# =====================================================
EXAM_QUESTIONS = [
    ("Fréquence absolue du nombre 5 dans [1,2,5,5,3,5] :", ["1", "2", "3", "4"], "3", "Fréquence"),
    ("Fréquence relative du '3' dans [1,3,3,3,2] :", ["0.2", "0.4", "0.6"], "0.6", "Fréquence"),
    ("Moyenne de [2,4,6,8] :", ["4", "5", "6"], "5", "Moyenne"),
    ("Moyenne pondérée valeurs=[2,3], poids=[3,2] :", ["2.4", "2.5", "2.6"], "2.4", "Moyenne"),
    ("Variance de [2,4,4,4,5,5,7,9] (approx) :", ["2.0", "2.5", "3.0"], "2.0", "Variance"),
    ("Écart-type (approx.) de [2,4,6,8] :", ["2", "2.24", "2.5"], "2.24", "Écart-type"),
    ("IQR de [1,3,5,7,9] :", ["4", "6", "8"], "4", "Statistique"),
    ("Distribution normale = ?", ["Symétrique en cloche", "Uniforme", "Discontinue"], "Symétrique en cloche", "Statistique"),
    ("La moyenne est sensible à :", ["Les valeurs extrêmes", "Le nombre d'observations"], "Les valeurs extrêmes", "Moyenne"),
    ("Analyse descriptive inclut :", ["Moyenne, variance, écart-type", "Clustering avancé"], "Moyenne, variance, écart-type", "Statistique"),
    ("Probabilité d'obtenir un roi dans un jeu de 52 cartes :", ["1/52", "1/13", "1/4"], "1/13", "Probabilité"),
    ("Probabilité somme 2 dés = 7 :", ["1/6", "1/8", "1/12"], "1/6", "Probabilité"),
    ("Probabilité que X>3 si X~U(1,5) :", ["0.25", "0.5", "0.75"], "0.5", "Probabilité"),
    ("Probabilité P(A∩B)= ? Si A et B indépendants et P(A)=0.4, P(B)=0.5", ["0.2", "0.25", "0.4"], "0.2", "Probabilité"),
    ("Probabilité P(A∪B)= ? Si P(A)=0.3, P(B)=0.4, P(A∩B)=0.1", ["0.6", "0.7", "0.8"], "0.6", "Probabilité"),
    ("L'IA en éducation permet surtout :", ["Remplacer le professeur", "Adapter le contenu au profil", "Créer des vidéos"], "Adapter le contenu au profil", "IA"),
    ("Un système adaptatif ajuste :", ["Vitesse et contenu selon le profil", "Taille des polices"], "Vitesse et contenu selon le profil", "IA"),
    ("Machine learning supervisé nécessite :", ["Données étiquetées", "Pas de données", "Un humain par étape"], "Données étiquetées", "IA"),
    ("Un cluster d'apprenants signifie :", ["Un groupe d'élèves similaires", "Un seul étudiant"], "Un groupe d'élèves similaires", "IA"),
    ("Recommandation pédagogique typique :", ["Exercices ciblés", "Copies papier", "Notes finales"], "Exercices ciblés", "IA")
]
TOTAL_EXAM_Q = len(EXAM_QUESTIONS)

EXPLANATIONS = {i: {"title": q[3], "text": "Explication détaillée ici."} for i, q in enumerate(EXAM_QUESTIONS)}

# =====================================================
# EXERCICES DATA
# =====================================================
EXERCICES_QUESTIONS = [
    {"q": "Quelle est la valeur de la moyenne d'un ensemble de données ?", "choices": ["Somme / effectif", "Valeur la plus fréquente", "Valeur centrale", "Ecart maximum"], "answer": "Somme / effectif", "concept": "Moyenne", "difficulty": 1},
    {"q": "La médiane est :", "choices": ["La valeur centrale", "La moyenne", "La variance", "Le min"], "answer": "La valeur centrale", "concept": "Médiane", "difficulty": 1},
    {"q": "L'écart-type mesure :", "choices": ["La dispersion", "La moyenne", "La tendance", "Le maximum"], "answer": "La dispersion", "concept": "Ecart-Type", "difficulty": 2},
    {"q": "Un outlier est :", "choices": ["Une donnée extrême", "La moyenne", "L’écart-type", "La médiane"], "answer": "Une donnée extrême", "concept": "Outliers", "difficulty": 3}
]
TOTAL_EXERCICES_Q = len(EXERCICES_QUESTIONS)

# =====================================================
# MODULES COURS
# =====================================================
MODULES = {
    "IA": {"icon": "🧠","badge":"Compréhension","lang":"Français / Wolof","desc":"Fondements, modèles symboliques, IA éthique, cas d'usage africains.","image":"https://images.unsplash.com/photo-1677442136019-21780ecad995?w=600","video":"https://www.youtube.com/embed/2ePf9rue1Ao","color":"#00f6ff"},
    "ML": {"icon": "🤖","badge":"Modélisation","lang":"Français","desc":"Régression, classification, pipelines ML.","image":"https://images.unsplash.com/photo-1591453089816-0fbb971b454c?w=600","video":"https://www.youtube.com/embed/8GBzamEdMOI","color":"#7bff6b"},
    "Proba": {"icon": "🎲","badge":"Analyse","lang":"Français / Wolof","desc":"Loi binomiale, loi normale, variables aléatoires.","image":"https://images.unsplash.com/photo-1754304342448-3eef0ab5ba9e?w=600","video":"https://www.youtube.com/embed/uzkc-qNVoOk","color":"#ffd84d"},
    "Stats": {"icon": "📊","badge":"Inférence","lang":"Français","desc":"Inférence, tests hypothèses, estimation.","image":"https://images.unsplash.com/photo-1584291527908-033f4d6542c8?w=600","video":"https://www.youtube.com/embed/132hCHFnAWA","color":"#b682ff"},
    "Algèbre": {"icon": "🧮","badge":"Calcul","lang":"Français","desc":"Vecteurs, matrices, systèmes linéaires.","image":"https://images.unsplash.com/photo-1754304342312-cace9df82c6e?w=600","video":"https://www.youtube.com/embed/fN332HsHJf0","color":"#ff6fb6"},
    "Python": {"icon": "🐍","badge":"Programmation","lang":"Français","desc":"Pandas, NumPy, fonctions, boucles.","image":"https://images.unsplash.com/photo-1649180556628-9ba704115795?w=600","video":"https://www.youtube.com/embed/oUJolR5bX6g","color":"#2ee3d7"},
    "Visualisation": {"icon": "📈","badge":"Visualisation","lang":"Français","desc":"Matplotlib, Seaborn, graphiques interactifs avec Plotly.","image":"https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=600","video":"https://www.youtube.com/embed/En_TX-mae8g","color":"#ffb86b"},
    "Éthique IA": {"icon": "⚖️","badge":"Responsabilité","lang":"Français","desc":"Biais algorithmiques, vie privée, IA inclusive.","image":"https://media.istockphoto.com/id/2192074737/fr/photo/%C3%A9thique-de-lia-texte-3d-lueur-bleu.webp","video":"https://www.youtube.com/embed/Ah1U1sDfQ2w","color":"#9fff6a"}
}

# =====================================================
# FONCTIONS UTILITAIRES
# =====================================================
def reset_exam():
    order = list(range(TOTAL_EXAM_Q))
    random.shuffle(order)
    st.session_state.questions_order = order
    st.session_state.cur_q_exam = 0
    st.session_state.responses_exam = []
    st.session_state.finished_exam = False

def reset_exercices():
    st.session_state.cur_q_exercices = 0
    st.session_state.responses_exercices = []
    st.session_state.finished_exercices = False
    st.session_state.start_time = 0

def compute_mastery_from_responses(responses, source="exam"):
    if not responses:
        if source == "exam":
            concepts = sorted({q[3] for q in EXAM_QUESTIONS})
        else:
            concepts = sorted({q["concept"] for q in EXERCICES_QUESTIONS})
        return {c: 0.0 for c in concepts}
    df = pd.DataFrame(responses)
    mastery = (df.groupby("concept")["correct"].mean() * 5).to_dict()
    return {k: round(v,2) for k,v in mastery.items()}

def recommander_module(comps, hist):
    scores = {}
    for m,lvl in comps.items():
        recent_err = sum(1 for mm,r in hist[-10:] if mm==m and r=="erreur")
        recent_succ = sum(1 for mm,r in hist[-10:] if mm==m and r=="succès")
        scores[m] = (1-lvl)**1.2 + recent_err*0.35 - recent_succ*0.25 + random.uniform(0,0.03)
    return max(scores,key=scores.get)

# =====================================================
# PAGE : ACCUEIL
# =====================================================
if st.session_state.page == "accueil":
    st.markdown("""
    
    """, unsafe_allow_html=True)
    if st.button("🚀 Commencer l'apprentissage"):
        st.session_state.page = "cours"
        st.rerun()
elif st.session_state.page == "cours":
   
    # ==================================================
    # 1. CONFIGURATION & DONNÉES INTELLIGENTES
    # ==================================================
    MODULES = {
        # NIVEAU 1 (BASES)
        "Python for Data": { "cat": "Code", "icon": "🐍", "color": "#00f2ff", "lvl": 1, "desc": "Maîtrisez Pandas & NumPy.", "video": "rfscVS0vtbw" },
        "Maths for AI":    { "cat": "Math", "icon": "🧮", "color": "#ff0055", "lvl": 1, "desc": "Algèbre linéaire & Matrices.", "video": "fNk_zzaMoSs" },
        "Statistiques":    { "cat": "Math", "icon": "📊", "color": "#ff0055", "lvl": 1, "desc": "Tests A/B & Inférence.", "video": "qBigTkBLU6g" },
        "SQL & Big Data":  { "cat": "Code", "icon": "🗄️", "color": "#00f2ff", "lvl": 1, "desc": "Requêtes avancées & NoSQL.", "video": "HXV3zeQKqGY" },
        
        # NIVEAU 2 (VERROUILLÉ SI NIV 1 < 50%)
        "Machine Learning":{ "cat": "IA", "icon": "🤖", "color": "#00ff99", "lvl": 2, "desc": "Regressions, SVM, XGBoost.", "video": "7eh4d6sabA0" },
        "Deep Learning":   { "cat": "IA", "icon": "🧠", "color": "#0088ff", "lvl": 2, "desc": "Réseaux de neurones (ANN).", "video": "aircAruvnKk" },
        "NLP & LLMs":      { "cat": "IA", "icon": "🗣️", "color": "#ff00ff", "lvl": 2, "desc": "Transformers, GPT, BERT.", "video": "CMrHM8a3hqw" },
        "Computer Vision": { "cat": "IA", "icon": "👁️", "color": "#0088ff", "lvl": 2, "desc": "CNN, YOLO, Segmentation.", "video": "OcycT1Jwsns" },
        
        # NIVEAU 3 (VERROUILLÉ SI NIV 2 < 50%)
        "Generative AI":   { "cat": "Pro", "icon": "✨", "color": "#ff00ff", "lvl": 3, "desc": "Stable Diffusion, RAG.", "video": "9zKuYvjFFS8" },
        "Time Series":     { "cat": "Pro", "icon": "⏳", "color": "#ffaa00", "lvl": 3, "desc": "ARIMA, Prophet, LSTM.", "video": "ZoJ2kLoe67I" },
        "Reinf. Learning": { "cat": "Pro", "icon": "🎮", "color": "#ffaa00", "lvl": 3, "desc": "Q-Learning, Agents autonomes.", "video": "JgvyzIkgxF0" },
        "Graph Neural Net":{ "cat": "Pro", "icon": "🕸️", "color": "#00ff99", "lvl": 3, "desc": "Analyse de graphes complexes.", "video": "ABCde12345" },

        # NIVEAU 4 (VERROUILLÉ SI NIV 3 < 50%)
        "MLOps & CI/CD":   { "cat": "Ops", "icon": "⚙️", "color": "#bf00ff", "lvl": 4, "desc": "Docker, Kubernetes, MLflow.", "video": "0qQCKAlfsZs" },
        "Cloud AI (AWS)":  { "cat": "Ops", "icon": "☁️", "color": "#bf00ff", "lvl": 4, "desc": "SageMaker, Cloud Computing.", "video": "ABCde12345" },
        "Data Storytelling":{ "cat": "Code", "icon": "📈", "color": "#00f2ff", "lvl": 4, "desc": "Dashboards pro avec Plotly.", "video": "a9UrKTVEeZA" },
        "AI Ethics & Law": { "cat": "Law", "icon": "⚖️", "color": "#ff0055", "lvl": 4, "desc": "RGPD, Biais, AI Act.", "video": "CfzO6iF3Y5o" }
    }

    # Ressources RAG (Simulées pour la remédiation)
    RAG_RESOURCES = {
        "Math": ["Article: Comprendre les P-Values intuitivement", "Vidéo: L'algèbre linéaire en 10min"],
        "Code": ["Cheatsheet Pandas PDF", "Exercice: 10 katas Python pour débutants"],
        "IA": ["Visualisation: Comment fonctionne un Neurone", "Article: Overfitting vs Underfitting"]
    }

    # --- INITIALISATION DE L'ÉTAT (STATE) ---
    if "eleve_competences" not in st.session_state: 
        st.session_state.eleve_competences = {k: random.uniform(0, 0.2) for k in MODULES}
    
    # Initialisation sécurisée pour éviter les erreurs de clé
    for mod in MODULES:
        if mod not in st.session_state.eleve_competences:
            st.session_state.eleve_competences[mod] = 0.0

    if "last_review" not in st.session_state: 
        st.session_state.last_review = {k: datetime.now() - timedelta(days=random.randint(1, 10)) for k in MODULES}
    
    # Gamification States
    if "focus_active" not in st.session_state: st.session_state.focus_active = False
    if "streak" not in st.session_state: st.session_state.streak = 5
    if "focus_timer" not in st.session_state: st.session_state.focus_timer = 25 * 60 # 25 min

    # ==================================================
    # 2. MODE FOCUS (POMODORO) - INTERFACE ALTERNATIVE
    # ==================================================
    if st.session_state.focus_active:
        # CSS spécifique pour cacher le sidebar et centrer le focus
        st.markdown("""
        <style>
            [data-testid="stSidebar"] {display: none;} 
            .focus-container {
                text-align: center; padding: 50px; background: #0e1117; 
                border: 2px solid #00f2ff; border-radius: 20px; 
                box-shadow: 0 0 50px rgba(0, 242, 255, 0.2);
                margin-top: 50px;
            }
            .timer { font-size: 6rem; font-weight: bold; color: white; font-family: monospace; text-shadow: 0 0 20px #00f2ff; margin: 20px 0; }
        </style>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown('<div class="focus-container">', unsafe_allow_html=True)
            st.markdown("<h1>🍅 MODE FOCUS ACTIVÉ</h1>", unsafe_allow_html=True)
            st.caption("Pas de distractions. Juste vous et le code.")
            
            # Minuteur
            mins, secs = divmod(st.session_state.focus_timer, 60)
            st.markdown(f'<div class="timer">{mins:02d}:{secs:02d}</div>', unsafe_allow_html=True)
            
            # Audio Lo-Fi (Lecteur HTML5 standard)
            st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3", format="audio/mp3")
            
            col_f1, col_f2 = st.columns(2)
            if col_f1.button("⏹️ Arrêter Focus", use_container_width=True):
                st.session_state.focus_active = False
                st.rerun()
            if col_f2.button("⏱️ -5 Min", use_container_width=True):
                st.session_state.focus_timer = max(0, st.session_state.focus_timer - 300)
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # C'est ici la correction : st.stop() arrête le script proprement au lieu de 'return'
        st.stop()

    # ==================================================
    # 3. LOGIQUE ADAPTATIVE & HELPER HTML
    # ==================================================
    def is_module_locked(module_lvl):
        """Vérifie si le module est verrouillé selon le niveau précédent."""
        if module_lvl == 1: return False
        # On récupère les modules du niveau précédent
        prev_lvl_mods = [s for m, s in st.session_state.eleve_competences.items() 
                         if m in MODULES and MODULES[m]['lvl'] == module_lvl - 1]
        
        if not prev_lvl_mods: return False # Sécurité
        avg_prev = np.mean(prev_lvl_mods)
        return avg_prev < 0.5 # Verrouillé si moyenne niveau N-1 < 50%

    def get_card_html(name, data, score, locked):
        pct = int(score * 100)
        
        # Style Conditionnel (Verrouillé ou Ouvert)
        if locked:
            opacity = "0.4"
            filter_css = "grayscale(100%)"
            border_col = "#555"
            icon_overlay = "<div style='position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); font-size:3rem; z-index:10;'>🔒</div>"
        else:
            opacity = "1"
            filter_css = "none"
            border_col = data['color']
            icon_overlay = ""

        # Barres de progression (HTML compacté)
        segments = ""
        for s in range(10):
            bg = data['color'] if s < (pct // 10) and not locked else "rgba(255,255,255,0.1)"
            glow = f"box-shadow: 0 0 8px {data['color']};" if s < (pct // 10) and not locked else ""
            segments += f"<div style='flex:1; height:5px; background:{bg}; margin-right:3px; border-radius:2px; {glow}'></div>"
        
        # HTML Minifié sur une ligne
        html = f"""<div class="course-card" style="border-left: 4px solid {border_col}; opacity:{opacity}; filter:{filter_css}; position:relative;">{icon_overlay}<div style="display:flex; align-items:center; margin-bottom:10px;"><span style="font-size:2rem; margin-right:15px;">{data['icon']}</span><div><div style="font-size:1.2rem; font-weight:bold; color:white;">{name}</div><span style="font-size:0.7rem; padding:2px 8px; border-radius:4px; background:{data['color']}20; color:{data['color']}; border:1px solid {data['color']};">{data['cat']}</span></div></div><p style="color:#bbb; font-size:0.85em; margin-bottom:12px; height:35px; overflow:hidden;">{data['desc']}</p><div style="display:flex; justify-content:space-between; font-size:0.8em; color:#ccc; margin-bottom:5px;"><span>{'Verrouillé' if locked else f'{pct}% Complété'}</span><span>Niveau {data['lvl']}</span></div><div style="display:flex; width:100%;">{segments}</div></div>"""
        return html

    # ==================================================
    # 4. CSS & STYLE GLOBAL
    # ==================================================
    st.markdown("""
    <style>
        .course-card {
            background: linear-gradient(145deg, rgba(25,25,35,0.9) 0%, rgba(15,15,20,0.95) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            transition: transform 0.3s ease;
        }
        .course-card:hover { transform: translateY(-4px); box-shadow: 0 10px 30px rgba(0,0,0,0.4); }
        .gamification-bar {
            background: rgba(255,255,255,0.03); border-radius: 12px; padding: 10px 20px;
            display: flex; justify-content: space-around; align-items: center; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.05);
        }
        .stat-item { text-align: center; }
        .stat-val { font-size: 1.2rem; font-weight: bold; color: white; }
        .stat-label { font-size: 0.8rem; color: #888; text-transform: uppercase; }
        .srs-box {
            background: rgba(255, 165, 0, 0.1); border-left: 4px solid orange; padding: 15px; border-radius: 8px; margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ==================================================
    # 5. MODALE (COURS + ÉDITEUR + IA RAG)
    # ==================================================
    @st.dialog("🎓 Studio d'Apprentissage", width="large")
    def open_modal(name):
        d = MODULES[name]
        s = st.session_state.eleve_competences[name]
        
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"## {d['icon']} {name}")
            st.caption(f"{d['cat']} • Niveau {d['lvl']}")
        with c2:
            st.metric("Votre Score", f"{int(s*100)}%")

        # 1. VIDEO
        st.video(f"https://www.youtube.com/watch?v={d['video']}")
        
        # 2. IA REMEDIATION (RAG SIMULÉ)
        if s < 0.5 and s > 0:
            st.error(f"📉 Difficulté détectée sur ce module (< 50%)")
            with st.expander("🆘 Zone de Renfort IA (Recommandations)", expanded=True):
                cat_res = RAG_RESOURCES.get(d['cat'], ["Article: Concepts Fondamentaux"])
                for res in cat_res:
                    st.markdown(f"- 🔗 [{res}](#)")
                st.caption("L'IA a sélectionné ces ressources pour combler vos lacunes spécifiques.")

        st.markdown("---")
        
        # 3. ÉDITEUR DE CODE
        st.subheader("💻 Défi de Code Interactif")
        st.caption("Prouvez votre compétence en codant la solution.")
        
        col_code, col_out = st.columns(2)
        with col_code:
            default_code = f"# Codez ici pour le module {name}\n\n# Exercice : Affichez 'Réussite' 5 fois\nfor i in range(5):\n    print('Réussite !')"
            code = st.text_area("Code Python", value=default_code, height=150, key=f"code_{name}")
            run = st.button("▶️ Exécuter", key=f"run_{name}")
        
        with col_out:
            st.markdown("**Terminal Output:**")
            if run:
                old_stdout = sys.stdout
                redirected_output = sys.stdout = StringIO()
                try:
                    exec(code) # Exécution réelle du code Python
                    res = redirected_output.getvalue()
                    st.code(res if res else "Code exécuté (aucune sortie).")
                    st.success("Compilé avec succès !")
                except Exception as e:
                    st.error(f"Erreur : {e}")
                finally:
                    sys.stdout = old_stdout

        st.markdown("---")
        if st.button("✅ Valider et Terminer", use_container_width=True, key=f"val_{name}"):
            st.session_state.eleve_competences[name] = min(1.0, s + 0.1)
            # Mise à jour SRS
            st.session_state.last_review[name] = datetime.now()
            st.toast("Progression enregistrée !", icon="🎉")
            st.rerun()

    # ==================================================
    # 6. HEADER GAMIFIÉ & FOCUS
    # ==================================================
    c_title, c_focus = st.columns([3, 1])
    with c_title:
        st.title("🚀 Neuro-Learning Hub")
    with c_focus:
        if st.button("🍅 Mode Focus", use_container_width=True, type="primary"):
            st.session_state.focus_active = True
            st.rerun()

    # BARRE DE GAMIFICATION
    is_night_owl = datetime.now().hour >= 22
    owl_badge = "🦉 Oiseau de Nuit" if is_night_owl else "☀️ Lève-tôt"
    
    st.markdown(f"""
    <div class="gamification-bar">
        <div class="stat-item"><div class="stat-val">🔥 {st.session_state.streak} Jours</div><div class="stat-label">Série</div></div>
        <div class="stat-item"><div class="stat-val">🥈 Argent</div><div class="stat-label">Ligue</div></div>
        <div class="stat-item"><div class="stat-val">{owl_badge}</div><div class="stat-label">Badge Actif</div></div>
        <div class="stat-item"><div class="stat-val">{int(sum(st.session_state.eleve_competences.values())*1000)} XP</div><div class="stat-label">Total XP</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ==================================================
    # 7. SRS (RÉPÉTITION ESPACÉE)
    # ==================================================
    # Trouve les modules à réviser (non vus depuis 3 jours)
    to_review = [m for m, date in st.session_state.last_review.items() 
                 if (datetime.now() - date).days > 3 and st.session_state.eleve_competences[m] > 0.1]
                 
    if to_review:
        st.markdown(f"""
        <div class="srs-box">
            <h3 style="margin:0">🧠 Flash Répétition (SRS)</h3>
            <p style="margin:5px 0">L'IA a détecté {len(to_review)} sujets qui commencent à s'effacer de votre mémoire.</p>
        </div>
        """, unsafe_allow_html=True)
        
        cols_rev = st.columns(min(len(to_review), 4))
        for i, m in enumerate(to_review[:4]):
            if cols_rev[i].button(f"↺ Réviser : {m}", key=f"srs_{m}"):
                open_modal(m)

    st.divider()

    # ==================================================
    # 8. CATALOGUE ADAPTATIF (GRILLE 2 COLONNES)
    # ==================================================
    st.subheader("📚 Arbre de Compétences (Parcours Adaptatif)")
    
    search = st.text_input("🔍", placeholder="Rechercher un module...", label_visibility="collapsed")
    
    cols = st.columns(2)
    idx = 0
    
    # Tri par Niveau pour l'effet "Arbre"
    sorted_modules = sorted(MODULES.items(), key=lambda x: x[1]['lvl'])
    
    for name, data in sorted_modules:
        if search.lower() in name.lower():
            score = st.session_state.eleve_competences[name]
            
            # LOGIQUE DE VERROUILLAGE (ADAPTIVE LEARNING)
            is_locked = is_module_locked(data['lvl'])
            
            # Génération HTML via la fonction Helper
            html_card = get_card_html(name, data, score, is_locked)
            
            with cols[idx % 2]:
                st.markdown(html_card, unsafe_allow_html=True)
                
                # Bouton Adaptatif
                if is_locked:
                    st.button(f"🔒 Niveau {data['lvl']-1} requis", key=f"lock_{name}", disabled=True, use_container_width=True)
                else:
                    if st.button(f"▶️ Accéder au module", key=f"open_{name}", use_container_width=True):
                        open_modal(name)
            
            idx += 1

elif st.session_state.page == "exercices":
   
    
    # ==================================================
    # 1. CONFIGURATION & STYLE (UI PREMIUM)
    # ==================================================
    
    try:
        from fpdf2 import FPDF
        FPDEF_AVAILABLE = True
    except ImportError:
        FPDEF_AVAILABLE = False

    # CSS : Ambiance "Laboratoire de Recherche IA"
    st.markdown("""
    <style>
        /* Animations */
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        
        /* Header de Niveau */
        .level-banner {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            padding: 25px; border-radius: 16px; border-bottom: 4px solid #00f6ff;
            text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.5); margin-bottom: 30px;
            animation: fadeIn 0.8s ease-out;
        }
        .level-title { font-size: 2.2rem; font-weight: 900; color: white; letter-spacing: 3px; text-transform: uppercase; }
        .level-desc { color: #00f6ff; font-size: 1.1rem; font-family: monospace; margin-top: 5px; }

        /* Carte Question */
        .question-card {
            background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px;
            padding: 35px; margin-bottom: 25px; border-left: 6px solid #00f6ff;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            animation: fadeIn 0.5s ease-out;
        }
        .q-text { font-size: 1.5rem; font-weight: 600; color: #ffffff; line-height: 1.5; }
        
        /* Feedback */
        .feedback-box { padding: 20px; border-radius: 12px; margin-top: 20px; animation: fadeIn 0.5s; }
        .success { background: rgba(46, 204, 113, 0.15); border: 1px solid #2ecc71; color: #2ecc71; }
        .error { background: rgba(231, 76, 60, 0.15); border: 1px solid #e74c3c; color: #e74c3c; }
        
        /* Fun Fact */
        .fun-fact {
            background: linear-gradient(90deg, #FF9A9E 0%, #FECFEF 100%);
            color: #444; padding: 15px; border-radius: 12px; 
            font-weight: bold; text-align: center; font-style: italic; margin-top: 20px;
            box-shadow: 0 4px 15px rgba(255, 154, 158, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)

    # ==================================================
    # 2. BANQUE DE 50 VRAIES QUESTIONS (Non répétitives)
    # ==================================================
    
    REAL_QUESTIONS = [
        # --- NIVEAU 1 : FONDAMENTAUX PYTHON ---
        {"q": "Pour afficher la phrase 'Hello World' dans la console, quelle syntaxe est correcte ?", "type": "mc", "choices": ["echo 'Hello World'", "print('Hello World')", "console.log('Hello World')", "System.out.print('Hello World')"], "answer": "print('Hello World')", "concept": "Syntaxe", "explication": "En Python, la fonction native pour l'affichage est `print()`."},
        {"q": "Comment assigner la valeur 10 à une variable nommée 'score' ?", "type": "mc", "choices": ["int score = 10", "score := 10", "score = 10", "var score = 10"], "answer": "score = 10", "concept": "Variables", "explication": "Python est typé dynamiquement : pas besoin de préciser le type (int)."},
        {"q": "Quel est le résultat de l'opération `10 % 3` (Modulo) ?", "type": "mc", "choices": ["3.33", "1", "3", "0"], "answer": "1", "concept": "Opérateurs", "explication": "Le modulo (%) retourne le reste de la division entière (10 = 3*3 + 1)."},
        {"q": "Comment créer une liste contenant les chiffres 1, 2 et 3 ?", "type": "mc", "choices": ["L = (1, 2, 3)", "L = {1, 2, 3}", "L = [1, 2, 3]", "L = <1, 2, 3>"], "answer": "L = [1, 2, 3]", "concept": "Listes", "explication": "Les listes en Python se définissent avec des crochets `[]`."},
        {"q": "Quelle est la position (index) de la première lettre dans la chaîne 'Python' ?", "type": "mc", "choices": ["1", "0", "-1", "A"], "answer": "0", "concept": "Indexation", "explication": "En informatique, et en Python, l'indexation commence toujours à 0."},
        {"q": "Quel mot-clé est utilisé pour définir une nouvelle fonction ?", "type": "mc", "choices": ["function", "def", "func", "define"], "answer": "def", "concept": "Fonctions", "explication": "On utilise `def` suivi du nom de la fonction."},
        {"q": "Quel type de donnée est le résultat de `5 > 10` ?", "type": "mc", "choices": ["Entier (int)", "Chaîne (str)", "Booléen (bool)", "Flottant (float)"], "answer": "Booléen (bool)", "concept": "Types", "explication": "Une comparaison renvoie toujours Vrai (True) ou Faux (False)."},
        {"q": "Comment ajouter l'élément 'X' à la fin d'une liste nommée `data` ?", "type": "mc", "choices": ["data.add('X')", "data.push('X')", "data.append('X')", "data.insert('X')"], "answer": "data.append('X')", "concept": "Méthodes", "explication": "La méthode `.append()` ajoute un élément à la fin de la liste."},
        {"q": "Quel symbole permet d'écrire un commentaire sur une seule ligne ?", "type": "mc", "choices": ["//", "/*", "<!--", "#"], "answer": "#", "concept": "Syntaxe", "explication": "Tout ce qui suit le `#` est ignoré par l'interpréteur Python."},
        {"q": "Quelle fonction retourne la taille (nombre d'éléments) d'une liste ?", "type": "mc", "choices": ["size()", "count()", "len()", "length()"], "answer": "len()", "concept": "Fonctions", "explication": "`len()` est la fonction universelle pour obtenir la longueur."},

        # --- NIVEAU 2 : LOGIQUE & CONTRÔLE ---
        {"q": "Quelle boucle utiliser si on veut répéter une action un nombre défini de fois ?", "type": "mc", "choices": ["while", "for", "repeat", "do...while"], "answer": "for", "concept": "Boucles", "explication": "La boucle `for` est idéale pour parcourir des séquences ou des intervalles connus."},
        {"q": "Quel mot-clé permet d'interrompre immédiatement une boucle ?", "type": "mc", "choices": ["stop", "exit", "break", "continue"], "answer": "break", "concept": "Contrôle", "explication": "`break` sort définitivement de la boucle. `continue` passe à l'itération suivante."},
        {"q": "Dans un dictionnaire `{'nom': 'Awa'}`, comment récupérer 'Awa' ?", "type": "mc", "choices": ["d[0]", "d.get_value()", "d['nom']", "d.nom"], "answer": "d['nom']", "concept": "Dictionnaires", "explication": "On accède aux valeurs d'un dictionnaire via leurs clés entre crochets."},
        {"q": "Quelle est la particularité principale d'un Tuple `(1, 2)` ?", "type": "mc", "choices": ["Il est immuable (non modifiable)", "Il est plus lent qu'une liste", "Il ne peut contenir que des chiffres", "Il est infini"], "answer": "Il est immuable (non modifiable)", "concept": "Types", "explication": "Contrairement aux listes, on ne peut pas modifier un tuple après sa création."},
        {"q": "Si `A = True` et `B = False`, que vaut `A and B` ?", "type": "mc", "choices": ["True", "False", "None", "Erreur"], "answer": "False", "concept": "Logique", "explication": "L'opérateur `and` exige que les DEUX conditions soient vraies."},
        {"q": "Que génère l'instruction `range(3)` ?", "type": "mc", "choices": ["1, 2, 3", "0, 1, 2", "0, 1, 2, 3", "1, 2"], "answer": "0, 1, 2", "concept": "Boucles", "explication": "`range(n)` génère une suite de 0 à n-1."},
        {"q": "Quelle erreur obtenez-vous en divisant un nombre par zéro ?", "type": "mc", "choices": ["ValueError", "SyntaxError", "ZeroDivisionError", "InfinityError"], "answer": "ZeroDivisionError", "concept": "Exceptions", "explication": "C'est une exception spécifique à l'arithmétique."},
        {"q": "Quelle structure permet de gérer les erreurs sans planter le programme ?", "type": "mc", "choices": ["if / else", "try / except", "do / catch", "error / resume"], "answer": "try / except", "concept": "Exceptions", "explication": "On 'essaie' (try) un code, et on 'attrape' (except) l'erreur si elle survient."},
        {"q": "Comment vérifier si la valeur 'pomme' est présente dans la liste `fruits` ?", "type": "mc", "choices": ["fruits.has('pomme')", "'pomme' in fruits", "check('pomme', fruits)", "fruits.contains('pomme')"], "answer": "'pomme' in fruits", "concept": "Opérateurs", "explication": "L'opérateur d'appartenance `in` est très puissant en Python."},
        {"q": "Pour utiliser les fonctions mathématiques avancées (racine carrée, pi...), que doit-on faire ?", "type": "mc", "choices": ["Rien, c'est inclus", "import math", "include math", "pip install math"], "answer": "import math", "concept": "Modules", "explication": "Il faut importer le module standard `math`."},

        # --- NIVEAU 3 : DATA SCIENCE (PANDAS & NUMPY) ---
        {"q": "Quelle est la bibliothèque standard pour manipuler des tableaux de données (DataFrames) ?", "type": "mc", "choices": ["NumPy", "Pandas", "Matplotlib", "Requests"], "answer": "Pandas", "concept": "Outils", "explication": "Pandas est l'outil de référence pour l'analyse de données structurées."},
        {"q": "Avec Pandas, quelle fonction permet de lire un fichier CSV ?", "type": "mc", "choices": ["pd.load_csv()", "pd.read_csv()", "pd.import()", "pd.csv()"], "answer": "pd.read_csv()", "concept": "Pandas", "explication": "`read_csv` charge les données dans un DataFrame."},
        {"q": "Que représente `NaN` dans un jeu de données ?", "type": "mc", "choices": ["Un nombre infini", "Une valeur manquante (Not a Number)", "Une erreur de calcul", "Un zéro"], "answer": "Une valeur manquante (Not a Number)", "concept": "Nettoyage", "explication": "C'est le marqueur standard pour les données absentes."},
        {"q": "Quelle méthode permet de voir les 5 premières lignes d'un DataFrame ?", "type": "mc", "choices": ["df.top()", "df.start()", "df.head()", "df.preview()"], "answer": "df.head()", "concept": "Exploration", "explication": "`head()` affiche l'en-tête du tableau."},
        {"q": "Quel est l'avantage principal d'un tableau NumPy (array) par rapport à une liste ?", "type": "mc", "choices": ["Il peut contenir du texte", "Il est beaucoup plus rapide pour les calculs", "Il est plus facile à lire", "Il a une taille infinie"], "answer": "Il est beaucoup plus rapide pour les calculs", "concept": "NumPy", "explication": "NumPy est optimisé en C pour les calculs vectoriels."},
        {"q": "Quelle fonction permet d'obtenir un résumé statistique (moyenne, min, max...) ?", "type": "mc", "choices": ["df.info()", "df.stats()", "df.describe()", "df.summary()"], "answer": "df.describe()", "concept": "Statistiques", "explication": "`describe()` génère un tableau de statistiques descriptives."},
        {"q": "Comment sélectionner uniquement la colonne 'Age' d'un DataFrame `df` ?", "type": "mc", "choices": ["df['Age']", "df(Age)", "df.select('Age')", "df -> Age"], "answer": "df['Age']", "concept": "Manipulation", "explication": "On utilise la syntaxe dictionnaire ou l'attribut `df.Age`."},
        {"q": "Quelle librairie est utilisée pour tracer des graphiques basiques ?", "type": "mc", "choices": ["Matplotlib", "Flask", "Django", "PyGame"], "answer": "Matplotlib", "concept": "Visualisation", "explication": "C'est la base de la visualisation en Python."},
        {"q": "Que fait l'opération de 'Data Cleaning' ?", "type": "mc", "choices": ["Supprimer toutes les données", "Corriger et préparer les données brutes", "Sauvegarder sur le disque", "Crypter les données"], "answer": "Corriger et préparer les données brutes", "concept": "Méthodologie", "explication": "C'est l'étape cruciale pour traiter les valeurs manquantes, doublons, etc."},
        {"q": "En statistique, qu'est-ce que la 'Médiane' ?", "type": "mc", "choices": ["La moyenne arithmétique", "La valeur la plus fréquente", "La valeur qui sépare la série en deux moitiés", "L'écart type"], "answer": "La valeur qui sépare la série en deux moitiés", "concept": "Statistiques", "explication": "50% des valeurs sont en dessous, 50% au-dessus."},

        # --- NIVEAU 4 : MACHINE LEARNING (THÉORIE) ---
        {"q": "Quel type d'apprentissage utilise des données déjà étiquetées (exemple: photos de chats/chiens) ?", "type": "mc", "choices": ["Supervisé", "Non-supervisé", "Renforcement", "Auto-supervisé"], "answer": "Supervisé", "concept": "Types ML", "explication": "On donne la réponse (label) au modèle pour qu'il apprenne."},
        {"q": "La Régression Logistique est utilisée pour...", "type": "mc", "choices": ["Prédire un nombre continu (prix)", "Classifier des données (Oui/Non)", "Grouper des données", "Générer du texte"], "answer": "Classifier des données (Oui/Non)", "concept": "Algorithmes", "explication": "Malgré son nom, c'est un classifieur binaire."},
        {"q": "Qu'est-ce que l'Overfitting (Sur-apprentissage) ?", "type": "mc", "choices": ["Le modèle est trop nul", "Le modèle apprend le bruit par cœur et ne généralise pas", "Le modèle est trop lent", "Le modèle manque de données"], "answer": "Le modèle apprend le bruit par cœur et ne généralise pas", "concept": "Problèmes", "explication": "Il est excellent sur les données d'entraînement, mais mauvais sur les nouvelles données."},
        {"q": "Quelle métrique est la plus adaptée pour évaluer une classification déséquilibrée ?", "type": "mc", "choices": ["L'Accuracy (Précision globale)", "Le F1-Score", "La MSE", "Le R-carré"], "answer": "Le F1-Score", "concept": "Métriques", "explication": "Le F1-Score combine précision et rappel, utile quand une classe est rare."},
        {"q": "À quoi sert l'ensemble de Test (Test Set) ?", "type": "mc", "choices": ["Entraîner le modèle", "Évaluer la performance finale objective", "À régler les hyperparamètres", "À rien"], "answer": "À évaluer la performance finale objective", "concept": "Méthodologie", "explication": "Ce sont des données que le modèle n'a JAMAIS vues."},
        {"q": "Quel algorithme regroupe des données sans étiquettes (Clustering) ?", "type": "mc", "choices": ["Linear Regression", "K-Means", "Decision Tree", "Random Forest"], "answer": "K-Means", "concept": "Algorithmes", "explication": "Il crée des groupes (clusters) basés sur la similarité."},
        {"q": "Dans Scikit-Learn, quelle méthode entraîne le modèle ?", "type": "mc", "choices": [".predict()", ".score()", ".fit()", ".train()"], "answer": ".fit()", "concept": "Scikit-Learn", "explication": "`model.fit(X, y)` lance l'apprentissage."},
        {"q": "Qu'est-ce qu'une 'Feature' (Caractéristique) ?", "type": "mc", "choices": ["Le résultat à prédire", "Une variable d'entrée (colonne)", "Une erreur", "Un algorithme"], "answer": "Une variable d'entrée (colonne)", "concept": "Vocabulaire", "explication": "Exemple : l'âge, le prix, la taille sont des features."},
        {"q": "Quel algorithme est basé sur une multitude d'arbres de décision ?", "type": "mc", "choices": ["SVM", "Random Forest", "KNN", "Naive Bayes"], "answer": "Random Forest", "concept": "Algorithmes", "explication": "C'est une méthode d'ensemble qui combine plusieurs arbres."},
        {"q": "La PCA (Analyse en Composantes Principales) sert à...", "type": "mc", "choices": ["Augmenter les données", "Réduire les dimensions", "Classifier", "Nettoyer"], "answer": "Réduire les dimensions", "concept": "Preprocessing", "explication": "Elle projette les données pour simplifier le problème."},

        # --- NIVEAU 5 : DEEP LEARNING & AVANCÉ ---
        {"q": "Quelle fonction d'activation est la plus populaire pour les couches cachées ?", "type": "mc", "choices": ["Sigmoid", "ReLU", "Linear", "Step"], "answer": "ReLU", "concept": "Réseaux de Neurones", "explication": "Rectified Linear Unit est simple et évite la disparition du gradient."},
        {"q": "Pour traiter des images, quel type de réseau est le plus performant ?", "type": "mc", "choices": ["RNN (Récurrent)", "CNN (Convolutif)", "MLP (Perceptron)", "K-Means"], "answer": "CNN (Convolutif)", "concept": "Vision", "explication": "Les CNN sont conçus pour détecter des motifs visuels."},
        {"q": "Qu'est-ce que la 'Backpropagation' ?", "type": "mc", "choices": ["L'initialisation du réseau", "L'algorithme qui met à jour les poids pour réduire l'erreur", "La prédiction finale", "Le nettoyage des données"], "answer": "L'algorithme qui met à jour les poids pour réduire l'erreur", "concept": "Entraînement", "explication": "C'est le cœur de l'apprentissage des réseaux de neurones."},
        {"q": "Pour le traitement du langage naturel (NLP) moderne, quelle architecture domine ?", "type": "mc", "choices": ["CNN", "Transformers", "SVM", "Arbres"], "answer": "Transformers", "concept": "NLP", "explication": "Architecture derrière BERT et GPT, basée sur l'attention."},
        {"q": "Quel framework de Deep Learning est développé par Facebook (Meta) ?", "type": "mc", "choices": ["TensorFlow", "PyTorch", "Keras", "Scikit-Learn"], "answer": "PyTorch", "concept": "Outils", "explication": "Très populaire dans la recherche."},
        {"q": "Que signifie 'Epoch' dans l'entraînement ?", "type": "mc", "choices": ["Une minute d'entraînement", "Un passage complet de toutes les données", "Une seule donnée", "Une erreur"], "answer": "Un passage complet de toutes les données", "concept": "Vocabulaire", "explication": "Si vous avez 1000 images, 1 epoch = le modèle a vu les 1000 images."},
        {"q": "Le 'Dropout' est une technique pour...", "type": "mc", "choices": ["Accélérer le calcul", "Éviter le sur-apprentissage (Overfitting)", "Sauvegarder le modèle", "Visualiser"], "answer": "Éviter le sur-apprentissage (Overfitting)", "concept": "Régularisation", "explication": "On désactive aléatoirement des neurones pour rendre le réseau plus robuste."},
        {"q": "Quel matériel est indispensable pour entraîner de gros modèles rapidement ?", "type": "mc", "choices": ["CPU", "GPU", "Disque Dur", "Souris"], "answer": "GPU", "concept": "Hardware", "explication": "Les cartes graphiques calculent les matrices en parallèle très vite."},
        {"q": "Qu'est-ce qu'un Tensor ?", "type": "mc", "choices": ["Un neurone", "Une matrice multidimensionnelle", "Une fonction", "Un paramètre"], "answer": "Une matrice multidimensionnelle", "concept": "Maths", "explication": "C'est la structure de données de base du Deep Learning."},
        {"q": "Quel type d'IA génère du contenu (texte, image) ?", "type": "mc", "choices": ["Discriminative", "Générative", "Analytique", "Descriptive"], "answer": "Générative", "concept": "GenAI", "explication": "Ex: ChatGPT, Midjourney."}
    ]

    # ==================================================
    # 3. FONCTIONS UTILITAIRES (FIX PDF & GRAPHS)
    # ==================================================
    
    def create_pdf_bytes(user_name, level_name, df, analysis_text):
        """
        Fonction Robuste pour générer le PDF.
        Le .encode('latin-1') est le coupable habituel si l'objet est déjà un bytearray.
        Ici, on s'assure de retourner un objet 'bytes' propre pour Streamlit.
        """
        if not FPDEF_AVAILABLE: return None
        
        pdf = FPDF()
        pdf.add_page()
        
        # En-tête
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"Rapport de Performance - {level_name}", ln=True, align='C')
        pdf.ln(10)
        
        # Infos
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Etudiant: {user_name}", ln=True)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%d/%m/%Y')}", ln=True)
        pdf.ln(5)
        
        # Stats
        score = int(df["correct"].mean() * 100)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"Score Obtenu: {score}%", ln=True)
        
        # Analyse
        pdf.set_font("Arial", "", 11)
        pdf.ln(5)
        # Nettoyage texte pour éviter erreurs encodage FPDF standard
        clean_analysis = analysis_text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 8, f"Analyse IA: {clean_analysis}")
        
        pdf.ln(15)
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 10, "Certifie par Elaan AI Platform.", ln=True, align='C')
        
        # --- FIX CRITIQUE ---
        # fpdf2 .output() retourne des bytes ou str.
        # Si c'est une chaine, on encode. Si c'est un bytearray, on convertit en bytes.
        raw_output = pdf.output(dest='S')
        
        if isinstance(raw_output, str):
            return raw_output.encode('latin-1')
        elif isinstance(raw_output, bytearray):
            return bytes(raw_output)
        else:
            return bytes(raw_output) # Cas ou c'est deja des bytes

    def generate_10_graphs(df, title_suffix=""):
        """Génère 10 graphiques analytiques Plotly."""
        graphs = []
        
        # 1. Barres : Score par Concept
        grp = df.groupby("concept")["correct"].mean().reset_index()
        fig1 = px.bar(grp, x="correct", y="concept", orientation='h', title="1. Maîtrise par Concept", color="correct", range_x=[0,1])
        graphs.append(fig1)
        
        # 2. Ligne : Vélocité
        df['q_num'] = range(1, len(df) + 1)
        fig2 = px.line(df, x="q_num", y="time", markers=True, title="2. Temps de Réflexion (s)")
        graphs.append(fig2)
        
        # 3. Pie : Validité
        fig3 = px.pie(df, names="correct", title="3. Taux de Succès Global", color_discrete_sequence=["#ff4b4b", "#00ff88"])
        graphs.append(fig3)
        
        # 4. Scatter : Temps vs Réussite
        fig4 = px.scatter(df, x="time", y="correct", color="concept", title="4. Corrélation Vitesse/Succès")
        graphs.append(fig4)
        
        # 5. Area : XP Cumulé
        df['xp'] = df['correct'].astype(int).cumsum() * 10
        fig5 = px.area(df, x="q_num", y="xp", title="5. Accumulation d'XP")
        graphs.append(fig5)
        
        # 6. Histogramme Temps
        fig6 = px.histogram(df, x="time", nbins=5, title="6. Distribution du Temps")
        graphs.append(fig6)
        
        # 7. Sunburst Erreurs
        errs = df[~df['correct']]
        if not errs.empty:
            fig7 = px.sunburst(errs, path=['concept'], title="7. Analyse des Erreurs")
        else:
            fig7 = go.Figure().add_annotation(text="Parfait !", showarrow=False)
            fig7.update_layout(title="7. Focus Erreurs")
        graphs.append(fig7)
        
        # 8. Gauge Score
        score = df['correct'].mean() * 100
        fig8 = go.Figure(go.Indicator(mode="gauge+number", value=score, title={'text': "8. Score Niveau"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#00f6ff"}}))
        graphs.append(fig8)
        
        # 9. Radar Profil (Simulé)
        cats = ['Mémoire', 'Logique', 'Vitesse', 'Attention', 'Précision']
        vals = [max(0.2, min(1, score/100 + random.uniform(-0.2, 0.2))) for _ in cats]
        fig9 = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill='toself'))
        fig9.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="9. Profil Cognitif")
        graphs.append(fig9)
        
        # 10. Funnel
        fig10 = px.funnel(pd.DataFrame({'Step':['Vues','Répondues','Justes'], 'Val':[len(df), len(df), df['correct'].sum()]}), x='Val', y='Step', title="10. Conversion")
        graphs.append(fig10)
        
        return graphs

    # ==================================================
    # 4. LOGIQUE DE SESSION
    # ==================================================
    
    if "exo_state" not in st.session_state:
        st.session_state.exo_state = {
            "q_idx": 0,
            "responses": [],
            "score_total": 0,
            "pause_analy": False
        }
    
    state = st.session_state.exo_state
    user_name = st.session_state.get("user_profile", {}).get("nom", "Etudiant")
    
    # --- FIN GLOBALE (50 Questions) ---
    if state["q_idx"] >= 50:
        st.balloons()
        st.markdown('<div class="level-banner"><div class="level-title">🏆 PARCOURS TERMINÉ</div></div>', unsafe_allow_html=True)
        
        df_all = pd.DataFrame(state["responses"])
        score_final = df_all["correct"].sum()
        
        c1, c2 = st.columns(2)
        c1.metric("Score Final", f"{score_final}/50")
        c2.metric("Précision", f"{score_final/50:.1%}")
        
        st.subheader("📊 Analyse Globale (10 Indicateurs)")
        graphs = generate_10_graphs(df_all, "Global")
        cols = st.columns(2)
        for i, g in enumerate(graphs):
            cols[i%2].plotly_chart(g, use_container_width=True)
            
        if FPDEF_AVAILABLE:
            # Appel fonction sécurisée
            pdf_data = create_pdf_bytes(user_name, "GLOBAL", df_all, "Parcours complet validé avec succès.")
            if pdf_data:
                st.download_button("📥 Télécharger Certificat Final", data=pdf_data, file_name="certificat_elaan.pdf", mime="application/pdf")
            
        if st.button("Recommencer"):
            st.session_state.exo_state = {"q_idx": 0, "responses": [], "score_total": 0, "pause_analy": False}
            st.rerun()
            
    # --- PAUSE BILAN NIVEAU ---
    elif state["q_idx"] > 0 and state["q_idx"] % 10 == 0 and state["pause_analy"]:
        level = state["q_idx"] // 10
        st.markdown(f'<div class="level-banner"><div class="level-title">BILAN NIVEAU {level}</div><div class="level-desc">Analyse de vos performances</div></div>', unsafe_allow_html=True)
        
        # Fun Fact
        facts = [
            "Le premier programmeur était une femme : Ada Lovelace.",
            "Python est utilisé par la NASA, Google et Netflix.",
            "90% des données mondiales ont moins de 2 ans.",
            "Le 'Bug' vient d'un papillon de nuit coincé dans un ordinateur en 1947.",
            "L'IA a battu le champion du monde de Go en 2016."
        ]
        st.markdown(f"<div class='fun-fact'>💡 Le saviez-vous ? {random.choice(facts)}</div>", unsafe_allow_html=True)
        
        # Analyse Niveau
        df_lvl = pd.DataFrame(state["responses"][-10:])
        score_lvl = df_lvl["correct"].mean()
        
        col1, col2 = st.columns(2)
        col1.metric("Score Niveau", f"{int(score_lvl*100)}%")
        
        if score_lvl > 0.8: col2.success("Excellent ! Continuez comme ça.")
        elif score_lvl > 0.5: col2.info("Bon travail, mais restez concentré.")
        else: col2.warning("Niveau difficile. Prenez le temps de relire.")
        
        with st.expander("📈 Voir l'analyse détaillée (10 Graphiques)", expanded=True):
            graphs = generate_10_graphs(df_lvl)
            gc = st.columns(2)
            for i, g in enumerate(graphs):
                gc[i%2].plotly_chart(g, use_container_width=True)
        
        if FPDEF_AVAILABLE:
            # Appel fonction sécurisée
            pdf_data = create_pdf_bytes(user_name, f"Niveau {level}", df_lvl, f"Score obtenu : {score_lvl:.1%}")
            if pdf_data:
                st.download_button(f"📄 Télécharger le rapport Niveau {level}", data=pdf_data, file_name=f"rapport_n{level}.pdf", mime="application/pdf")
            
        if st.button(f"🚀 Passer au Niveau {level+1}", type="primary", use_container_width=True):
            state["pause_analy"] = False
            st.rerun()

    # --- QUESTIONNAIRE STANDARD ---
    else:
        q_data = REAL_QUESTIONS[state["q_idx"]]
        lvl_current = (state["q_idx"] // 10) + 1
        q_num = (state["q_idx"] % 10) + 1
        
        st.markdown(f"""
        <div class="level-banner">
            <div class="level-title">NIVEAU {lvl_current}</div>
            <div class="level-desc">Question {q_num}/10 • {q_data['concept']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(state["q_idx"]/50)
        
        st.markdown(f"<div class='question-card'><div class='q-text'>{q_data['q']}</div></div>", unsafe_allow_html=True)
        
        start = time.time()
        choice = st.radio("Votre réponse :", q_data['choices'], key=f"q{state['q_idx']}")
        
        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button("Valider", type="primary", use_container_width=True):
                correct = (choice == q_data['answer'])
                
                if correct:
                    st.markdown(f"<div class='feedback-success'>✅ <b>Exact !</b> {q_data['explication']}</div>", unsafe_allow_html=True)
                    if random.random() > 0.7: st.balloons()
                else:
                    st.markdown(f"<div class='feedback-error'>❌ <b>Erreur.</b> Réponse : {q_data['answer']}.<br><i>{q_data['explication']}</i></div>", unsafe_allow_html=True)
                
                state["responses"].append({
                    "q_index": state["q_idx"],
                    "correct": correct,
                    "time": time.time() - start,
                    "concept": q_data["concept"]
                })
                
                time.sleep(2) # Pause pour lire
                state["q_idx"] += 1
                if state["q_idx"] % 10 == 0: state["pause_analy"] = True
                st.rerun()
                
        with c2:
            if st.button("💡 Indice"):
                st.info(f"Indice : {q_data['concept']}")
                # =====================================================
# 9. PAGE: ESPACE ENSEIGNANT — SINGULARITY V9 (STABLE & NO DEPENDENCY)
# =====================================================
elif st.session_state.page == "enseignant":
    # --- IMPORTS ---
  

    # --- STYLE CSS "SINGULARITY NEON" ---
    st.markdown(f"""
    <style>
        /* Fond sombre tech */
        .stApp {{
            background-color: #050505;
        }}
        
        /* Cartes KPI Holographiques */
        .kpi-panel {{
            background: linear-gradient(145deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }}
        .kpi-panel:hover {{
            transform: translateY(-5px);
            border-color: {THEME['primary']};
            box-shadow: 0 0 25px rgba(0, 242, 255, 0.2);
        }}
        .kpi-value {{
            font-size: 2.5rem;
            font-weight: 800;
            color: white;
            text-shadow: 0 0 10px rgba(0,0,0,0.5);
        }}
        .kpi-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #888;
            margin-bottom: 5px;
        }}
        
        /* Titres Sections */
        .section-header {{
            font-size: 1.4rem;
            font-weight: 700;
            margin: 40px 0 20px 0;
            padding-left: 15px;
            border-left: 4px solid {THEME['secondary']};
            background: linear-gradient(90deg, rgba(112,0,255,0.1), transparent);
            color: white;
            padding: 10px;
            border-radius: 0 10px 10px 0;
        }}
    </style>
    """, unsafe_allow_html=True)

    # --- GÉNÉRATION DE DONNÉES (CACHE) ---
    @st.cache_data
    def get_v9_data():
        np.random.seed(2025)
        n = 200
        names = [f"Agent-{i:03d}" for i in range(n)]
        groups = np.random.choice(["Alpha", "Beta", "Gamma", "Omega"], n)
        
        df = pd.DataFrame({
            "ID": range(1000, 1000+n),
            "Nom": names,
            "Groupe": groups,
            "Maths": np.random.normal(65, 15, n).clip(0, 100),
            "Python": np.random.normal(60, 20, n).clip(0, 100),
            "DeepLearning": np.random.normal(50, 25, n).clip(0, 100),
            "SoftSkills": np.random.normal(75, 10, n).clip(0, 100),
            "Assiduité": np.random.randint(20, 100, n),
            "Heures_Code": np.random.randint(5, 150, n),
        })
        
        # Score calculé
        df["Score_Global"] = (df["Maths"]*0.2 + df["Python"]*0.3 + df["DeepLearning"]*0.4 + df["SoftSkills"]*0.1)
        
        # Statut (Fix du bug np.select)
        conds = [(df["Assiduité"] < 40), (df["Score_Global"] < 60), (df["Score_Global"] >= 60)]
        choices = ["Critique", "Surveillance", "Optimal"]
        df["Statut"] = np.select(conds, choices, default="Inconnu")
        
        # Cluster
        df["Cluster"] = np.where(df["Score_Global"] > 75, "Elite", 
                        np.where(df["Score_Global"] > 50, "Standard", "Risque"))
        
        return df

    df_raw = get_v9_data()

    # --- SIDEBAR FILTRES ---
    with st.sidebar:
        st.header("🎛️ Commandes")
        sel_grp = st.multiselect("Cohorte", ["Alpha", "Beta", "Gamma", "Omega"], default=["Alpha", "Beta"])
        sel_sts = st.multiselect("Statut", ["Optimal", "Surveillance", "Critique"], default=["Optimal", "Surveillance", "Critique"])
        
        df = df_raw[df_raw["Groupe"].isin(sel_grp) & df_raw["Statut"].isin(sel_sts)]
        
        st.divider()
        st.metric("Agents Actifs", len(df))
        st.download_button("📥 Export CSV", df.to_csv(), "data.csv", "text/csv")

    # --- HEADER ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f'<h1 style="font-size:2.8rem; margin:0;">SamaLearn <span style="color:{THEME["primary"]}">SINGULARITY</span></h1>', unsafe_allow_html=True)
        st.caption("Système d'Intelligence Pédagogique Augmentée v9.0")
    with c2:
        st.markdown(f"""
        <div style="text-align:right; padding:10px; background:rgba(0,242,255,0.05); border-radius:8px; border:1px solid {THEME['primary']};">
            <div style="color:{THEME['primary']}; font-weight:bold;">● LIVE MONITOR</div>
            <div style="color:white; font-size:1.2rem; font-weight:bold;">{df['Score_Global'].mean():.1f} <span style="font-size:0.6rem;">MOYENNE</span></div>
        </div>
        """, unsafe_allow_html=True)

    # --- ZONE 1 : KPIS ---
    k1, k2, k3, k4 = st.columns(4)
    
    # Calcul sécurisé des métriques
    m_dl = df['DeepLearning'].mean() if not df.empty else 0
    m_att = df['Assiduité'].mean() if not df.empty else 0
    m_risk = len(df[df['Statut']=='Critique'])
    
    metrics = [
        {"l": "Effectif", "v": len(df), "c": "white"},
        {"l": "Performance IA", "v": f"{m_dl:.1f}%", "c": THEME['secondary']},
        {"l": "Engagement", "v": f"{m_att:.1f}%", "c": "#ffaa00"},
        {"l": "Alertes", "v": m_risk, "c": "#ff2b2b"},
    ]
    
    for col, m in zip([k1, k2, k3, k4], metrics):
        with col:
            st.markdown(f"""
            <div class="kpi-panel">
                <div class="kpi-label">{m['l']}</div>
                <div class="kpi-value" style="color:{m['c']}">{m['v']}</div>
            </div>
            """, unsafe_allow_html=True)

    # =====================================================
    # ZONE 2 : CORE INTELLIGENCE (LES 4 MODULES)
    # =====================================================
    st.markdown('<div class="section-header">🧠 CORE INTELLIGENCE ARTIFICIELLE</div>', unsafe_allow_html=True)
    
    t1, t2, t3, t4 = st.tabs(["🧬 DEEP LEARNING", "🌌 CLUSTERING 3D", "🤖 ML REGRESSION", "🔮 PRÉDICTIONS"])

    # TAB 1: Deep Learning Monitor
    with t1:
        c_dl1, c_dl2 = st.columns([2, 1])
        with c_dl1:
            st.markdown("##### 📉 Courbe d'Apprentissage (Simulation)")
            epochs = list(range(1, 31))
            loss = [0.8 * (0.9 ** i) + np.random.normal(0, 0.02) for i in epochs]
            acc = [0.6 + 0.35 * (1 - 0.85 ** i) + np.random.normal(0, 0.01) for i in epochs]
            
            fig_dl = go.Figure()
            fig_dl.add_trace(go.Scatter(x=epochs, y=loss, mode='lines', name='Loss', line=dict(color='#ff2b2b', width=3)))
            fig_dl.add_trace(go.Scatter(x=epochs, y=acc, mode='lines', name='Accuracy', yaxis='y2', line=dict(color='#00ff94', width=3)))
            
            fig_dl.update_layout(
                xaxis=dict(title="Epochs", showgrid=False),
                yaxis=dict(title=dict(text="Perte", font=dict(color="#ff2b2b")), showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                yaxis2=dict(title=dict(text="Précision", font=dict(color="#00ff94")), overlaying='y', side='right'),
                height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='white'), legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_dl, use_container_width=True)
        with c_dl2:
            st.markdown("##### ⚙️ Paramètres")
            st.info("Modèle: Neural Network v4")
            st.progress(92, text="Précision Modèle: 92%")
            st.progress(15, text="Taux d'Erreur: 0.08")

    # TAB 2: Clustering 3D
    with t2:
        st.markdown("##### 🌌 Cartographie 3D des Profils")
        if not df.empty:
            fig_3d = px.scatter_3d(df, x='Maths', y='DeepLearning', z='Heures_Code',
                                 color='Cluster', size='Assiduité',
                                 color_discrete_map={'Elite': '#00ff94', 'Standard': '#00f2ff', 'Risque': '#ff2b2b'},
                                 opacity=0.8)
            fig_3d.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                               scene=dict(bgcolor='rgba(0,0,0,0)', xaxis=dict(backgroundcolor='rgba(0,0,0,0)')))
            st.plotly_chart(fig_3d, use_container_width=True)

    # TAB 3: ML Regression (SANS STATSMODELS - FIX CORRIGÉ)
    with t3:
        st.markdown("##### 🤖 Analyse de Corrélation (Régression)")
        if not df.empty:
            # Calcul manuel de la régression pour éviter l'erreur statsmodels
            x = df["Heures_Code"]
            y = df["Score_Global"]
            # Ajustement polynomial degré 1 (Ligne droite)
            m, b = np.polyfit(x, y, 1)
            
            fig_ml = px.scatter(df, x="Heures_Code", y="Score_Global", color="Cluster",
                              title="Impact du Code sur la Performance",
                              color_discrete_map={'Elite': '#00ff94', 'Standard': '#00f2ff', 'Risque': '#ff2b2b'})
            
            # Ajout manuel de la ligne de tendance
            x_line = np.array([x.min(), x.max()])
            y_line = m * x_line + b
            fig_ml.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Tendance', line=dict(color='white', dash='dash')))
            
            fig_ml.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.05)', font=dict(color='white'))
            st.plotly_chart(fig_ml, use_container_width=True)

    # TAB 4: Prédictions
    with t4:
        st.markdown("##### 🔮 Projection Temporelle (4 Semaines)")
        dates = pd.date_range(start=datetime.now(), periods=5)
        base = df['Score_Global'].mean() if not df.empty else 50
        
        y_opt = [base * (1 + 0.03*i) for i in range(5)]
        y_avg = [base * (1 + 0.01*i) for i in range(5)]
        y_pes = [base * (1 - 0.02*i) for i in range(5)]
        
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=dates, y=y_opt, name='Optimiste', line=dict(color='#00ff94', dash='dash')))
        fig_p.add_trace(go.Scatter(x=dates, y=y_avg, name='IA Projection', line=dict(color='#00f2ff', width=4)))
        fig_p.add_trace(go.Scatter(x=dates, y=y_pes, name='Pessimiste', line=dict(color='#ff2b2b', dash='dot')))
        
        fig_p.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                          font=dict(color='white'), xaxis_title="Semaines", yaxis_title="Score Prédit")
        st.plotly_chart(fig_p, use_container_width=True)

    # =====================================================
    # ZONE 3 : ANALYSE GLOBALE (GRAPHIQUES SUPLÉMENTAIRES)
    # =====================================================
    st.markdown('<div class="section-header">📊 DASHBOARD ANALYTIQUE</div>', unsafe_allow_html=True)
    
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("##### Distribution des Scores")
        if not df.empty:
            fig = px.histogram(df, x="Score_Global", color="Groupe", barmode="overlay", nbins=20)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
    with g2:
        st.markdown("##### Répartition par Statut")
        if not df.empty:
            fig = px.pie(df, names="Statut", hole=0.6, color="Statut", 
                       color_discrete_map={'Critique':'#ff2b2b', 'Surveillance':'#ffaa00', 'Optimal':'#00ff94'})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # ZONE 4 : TABLEAU FINAL
    # =====================================================
    st.markdown('<div class="section-header">📋 REGISTRE DES AGENTS</div>', unsafe_allow_html=True)
    
    if not df.empty:
        st.dataframe(
            df,
            column_config={
                "Nom": st.column_config.TextColumn("Agent", width="medium"),
                "Score_Global": st.column_config.ProgressColumn("Performance", format="%.1f", min_value=0, max_value=100),
                "Statut": st.column_config.SelectboxColumn("État", options=["Optimal", "Surveillance", "Critique"]),
                "Cluster": st.column_config.TextColumn("Profil IA"),
                "Maths": st.column_config.NumberColumn("Maths"),
                "Python": st.column_config.NumberColumn("Python"),
                "DeepLearning": st.column_config.NumberColumn("DL"),
            },
            hide_index=True,
            use_container_width=True,
            height=500
        )
    else:
        st.warning("Aucune donnée disponible avec les filtres actuels.")