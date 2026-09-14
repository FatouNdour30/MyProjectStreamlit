# -*- coding: utf-8 -*-
"""
SamaLearn — ULTRA FINAL PRO (VERSION COMPLÈTE CORRIGÉE)
Intelligence Adaptative, Émotionnelle & Longitudinale
Made in Africa  • L3 Data Science • Ndeye Fatou NDOUR (2025)
"""
import streamlit as st
import pandas as pd
import numpy as np
import random
import ast # C'est le nouvel import indispensable pour le parsingort time
import base64

@st.cache_data
def get_base64_image(image_path):
    """Encode une image locale en base64 pour l'intégrer directement dans du HTML."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
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
import streamlit as st
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Dans app.py, après l'importation de Pandas (pd) :

# ... (votre code continue) ... 

# ----------------------------------------------------
# CHARGEMENT DES DONNÉES HISTORIQUES
# ----------------------------------------------------
LEARNING_TRACE_DF = None

try:
    # Utilisez le nom exact de votre fichier de traces d'apprentissage
    # Fixe : spécifiez explicitement le séparateur (la virgule)
    LEARNING_TRACE_DF = pd.read_csv("learning_traces.csv", sep=',')

    # Conversion des dates pour la logique du Recommender
    if 'timestamp' in LEARNING_TRACE_DF.columns:
        LEARNING_TRACE_DF['timestamp'] = pd.to_datetime(LEARNING_TRACE_DF['timestamp'])
        
except FileNotFoundError:
    print("❌ Erreur : Le fichier 'learning_traces.csv' est introuvable.")
    LEARNING_TRACE_DF = pd.DataFrame()
# ==============================================================================
# FONCTIONS DE GESTION DU QUIZ ET NAVIGATION
# ==============================================================================

import random
from datetime import datetime # Nécessaire pour mettre à jour la date de révision
# Dans app.py :
from recommendation_engine import AdaptiveRecommender, UserProfileMinimal, LearningTrace
def reset_session_for_menu():
    """ Réinitialise les clés de session pour retourner au menu principal. """
    st.session_state.current_level = None
    st.session_state.exam_questions = []
    st.session_state.quiz_submitted = False
    st.session_state.quiz_history = []
    st.session_state.current_q_index = 0
    st.session_state.score = 0
    st.rerun()

def show_quiz_page():
    """ 
    Affiche la question en cours, gère la soumission, appelle le FormativeFeedbackEngine 
    et gère la progression du quiz.
    """
    
    # S'assurer que le moteur de feedback est initialisé
    if 'feedback_engine' not in st.session_state:
        st.session_state.feedback_engine = FormativeFeedbackEngine()
        
    q_index = st.session_state.current_q_index
    questions: List['Question'] = st.session_state.exam_questions
    current_q: 'Question' = questions[q_index]
    
    # --- HEADER ET PROGRESSION ---
    lvl_id = st.session_state.current_level
    mod_title = LEVELS[lvl_id]['title']
    
    st.markdown(f"## 📚 Évaluation : {mod_title}")
    st.markdown(f"### Question {q_index + 1} / {len(questions)}")
    st.progress((q_index + 1) / len(questions))
    st.divider()

    # --- AFFICHAGE DE LA QUESTION ---
    with st.container(border=True):
        st.markdown(f"**Niveau Bloom :** *{current_q.bloom_level.name}*")
        st.markdown(f"#### {current_q.text}")
        
        # Préparation des options pour l'affichage
        options = current_q.distractors.keys()
        all_options_list = [current_q.correct_answer] + list(options)
        random.shuffle(all_options_list) 
        
        # Le sélecteur de réponse de l'utilisateur
        user_choice = st.radio(
            "Votre réponse :",
            options=all_options_list,
            key=f"q_radio_{q_index}",
            index=None
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- LOGIQUE DE SOUMISSION ET FEEDBACK ---
    
    if st.button("Soumettre la Réponse 🎯", key=f"submit_q_{q_index}", type="primary"):
        if user_choice is None:
            st.warning("Veuillez sélectionner une réponse avant de soumettre.")
            return

        # 1. APPEL au FormativeFeedbackEngine (Action requise C)
        feedback: 'FeedbackResponse' = st.session_state.feedback_engine.generate(
            question=current_q,
            user_answer=user_choice,
            user_profile=st.session_state.user_profile
        )

        # 2. Mise à jour du score et de l'historique (Action requise C)
        is_correct = feedback.is_correct
        if is_correct:
            st.session_state.score += 1
            st.success("✅ Réponse Correcte ! Poursuivons l'apprentissage.")
        else:
            st.error("❌ Réponse Incorrecte. L'IA analyse votre erreur...")

        st.session_state.quiz_history.append({
            "q_id": current_q.id,
            "is_correct": is_correct, 
            "feedback": feedback
        })
        
        # 3. Affichage du Feedback
        with st.expander("🔬 Analyse et Feedback Adaptatif", expanded=True):
            st.markdown(f"**Diagnostic :** {feedback.details}")
            st.markdown(f"**Message IA :** *{feedback.message}*")
            st.info(f"Prochaine Action : {feedback.next_action}")

        # 4. Avancer à la question suivante ou finir (Action requise C)
        if q_index < len(questions) - 1:
            if st.button("Question Suivante ➡️", key="next_q_btn", use_container_width=True):
                st.session_state.current_q_index += 1
                st.rerun()
        else:
            # Fin du quiz
            st.session_state.quiz_submitted = True
            mod_title = LEVELS[st.session_state.current_level]['title']
            st.session_state.last_review_dates[mod_title] = datetime.now()
            st.success("🎉 Module terminé ! Cliquez ci-dessous pour le rapport final.")
            
            if st.button("Afficher le Rapport Final 🏆", key="show_dash_btn", use_container_width=True):
                # Appel à show_dashboard_final() se fera via le rerender de l'app principal
                st.rerun() 

    st.button("Annuler et Retourner au Menu", key="cancel_quiz_btn", on_click=reset_session_for_menu, use_container_width=True)
# 1. DONNÉES : LES 20 NIVEAUX D'ENTRAÎNEMENT

# À METTRE AU DÉBUT DU FICHIER (HORS DES IF/ELIF)
def show_dashboard_final():
    """ Affiche le Dashboard Final """
    st.balloons()
 # Récupération automatique des données depuis la session
    score = st.session_state.score
    history = st.session_state.quiz_history
    lvl_id = st.session_state.current_level
    
    # Sécurité pour récupérer le titre
    if lvl_id in LEVELS:
        mod_title = LEVELS[lvl_id]['title']
    else:
        mod_title = "Module"
        
    total = 15
    percent = (score / total) * 100
    
    # --- HEADER ---
    st.markdown(f"## 🚀 Rapport : {mod_title}")
    st.divider()

    # --- KPI ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score Final", f"{score} / {total}")
    with col2:
        st.metric("Précision", f"{percent:.1f}%")
    with col3:
        if score >= 12:
            st.success("🥇 EXPERT")
        elif score >= 8:
            st.warning("🥈 INTERMÉDIAIRE")
        else:
            st.error("🥉 DÉBUTANT")

    # --- PROGRESSION ---
    st.write("")
    st.write("**Maîtrise du module :**")
    st.progress(percent / 100)
    
    st.divider()
    
    # --- GRILLE VISUELLE ---
    st.subheader("🔍 Analyse des réponses")
    
    # Grille 5 colonnes x 3 lignes
    for row in range(3):
        cols = st.columns(5)
        for col in range(5):
            idx = row * 5 + col
            if idx < 15:
                is_correct = history[idx]['is_correct'] if idx < len(history) else False
                
                with cols[col]:
                    bg = "#e8f5e9" if is_correct else "#ffebee"
                    border = "green" if is_correct else "red"
                    icon = "✅" if is_correct else "❌"
                    
                    st.markdown(
                        f"""
                        <div style="background-color:{bg}; border:1px solid {border}; border-radius:8px; padding:10px; text-align:center; margin-bottom:10px;">
                            <div style="font-weight:bold; font-size:12px; color:#555;">Q{idx+1}</div>
                            <div style="font-size:22px;">{icon}</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
    
    st.divider()
    
    # --- ACTIONS ---
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("🔄 Retour au Menu Principal", key="btn_dash_return_fix", type="primary", use_container_width=True):
            st.session_state.current_level = None
            st.session_state.exam_questions = []
            st.session_state.quiz_submitted = False
            st.session_state.quiz_history = []
            st.rerun()   
# ==============================================================================

LEVELS = {
    1: {"title": "Python: Les Fondations", "icon": "🐍", "desc": "Syntaxe, Variables, Boucles", "color": "#FFD43B"},
    2: {"title": "Python Avancé", "icon": "🚀", "desc": "Générateurs, Décorateurs, OPP", "color": "#306998"},
    3: {"title": "Structures de Données", "icon": "📚", "desc": "Listes, Dicos, Sets, Complexité", "color": "#FFD43B"},
    4: {"title": "Algorithmique", "icon": "⚙️", "desc": "Tri, Recherche, Récursivité", "color": "#306998"},
    5: {"title": "NumPy & Matrices", "icon": "🔢", "desc": "Calcul vectoriel, Broadcasting", "color": "#013243"},
    6: {"title": "Pandas: Manipulation", "icon": "🐼", "desc": "DataFrames, Séries, Indexing", "color": "#150458"},
    7: {"title": "Pandas: Nettoyage", "icon": "🧹", "desc": "Missing Values, Duplicates, Types", "color": "#150458"},
    8: {"title": "Data Viz (Matplotlib)", "icon": "📊", "desc": "Plots statiques, Personnalisation", "color": "#F26627"},
    9: {"title": "Viz Interactive", "icon": "📈", "desc": "Plotly, Streamlit, Dashboards", "color": "#F26627"},
    10: {"title": "SQL: Les Bases", "icon": "🗄️", "desc": "SELECT, WHERE, ORDER BY", "color": "#00758F"},
    11: {"title": "SQL: Avancé", "icon": "🔗", "desc": "JOINS, Window Functions, CTE", "color": "#00758F"},
    12: {"title": "Probabilités", "icon": "🎲", "desc": "Lois, Bayes, Espérance", "color": "#E91E63"},
    13: {"title": "Statistiques Inférentielles", "icon": "📉", "desc": "Tests A/B, P-value, Confiance", "color": "#E91E63"},
    14: {"title": "ML: Supervisé", "icon": "🎯", "desc": "Régression, Classification, KNN", "color": "#F39C12"},
    15: {"title": "ML: Non-Supervisé", "icon": "🧩", "desc": "K-Means, PCA, Clustering", "color": "#F39C12"},
    16: {"title": "Évaluation de Modèles", "icon": "⚖️", "desc": "ROC, AUC, Precision/Recall", "color": "#F39C12"},
    17: {"title": "Deep Learning (ANN)", "icon": "🧠", "desc": "Réseaux de neurones, Backprop", "color": "#D32F2F"},
    18: {"title": "Computer Vision (CNN)", "icon": "👁️", "desc": "Images, Convolution, Pooling", "color": "#D32F2F"},
    19: {"title": "NLP & Transformers", "icon": "🗣️", "desc": "Texte, Tokenization, BERT/GPT", "color": "#9C27B0"},
    20: {"title": "MLOps & Production", "icon": "🚢", "desc": "Docker, Git, API, Cloud", "color": "#555555"}
}

# Fonction Dashboard (Reste identique à la précédente réponse, pas besoin de la changer)
# (Assurez-vous d'avoir gardé la fonction show_dashboard_final du message précédent)

# =====================================================
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import logging

# Configuration du logging
logger = logging.getLogger(__name__)

# --- 1. DÉFINITIONS & ENUMS (STRUCTURE) ---

class BloomLevel(Enum):
    REMEMBER = 1    # Mémoriser
    UNDERSTAND = 2  # Comprendre
    APPLY = 3       # Appliquer
    ANALYZE = 4     # Analyser
    EVALUATE = 5    # Évaluer
    CREATE = 6      # Créer

class ErrorType(Enum):
    NONE = "none"
    CONCEPTUAL = "conceptual"   # Ne comprend pas le 'pourquoi'
    PROCEDURAL = "procedural"   # Erreur dans la méthode/calcul
    ATTENTION = "attention"     # Faute d'inattention probable
    MISCONCEPTION = "misconception" # Croyance erronée ancrée

@dataclass
class Question:
    id: str
    text: str
    concept: str
    correct_answer: str
    distractors: Dict[str, ErrorType] # Map les mauvaises réponses à un type d'erreur
    bloom_level: BloomLevel
    resources: Dict[str, str] # Liens vers vidéos/cours

@dataclass
class UserProfile:
    id: str
    competences: Dict[str, float] # Score entre 0.0 et 1.0
    
    def update_competence(self, concept: str, delta: float):
        """Mise à jour sécurisée avec bornes."""
        current = self.competences.get(concept, 0.5)
        new_score = max(0.0, min(1.0, current + delta))
        self.competences[concept] = round(new_score, 3)

@dataclass
class FeedbackResponse:
    """Structure de sortie standardisée pour l'UI."""
    is_correct: bool
    message: str
    details: str
    resources: List[str] = field(default_factory=list)
    next_action: str = ""
    bloom_adjustment: str = "" # Indique si on monte ou descend dans la taxonomie

# --- 2. MOTEUR DE FEEDBACK (LOGIQUE) ---

class FormativeFeedbackEngine:
    """Générateur de feedback basé sur les sciences cognitives."""

    def __init__(self):
        # Ici, on pourrait injecter des services de recommandation externes
        pass

    def _diagnose_error(self, question: Question, user_answer: str) -> ErrorType:
        """Identifie le type d'erreur basé sur les distracteurs prédéfinis."""
        if user_answer == question.correct_answer:
            return ErrorType.NONE
        # Retourne le type d'erreur associé à la mauvaise réponse, ou PROCÉDURAL par défaut
        return question.distractors.get(user_answer, ErrorType.PROCEDURAL)

    def _get_scaffolding(self, error_type: ErrorType, question: Question) -> tuple[str, List[str]]:
        """Génère l'étayage (scaffolding) selon le type d'erreur."""
        
        if error_type == ErrorType.CONCEPTUAL:
            msg = f"🔍 Il semble y avoir une confusion sur le concept de **{question.concept}**."
            res = [question.resources.get('video', '#'), question.resources.get('schema', '#')]
            return msg, res
            
        elif error_type == ErrorType.PROCEDURAL:
            msg = "⚙️ La méthode semble comprise, mais il y a une erreur dans l'exécution."
            res = [question.resources.get('step_by_step', '#')]
            return msg, res
            
        elif error_type == ErrorType.MISCONCEPTION:
            msg = "⚠️ Attention, c'est un piège classique (intuition trompeuse)."
            return msg, []
            
        return "❌ Réponse incorrecte. Prenons le temps de revoir la consigne.", []

    def generate(self, question: Question, user_answer: str, user_profile: UserProfile) -> FeedbackResponse:
        """
        Fonction principale.
        Génère un feedback formatif et met à jour le modèle apprenant.
        """
        try:
            error_type = self._diagnose_error(question, user_answer)
            is_correct = (error_type == ErrorType.NONE)

            # 1. Gestion de la réponse CORRECTE (Renforcement & Extension)
            if is_correct:
                # Logique adaptative : Si l'élève maîtrise, on propose le niveau supérieur
                next_bloom = BloomLevel(min(question.bloom_level.value + 1, 6))
                
                # Mise à jour profil (Renforcement positif)
                user_profile.update_competence(question.concept, 0.05)
                
                return FeedbackResponse(
                    is_correct=True,
                    message=f"✅ Excellent ! Vous maîtrisez {question.concept}.",
                    details=f"Vous avez réussi un exercice de niveau {question.bloom_level.name}.",
                    next_action=f"Défi suivant : Passons au niveau {next_bloom.name} (Analyse/Création).",
                    bloom_adjustment="UP",
                    resources=[question.resources.get('advanced', '#')]
                )

            # 2. Gestion de la réponse INCORRECTE (Remédiation)
            else:
                details_msg, resources = self._get_scaffolding(error_type, question)
                
                # Mise à jour profil (Léger malus pour ajuster le niveau futur)
                # On punit moins une erreur procédurale qu'une erreur conceptuelle
                penalty = -0.05 if error_type == ErrorType.PROCEDURAL else -0.10
                user_profile.update_competence(question.concept, penalty)

                return FeedbackResponse(
                    is_correct=False,
                    message=f"Pas tout à fait. Analysons cela ensemble.",
                    details=details_msg,
                    resources=resources,
                    next_action="Réessayez avec cet indice ou consultez la ressource.",
                    bloom_adjustment="STAY"
                )

        except Exception as e:
            logger.error(f"Erreur génération feedback: {e}")
            return FeedbackResponse(False, "Une erreur est survenue.", str(e))

# --- 3. EXEMPLE D'UTILISATION ---

# Données fictives (Mock)
q1 = Question(
    id="Q101",
    text="Quelle est la dérivée de x^2 ?",
    concept="Dérivées",
    correct_answer="2x",
    distractors={
        "x": ErrorType.CONCEPTUAL,       # Confond dérivée et division ?
        "2": ErrorType.PROCEDURAL,       # Oubli de la variable
        "x^3/3": ErrorType.MISCONCEPTION # Confond avec l'intégrale
    },
    bloom_level=BloomLevel.APPLY,
    resources={
        "video": "http://vid.eo/concept",
        "step_by_step": "http://doc/steps",
        "advanced": "http://doc/optimisation"
    }
)

student = UserProfile(id="SUD-001", competences={"Dérivées": 0.4})
engine = FormativeFeedbackEngine()

# Simulation : L'élève répond faux (misconception)
user_ans = "x^3/3" 
feedback = engine.generate(q1, user_ans, student)

print("--- FEEDBACK ÉLÈVE ---")
print(f"Statut: {'✅ Bravo' if feedback.is_correct else '🔴 A revoir'}")
print(f"Message: {feedback.message}")
print(f"Détails: {feedback.details}")
print(f"Action: {feedback.next_action}")
print(f"Ressources: {feedback.resources}")
print("-" * 20)
print(f"Nouveau profil compétence 'Dérivées': {student.competences['Dérivées']}")

# =====================================================
# ==============================================================================
# 3. FONCTIONNALITÉS PRINCIPALES : LE QUIZ ADAPTATIF
# ==============================================================================

def show_quiz_page():
    """ Affiche la question en cours, gère la soumission et le feedback. """
    
    # Initialisation du moteur de feedback (à faire une fois)
    if 'feedback_engine' not in st.session_state:
        st.session_state.feedback_engine = FormativeFeedbackEngine()
        
    # Vérification des états requis
    if not st.session_state.exam_questions:
        st.error("Erreur: Aucune question chargée pour ce module.")
        st.session_state.current_level = None
        st.rerun()
        return

    q_index = st.session_state.current_q_index
    questions: List[Question] = st.session_state.exam_questions
    current_q: Question = questions[q_index]
    
    # --- Affichage du Header du Quiz ---
    lvl_id = st.session_state.current_level
    mod_title = LEVELS[lvl_id]['title']
    
    st.markdown(f"## 📚 Évaluation : {mod_title}")
    st.markdown(f"### Question {q_index + 1} / {len(questions)}")
    st.progress((q_index) / len(questions))
    st.divider()

    # --- Affichage de la Question ---
    with st.container(border=True):
        st.markdown(f"**Niveau Bloom :** *{current_q.bloom_level.name}*")
        st.markdown(f"#### {current_q.text}")
        
        # Le sélecteur de réponse
        options = current_q.distractors.keys()
        all_options_list = [current_q.correct_answer] + list(options)
        
        # Utiliser des boutons radio pour le choix
        user_choice = st.radio(
            "Votre réponse :",
            options=all_options_list,
            key=f"q_radio_{q_index}",
            index=None # Aucune sélection par défaut
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Zone de Soumission et Feedback ---
    
    if st.button("Soumettre la Réponse 🎯", key=f"submit_q_{q_index}", type="primary"):
        if user_choice is None:
            st.warning("Veuillez sélectionner une réponse avant de soumettre.")
            return

        # 1. Génération du Feedback Adaptatif
        feedback: FeedbackResponse = st.session_state.feedback_engine.generate(
            question=current_q,
            user_answer=user_choice,
            user_profile=st.session_state.user_profile
        )

        # 2. Traitement du Score et de l'Historique
        is_correct = feedback.is_correct
        if is_correct:
            st.session_state.score = st.session_state.score + 1
            st.success("✅ Réponse Correcte ! Poursuivons l'apprentissage.")
        else:
            st.error("❌ Réponse Incorrecte. L'IA analyse votre erreur...")

        # Ajout à l'historique du quiz pour le tableau de bord
        st.session_state.quiz_history.append({
            "q_id": current_q.id,
            "is_correct": is_correct,
            "user_ans": user_choice,
            "feedback": feedback
        })
        
        # 3. Affichage du Feedback Formatif
        with st.expander("🔬 Analyse et Feedback Adaptatif", expanded=True):
            st.markdown(f"**Message IA :** *{feedback.message}*")
            st.markdown(f"**Diagnostic :** {feedback.details}")
            
            if feedback.resources:
                st.markdown("**Ressources suggérées pour la remédiation/l'extension :**")
                for res in feedback.resources:
                    st.markdown(f"- 🔗 {res}")
            
            st.info(f"Prochaine Action Adaptative : {feedback.next_action}")
            st.markdown(f"*(Ajustement du profil de compétence '{current_q.concept}' : **{feedback.bloom_adjustment}**)*")

        
        # 4. Bouton pour passer à la question suivante/finir
        if q_index < len(questions) - 1:
            if st.button("Question Suivante ➡️", key="next_q_btn", use_container_width=True):
                st.session_state.current_q_index += 1
                st.rerun()
        else:
            # Fin du quiz
            st.session_state.quiz_submitted = True
            st.session_state.last_review_dates[mod_title] = datetime.now() # Mise à jour pour le Recommender
            st.success("🎉 Module terminé ! Cliquez ci-dessous pour le rapport final.")
            
            if st.button("Afficher le Rapport Final 🏆", key="show_dash_btn", use_container_width=True):
                st.rerun()

    # Afficher le bouton Retour au Menu si rien n'est soumis
# ==============================================================================
# 3. FONCTIONNALITÉS PRINCIPALES : LE QUIZ ADAPTATIF
# (Nécessite les classes Question et FormativeFeedbackEngine)

# ==============================================================================

def reset_session_for_menu():
    """ Fonction utilitaire pour retourner au menu principal. """
    st.session_state.current_level = None
    st.session_state.exam_questions = []
    st.session_state.quiz_submitted = False
    st.session_state.quiz_history = []
    
def show_quiz_page():
    """ Affiche la question en cours, gère la soumission et le feedback. """
    
    # Initialisation du moteur de feedback
    if 'feedback_engine' not in st.session_state:
        # Assurez-vous que la classe FormativeFeedbackEngine est définie
        st.session_state.feedback_engine = FormativeFeedbackEngine() 
        
    q_index = st.session_state.current_q_index
    questions: List['Question'] = st.session_state.exam_questions
    current_q: 'Question' = questions[q_index]
    
    # --- Affichage du Header du Quiz ---
    lvl_id = st.session_state.current_level
    mod_title = LEVELS[lvl_id]['title']
    
    st.markdown(f"## 📚 Évaluation : {mod_title}")
    st.markdown(f"### Question {q_index + 1} / {len(questions)}")
    st.progress((q_index) / len(questions))
    st.divider()

    # --- Affichage de la Question ---
    with st.container(border=True):
        st.markdown(f"**Niveau Bloom :** *{current_q.bloom_level.name}*")
        st.markdown(f"#### {current_q.text}")
        
        # Le sélecteur de réponse (Liste des options : Correcte + Distracteurs)
        options = current_q.distractors.keys()
        all_options_list = [current_q.correct_answer] + list(options)
        random.shuffle(all_options_list) # Mélanger pour que la bonne réponse ne soit pas toujours la première
        
        user_choice = st.radio(
            "Votre réponse :",
            options=all_options_list,
            key=f"q_radio_{q_index}",
            index=None
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Zone de Soumission et Feedback ---
    
    if st.button("Soumettre la Réponse 🎯", key=f"submit_q_{q_index}", type="primary"):
        if user_choice is None:
            st.warning("Veuillez sélectionner une réponse avant de soumettre.")
            return

        # 1. Génération du Feedback Adaptatif (appel au moteur)
        feedback: 'FeedbackResponse' = st.session_state.feedback_engine.generate(
            question=current_q,
            user_answer=user_choice,
            user_profile=st.session_state.user_profile
        )

        # 2. Mise à jour de l'État (Score et Historique)
        is_correct = feedback.is_correct
        if is_correct:
            st.session_state.score += 1
            st.success("✅ Réponse Correcte ! Poursuivons l'apprentissage.")
        else:
            st.error("❌ Réponse Incorrecte. L'IA analyse votre erreur...")

        st.session_state.quiz_history.append({"is_correct": is_correct, "feedback": feedback})
        
        # 3. Affichage du Feedback Formatif
        with st.expander("🔬 Analyse et Feedback Adaptatif", expanded=True):
            st.markdown(f"**Diagnostic :** {feedback.details}")
            st.markdown(f"**Message IA :** *{feedback.message}*")
            if feedback.resources:
                st.markdown("**Ressources suggérées :**")
                for key, link in feedback.resources.items():
                     st.markdown(f"- 🔗 {key}: {link}") # Afficher les ressources
            st.info(f"Prochaine Action : {feedback.next_action}")

        # 4. Bouton pour passer à la question suivante/finir
        if q_index < len(questions) - 1:
            if st.button("Question Suivante ➡️", key="next_q_btn", use_container_width=True):
                st.session_state.current_q_index += 1
                st.rerun()
        else:
            st.session_state.quiz_submitted = True
            mod_title = LEVELS[st.session_state.current_level]['title']
            st.session_state.last_review_dates[mod_title] = datetime.now()
            st.success("🎉 Module terminé ! Cliquez ci-dessous pour le rapport final.")
            
            if st.button("Afficher le Rapport Final 🏆", key="show_dash_btn", use_container_width=True):
                st.rerun()

    st.button("Annuler et Retourner au Menu", key="cancel_quiz_btn", on_click=reset_session_for_menu, use_container_width=True)   
# 1. CONFIGURATION & CONSTANTES
# =====================================================
st.set_page_config(page_title="SamaLearn Ultra", page_icon="💠", layout="wide")

import logging
from statistics import mean
from typing import List, Dict

# --- DÉBUT DU MOTEUR DE RECOMMANDATION (À COLLER DANS APP.PY) ---

# Configuration basique des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RecommenderEngine")

class RecommenderSystem:
    """
    Moteur de recommandation pédagogique adaptatif intégré.
    """

    def __init__(self, curriculum: Dict):
        self.curriculum = curriculum

    def get_recommendations(self, user_profile: Dict, historical_traces: List[Dict]) -> List[Dict]:
        try:
            competences = user_profile.get("competences", {})
            last_reviews = user_profile.get("last_review", {})
            
            if not competences:
                return self._cold_start()

            # 1. Calcul de la difficulté cible (Zone Proximale de Développement)
            # On vise un peu au-dessus de la moyenne de l'élève
            vals = list(competences.values())
            avg_competence = sum(vals) / len(vals) if vals else 0
            target_difficulty = avg_competence + 0.15 

            scored_modules = []
            
            # On regarde tous les modules disponibles
            for module in self.curriculum.keys():
                score = 0
                current_mastery = competences.get(module, 0.0)
                
                # A. Priorité aux compétences faibles (40%)
                score += (1.0 - current_mastery) * 0.4
                
                # B. Révision espacée (30%)
                days_elapsed = 0
                if module in last_reviews:
                    # Gestion format date
                    l_date = last_reviews[module]
                    if isinstance(l_date, str):
                        try:
                            l_date = datetime.strptime(l_date, "%Y-%m-%d %H:%M:%S")
                        except:
                            l_date = datetime.now()
                    
                    if isinstance(l_date, datetime):
                        days_elapsed = (datetime.now() - l_date).days
                        if days_elapsed > 3: # Seuil d'oubli
                            score += 0.3
                else:
                    # Jamais vu = priorité boostée
                    score += 0.1

                # C. Niveau adapté (20%)
                mod_lvl = self.curriculum.get(module, {}).get("lvl", 1) * 0.25 # Normalisation approx 0-1
                if abs(mod_lvl - target_difficulty) < 0.2:
                    score += 0.2

                # Explication simple
                reason = self._generate_reason(module, current_mastery, days_elapsed)
                
                scored_modules.append({
                    "module": module,
                    "score": score,
                    "reason": reason
                })

            # Retourner le Top 1
            return sorted(scored_modules, key=lambda x: x["score"], reverse=True)[:1]

        except Exception as e:
            print(f"Erreur Recommender: {e}")
            return []

    def _generate_reason(self, module, mastery, days):
        if days > 5: return f"Cela fait {days} jours sans pratique sur '{module}'. Révision conseillée !"
        if mastery < 0.3: return f"Renforcez vos bases sur '{module}' pour progresser."
        if mastery > 0.8: return f"Vous excellez en '{module}', passez au niveau supérieur !"
        return f"Le module '{module}' est parfaitement adapté à votre niveau actuel."

    def _cold_start(self):
        return [{"module": "Introduction", "reason": "Bienvenue ! Commencez par les bases.", "score": 1}]

# --- FIN DU MOTEUR DE RECOMMANDATION ---
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
# GESTION DU LOGIN ET INITIALISATION DE SESSION
# =====================================================

# 1. Initialisation de l'état de session si les clés n'existent pas
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None

# 2. LOGIQUE DE CONNEXION : Si l'utilisateur n'est pas connecté
if st.session_state.user_profile is None:
    
    # Création des deux colonnes (Texte vs Image)
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
        st.markdown("", unsafe_allow_html=True)
        nom_saisi = st.text_input("Username", placeholder="Entrez votre Identifiant Agent...", key="login_name_input")
        
        # Espace
        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
        
        # Bouton (Action)
        c_btn, c_void = st.columns([1, 1.5]) 
        with c_btn:
            if st.button("Lancer la Session ➜", use_container_width=True):
                # 🚨 CORRECTION D'INDENTATION : L'imbrication est correcte maintenant
                if nom_saisi:
                    # Petit effet de loading "IA"
                    with st.spinner("Analyse du profil neuronal..."):
                        import time
                        time.sleep(1.2)
                        
                    # Définition des compétences initiales pour tous les LEVELS
                    initial_competences = {
                        LEVELS[lvl_id]['title']: 0.5
                        for lvl_id in LEVELS
                    }
                    
                    # 🚨 CORRECTION CRITIQUE : Création de la DATACLASS UserProfile
                    st.session_state.user_profile = UserProfile(
                        id=nom_saisi,
                        competences=initial_competences
                    )
                    
                    # Initialisation des clés de navigation pour le Recommender System
                    if 'last_review_dates' not in st.session_state:
                         st.session_state.last_review_dates = {}
                         
                    # Initialisation complète de l'état de navigation pour le rerender
                    st.session_state.current_level = None
                    st.session_state.quiz_submitted = False
                    st.session_state.current_q_index = 0
                    st.session_state.score = 0
                    st.session_state.quiz_history = []
                    
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
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <img src="data:image/jpeg;base64,{get_base64_image('assets/login_illustration.jpg')}"
             width="100%" class="floating-img">
        """, unsafe_allow_html=True)
        
    # 🚨 GARDE-FOU ESSENTIEL : Arrêter l'exécution ici si non connecté
    st.stop()

# =====================================================
# LOGIQUE POST-LOGIN (Début de l'application réelle)
# =====================================================
# 🚨 Correction de l'accès aux données (on utilise le .id de la dataclass)
user = st.session_state.user_profile
nom = user.id # Maintenant sécurisé car user_profile est un objet UserProfile

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
nom = user.id

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

# =====================================================
# 1. PAGE: ACCUEIL DASHBOARD — SAMA LEARN AI
# =====================================================
if st.session_state.page == "accueil": # Ou 'elif' si c'est à la suite d'autres pages
    
    # --- IMPORT NÉCESSAIRE ---
    from scipy.optimize import curve_fit

    # --- DÉFINITION DE LA FONCTION DE PROJECTION (AVEC STYLE) ---
    def create_learning_curve_graph():
        """Courbe d'apprentissage selon la Loi de Puissance (Power Law)"""
        # 1. Simulation de données (puisque user_traces n'est pas défini dans ce snippet)
        # On simule une progression réaliste
        x_data = np.linspace(1, 50, 20) # Heures cumulées
        # Score = 15 * x^0.4 (loi de puissance) + bruit aléatoire
        y_data = 15 * np.power(x_data, 0.4) + np.random.normal(0, 2, len(x_data))
        y_data = np.clip(y_data, 0, 100) # Borner entre 0 et 100
        
        practice_hours = x_data
        scores = y_data
        
        # 2. Modélisation : Performance = a * (pratique)^b
        def power_law(x, a, b): 
            return a * np.power(x, b)
        
        try:
            # Ajustement du modèle (Curve Fit)
            params, _ = curve_fit(power_law, practice_hours, scores, p0=[10, 0.5], maxfev=5000)
            
            fig = go.Figure()
            
            # Points réels
            fig.add_trace(go.Scatter(
                x=practice_hours, y=scores, 
                mode='markers', name='Données réelles',
                marker=dict(color='#F72585', size=8, line=dict(width=1, color='white'))
            ))
            
            # Projection IA
            x_pred = np.linspace(0, max(practice_hours)*1.5, 100)
            y_pred = power_law(x_pred, *params)
            
            fig.add_trace(go.Scatter(
                x=x_pred, y=y_pred, 
                mode='lines', name='Projection IA',
                line=dict(color='#00F0FF', width=3, shape='spline')
            ))
            
            # Application du style Neon
            fig = style_fig(fig)
            fig.update_layout(
                title="Modèle Cognitif (Loi de Puissance)",
                xaxis_title="Heures de Pratique",
                yaxis_title="Score de Compétence",
                height=250,
                legend=dict(orientation="h", y=1.1)
            )
            return fig
            
        except Exception as e:
            return go.Figure().add_annotation(text="Données insuffisantes pour projection", showarrow=False)

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
        # --- GRAPH 9: BARRES ---
        fig_bar = go.Figure(go.Bar(
            x=[85, 62, 94, 78], y=['Python', 'SQL', 'Machine Learning', 'Statistics'], orientation='h',
            marker=dict(
                color=['#00F0FF', '#333', '#7B61FF', '#333'],
                cornerradius=5
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

    # ================= ZONE 5 : PROJECTION NEURONALE (NOUVEAU) =================
    st.markdown("---")
    st.markdown('<div class="card-header"><span>🔮 Projection Neuronale (Loi de Puissance)</span><span class="badge badge-live">AI MODEL</span></div>', unsafe_allow_html=True)
    
    # Appel de la nouvelle fonction intégrée
    fig_power = create_learning_curve_graph()
    st.plotly_chart(fig_power, use_container_width=True)
        
        
elif st.session_state.page == "cours":
    st.title("⚡ Mes Cours")
    st.markdown('<div class="glass-panel"><h3>Modules d\'apprentissage</h3><p>Contenu personnalisé selon votre profil.</p></div>', unsafe_allow_html=True)
elif st.session_state.page == "exercices":
    
    # --- A. GESTION DE L'ÉTAT (STATE MANAGEMENT) ---
    if "current_level" not in st.session_state: 
        st.session_state.current_level = None
    if "quiz_submitted" not in st.session_state: 
        st.session_state.quiz_submitted = False
    if "q_index" not in st.session_state: 
        st.session_state.q_index = 0
    if "score" not in st.session_state: 
        st.session_state.score = 0
    if "answer_validated" not in st.session_state: 
        st.session_state.answer_validated = False


  

    # --------------------------------------------------------------------------
    # VUE 2 : LA CARTE DES 20 NIVEAUX (MENU PRINCIPAL)
    # --------------------------------------------------------------------------
    
# =====================================================
# =====================================================
# 3. PAGE: CENTRE D'EXAMEN (FINAL VERSION: ALL MODULES + PRO CERTIFICATE)
# =====================================================
# =====================================================
# 3. PAGE: CENTRE D'EXAMEN (FINAL VERSION: ALL MODULES + PRO CERTIFICATE + CSV EXPORT)
# =====================================================
elif st.session_state.page == "examen":
    
    # --- IMPORTS LOCAUX ---
    import time
    import numpy as np
    import hashlib
    import datetime
    import random
    # [AJOUT] Import Pandas pour l'export CSV
    import pandas as pd
    from scipy.optimize import minimize

    # VERIFICATION DE LA LIBRAIRIE PDF
    try:
        from fpdf import FPDF
    except ImportError:
        st.error("⚠️ Erreur Critique : La librairie 'fpdf' n'est pas installée. Veuillez exécuter : pip install fpdf")
        st.stop()

    # =========================================================================
    # 0. CONFIGURATION & UTILITAIRES
    # =========================================================================
    
    # Fonction de nettoyage des caractères pour FPDF (Latin-1)
    def clean_text(text):
        if not isinstance(text, str): return str(text)
        # Remplace les caractères non supportés par FPDF standard
        replacements = {
            "’": "'", "–": "-", "“": '"', "”": '"', "œ": "oe", "€": "EUR"
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text.encode('latin-1', 'replace').decode('latin-1')

    # Générateur de feedbacks IA
    def generate_ai_feedback(tag, score):
        feedbacks = {
            "strength": [
                "Architecture modulaire et scalable detectee.",
                "Excellente maitrise des structures de donnees.",
                "Gestion robuste des exceptions et cas limites.",
                "Code propre, respectant scrupuleusement les standards."
            ],
            "weakness": [
                "L'optimisation memoire (complexite spatiale) est perfectible.",
                "La documentation des fonctions critiques est manquante.",
                "Attention a la complexite cyclomatique dans les boucles.",
                "Gestion des dependances externes a securiser."
            ],
            "quotes": [
                "Talk is cheap. Show me the code. - Linus Torvalds",
                "Code is like humor. When you have to explain it, its bad.",
                "Simplicity is the soul of efficiency. - Austin Freeman",
                "Quality is not an act, it is a habit. - Aristotle"
            ],
            "recommendation": [
                "Approfondir les Design Patterns (Gang of Four).",
                "Explorer l'asynchronisme pour la haute performance.",
                "Implementer une pipeline CI/CD plus stricte.",
                "Contribuer a des projets Open Source complexes."
            ]
        }
        
        # Logique de sélection
        strength = feedbacks["strength"][0] if score > 1.5 else feedbacks["strength"][1]
        
        return {
            "strength": strength,
            "weakness": random.choice(feedbacks["weakness"]),
            "quote": random.choice(feedbacks["quotes"]),
            "recommendation": random.choice(feedbacks["recommendation"]),
            "technical_audit": f"Code Quality: {random.randint(94,99)}/100 | Security Scan: PASSED"
        }

    # =========================================================================
    # 0. BASE DE DONNÉES DU CONTENU (15 MODULES COMPLETS)
    # =========================================================================
    CONTENT_DB = {
        "Python": {
            "questions": [
                {"q": "Quelle est la différence principale entre une liste et un tuple en Python ?", "opts": ["Les tuples sont immutables, listes modifiables", "Les listes ne contiennent que des nombres", "Les tuples n'ont pas d'index", "Les listes sont toujours triées"], "idx": 0, "b": -1.5},
                {"q": "Que fait le mot-clé @staticmethod dans une classe ?", "opts": ["Crée une méthode liée à l’instance", "Crée une méthode qui n’accède ni à self ni à cls", "Remplace le constructeur __init__", "Transforme une fonction en propriété"], "idx": 1, "b": 0.5},
                {"q": "Quel pattern garantit qu’une classe n’a qu’une seule instance ?", "opts": ["Factory", "Observer", "Singleton", "Strategy"], "idx": 2, "b": 0.0},
                {"q": "Quelle est la bonne manière de gérer des exceptions spécifiques ?", "opts": ["Utiliser if à la place", "try/except en ciblant l’exception précise", "Ignorer les erreurs", "Ré-essayer sans limites"], "idx": 1, "b": -1.0},
                {"q": "Pourquoi utiliser des décorateurs en Python ?", "opts": ["Pour exécuter en parallèle", "Modifier le comportement d’une fonction sans changer son code", "Créer des classes anonymes", "Écrire des commentaires"], "idx": 1, "b": 1.0},
                {"q": "Que contient un fichier __init__.py dans un package ?", "opts": ["Code DB", "Indicateur de package Python et init code", "Variables OS", "Rien (obsolète)"], "idx": 1, "b": -0.5},
                {"q": "Le « duck typing » signifie :", "opts": ["Forcer les types", "Fonctions prédéfinies", "S’intéresser au comportement plutôt qu’au type", "Compiler avant exécution"], "idx": 2, "b": 1.5},
                {"q": "Différence deepcopy vs shallow copy ?", "opts": ["deepcopy copie récursivement, shallow copie la référence", "shallow est plus lent", "deepcopy = affectation simple", "Aucune"], "idx": 0, "b": 1.2},
                {"q": "Pourquoi écrire des tests unitaires ?", "opts": ["Rallonger le code", "Valider comportements et empêcher régressions", "Remplacer la doc", "Empêcher le déploiement"], "idx": 1, "b": -2.0},
                {"q": "Qu’est-ce que le principe SOLID ?", "opts": ["Protocole réseau", "5 principes pour classes maintenables", "Base NoSQL", "Style de variable"], "idx": 1, "b": 0.8},
            ],
            "tasks": [
                "Mini-projet API : Créer une API REST (CRUD) modulaire + README et tests unitaires.",
                "Design Pattern Report : Implémenter 2 patterns (ex: Singleton, Factory) avec UML.",
                "Qualité & Tests : Fournir un dossier tests/ avec coverage report."
            ]
        },
        "DSA": {
            "questions": [
                {"q": "Complexité moyenne du Quicksort ?", "opts": ["O(n)", "O(n log n)", "O(n²)", "O(log n)"], "idx": 1, "b": 0.5},
                {"q": "Quelle structure utilise LIFO ?", "opts": ["File", "Pile (Stack)", "Arbre", "Graphe"], "idx": 1, "b": -1.5},
                {"q": "Quand privilégier un dictionnaire (hash) ?", "opts": ["Recherche en O(1) par clé", "Maintenir ordre trié", "Stocker grand texte", "Itérer en O(n²)"], "idx": 0, "b": -0.5},
                {"q": "Algorithme de plus court chemin (poids positifs) ?", "opts": ["Dijkstra", "Kruskal", "Prim", "Bubble sort"], "idx": 0, "b": 0.8},
                {"q": "BFS est utilisé pour :", "opts": ["Trouver cycles", "Explorer niveau par niveau", "Trier nœuds", "Compresser"], "idx": 1, "b": 0.2},
                {"q": "Qu’est-ce qu’un heap permet d’obtenir rapidement ?", "opts": ["Min/Max en O(1)", "Recherche par nom", "Copie profonde", "Itération ordonnée"], "idx": 0, "b": 1.5},
                {"q": "Complexité recherche linéaire ?", "opts": ["O(1)", "O(n)", "O(n log n)", "O(n²)"], "idx": 1, "b": -2.5},
                {"q": "Quel algo est « greedy » ?", "opts": ["Dijkstra", "Kruskal", "Merge sort", "Binary search"], "idx": 1, "b": 1.0},
                {"q": "Détecter un cycle dans un graphe dirigé ?", "opts": ["Tri topologique / DFS marquage", "Intersection sets", "Impossible", "BFS uniquement"], "idx": 0, "b": 2.0},
                {"q": "Table de hachage vs Arbre de recherche ?", "opts": ["Ordre important", "Accès rapide clé sans ordre", "Parcours ordonnés", "Peu d'éléments"], "idx": 1, "b": 0.5},
            ],
            "tasks": [
                "Implement & Analyse : Quicksort vs MergeSort (temps sur datasets variés).",
                "Projet Algorithme : Problème d’optimisation (planification) + analyse complexité.",
                "Challenge Graphe : Détecter composantes connexes et plus court chemin."
            ]
        },
        "DataAnalysis": {
            "questions": [
                {"q": "Qu’est-ce que le broadcasting en NumPy ?", "opts": ["Mélanger fichiers", "Étendre dimensions compatibles pour opérations", "Trier array", "Compiler tableau"], "idx": 1, "b": 1.0},
                {"q": "a.shape pour np.array([1,2,3]) ?", "opts": ["(3,)", "(1,3)", "(3,1)", "None"], "idx": 0, "b": -1.0},
                {"q": ".loc vs .iloc ?", "opts": [".loc=position, .iloc=label", ".loc=label, .iloc=position", "Identiques", "Suppression"], "idx": 1, "b": -0.5},
                {"q": "Méthode pour supprimer les NaN ?", "opts": ["dropna()", "remove_na()", "clear()", "nan_delete()"], "idx": 0, "b": -2.0},
                {"q": "Que fait groupby ?", "opts": ["Trie colonnes", "Regroupe par clé pour agrégations", "Convertit en liste", "Supprime doublons"], "idx": 1, "b": 0.0},
                {"q": "np.reshape sert à :", "opts": ["Changer forme sans modifier données", "Trier", "Indexer", "Remplacer NaN"], "idx": 0, "b": 0.5},
                {"q": "Calculer corrélation entre colonnes ?", "opts": ["df.corr()", "df.join()", "df.concat()", "df.merge()"], "idx": 0, "b": -1.5},
                {"q": "Avantage NumPy sur listes ?", "opts": ["Flexibilité", "Performance calcul vectoriel", "Moins de fonctions", "Plus lent"], "idx": 1, "b": -1.0},
                {"q": "Qu’est-ce qu’un mask ?", "opts": ["Fonction math", "Booléen pour filtrer", "Affichage", "Type objet"], "idx": 1, "b": 0.8},
                {"q": ".pivot_table sert à :", "opts": ["Créer table pivot (synthèse)", "Supprimer colonnes", "Fusionner", "Convertir array"], "idx": 0, "b": 1.2},
            ],
            "tasks": [
                "Nettoyage complet : Dataset sale -> imputation, standardisation + Notebook.",
                "Analyse exploratoire : Rapport EDA (stats, corrs, graphs) + insights.",
                "Transformation pipeline : ETL simple (ingestion -> clean -> aggr -> export)."
            ]
        },
        "DataViz": {
             "questions": [
                {"q": "Quand privilégier un scatterplot ?", "opts": ["Relation 2 variables continues", "Résumé distribution", "Série temporelle", "Catégoriel"], "idx": 0, "b": -1.0},
                {"q": "Qu’est-ce qu’une heatmap ?", "opts": ["Distribution univariée", "Matrice colorée (ex: corrélation)", "Histogramme bivarié", "Nuage de points"], "idx": 1, "b": -0.5},
                {"q": "Librairie haut niveau sur Matplotlib ?", "opts": ["Seaborn", "Requests", "NumPy", "Flask"], "idx": 0, "b": -2.0},
                {"q": "Pourquoi ajouter une légende ?", "opts": ["Décorer", "Associer éléments et séries", "Augmenter taille", "Trier"], "idx": 1, "b": -2.5},
                {"q": "Que fait plt.tight_layout() ?", "opts": ["Resserre police", "Ajuste marges (évite chevauchement)", "Compresse image", "Rien"], "idx": 1, "b": 0.5},
                {"q": "Quel plot pour distribution continue ?", "opts": ["barplot", "histogram / kde", "pie chart", "network"], "idx": 1, "b": -0.8},
                {"q": "Pourquoi utiliser des subplots ?", "opts": ["Plusieurs graphiques même figure", "Supprimer axes", "Agrandir titre", "Stocker données"], "idx": 0, "b": 0.0},
                {"q": "Graphique pour parts relatives ?", "opts": ["Line plot", "Pie chart", "Scatter", "Heatmap"], "idx": 1, "b": -1.5},
                {"q": "Sauvegarder en PNG ?", "opts": ["savefig()", "export()", "write_png()", "to_image()"], "idx": 0, "b": -1.0},
                {"q": "Le storytelling en data viz ?", "opts": ["Code", "Structurer pour guider décision", "Nettoyage", "Base de données"], "idx": 1, "b": 1.0},
            ],
            "tasks": [
                "Dashboard interactif : 3 graphiques (trend, dist, corr) + export.",
                "Storytelling report : Slides/PDF expliquant un insight business.",
                "Visual Audit : Refonte d'un mauvais graphique + justification."
            ]
        },
        "SQL": {
             "questions": [
                {"q": "INNER JOIN retourne :", "opts": ["Tout", "Correspondance dans les deux tables", "Gauche seulement", "Droite seulement"], "idx": 1, "b": -1.0},
                {"q": "GROUP BY sert à :", "opts": ["Filtrer", "Agréger par clé", "Détruire index", "Réorganiser"], "idx": 1, "b": -0.5},
                {"q": "Qu’est-ce qu’un index DB ?", "opts": ["Fichier image", "Structure accélérant recherches", "Trigger", "Vue"], "idx": 1, "b": 0.0},
                {"q": "COUNT(*) compte :", "opts": ["Valeurs non nulles", "Toutes les lignes", "Colonnes", "Rien"], "idx": 1, "b": -1.5},
                {"q": "HAVING est utilisé avec :", "opts": ["WHERE", "GROUP BY (filtre agrégats)", "ORDER BY", "JOIN"], "idx": 1, "b": 0.5},
                {"q": "CTE (WITH) sert à :", "opts": ["Requête temporaire réutilisable", "Indexer", "Supprimer table", "Importer CSV"], "idx": 0, "b": 1.0},
                {"q": "TRANSACTION garantit :", "opts": ["Vitesse", "ACID (Atomicité...)", "Commit auto", "No rollback"], "idx": 1, "b": 1.5},
                {"q": "LEFT JOIN retourne :", "opts": ["Intersection", "Tout table gauche + correspondances", "Droite seulement", "Milieu"], "idx": 1, "b": -0.8},
                {"q": "DELETE vs TRUNCATE ?", "opts": ["Identique", "DELETE=row by row (rollback), TRUNCATE=reset fast", "TRUNCATE ajoute col", "DELETE supprime DB"], "idx": 1, "b": 0.8},
                {"q": "Pourquoi normaliser ?", "opts": ["Augmenter redondance", "Réduire redondance/anomalies", "Supprimer index", "Améliorer UI"], "idx": 1, "b": 0.5},
            ],
            "tasks": [
                "Schema design : ERD pour une app + scripts SQL création.",
                "Requêtes complexes : 10 requêtes (JOIN, CTE, Aggr) + résultat CSV.",
                "Optimisation : Proposer indexs et expliquer plan d’exécution."
            ]
        },
        "ML_Core": {
            "questions": [
                {"q": "Supervisé vs Non supervisé ?", "opts": ["Supervised utilise labels", "Inverse", "Aucun utilise données", "Identiques"], "idx": 0, "b": -2.0},
                {"q": "Overfitting ?", "opts": ["Modèle simple", "Mémorise bruit, échoue sur nouvelles données", "Souhaitable", "Dataset vide"], "idx": 1, "b": -1.0},
                {"q": "Pourquoi standardiser features ?", "opts": ["Esthétique", "Centrer/échelle, améliorer convergence", "Supprimer outliers", "Augmenter variance"], "idx": 1, "b": 0.5},
                {"q": "Matrice de confusion mesure :", "opts": ["Nb features", "Performance classification (TP,FP...)", "Temps", "Disque"], "idx": 1, "b": -0.5},
                {"q": "Cross-validation ?", "opts": ["Jamais", "Évaluer robustesse, éviter overfitting", "Pour régression", "Gérer missing"], "idx": 1, "b": 0.8},
                {"q": "Algorithme non-paramétrique ?", "opts": ["Linear Reg", "KNN", "Logistic Reg", "Naive Bayes"], "idx": 1, "b": 1.2},
                {"q": "Metric dataset déséquilibré ?", "opts": ["Accuracy", "F1-score / Precision-Recall", "MSE", "AUC inutile"], "idx": 1, "b": 1.0},
                {"q": "Validation hold-out ?", "opts": ["Diviser train/test une fois", "Cross-valid 10x", "Même jeu", "Réentrainement infini"], "idx": 0, "b": -0.5},
                {"q": "Pourquoi feature selection ?", "opts": ["Plus de features", "Réduire bruit, perf & interprétabilité", "Plus de modèles", "Remplacer Tuning"], "idx": 1, "b": 0.5},
                {"q": "Rôle de GridSearchCV ?", "opts": ["Architecture réseau", "Chercher hyperparamètres optimaux", "Sauvegarder", "Visualiser"], "idx": 1, "b": 0.2},
            ],
            "tasks": [
                "Pipeline ML : Preprocess, train, eval, matrice confusion.",
                "Feature engineering : Documenter transfos et impact performance.",
                "Comparatif : Tester 3 modèles (LogReg, RF, SVM), justifier choix."
            ]
        },
        "DeepLearning": {
            "questions": [
                {"q": "Backpropagation ?", "opts": ["Prétraitement", "Mise à jour poids via gradient", "Type couche", "Dataset"], "idx": 1, "b": 0.0},
                {"q": "Pourquoi ReLU ?", "opts": ["Lent", "Non-linéarité, évite vanishing gradient", "Encoder texte", "Normaliser"], "idx": 1, "b": 0.5},
                {"q": "Dropout ?", "opts": ["Augmenter data", "Régulariser (désactive neurones aléatoires)", "Images->Vecteurs", "Sauvegarder"], "idx": 1, "b": 0.2},
                {"q": "CNN vs MLP ?", "opts": ["Tabulaire", "Images/spatial patterns", "Time series", "Clustering"], "idx": 1, "b": -1.0},
                {"q": "Embedding layer ?", "opts": ["Réduire dimensionnalité catégorielle", "Output", "Optimiseur", "Métrique"], "idx": 0, "b": 1.0},
                {"q": "Vanishing gradient ?", "opts": ["Entraînement impossible couches profondes", "Accélère", "Précision", "Reset poids"], "idx": 0, "b": 1.5},
                {"q": "Séquences longues ?", "opts": ["CNN", "LSTM/GRU", "PCA", "Dropout"], "idx": 1, "b": 0.5},
                {"q": "Learning rate ?", "opts": ["Batch size", "Taux maj des poids (amplitude)", "Epochs", "Dataset size"], "idx": 1, "b": -0.5},
                {"q": "Softmax en multiclasse ?", "opts": ["Probas sommant à 1", "Normalise images", "Supprime features", "Incrémente"], "idx": 0, "b": -0.8},
                {"q": "EarlyStopping ?", "opts": ["Arrêt si métrique stagne (anti-overfit)", "Logs", "Augmente LR", "Calcul F1"], "idx": 0, "b": 0.0},
            ],
            "tasks": [
                "CNN Image Classifier : CIFAR/MNIST, courbes learning, export.",
                "Seq2Seq / LSTM : Prédiction séquentielle simple + architecture.",
                "Ablation Study : Varier LR et Dropout, analyser impact."
            ]
        },
        "NLP": {
            "questions": [
                {"q": "Tokenization subword (BPE) ?", "opts": ["Phrases vides", "Vocab limité + OOV (découpe mots)", "Optimisation GPU", "Tri"], "idx": 1, "b": 1.5},
                {"q": "Stemming vs Lemmatization ?", "opts": ["Idem", "Stem=coupe mécanique, Lemma=linguistique", "Retire stopwords", "Corrige grammaire"], "idx": 1, "b": 0.5},
                {"q": "TF-IDF ?", "opts": ["Pondérer rareté termes", "Encoder images", "Normaliser", "Trier"], "idx": 0, "b": -0.5},
                {"q": "Embedding ?", "opts": ["Image", "Vecteur dense représentant mot", "Hyperparam", "Fichier"], "idx": 1, "b": -1.0},
                {"q": "Attention mask ?", "opts": ["Masque padding", "Augmente LR", "Réduit data", "Trie"], "idx": 0, "b": 1.0},
                {"q": "Fine-tuning ?", "opts": ["Créer vocab", "Adapter modèle pré-entraîné", "Diminuer dataset", "Générer images"], "idx": 1, "b": 0.0},
                {"q": "Masked Language Modeling (MLM) ?", "opts": ["Masquer/prédire mots (BERT)", "Tokenizer", "Optimiseur", "RL"], "idx": 0, "b": 1.2},
                {"q": "Tronquer séquences ?", "opts": ["Coût mémoire/calcul", "Vocabulaire", "Stopwords", "Erreur"], "idx": 0, "b": -0.5},
                {"q": "Modèle pré-entraîné ?", "opts": ["Transférer connaissances (large corpus)", "Sans poids", "Dataset", "Tokenizer"], "idx": 0, "b": -1.5},
                {"q": "Perplexité ?", "opts": ["Mesure erreur (bas=mieux)", "Taille vocab", "OOV", "Tokenization"], "idx": 0, "b": 1.8},
            ],
            "tasks": [
                "Sentiment Analysis : Pipeline complet, rapport métriques/erreurs.",
                "Fine-tuning : Adapter petit modèle (ex: DistilBERT) classification.",
                "Tokenization Study : Comparer Word vs Subword, analyser."
            ]
        },
        "CV": {
            "questions": [
                {"q": "YOLO est pour :", "opts": ["Classification", "Détection objets temps réel", "Compression", "Segmentation"], "idx": 1, "b": -0.5},
                {"q": "Kernel/filtre ?", "opts": ["Hyperparam", "Matrice extrayant features locaux", "Dataset", "Loss"], "idx": 1, "b": 0.0},
                {"q": "MaxPooling ?", "opts": ["Supprimer", "Réduire dimensions (valeur max)", "Couleurs", "Agrandir"], "idx": 1, "b": -1.0},
                {"q": "Normalisation image ?", "opts": ["Taille", "Échelle pixels (0-1)", "Texte", "Bruit"], "idx": 1, "b": -1.5},
                {"q": "Classification vs Detection ?", "opts": ["Localise", "Detection=objets+boxes, Classif=label global", "Idem", "Detection simple"], "idx": 1, "b": -0.5},
                {"q": "Bounding box ?", "opts": ["Convolution", "Rectangle localisant objet", "Loss", "Graphe"], "idx": 1, "b": -2.0},
                {"q": "Augmentation d’images ?", "opts": ["Réduire", "Robustesse (simuler variations)", "Normaliser", "Compresser"], "idx": 1, "b": 0.5},
                {"q": "NMS (Non-Max Suppression) ?", "opts": ["Supprimer boxes redondantes", "Optimisation", "Augmentation", "Kernel"], "idx": 0, "b": 1.5},
                {"q": "Format pipeline CV ?", "opts": ["CSV", "Images + Annotations (COCO/VOC)", "Excel", "TXT"], "idx": 1, "b": 0.0},
                {"q": "Segmentation sémantique ?", "opts": ["Rectangles", "Classer chaque pixel", "Binaire", "Compression"], "idx": 1, "b": 1.0},
            ],
            "tasks": [
                "Détection temps réel : Vidéo/Webcam, rapport FPS/Précision.",
                "Segmentation : Masque classes sur dataset simple.",
                "Annotation : Annoter images (COCO/VOC) + expliquer pipeline."
            ]
        },
        "BigData": {
            "questions": [
                {"q": "RDD ?", "opts": ["Resilient Distributed Dataset", "Random Dump", "VM", "Texte"], "idx": 0, "b": -1.0},
                {"q": "Transformation vs Action ?", "opts": ["Transfo calcule", "Action déclenche (lazy)", "Idem", "Rien"], "idx": 1, "b": 0.0},
                {"q": "Lazy evaluation ?", "opts": ["Immédiat", "Calcul différé jusqu’à action", "Supprime partitions", "Mémoire"], "idx": 1, "b": 0.5},
                {"q": "map vs flatMap ?", "opts": ["map=liste, flatMap=aplatit", "Même", "Impossible", "Supprime"], "idx": 0, "b": 0.2},
                {"q": "Éviter le shuffle ?", "opts": ["Lent/Coûteux (réseau/IO)", "Précision", "Supprime lignes", "Crypte"], "idx": 0, "b": 1.0},
                {"q": "Persist vs Cache ?", "opts": ["Idem", "Persist=options stockage, Cache=RAM", "Disque seul", "Rien"], "idx": 1, "b": 0.8},
                {"q": "Partition Spark ?", "opts": ["Fragmentation logique (parallélisme)", "Fichier", "Colonne", "Log"], "idx": 0, "b": -0.5},
                {"q": "Spark SQL ?", "opts": ["SQL sur DataFrames", "Utilisateurs", "Cluster", "Sécurité"], "idx": 0, "b": -1.5},
                {"q": "coalesce vs repartition ?", "opts": ["coalesce sans shuffle, repartition shuffle", "Inverse", "Idem", "Inutile"], "idx": 0, "b": 1.5},
                {"q": "Structured Streaming ?", "opts": ["Batch", "Flux continu structuré", "Interactif", "Viz"], "idx": 1, "b": 1.2},
            ],
            "tasks": [
                "Pipeline ETL Spark : Ingestion volu, transfo, export parquet.",
                "Optimisation : Plan exécution, partitioning/cache, gains.",
                "Streaming demo : Ingestion temps réel et agrégation."
            ]
        },
        "CloudSec": {
            "questions": [
                {"q": "IAM ?", "opts": ["Identity & Access Mgmt", "Internet", "Image", "API"], "idx": 0, "b": -2.0},
                {"q": "VPC ?", "opts": ["Virtual Private Cloud (réseau isolé)", "Variable", "Page", "Rien"], "idx": 0, "b": -1.5},
                {"q": "Symétrique vs Asymétrique ?", "opts": ["Sym=même clé, Asym=privée/publique", "Inverse", "Idem", "Aucun"], "idx": 0, "b": -0.5},
                {"q": "Security group ?", "opts": ["Firewall instance (in/out)", "DB", "Container", "Clé"], "idx": 0, "b": -0.8},
                {"q": "KMS ?", "opts": ["Key Management Service", "Logs", "Web", "Images"], "idx": 0, "b": 0.0},
                {"q": "Responsabilité partagée ?", "opts": ["Provider fait tout", "Partage Provider/Client", "Client fait tout", "Aucun"], "idx": 1, "b": -1.0},
                {"q": "MFA ?", "opts": ["Multi-facteur auth", "Firewall", "Protocole", "DB"], "idx": 0, "b": -2.5},
                {"q": "S3 vs EBS ?", "opts": ["S3=Objet, EBS=Bloc (VM)", "Inverse", "Idem", "Provider"], "idx": 0, "b": 0.5},
                {"q": "Chiffrer au repos ?", "opts": ["Taille", "Protéger vol physique/accès", "Latence", "Coût"], "idx": 1, "b": 0.2},
                {"q": "Audit cloud ?", "opts": ["Vérif config/conformité", "Backup", "Déploiement", "Suppression"], "idx": 0, "b": 0.8},
            ],
            "tasks": [
                "Secure deployment : Archi VPC, SG, IAM pour web app.",
                "PenTest Report (simu) : Vecteurs testés, corrections.",
                "Compliance checklist : Chiffrement, MFA, logs, conformité."
            ]
        },
        "MLOps": {
             "questions": [
                {"q": "Docker utile car :", "opts": ["Remplace OS", "Conteneurise (portabilité/iso)", "Remplace GPU", "Backup"], "idx": 1, "b": -1.5},
                {"q": "Kubernetes orchestre :", "opts": ["VM", "Containers (deploy/scale)", "DB", "Local"], "idx": 1, "b": -0.5},
                {"q": "CI ?", "opts": ["Continuous Integration", "Code Insp", "Container", "Cloud"], "idx": 0, "b": -2.0},
                {"q": "Versionner modèles ?", "opts": ["Poids", "Reproductibilité/Traçabilité", "Coût", "Logs"], "idx": 1, "b": 0.0},
                {"q": "Rollback ?", "opts": ["Supprimer", "Revenir version stable", "Augmenter LR", "Reset"], "idx": 1, "b": -0.8},
                {"q": "Artifact ML ?", "opts": ["Fichier produit (modèle, metrics)", "Erreur", "Container", "Dataset"], "idx": 0, "b": 0.5},
                {"q": "Canary deployment ?", "opts": ["Global", "Progressif (petit groupe)", "Logs", "Backup"], "idx": 1, "b": 1.0},
                {"q": "Monitoring ML Prod ?", "opts": ["Drift, latence, erreurs", "Batch", "Convertir", "Images"], "idx": 0, "b": 0.8},
                {"q": "Tests auto modèles ?", "opts": ["Eviter", "Garantir qualité avant deploy", "Ralentir", "Supprimer"], "idx": 1, "b": -1.2},
                {"q": "Pipeline ML ?", "opts": ["Ingest->Prep->Train->Deploy", "Dataset", "Modèle", "Container"], "idx": 0, "b": -1.0},
            ],
            "tasks": [
                "Pipeline CI/CD : Build/Test/Deploy modèle, artifacts.",
                "Container & K8s : Manifeste déploiement, scaling.",
                "Monitoring Plan : Métriques (drift, latence), alertes."
            ]
        },
        "TimeSeries": {
            "questions": [
                {"q": "Stationnarité ?", "opts": ["Moyenne/variance constantes", "Monotone", "Binaire", "Aucun"], "idx": 0, "b": 0.5},
                {"q": "ARIMA ?", "opts": ["AR, I, MA", "Arch...", "Image", "None"], "idx": 0, "b": -0.5},
                {"q": "Prophet ?", "opts": ["Tendances/Saisons/Events (facile)", "Classif", "Cluster", "Img"], "idx": 0, "b": -1.0},
                {"q": "Rolling forecast ?", "opts": ["Prédictions séquentielles + update", "Tout dataset", "Aléatoire", "None"], "idx": 0, "b": 1.0},
                {"q": "Tendance vs Saison ?", "opts": ["Long terme vs Périodique", "Inverse", "Idem", "Aucun"], "idx": 0, "b": -1.5},
                {"q": "Differencing ?", "opts": ["Saison", "Rendre stationnaire", "Variance", "Normaliser"], "idx": 1, "b": 1.2},
                {"q": "LSTM utile car ?", "opts": ["Pas mémoire", "Apprend dépendances longues", "Compresse", "Remplace ARIMA"], "idx": 1, "b": 0.0},
                {"q": "AIC/BIC ?", "opts": ["Choisir modèle (pénalise complexité)", "RMSE", "Plot", "Save"], "idx": 0, "b": 1.5},
                {"q": "Metric continuous ?", "opts": ["Accuracy", "RMSE / MAE", "F1", "Precision"], "idx": 1, "b": -1.0},
                {"q": "Anomaly detection temp ?", "opts": ["Diffère du comportement attendu", "Trier", "Supprimer", "Aucun"], "idx": 0, "b": 0.5},
            ],
            "tasks": [
                "Forecasting pipeline : ARIMA vs Prophet, comparaison.",
                "LSTM experiment : Seq2Seq, logs training.",
                "Anomaly detection : Sur série tempo, rapport alertes."
            ]
        },
        "CyberDef": {
            "questions": [
                {"q": "IDS vs IPS ?", "opts": ["IDS détecte, IPS bloque", "Inverse", "Idem", "Aucun"], "idx": 0, "b": -1.0},
                {"q": "False positive ?", "opts": ["Correct", "Alerte menace inexistante", "Manqué", "Coût"], "idx": 1, "b": -1.5},
                {"q": "Isolation Forest ?", "opts": ["Cluster", "Anomaly detection non-sup", "Classif", "Compresse"], "idx": 1, "b": 1.0},
                {"q": "SIEM ?", "opts": ["Security Info & Event Mgmt (logs)", "IM", "Image", "Aucun"], "idx": 0, "b": -0.5},
                {"q": "Honeypot ?", "opts": ["Données", "Leurre pour étudier attaquants", "Logs", "Backup"], "idx": 1, "b": 0.0},
                {"q": "Concept drift (Sec) ?", "opts": ["Changement distribution data", "Perte", "Mémoire", "Aucun"], "idx": 0, "b": 1.2},
                {"q": "Réduire faux positifs ?", "opts": ["Eviter alerte fatigue", "Coûts", "Perdre logs", "Aucun"], "idx": 0, "b": -0.8},
                {"q": "Metric anomaly ?", "opts": ["Accuracy", "Precision/Recall/F1", "RMSE", "AUC inutile"], "idx": 1, "b": 0.5},
                {"q": "Host vs Network IDS ?", "opts": ["Logs machine vs Trafic réseau", "Inverse", "Idem", "Aucun"], "idx": 0, "b": -0.5},
                {"q": "Combiner Stat + ML ?", "opts": ["Lourd", "Robustesse, moins d'erreurs", "Aucun", "Rien"], "idx": 1, "b": 0.8},
            ],
            "tasks": [
                "Logs Anomaly : Feature extraction, modèle, rapport.",
                "CTF mini : Intrusion simulée, remediation.",
                "SIEM mock : Ingestion -> Correlation -> Alerting."
            ]
        },
        "Ethics": {
            "questions": [
                {"q": "GDPR impose :", "opts": ["Rien", "Protection données perso / droits", "Marketing", "IA"], "idx": 1, "b": -2.0},
                {"q": "Biais algorithmique ?", "opts": ["Erreur hard", "Comportement inéquitable (data/model)", "Optimisation", "Aucun"], "idx": 1, "b": -1.0},
                {"q": "Droit à l'oubli ?", "opts": ["Effacer données perso", "Mdp", "Modèle", "Aucun"], "idx": 0, "b": -1.5},
                {"q": "Fairness metric ?", "opts": ["Perf", "Inégalité traitement groupes", "Taille", "Aucun"], "idx": 1, "b": 0.5},
                {"q": "Auditer modèle ?", "opts": ["Vérifier biais/robustesse/légal", "Ralentir", "Supprimer", "Aucun"], "idx": 0, "b": -0.5},
                {"q": "XAI (Explainable AI) ?", "opts": ["Décisions interprétables", "Dataset", "Lib", "Aucun"], "idx": 0, "b": 0.0},
                {"q": "Data minimization ?", "opts": ["Collecter le nécessaire", "Tout", "Garder tout", "Aucun"], "idx": 0, "b": 0.8},
                {"q": "Fairness vs Accuracy ?", "opts": ["Toujours up", "Trade-off (arbitrage nécessaire)", "Aucun", "Idem"], "idx": 1, "b": 1.0},
                {"q": "Discrimination indirecte ?", "opts": ["Sans data", "Via variables corrélées", "Erreur", "Aucun"], "idx": 1, "b": 1.5},
                {"q": "Outils détection biais ?", "opts": ["AI Fairness 360, What-If...", "Texte", "Cloud", "Aucun"], "idx": 0, "b": 0.5},
            ],
            "tasks": [
                "Audit Fairness : Mesurer metrics sur modèle, corrections.",
                "GDPR Report : Flux données, consentement, retention.",
                "Explainability : LIME/SHAP sur décisions clés."
            ]
        }
    }

    # =========================================================================
    # 1. MOTEUR IA : THEORY RESPONSE ITEM (IRT 3PL)
    # =========================================================================
    class PowerfulAdaptiveExam:
        def __init__(self, question_bank):
            self.questions = {}
            for q in question_bank:
                self.questions[q['id']] = {
                    **q,
                    'a': q.get('a', 1.0),
                    'b': q.get('b', 0.0),
                    'c': q.get('c', 0.25)
                }
            self.theta = 0.0
            self.history = []
            self.min_theta = -3.0
            self.max_theta = 3.0

        def _sigmoid_3pl(self, theta, a, b, c):
            exponent = np.clip(-a * (theta - b), -20, 20)
            return c + (1 - c) / (1 + np.exp(exponent))

        def _fisher_information(self, theta, question_id):
            q = self.questions[question_id]
            P = self._sigmoid_3pl(theta, q['a'], q['b'], q['c'])
            Q = 1 - P
            return (q['a']**2 * Q / P) * ((P - q['c']) / (1 - q['c']))**2

        def estimate_theta_mle(self, response_history):
            if not response_history: return 0.0
            
            def neg_log_likelihood(theta):
                log_l = 0
                for resp in response_history:
                    q = self.questions[resp['id']]
                    prob = self._sigmoid_3pl(theta[0], q['a'], q['b'], q['c'])
                    prob = np.clip(prob, 1e-9, 1.0 - 1e-9)
                    if resp['correct']:
                        log_l += np.log(prob)
                    else:
                        log_l += np.log(1 - prob)
                return -log_l
            
            result = minimize(neg_log_likelihood, x0=[self.theta], bounds=[(self.min_theta, self.max_theta)], method='L-BFGS-B')
            return result.x[0]

        def select_next_question(self, response_history):
            if response_history:
                self.history = [r['id'] for r in response_history]
                self.theta = self.estimate_theta_mle(response_history)
            
            candidates = [qid for qid in self.questions if qid not in self.history]
            if not candidates: return None

            best_q_id = max(candidates, key=lambda qid: self._fisher_information(self.theta, qid))
            
            total_info = sum(self._fisher_information(self.theta, r['id']) for r in response_history)
            if total_info > 1e-5:
                confidence = max(0, 100 * (1 - (1 / np.sqrt(total_info))))
            else:
                confidence = 0
            
            return {
                "question": self.questions[best_q_id],
                "theta": round(self.theta, 3),
                "confidence": round(confidence, 1)
            }

    # =========================================================================
    # 2. GÉNÉRATEUR DE CONTENU RÉEL
    # =========================================================================
    def generate_real_questions(exam_tag):
        """Récupère les questions de la DB et les formate pour le moteur IRT"""
        # Mapping Tag -> DB Key
        tag_map = {
            "Python": "Python", "DSA": "DSA", "DataAnalysis": "DataAnalysis", 
            "DataViz": "DataViz", "SQL": "SQL", "ML": "ML_Core", 
            "DL": "DeepLearning", "NLP": "NLP", "CV": "CV", 
            "BigData": "BigData", "CloudSec": "CloudSec", "MLOps": "MLOps", 
            "TimeSeries": "TimeSeries", "Cyber": "CyberDef", "Ethics": "Ethics"
        }
        
        db_key = tag_map.get(exam_tag, "Python") # Fallback
        raw_data = CONTENT_DB.get(db_key, CONTENT_DB["Python"])
        
        bank = []
        for i, item in enumerate(raw_data["questions"]):
            bank.append({
                "id": f"{exam_tag}_{i}",
                "content": f"**[{exam_tag}]**<br>{item['q']}",
                "options": item['opts'],
                "correct_index": item['idx'],
                "b": item['b'],       # Difficulté manuelle
                "a": np.random.uniform(1.0, 2.5), # Discrimination haute
                "c": 0.25             # Guessing standard
            })
        return bank, raw_data.get("tasks", [])

    # =========================================================================
    # 3. STATE & EXAM DB CONFIGURATION (UPDATED)
    # =========================================================================
    if "exam_view" not in st.session_state: st.session_state.exam_view = "catalog"
    if "selected_exam_id" not in st.session_state: st.session_state.selected_exam_id = None
    if "adaptive_engine" not in st.session_state: st.session_state.adaptive_engine = None
    if "exam_history" not in st.session_state: st.session_state.exam_history = []
    if "current_q_data" not in st.session_state: st.session_state.current_q_data = None
    if "practical_tasks" not in st.session_state: st.session_state.practical_tasks = []
    if "project_uploaded" not in st.session_state: st.session_state.project_uploaded = False
    if "ai_analysis_done" not in st.session_state: st.session_state.ai_analysis_done = None
    if "user_fullname" not in st.session_state: st.session_state.user_fullname = ""

    # --- BANQUE D'EXAMENS ÉLARGIE (15 MODULES) ---
    EXAM_DB = {
        "PY-101":   { "title": "Python Architect", "tag": "Python", "level": "Intermédiaire", "xp": 500, "color": "#3776ab", "icon": "🐍" },
        "DSA-500":  { "title": "Data Struct & Algo", "tag": "DSA", "level": "Difficile", "xp": 800, "color": "#ff4b4b", "icon": "⚡" },
        "DA-200":   { "title": "NumPy & Pandas", "tag": "DataAnalysis", "level": "Intermédiaire", "xp": 600, "color": "#1565c0", "icon": "📊" },
        "VIZ-300":  { "title": "Data Visualization", "tag": "DataViz", "level": "Créatif", "xp": 550, "color": "#ff61f6", "icon": "📈" },
        "SQL-300":  { "title": "Advanced SQL DB", "tag": "SQL", "level": "Difficile", "xp": 700, "color": "#00758f", "icon": "💾" },
        "ML-202":   { "title": "Machine Learning Core", "tag": "ML", "level": "Expert", "xp": 1000, "color": "#F7931E", "icon": "🤖" },
        "DL-NET":   { "title": "Deep Learning & NN", "tag": "DL", "level": "Grand Maître", "xp": 1500, "color": "#8e24aa", "icon": "🧠" },
        "NLP-TXT":  { "title": "NLP Processing", "tag": "NLP", "level": "Expert", "xp": 1300, "color": "#00acc1", "icon": "📝" },
        "CV-IMG":   { "title": "Computer Vision", "tag": "CV", "level": "Expert", "xp": 1400, "color": "#ef5350", "icon": "📸" },
        "BLK-101":  { "title": "Big Data & Spark", "tag": "BigData", "level": "Difficile", "xp": 1100, "color": "#e65100", "icon": "☁️" },
        "SEC-CLD":  { "title": "Cloud Security Ops", "tag": "CloudSec", "level": "Expert", "xp": 950, "color": "#2e7d32", "icon": "🔒" },
        "MLOPS-CI": { "title": "MLOps & CI/CD", "tag": "MLOps", "level": "Difficile", "xp": 1200, "color": "#0277bd", "icon": "⚙️" },
        "TIME-SER": { "title": "Time Series Forecast", "tag": "TimeSeries", "level": "Intermédiaire", "xp": 900, "color": "#00838f", "icon": "⏱️" },
        "CY-66":    { "title": "Cyber & Anomaly", "tag": "Cyber", "level": "Grand Maître", "xp": 2000, "color": "#00FF00", "icon": "🛡️" },
        "ETH-AI":   { "title": "Ethics & Responsible AI", "tag": "Ethics", "level": "Essentiel", "xp": 400, "color": "#ad1457", "icon": "⚖️" },
    }

    # --- STYLE CSS ---
    st.markdown("""
    <style>
        .exam-card { 
            background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; 
            transition: all 0.2s ease-in-out; position: relative; overflow: hidden;
        }
        .exam-card:hover { transform: translateY(-5px); border-color: #58a6ff; box-shadow: 0 5px 15px rgba(0,0,0,0.5); }
        .tag-badge { 
            background: rgba(255,255,255,0.1); padding: 2px 8px; border-radius: 4px; 
            font-size: 0.7em; text-transform: uppercase; letter-spacing: 1px;
        }
        .task-card {
            background: rgba(0, 255, 0, 0.05); border: 1px solid #00FF00; padding: 15px;
            border-radius: 8px; margin-bottom: 10px; font-family: 'Courier New'; color: #ccffcc;
        }
        .cert-card {
            background: linear-gradient(135deg, #FFD700 0%, #B8860B 100%);
            color: black; padding: 20px; border-radius: 10px; margin-top: 15px;
            font-family: 'Courier New'; text-align: center; border: 2px solid #FFF;
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.4);
        }
        .cert-hash {
            font-size: 0.7em; opacity: 0.8; word-break: break-all; margin-top: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    # =========================================================================
    
    # VUE 1 : CATALOGUE (GRID)
    # =========================================================================
    if st.session_state.exam_view == "catalog":
        c1, c2 = st.columns([3, 1])
        with c1:
            st.title("🛡️ Centre de Certification")
            st.caption("Sélectionnez un module pour calibrer votre profil via IRT Engine v3.0")
        with c2:
            st.text_input("🔍 Filtrer...", placeholder="Python, Cloud...", label_visibility="collapsed")

        cols = st.columns(3)
        exam_list = list(EXAM_DB.items())
        
        for idx, (eid, d) in enumerate(exam_list):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="exam-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <span style="font-size:1.5em;">{d['icon']}</span>
                        <span class="tag-badge" style="color:{d['color']}; border:1px solid {d['color']};">{d['tag']}</span>
                    </div>
                    <h3 style="margin:0; font-size:1.1em; color:white;">{d['title']}</h3>
                    <p style="color:#8b949e; font-size:0.85em; margin-top:5px;">Niveau: {d['level']}</p>
                    <div style="margin-top:15px; display:flex; justify-content:space-between; align-items:center;">
                        <span style="color:#e3b341; font-weight:bold;">+{d['xp']} XP</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Lancer Diagnostique", key=f"btn_{eid}", use_container_width=True):
                    # --- INITIALISATION DU LAB ---
                    bank, tasks = generate_real_questions(d['tag']) # Charge VRAIES questions
                    st.session_state.adaptive_engine = PowerfulAdaptiveExam(bank)
                    st.session_state.practical_tasks = tasks
                    st.session_state.exam_history = []
                    st.session_state.current_q_data = None
                    st.session_state.project_uploaded = False
                    st.session_state.ai_analysis_done = None
                    st.session_state.selected_exam_id = eid
                    st.session_state.exam_view = "boot"
                    st.rerun()
                st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # VUE 2 : SÉQUENCE DE BOOT (ANIMATION)
    # =========================================================================
    elif st.session_state.exam_view == "boot":
        exam = EXAM_DB[st.session_state.selected_exam_id]
        placeholder = st.empty()
        
        steps = [
            "Initializing Secure Kernel...",
            "Loading 3PL IRT Parameters...",
            f"Loading Content: {len(st.session_state.adaptive_engine.questions)} modules...",
            f"Connecting to {exam['tag']} Neural Node...",
            "Encrypting Session...",
            "SYSTEM READY."
        ]
        
        log_text = ""
        for step in steps:
            log_text += f"> {step}\n"
            placeholder.markdown(f"""
            <div style="display:flex; align-items:center; justify-content:center; height:400px;">
                <div style="background:black; color:#00FF00; font-family:'Courier New'; padding:30px; border-radius:10px; width:600px; border:1px solid #333; box-shadow:0 0 20px rgba(0,255,0,0.1);">
                    <h2 style="border-bottom:1px solid #00FF00; padding-bottom:10px;">BOOT_SEQUENCE // {exam['tag']}</h2>
                    <pre style="white-space: pre-wrap;">{log_text}</pre>
                    <span style="animation: blink 1s infinite;">_</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.5)
            
        st.session_state.exam_view = "lab"
        st.rerun()

    # =========================================================================
    # VUE 3 : LAB ADAPTATIF (INTERFACE SPLIT)
    # =========================================================================
    elif st.session_state.exam_view == "lab":
        engine = st.session_state.adaptive_engine
        history = st.session_state.exam_history
        exam = EXAM_DB[st.session_state.selected_exam_id]

        # Sélectionner la prochaine question si aucune n'est active
        if st.session_state.current_q_data is None:
            st.session_state.current_q_data = engine.select_next_question(history)

        data = st.session_state.current_q_data
        
        # Variables de HUD
        theta_val = data['theta'] if data else engine.theta
        conf_val = data['confidence'] if data else 0.0
        q_num = len(history) + 1

        # HUD Supérieur
        col1, col2, col3 = st.columns(3)
        col1.metric("Niveau (Theta)", f"{theta_val:.2f}", delta_color="off")
        col2.metric("Confiance IA", f"{conf_val}%")
        col3.metric("Questions", f"{len(history)}")
        
        st.progress(min(100, int(conf_val)), text="Précision de l'évaluation IA")

        # --- FIN DU QUIZ THÉORIQUE / DÉBUT PHASE PRATIQUE ---
        # Stop si Confiance > 90% OU plus de 10 questions OU plus de questions dispos
        if data is None or conf_val > 90 or len(history) >= 10:
            
            final_score = float(theta_val) if isinstance(theta_val, (int, float)) else 0.0
            score_percent = int(np.clip((final_score + 3) / 6 * 100, 0, 100))
            
            st.success(f"🏁 **Phase Théorique Validée !** Score: {score_percent}/100")
            
            # =========================================================================
            # [MODIFICATION] GENERATION CSV PANDAS POUR VISUALISATION
            # =========================================================================
            # On reconstruit l'historique pour avoir les points de données étape par étape
            csv_data = []
            temp_history_for_calc = []
            
            # On rejoue l'histoire pour calculer le Theta à chaque étape
            for i, record in enumerate(history):
                temp_history_for_calc.append(record)
                step_theta = engine.estimate_theta_mle(temp_history_for_calc)
                
                csv_data.append({
                    "step": i + 1,
                    "exam_tag": exam['tag'],
                    "question_id": record['id'],
                    "user_success": 1 if record['correct'] else 0,
                    "theta_evolution": round(step_theta, 4),
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # Création du DataFrame et conversion CSV
            df_export = pd.DataFrame(csv_data)
            csv_file = df_export.to_csv(index=False).encode('utf-8')
            
            with st.expander("📊 Données de session (Analyse)", expanded=False):
                st.caption("Téléchargez ces données pour visualiser votre courbe d'apprentissage.")
                st.dataframe(df_export)
                st.download_button(
                    label="📥 Télécharger les données (.csv)",
                    data=csv_file,
                    file_name=f"exam_data_{exam['tag']}_{int(time.time())}.csv",
                    mime="text/csv",
                    key="download_csv_btn"
                )
            # =========================================================================
            # [FIN MODIFICATION]
            # =========================================================================

            # --- LAYOUT RÉSULTATS + TASKS ---
            res_c1, res_c2 = st.columns(2)
            
            with res_c1:
                st.markdown("### 📊 Rapport de Performance")
                st.write(f"Votre niveau théorique a été calibré à **Theta: {final_score:.2f}**.")
                st.info("Pour obtenir votre certification officielle, veuillez compléter le projet pratique ci-contre et soumettre vos résultats.")

            with res_c2:
                st.markdown("### 📜 Mission Pratique (Obligatoire)")
                if st.session_state.practical_tasks:
                    for task in st.session_state.practical_tasks:
                        st.markdown(f"<div class='task-card'>✅ {task}</div>", unsafe_allow_html=True)
                
                # Lien de téléchargement des consignes
                consignes_text = f"CAHIER DES CHARGES - {exam['title']}\n\n"
                consignes_text += "OBJECTIFS:\n"
                for t in st.session_state.practical_tasks:
                    consignes_text += f"- {t}\n"
                consignes_text += "\nLIVRABLES ATTENDUS:\n- Code source commenté\n- Rapport d'analyse\n- Capture d'écran des résultats"

                st.download_button(
                    "📥 Télécharger le Cahier des Charges (TXT)", 
                    data=consignes_text,
                    file_name=f"{exam['tag']}_Instructions.txt",
                    mime="text/plain"
                )

            st.markdown("---")

            # --- ZONE DE SOUMISSION (UPLOAD) ---
            st.markdown("### 📤 Soumission du Projet & Certification")
            
            # Container de soumission
            with st.container():
                upload_col, cert_col = st.columns([1, 1.2])
                
                with upload_col:
                    st.warning("⚠️ Action requise : Déposez votre projet (Zip/PDF) pour débloquer la certification.")
                    uploaded_file = st.file_uploader("Déposer votre fichier ici", type=['zip', 'pdf', 'rar'], key="project_upload")
                    
                    if uploaded_file is not None:
                        st.session_state.project_uploaded = True
                        if not st.session_state.ai_analysis_done:
                            with st.spinner("🤖 L'IA analyse votre code source... Recherche de vulnérabilités..."):
                                time.sleep(2.5) # Simulation
                                st.session_state.ai_analysis_done = generate_ai_feedback(exam['tag'], final_score)
                        
                        st.success("✅ Projet reçu et validé par le système !")
                        
                        # AFFICHAGE FEEDBACK IA
                        feedback = st.session_state.ai_analysis_done
                        if feedback:
                            with st.expander("👁️ Voir l'analyse détaillée de l'IA", expanded=True):
                                st.markdown(f"**Points Forts:** {feedback['strength']}")
                                st.markdown(f"**Points d'attention:** {feedback['weakness']}")
                                st.markdown(f"**Recommandation:** {feedback['recommendation']}")
                                st.info(f"💡 *{feedback['quote']}*")

                    else:
                        st.session_state.project_uploaded = False
                        st.session_state.ai_analysis_done = None

                with cert_col:
                    if st.session_state.project_uploaded:
                        # --- FORMULAIRE NOM ---
                        st.markdown("#### 👤 Identité du Candidat")
                        user_name = st.text_input("Entrez votre Prénom et Nom pour le certificat", value=st.session_state.user_fullname)
                        if user_name: st.session_state.user_fullname = user_name

                        if st.session_state.user_fullname:
                            # --- GÉNÉRATION HASH ---
                            timestamp = datetime.datetime.now().strftime("%d/%m/%Y")
                            raw_string = f"{st.session_state.selected_exam_id}_{final_score}_{timestamp}_{st.session_state.user_fullname}_SAMA"
                            cert_hash = hashlib.sha256(raw_string.encode()).hexdigest()
                            fb = st.session_state.ai_analysis_done

                            st.markdown(f"""
                            <div class="cert-card">
                                <h3>🏆 CERTIFICAT BLOCKCHAIN</h3>
                                <div style="font-size:1.2em; margin:5px 0;">Délivré à: <b>{st.session_state.user_fullname}</b></div>
                                <div style="font-size:1.5em; margin:10px 0; font-weight:bold;">{exam['tag']} EXPERT</div>
                                <div style="font-size:0.9em;">Score Global: {score_percent}%</div>
                                <div class="cert-hash">SHA-256: {cert_hash}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # --- GÉNÉRATION DU VRAI PDF PRO (DESIGN PREMIUM) ---
                            
                            class ProPDF(FPDF):
                                def header(self):
                                    pass # On dessine tout manuellement pour le certificat
                                def footer(self):
                                    pass

                            # Création PDF Paysage A4
                            pdf = ProPDF(orientation='L', unit='mm', format='A4')
                            pdf.add_page()
                            
                            # 1. BORDURES & FOND
                            # Couleur Or pour la bordure
                            pdf.set_draw_color(218, 165, 32) # Goldenrod
                            pdf.set_line_width(2)
                            pdf.rect(5, 5, 287, 200) # Cadre extérieur
                            
                            pdf.set_draw_color(10, 25, 50) # Dark Navy
                            pdf.set_line_width(0.5)
                            pdf.rect(8, 8, 281, 194) # Cadre intérieur fin

                            # 2. EN-TÊTE BLEU NUIT
                            pdf.set_fill_color(10, 25, 50) # Dark Navy
                            pdf.rect(8, 8, 281, 35, 'F') # Bandeau haut
                            
                            pdf.set_y(15)
                            pdf.set_font("Arial", 'B', 28)
                            pdf.set_text_color(255, 255, 255) # Blanc
                            pdf.cell(0, 10, "SAMA LEARN CERTIFICATION", 0, 1, 'C')
                            
                            pdf.set_font("Arial", 'I', 12)
                            pdf.set_text_color(200, 200, 200) # Gris clair
                            pdf.cell(0, 10, "Center of Excellence for Artificial Intelligence", 0, 1, 'C')

                            # 3. TITRE "CERTIFICAT"
                            pdf.ln(25)
                            pdf.set_text_color(218, 165, 32) # Or
                            pdf.set_font('Times', 'B', 42)
                            pdf.cell(0, 20, "CERTIFICAT D'EXCELLENCE", 0, 1, 'C')
                            
                            # 4. DECERNÉ À
                            pdf.ln(5)
                            pdf.set_text_color(80, 80, 80) # Gris foncé
                            pdf.set_font('Arial', 'I', 14)
                            pdf.cell(0, 10, clean_text("Ce document atteste que"), 0, 1, 'C')
                            
                            # 5. NOM DU CANDIDAT (GROS)
                            pdf.ln(2)
                            pdf.set_text_color(10, 25, 50) # Bleu nuit
                            pdf.set_font('Arial', 'B', 32)
                            pdf.cell(0, 15, clean_text(st.session_state.user_fullname.upper()), 0, 1, 'C')
                            # Ligne décorative sous le nom
                            pdf.set_draw_color(218, 165, 32)
                            pdf.line(80, 105, 217, 105) 
                            
                            # 6. DETAILS DU MODULE
                            pdf.ln(10)
                            pdf.set_text_color(50, 50, 50)
                            pdf.set_font('Arial', '', 14)
                            pdf.cell(0, 8, clean_text(f"A validé avec succès le parcours expert :"), 0, 1, 'C')
                            pdf.set_font('Arial', 'B', 16)
                            pdf.cell(0, 10, clean_text(f"{exam['title']} ({exam['tag']})"), 0, 1, 'C')
                            
                            pdf.set_font('Arial', 'B', 12)
                            pdf.set_text_color(100, 100, 100)
                            pdf.cell(0, 8, f"Score Final: {score_percent}/100 (Theta: {final_score:.2f})", 0, 1, 'C')

                            # 7. BOITE FEEDBACK IA (DESIGN TECH)
                            pdf.set_y(140)
                            pdf.set_x(40)
                            pdf.set_fill_color(245, 245, 245) # Gris très clair
                            pdf.set_draw_color(200, 200, 200)
                            pdf.rect(40, 140, 217, 40, 'FD')
                            
                            pdf.set_xy(45, 142)
                            pdf.set_font('Courier', 'B', 10)
                            pdf.set_text_color(0, 100, 0) # Vert Terminal
                            pdf.cell(0, 6, "AI_ANALYSIS_REPORT // STATUS: VERIFIED", 0, 1, 'L')
                            
                            pdf.set_x(45)
                            pdf.set_font('Arial', '', 9)
                            pdf.set_text_color(0, 0, 0)
                            pdf.multi_cell(205, 5, clean_text(f"[+] Strength: {fb['strength']}"))
                            pdf.set_x(45)
                            pdf.multi_cell(205, 5, clean_text(f"[!] Recommendation: {fb['recommendation']}"))
                            
                            # 8. PIED DE PAGE & SCEAU
                            pdf.set_y(185)
                            
                            # Date (Gauche)
                            pdf.set_x(20)
                            pdf.set_font('Arial', '', 10)
                            pdf.cell(60, 5, f"Date: {timestamp}", 0, 0, 'L')
                            
                            # Hash (Centre)
                            pdf.set_x(80)
                            pdf.set_font('Courier', '', 7)
                            pdf.set_text_color(128, 128, 128)
                            pdf.cell(140, 5, f"BLOCKCHAIN HASH: {cert_hash}", 0, 0, 'C')
                            
                            # Sceau (Droite) - Carré car circle n'existe pas par défaut dans fpdf basic
                            pdf.set_draw_color(10, 25, 50)
                            pdf.set_line_width(1)
                            pdf.rect(248, 163, 25, 25) # x, y, w, h
                            
                            pdf.set_xy(248, 168)
                            pdf.set_font('Arial', 'B', 6)
                            pdf.set_text_color(10, 25, 50)
                            pdf.cell(25, 3, "SAMA LEARN", 0, 1, 'C')
                            pdf.set_x(248)
                            pdf.cell(25, 3, "OFFICIAL", 0, 1, 'C')

                            # Conversion en bytes
                            pdf_bytes = pdf.output(dest='S').encode('latin-1', 'replace')
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.download_button(
                                label="🏆 TÉLÉCHARGER MON CERTIFICAT PRO (PDF)",
                                data=pdf_bytes,
                                file_name=f"CERTIFICAT_{exam['tag']}_{st.session_state.user_fullname.replace(' ', '_')}.pdf",
                                mime="application/pdf",
                                key="dl_cert_btn",
                                type="primary",
                                use_container_width=True
                            )
                            st.balloons()
                        else:
                            st.warning("👉 Veuillez entrer votre nom pour générer le certificat.")
                    else:
                        # --- MODE VEROUILLÉ ---
                        st.markdown("""
                        <div style="opacity:0.5; filter:grayscale(1); border:2px dashed #555; padding:20px; border-radius:10px; text-align:center;">
                            <h3>🔒 CERTIFICATION VERROUILLÉE</h3>
                            <p>En attente de la soumission du projet...</p>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown("---")
            if st.button("Retour au Catalogue", type="secondary"):
                st.session_state.exam_view = "catalog"
                st.rerun()
        
        else:
            # --- INTERFACE DE QUESTION (ACTIVE PENDANT LE QUIZ) ---
            col_q, col_term = st.columns([1.2, 1])
            
            with col_q:
                q = data['question']
                st.markdown(f"### Question {q_num}")
                st.write(q['content'], unsafe_allow_html=True)
                
                choice = st.radio("Votre réponse :", q['options'], key=f"q_radio_{q_num}")
                
                if st.button("Valider la réponse", type="primary", use_container_width=True):
                    correct_idx = q['correct_index']
                    user_idx = q['options'].index(choice)
                    is_correct = (user_idx == correct_idx)
                    
                    if is_correct:
                        st.toast("✅ Correct ! Difficulté augmentée.", icon="🚀")
                    else:
                        st.toast("❌ Incorrect. Ajustement du niveau...", icon="📉")
                    
                    st.session_state.exam_history.append({"id": q['id'], "correct": is_correct})
                    st.session_state.current_q_data = None
                    time.sleep(0.5)
                    st.rerun()

            with col_term:
                st.markdown("### 🖥️ Live Logs")
                logs_html = f"""
                <div style="background:#000; color:#33FF33; font-family:'Courier New', monospace; padding:15px; border-radius:6px; height:400px; overflow-y:auto; border:1px solid #333; font-size:0.85em;">
                    <span>root@sama-node:~# adaptive_engine --status running</span><br>
                    <span style="color:#666;">[INFO] Current Estimation: {theta_val}</span><br>
                    <span style="color:#666;">[INFO] Question ID: {q['id']} loaded.</span><br>
                    <br>
                    <span style="color:white;">> Analyzing user interaction...</span><br>
                    <span style="color:white;">> Gaze tracking: Nominal</span><br>
                    <br>
                    <span style="color:#FFFF00;">Waiting for input...</span><span style="animation:blink 1s infinite;">_</span>
                </div>
                """
                st.markdown(logs_html, unsafe_allow_html=True)





    
# =====================================================
# 6. PAGE: EXERCICES (IDE SIMULÉ)
elif st.session_state.page == "exercices":
    
    # --- 1. INITIALISATION DES ÉTATS ---
    if "current_level" not in st.session_state: 
        st.session_state.current_level = None
    if "quiz_submitted" not in st.session_state: 
        st.session_state.quiz_submitted = False
    if "q_index" not in st.session_state: 
        st.session_state.q_index = 0
    if "score" not in st.session_state: 
        st.session_state.score = 0
    if "answer_validated" not in st.session_state: 
        st.session_state.answer_validated = False

    # --------------------------------------------------------------------------
    # VUE 1 : RAPPORT DE PERFORMANCE


    # --------------------------------------------------------------------------
    # VUE 2 : MENU DES 20 NIVEAUX
    # --------------------------------------------------------------------------
   
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
        
        st.stop()

    # ==================================================
    # 3. LOGIQUE ADAPTATIVE & HELPER HTML
    # ==================================================
    def is_module_locked(module_lvl):
        """Vérifie si le module est verrouillé selon le niveau précédent."""
        if module_lvl == 1: return False
        prev_lvl_mods = [s for m, s in st.session_state.eleve_competences.items() 
                         if m in MODULES and MODULES[m]['lvl'] == module_lvl - 1]
        
        if not prev_lvl_mods: return False
        avg_prev = np.mean(prev_lvl_mods)
        return avg_prev < 0.5 

    def get_card_html(name, data, score, locked):
        pct = int(score * 100)
        
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

        segments = ""
        for s in range(10):
            bg = data['color'] if s < (pct // 10) and not locked else "rgba(255,255,255,0.1)"
            glow = f"box-shadow: 0 0 8px {data['color']};" if s < (pct // 10) and not locked else ""
            segments += f"<div style='flex:1; height:5px; background:{bg}; margin-right:3px; border-radius:2px; {glow}'></div>"
        
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

        st.video(f"https://www.youtube.com/watch?v={d['video']}")
        
        if s < 0.5 and s > 0:
            st.error(f"📉 Difficulté détectée sur ce module (< 50%)")
            with st.expander("🆘 Zone de Renfort IA (Recommandations)", expanded=True):
                cat_res = RAG_RESOURCES.get(d['cat'], ["Article: Concepts Fondamentaux"])
                for res in cat_res:
                    st.markdown(f"- 🔗 [{res}](#)")
                st.caption("L'IA a sélectionné ces ressources pour combler vos lacunes spécifiques.")

        st.markdown("---")
        
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
                    exec(code)
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
        # Note : key ajoutée pour éviter l'erreur DuplicateElementId
        if st.button("🍅 Mode Focus", use_container_width=True, type="primary", key="header_focus_btn"):
            st.session_state.focus_active = True
            st.rerun()

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
    # 7. MOTEUR DE RECOMMANDATION (IA INTELLIGENTE)
    # ==================================================
    try:
        # Initialiser le moteur
        rec_sys = RecommenderSystem(curriculum=MODULES)

        # Préparer le profil utilisateur
        user_profile = {
            "competences": st.session_state.eleve_competences,
            # Conversion sécurisée des dates en string pour le moteur
            "last_review": {k: str(v) for k, v in st.session_state.last_review.items()} 
        }

        # Récupérer l'historique (ou vide si inexistant)
        history = st.session_state.get("history", [])

        # Générer les recommandations
        recommendations = rec_sys.get_recommendations(user_profile, history)

        # Afficher la meilleure recommandation
        if recommendations:
            top_rec = recommendations[0]
            st.info(f"💡 **Conseil de l'IA :** {top_rec['reason']} (Module suggéré : **{top_rec['module']}**)")
            
    except Exception as e:
        # Fallback silencieux en cas d'erreur moteur
        st.caption(f"IA en cours de calibrage... ({e})")


    # ==================================================
    # 8. SRS (RÉPÉTITION ESPACÉE)
    # ==================================================
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
    # 9. CATALOGUE ADAPTATIF (GRILLE 2 COLONNES)
    # ==================================================
    st.subheader("📚 Arbre de Compétences (Parcours Adaptatif)")
    
    search = st.text_input("🔍", placeholder="Rechercher un module...", label_visibility="collapsed")
    
    cols = st.columns(2)
    idx = 0
    
    sorted_modules = sorted(MODULES.items(), key=lambda x: x[1]['lvl'])
    
    for name, data in sorted_modules:
        if search.lower() in name.lower():
            score = st.session_state.eleve_competences[name]
            is_locked = is_module_locked(data['lvl'])
            html_card = get_card_html(name, data, score, is_locked)
            
            with cols[idx % 2]:
                st.markdown(html_card, unsafe_allow_html=True)
                if is_locked:
                    st.button(f"🔒 Niveau {data['lvl']-1} requis", key=f"lock_{name}", disabled=True, use_container_width=True)
                else:
                    if st.button(f"▶️ Accéder au module", key=f"open_{name}", use_container_width=True):
                        open_modal(name)
            idx += 1

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

   
            
import streamlit as st
import random
import time

# ==============================================================================
# 1. CONFIGURATION (Première ligne obligatoire)


# Styles CSS pour le design personnalisé
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DONNÉES GLOBALES (VOTRE LISTE DE 20 NIVEAUX)
# ==============================================================================

LEVELS = {
    1: {"title": "Python: Les Fondations", "icon": "🐍", "desc": "Syntaxe, Variables, Boucles", "color": "#FFD43B"},
    2: {"title": "Python Avancé", "icon": "🚀", "desc": "Générateurs, Décorateurs, OPP", "color": "#306998"},
    3: {"title": "Structures de Données", "icon": "📚", "desc": "Listes, Dicos, Sets, Complexité", "color": "#FFD43B"},
    4: {"title": "Algorithmique", "icon": "⚙️", "desc": "Tri, Recherche, Récursivité", "color": "#306998"},
    5: {"title": "NumPy & Matrices", "icon": "🔢", "desc": "Calcul vectoriel, Broadcasting", "color": "#013243"},
    6: {"title": "Pandas: Manipulation", "icon": "🐼", "desc": "DataFrames, Séries, Indexing", "color": "#150458"},
    7: {"title": "Pandas: Nettoyage", "icon": "🧹", "desc": "Missing Values, Duplicates, Types", "color": "#150458"},
    8: {"title": "Data Viz (Matplotlib)", "icon": "📊", "desc": "Plots statiques, Personnalisation", "color": "#F26627"},
    9: {"title": "Viz Interactive", "icon": "📈", "desc": "Plotly, Streamlit, Dashboards", "color": "#F26627"},
    10: {"title": "SQL: Les Bases", "icon": "🗄️", "desc": "SELECT, WHERE, ORDER BY", "color": "#00758F"},
    11: {"title": "SQL: Avancé", "icon": "🔗", "desc": "JOINS, Window Functions, CTE", "color": "#00758F"},
    12: {"title": "Probabilités", "icon": "🎲", "desc": "Lois, Bayes, Espérance", "color": "#E91E63"},
    13: {"title": "Statistiques Inférentielles", "icon": "📉", "desc": "Tests A/B, P-value, Confiance", "color": "#E91E63"},
    14: {"title": "ML: Supervisé", "icon": "🎯", "desc": "Régression, Classification, KNN", "color": "#F39C12"},
    15: {"title": "ML: Non-Supervisé", "icon": "🧩", "desc": "K-Means, PCA, Clustering", "color": "#F39C12"},
    16: {"title": "Évaluation de Modèles", "icon": "⚖️", "desc": "ROC, AUC, Precision/Recall", "color": "#F39C12"},
    17: {"title": "Deep Learning (ANN)", "icon": "🧠", "desc": "Réseaux de neurones, Backprop", "color": "#D32F2F"},
    18: {"title": "Computer Vision (CNN)", "icon": "👁️", "desc": "Images, Convolution, Pooling", "color": "#D32F2F"},
    19: {"title": "NLP & Transformers", "icon": "🗣️", "desc": "Texte, Tokenization, BERT/GPT", "color": "#9C27B0"},
    20: {"title": "MLOps & Production", "icon": "🚢", "desc": "Docker, Git, API, Cloud", "color": "#555555"}
}

# ==============================================================================
# 3. FONCTIONS UTILITAIRES
# ==============================================================================
# ==============================================================================
# 1. DONNÉES : MISE À JOUR DE LA FONCTION get_questions_for_level
# ==============================================================================
import random
import math
from typing import List

# Assurez-vous que Question, BloomLevel, ErrorType, et LEVELS sont définis ici

def get_questions_for_level(level_id) -> List['Question']:
    """
    Génère 15 questions en utilisant la dataclass Question, indispensable 
    pour le moteur de feedback formatif.
    """
    topic = LEVELS[level_id]['title']
    questions = []
    required_q = 15
    
    # Définition des Concepts, Types d'Erreur et Niveaux de Bloom
    q_data_pool = [
        ("la syntaxe de base", BloomLevel.REMEMBER, "Mémorisation des commandes."),
        ("l'utilisation des structures", BloomLevel.UNDERSTAND, "Comprendre le 'comment'."),
        ("l'optimisation du code", BloomLevel.APPLY, "Savoir mettre en œuvre une solution efficace."),
        ("l'analyse des performances", BloomLevel.ANALYZE, "Identifier les goulots d'étranglement."),
        ("l'architecture du module", BloomLevel.EVALUATE, "Juger la pertinence d'un design."),
        ("la conception de nouveaux outils", BloomLevel.CREATE, "Synthétiser de nouvelles solutions.")
    ]
    
    extended_q_data = random.choices(q_data_pool, k=required_q)
    
    correct_ans_text = "Approche recommandée (Standard industrie)"
    
    # Mapping des mauvaises réponses aux types d'erreurs (pour le feedback)
    distractors_map = {
        "Approche naïve (Fonctionnelle mais lente)": ErrorType.PROCEDURAL, 
        "Approche dépréciée (Obsolète)": ErrorType.MISCONCEPTION, 
        "Approche incorrecte (Bug potentiel)": ErrorType.CONCEPTUAL
    }
    
    all_options = [correct_ans_text] + list(distractors_map.keys())
    
    for i in range(required_q):
        concept_name, bloom_level, description = extended_q_data[i]
        
        # 1. Mélanger les options pour éviter que la bonne réponse soit toujours à la même place
        options_shuffled = all_options[:]
        random.shuffle(options_shuffled)
        
        # 2. Re-créer le dictionnaire des distracteurs avec les options mélangées
        current_distractors = {}
        for ans in options_shuffled:
            if ans != correct_ans_text:
                # On utilise la valeur ErrorType stockée dans distractors_map
                current_distractors[ans] = distractors_map.get(ans, ErrorType.PROCEDURAL)

        # 3. Création de l'objet Question
        q = Question(
            id=f"{topic[:4].replace(' ', '')}-{i+1}",
            text=f"Q{i+1}. **({bloom_level.name})** Sur le module **{topic}**, concernant **{concept_name}**, quelle est l'approche recommandée ?",
            concept=topic, 
            correct_answer=correct_ans_text, 
            distractors=current_distractors,
            bloom_level=bloom_level,
            resources={
                "video": f"Lien Vidéo: Introduction à {concept_name}",
                "step_by_step": f"Guide pratique: {description}",
                "advanced": f"Challenge: Optimisation de {concept_name}"
            }
        )
        questions.append(q)
        
    return questions
    # --- ACTIONS ---
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("🔄 Retour au Menu Principal", key="btn_dash_return", type="primary", use_container_width=True):
            st.session_state.current_level = None
            st.session_state.exam_questions = []
            st.session_state.quiz_submitted = False
            st.session_state.quiz_history = []
            st.rerun()

# ==============================================================================
# 4. FONCTION PRINCIPALE (MAIN)
# ==============================================================================

def main():
    # --- A. INITIALISATION DU STATE ---
    # Initialisation sécurisée des variables de session
    defaults = {
        "page": "exercices",
        "current_level": None,
        "quiz_submitted": False,
        "q_index": 0,
        "score": 0,
        "answer_validated": False,
        "last_res": None,
        "exam_questions": [],
        "quiz_history": []
    }
    
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # --- B. ROUTAGE DE L'APPLICATION ---
    
    # NOTE: Si vous avez d'autres pages (cours, profil), gérez les ici via st.sidebar
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime

# ==============================================================================
# 1. CONFIGURATION & STYLE
# ==============================================================================
try:
    st.set_page_config(
        page_title="DataMaster Pro | Audit IA",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
except:
    pass

# Gestion PDF
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# CSS "Laboratoire de Recherche / Corporate"
st.markdown("""
<style>
    /* Global Theme */
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* Headers */
    h1 { 
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        font-weight: 800; text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Cards Module */
    .level-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px; padding: 20px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .level-card:hover { transform: translateY(-5px); border-color: #00C9FF; box-shadow: 0 0 15px rgba(0, 201, 255, 0.3); }
    
    /* Question Interface */
    .question-box {
        background: #161b22; border-left: 5px solid #00C9FF;
        padding: 25px; border-radius: 8px; margin-bottom: 20px;
        font-size: 1.3rem; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Feedback Colors */
    .feedback-ok { background: rgba(0, 255, 127, 0.1); border: 1px solid #00ff7f; color: #00ff7f; padding: 15px; border-radius: 8px; }
    .feedback-ko { background: rgba(255, 99, 71, 0.1); border: 1px solid #ff6347; color: #ff6347; padding: 15px; border-radius: 8px; }
    
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white; border: none; font-weight: 600; padding: 0.5rem 1rem;
        border-radius: 8px; width: 100%; transition: all 0.3s;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
        color: #000;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. BANQUE DE QUESTIONS (20 MODULES)
# ==============================================================================

def q(question, options, correct_idx, concept, expl=""):
    return {
        "question": question,
        "options": options,
        "correct_index": correct_idx,
        "concept": concept,
        "explanation": expl if expl else f"Réponse attendue : {options[correct_idx]}"
    }

QUESTION_BANK = {
    1: [
        q("Laquelle de ces instructions est correctement écrite en Python pour afficher un texte ?", ["print 'Hello'", "echo('Hello')", "print('Hello')", "afficher('Hello')"], 2, "Syntaxe"),
        q("Dans Python, que se passe-t-il si vous oubliez l’indentation dans une structure comme `if` ?", ["Le programme s’exécute normalement", "Python corrige automatiquement", "Une erreur IndentationError est générée", "Le programme s’arrête sans message"], 2, "Syntaxe"),
        q("Quelle est la bonne manière d’écrire un commentaire en Python ?", ["// Commentaire", "# Commentaire", "<!-- Commentaire -->", "/* Commentaire */"], 1, "Syntaxe"),
        q("Quelle phrase décrit correctement une variable en Python ?", ["Espace mémoire stockant une valeur", "Exécute le code", "Crée des fichiers", "Supprime les données"], 0, "Variables"),
        q("Laquelle de ces affectations est valide en Python ?", ["2age = 18", "age = 18", "age-1 = 18", "age 18 ="], 1, "Variables"),
        q("Quelle est la valeur du type de la variable suivante : x = '25' ?", ["Entier", "Chaîne de caractères", "Flottant", "Booléen"], 1, "Types"),
        q("Si x = 5 puis x = x + 3, quelle sera la nouvelle valeur de x ?", ["3", "5", "8", "53"], 2, "Variables"),
        q("Laquelle de ces boucles affiche les nombres de 1 à 5 ?", ["for i in range(1, 6): print(i)", "for i in range(5): print(i)", "while i < 5: print(i)", "for i in (1,2,3,4,5): stop(i)"], 0, "Boucles"),
        q("Quel est le rôle de la boucle `while` en Python ?", ["Répéter un nombre fixe de fois", "Répéter tant qu’une condition est vraie", "Répéter une seule fois", "Répéter après erreur"], 1, "Boucles"),
        q("Que se passe-t-il si la condition d’un `while` n’est jamais fausse ?", ["Arrêt normal", "Nouvelle condition auto", "Boucle infinie", "Saut de boucle"], 2, "Boucles"),
        q("Quel mot-clé permet de sortir immédiatement d’une boucle ?", ["stop", "break", "exit", "continue"], 1, "Contrôle"),
        q("Quel mot-clé permet de passer à l’itération suivante sans exécuter le reste ?", ["break", "skip", "continue", "pass"], 2, "Contrôle"),
        q("Quel code affiche chaque lettre d’un mot stocké dans `mot` ?", ["for lettre in mot: print(lettre)", "for mot in lettre: print(mot)", "while mot: print(mot[0])", "print(mot)"], 0, "Itération"),
        q("Si total = 0 et boucle range(3): total = total + 1. Valeur finale ?", ["1", "2", "3", "4"], 2, "Logique"),
        q("Affirmation correcte sur une variable dans une boucle ?", ["Ne change jamais", "Peut être mise à jour à chaque itération", "Doit commencer par majuscule", "Doit être supprimée après"], 1, "Variables")
    ],
    2: [
        q("Quelle instruction crée une liste contenant 2, 4, 6 ?", ["liste = 2, 4, 6", "liste = [2, 4, 6]", "liste = (2, 4, 6)", "liste = {2, 4, 6}"], 1, "Listes"),
        q("Si nombres = [10, 20, 30], quel est nombres[1] ?", ["10", "20", "30", "Erreur"], 1, "Indexation"),
        q("Quelle méthode ajoute un élément à la fin d’une liste ?", ["push()", "add()", "append()", "insert()"], 2, "Méthodes"),
        q("items=[1,2,3]; items.append(4). Nouvelle valeur ?", ["[1,2,3]", "[4,1,2,3]", "[1,2,3,4]", "[1,2,3,[4]]"], 2, "Listes"),
        q("Quelle instruction supprime l’élément à l’index 2 ?", ["remove(liste, 2)", "liste.remove(2)", "del liste[2]", "liste.pop(5)"], 2, "Listes"),
        q("Syntaxe pour dico nom:Awa, age:25 ?", ["{'nom'='Awa'...}", "{nom:'Awa'...}", "['nom':'Awa'...]", "{'nom': 'Awa', 'age': 25}"], 3, "Dicos"),
        q("Accéder à la valeur de 'ville' dans personne ?", ["personne('ville')", "personne['ville']", "personne.ville", "personne.get['ville']"], 1, "Dicos"),
        q("Ajouter une nouvelle clé à un dictionnaire ?", ["dico.insert()", "dico.push()", "dico['clé'] = val", "dico.add()"], 2, "Dicos"),
        q("d={'a':1}; d['a']=10. Résultat ?", ["{'a':1}", "{'a':10}", "{'a':1, 'a':10}", "Erreur"], 1, "Dicos"),
        q("Vérifier si une clé existe dans un dico ?", ["'clé' inside dico", "dico.has('clé')", "'clé' in dico", "dico['clé'] == True"], 2, "Dicos"),
        q("Avantage principal d'un Set ?", ["Trié auto", "Doublons acceptés", "Éléments uniques", "Accès par index"], 2, "Sets"),
        q("Créer un set avec 1, 2, 3 ?", ["s=[1,2,3]", "s=(1,2,3)", "s={1,2,3}", "s=set[1,2,3]"], 2, "Sets"),
        q("s={1,2,2,3}. print(s) ?", ["{1,2,2,3}", "[1,2,3]", "{1,2,3}", "Erreur"], 2, "Sets"),
        q("Ajouter un élément à un set ?", ["push()", "add()", "append()", "insert()"], 1, "Sets"),
        q("a={1,2,3}, b={3,4,5}. a & b ?", ["{1,2,3,4,5}", "{3}", "{1,2}", "[]"], 1, "Sets")
    ],
    3: [
        q("Objectif principal d'un tri ?", ["Trouver un élément", "Réorganiser l'ordre", "Supprimer doublons", "Créer liste"], 1, "Tri"),
        q("Principe du tri par sélection ?", ["Insérer à sa place", "Échanger successifs", "Trouver petit et placer début", "Diviser en deux"], 2, "Tri"),
        q("Tri le plus rapide sur gros ensembles ?", ["Bubble sort", "Quick sort", "Insertion sort", "Selection sort"], 1, "Tri"),
        q("Algorithme efficace sur liste presque triée ?", ["Quick sort", "Merge sort", "Insertion sort", "Bubble sort"], 2, "Tri"),
        q("Tri basé sur 'diviser pour régner' ?", ["Insertion", "Quick sort", "Bubble", "Selection"], 1, "Tri"),
        q("Recherche linéaire consiste à ?", ["Chercher au hasard", "Comparer un par un", "Diviser par deux", "Trier auto"], 1, "Recherche"),
        q("Condition pour recherche dichotomique ?", ["Contient doublons", "Liste triée", "Liste petite", "Liste strings"], 1, "Recherche"),
        q("Recherche binaire: élément < milieu. Action ?", ["Chercher partout", "Chercher droite", "Chercher gauche", "Arrêter"], 2, "Recherche"),
        q("Inconvénient recherche linéaire ?", ["Nécessite tri", "Lente pour grandes listes", "Seulement chiffres", "Erreur"], 1, "Recherche"),
        q("Moins de comparaisons sur grande liste triée ?", ["Linéaire", "Aléatoire", "Dichotomique", "Séquentielle"], 2, "Recherche"),
        q("Que mesure Big-O ?", ["Mémoire", "Max étapes vs taille", "Vitesse CPU", "Lignes code"], 1, "Complexité"),
        q("Complexité tri à bulles ?", ["O(n)", "O(log n)", "O(n log n)", "O(n²)"], 3, "Complexité"),
        q("Complexité moyenne Quick Sort ?", ["O(n²)", "O(n)", "O(log n)", "O(n log n)"], 3, "Complexité"),
        q("Complexité recherche linéaire ?", ["O(1)", "O(n)", "O(n²)", "O(log n)"], 1, "Complexité"),
        q("Complexité O(1) signifie ?", ["Dépend taille", "Augmente avec n", "Temps constant", "Double avec taille"], 2, "Complexité")
    ],
    4: [
        q("Objet principal de NumPy ?", ["DataFrame", "ndarray", "Series", "MatrixList"], 1, "Structures"),
        q("Créer un tableau NumPy correctement ?", ["np.array([1,2])", "np.list([1,2])", "np.matrix([1,2])", "np.table([1,2])"], 0, "Syntaxe"),
        q("a.shape pour a=[1,2,3] ?", ["(1,3)", "(3,)", "(3,1)", "[3]"], 1, "Dimensions"),
        q("Matrice 2x2 de zéros ?", ["np.zero(2,2)", "np.zeros((2,2))", "np.null((2,2))", "np.empty"], 1, "Initialisation"),
        q("Opération A @ B ?", ["Addition", "Concaténation", "Mult. élément", "Mult. matricielle"], 3, "Opérations"),
        q("Transposée de [[1,2],[3,4]] ?", ["[[1,2],[3,4]]", "[[1,3],[2,4]]", "[1,3,2,4]", "Erreur"], 1, "Opérations"),
        q("Somme des éléments de a ?", ["a.sum()", "np.add(a)", "sum(a, axis=0)", "a.total()"], 0, "Aggrégations"),
        q("Produit élément par élément ?", ["A @ B", "A * B", "A.dot(B)", "A.mult(B)"], 1, "Opérations"),
        q("Le broadcasting permet de ?", ["Convertir en DF", "Code GPU", "Étendre dimensions auto", "Convertir en liste"], 2, "Broadcasting"),
        q("Opération valide par broadcasting ?", ["Matrice 3x3 + Scalaire", "Matrice + String", "Vecteurs incompatibles", "Aucune"], 0, "Broadcasting"),
        q("Shape de (3,1) + (3,) ?", ["(3,)", "(3,1)", "(3,3)", "Impossible"], 2, "Broadcasting"),
        q("Changer forme d'un tableau ?", ["reshape()", "resize()", "reform()", "shape()"], 0, "Manipulation"),
        q("a=arange(9).reshape(3,3). a[1,:] ?", ["3", "[3,4,5]", "[1,2,3]", "[6,7,8]"], 1, "Slicing"),
        q("Shape (5,1) représente ?", ["Vecteur ligne", "Matrice 5x1", "Matrice 1x5", "Vide"], 1, "Dimensions"),
        q("Moyenne d'un tableau ?", ["np.mean(a)", "np.avg(a)", "a.mean()", "Toutes"], 3, "Statistiques")
    ],
    5: [
        q("Objet cœur de Pandas ?", ["ndarray", "DataFrame", "Matrix", "Table"], 1, "Structures"),
        q("Créer DF depuis dico ?", ["pd.DataFrame({...})", "pd.Frame({...})", "pd.Table({...})", "pd.Dict({...})"], 0, "Création"),
        q("df.shape retourne ?", ["Colonnes", "Valeurs uniques", "Lignes et Colonnes", "Première ligne"], 2, "Propriétés"),
        q("Sélectionner colonne 'Age' ?", ["df('Age')", "df.Age()", "df['Age']", "df.get(Age)"], 2, "Sélection"),
        q("Sélectionner 'Nom' et 'Pays' ?", ["df['Nom','Pays']", "df[['Nom','Pays']]", "df.select()", "df.cols()"], 1, "Sélection"),
        q("Filtrer Age > 18 ?", ["df.Age > 18", "df[df['Age'] > 18]", "df('Age') > 18", "df.where()"], 1, "Filtrage"),
        q("df.iloc[0] sélectionne ?", ["Dernière ligne", "Première ligne", "Première col", "Supprime ligne"], 1, "Indexation"),
        q("df.loc[2] sélectionne ?", ["Ligne index 2", "2ème ligne", "Col 2", "Erreur"], 0, "Indexation"),
        q("Supprimer une colonne ?", ["remove()", "drop(axis=1)", "delete()", "clear()"], 1, "Manipulation"),
        q("Ajouter colonne 'Score' ?", ["df.add()", "df.insert()", "df['Score']=val", "df.push()"], 2, "Manipulation"),
        q("Supprimer lignes avec NaN ?", ["dropna()", "clean()", "remove_na()", "fill()"], 0, "Nettoyage"),
        q("Remplacer NaN par 0 ?", ["replace()", "fillna(0)", "fill()", "dropna(0)"], 1, "Nettoyage"),
        q("Résumé statistique ?", ["summary()", "describe()", "stats()", "info()"], 1, "Stats"),
        q("Compter valeurs uniques ?", ["unique()", "nunique()", "count()", "distinct()"], 1, "Stats"),
        q("Fusionner deux DF ?", ["append()", "join()", "merge()", "mix()"], 2, "Fusion")
    ],
    6: [
        q("Que fait plt.plot() ?", ["Trace une courbe", "Affiche une image", "Crée un tableau", "Sauvegarde"], 0, "Matplotlib Base"),
        q("Afficher le graphique ?", ["plt.show()", "plt.display()", "plt.render()", "plt.draw()"], 0, "Matplotlib Base"),
        q("Ajouter un titre ?", ["plt.header()", "plt.title()", "plt.name()", "plt.top()"], 1, "Matplotlib Base"),
        q("Afficher une légende ?", ["plt.keys()", "plt.legend()", "plt.labels()", "plt.map()"], 1, "Matplotlib Base"),
        q("Changer couleur en rouge ?", ["c='red'", "color='r'", "col='red'", "Toutes"], 3, "Matplotlib Style"),
        q("Fonction Seaborn pour histogramme ?", ["sns.histplot()", "sns.bar()", "sns.count()", "sns.dist()"], 0, "Seaborn Base"),
        q("Scatterplot avec Seaborn ?", ["sns.scatter()", "sns.scatterplot()", "sns.points()", "sns.dots()"], 1, "Seaborn Base"),
        q("Différence Seaborn vs Matplotlib ?", ["Seaborn est bas niveau", "Seaborn est basé sur Matplotlib", "Incompatibles", "Aucune"], 1, "Concepts"),
        q("Changer taille figure Matplotlib ?", ["plt.size()", "plt.figure(figsize=(x,y))", "plt.resize()", "plt.dim()"], 1, "Matplotlib Avancé"),
        q("Grille de sous-graphes ?", ["plt.subplots()", "plt.grids()", "plt.multi()", "plt.matrix()"], 0, "Matplotlib Avancé"),
        q("Relation 3 variables Seaborn ?", ["hue='var3'", "size='var3'", "style='var3'", "Toutes"], 3, "Seaborn Avancé"),
        q("Jointplot sert à ?", ["3D plot", "Distribution jointe + scatter", "Animation", "Réseau"], 1, "Seaborn Avancé"),
        q("Heatmap de corrélation ?", ["sns.heatmap(df.corr())", "sns.corr()", "sns.map()", "sns.hot()"], 0, "Seaborn Avancé"),
        q("Enregistrer figure ?", ["plt.save()", "plt.savefig()", "plt.export()", "plt.download()"], 1, "Export"),
        q("Boxplot sert à ?", ["Voir tendances", "Voir distributions & outliers", "Voir corrélations", "Voir séries"], 1, "Stats Viz")
    ],
    7: [
        q("Que signifie SQL ?", ["Structured Query Language", "Strong Query Logic", "Simple Question List", "System Query Level"], 0, "Base"),
        q("Sélectionner toutes les colonnes ?", ["SELECT ALL", "SELECT *", "GET ALL", "FETCH *"], 1, "Select"),
        q("Filtrer les résultats ?", ["FILTER BY", "WHERE", "HAVING", "WHEN"], 1, "Filtres"),
        q("Trier les résultats ?", ["SORT BY", "ORDER BY", "ARRANGE BY", "GROUP BY"], 1, "Tri"),
        q("Clause pour limiter le nombre de lignes ?", ["TOP", "LIMIT", "MAX", "CAP"], 1, "Limites"),
        q("Sélectionner valeurs uniques ?", ["UNIQUE", "DISTINCT", "DIFFERENT", "SINGLE"], 1, "Select"),
        q("Différence WHERE vs HAVING ?", ["Aucune", "HAVING pour agrégations", "WHERE après GROUP BY", "HAVING avant WHERE"], 1, "Avancé"),
        q("Clé primaire (Primary Key) ?", ["Identifiant unique", "Clé étrangère", "Index texte", "Valeur nulle"], 0, "Modélisation"),
        q("INNER JOIN retourne ?", ["Tout", "Lignes correspondantes aux 2 tables", "Table gauche", "Table droite"], 1, "Jointures"),
        q("Fonction d'agrégation ?", ["SUM()", "ADD()", "TOTAL()", "PLUS()"], 0, "Agrégations"),
        q("Grouper les résultats ?", ["GROUP BY", "CLUSTER BY", "AGGREGATE BY", "COLLECT"], 0, "Agrégations"),
        q("Compter toutes les lignes ?", ["COUNT(*)", "SUM(*)", "TOTAL(*)", "NUMBER(*)"], 0, "Agrégations"),
        q("Window Function pour rang ?", ["RANK()", "ORDER()", "LEVEL()", "STEP()"], 0, "Avancé"),
        q("Sous-requête (Subquery) ?", ["Requête imbriquée", "Requête lente", "Requête internet", "Requête serveur"], 0, "Complexe"),
        q("Performance JOIN dépend de ?", ["Indexation", "Noms colonnes", "Wifi", "Disque"], 0, "Performance")
    ],
    8: [
        q("Différence variable qualitative/quantitative ?", ["Texte vs Nombre", "Discret vs Continu", "Mesurable vs Catégorique", "Aucune"], 2, "Bases"),
        q("Événement en probabilité ?", ["Sous-ensemble de l'univers", "Erreur", "Variable", "Fonction"], 0, "Bases"),
        q("Espérance mathématique ?", ["Moyenne pondérée", "Variance", "Médiane", "Maximum"], 0, "Bases"),
        q("Variance mesure ?", ["La dispersion", "La tendance", "L'asymétrie", "La pointe"], 0, "Bases"),
        q("Loi Normale caractérisée par ?", ["Moyenne et Ecart-type", "Lambda", "Degrés liberté", "Min Max"], 0, "Lois"),
        q("Intervalle ±1σ loi normale ?", ["68%", "95%", "99%", "50%"], 0, "Lois"),
        q("Hypothèse Nulle (H0) ?", ["Pas d'effet/différence", "Effet prouvé", "Erreur modèle", "Hypothèse chercheur"], 0, "Tests"),
        q("P-value < 0.05 signifie ?", ["Rejet H0 (Significatif)", "Accepte H0", "Erreur calcul", "Données fausses"], 0, "Tests"),
        q("Test Student (t-test) compare ?", ["Moyennes", "Variances", "Proportions", "Lois"], 0, "Tests"),
        q("Corrélation 0.9 signifie ?", ["Forte relation positive", "Forte relation négative", "Pas de relation", "Erreur"], 0, "Stats"),
        q("Théorème Central Limite ?", ["Distribution moyenne tend vers Normale", "Tout est aléatoire", "Moyenne = Médiane", "Variance nulle"], 0, "Théorèmes"),
        q("Erreur Type I ?", ["Faux Positif", "Faux Négatif", "Vrai Positif", "Vrai Négatif"], 0, "Tests"),
        q("Régression linéaire estime ?", ["Relation linéaire variables", "Densité", "Classification", "Groupes"], 0, "Modèles"),
        q("Loi de Poisson utilisée pour ?", ["Événements rares/temps", "Jeux hasard", "Tailles humaines", "Salaires"], 0, "Lois"),
        q("Test ANOVA compare ?", ["Plusieurs moyennes", "Deux variances", "Une proportion", "Deux médianes"], 0, "Tests")
    ],
    9: [
        q("Supervisé vs Non-supervisé ?", ["Labels vs Pas de labels", "Rapide vs Lent", "Image vs Texte", "Simple vs Complexe"], 0, "Bases"),
        q("Variable cible (Target) ?", ["Ce qu'on veut prédire", "Les données d'entrée", "Le bruit", "L'erreur"], 0, "Bases"),
        q("Train/Test Split sert à ?", ["Évaluer généralisation", "Augmenter données", "Nettoyer données", "Accélérer"], 0, "Méthodo"),
        q("Overfitting ?", ["Apprend bruit par cœur", "Trop simple", "Pas assez d'entraînement", "Modèle parfait"], 0, "Problèmes"),
        q("Normaliser données ?", ["Mettre à même échelle", "Supprimer", "Trier", "Dupliquer"], 0, "Prep"),
        q("Validation croisée ?", ["Test robuste sur plis", "Valider manuellement", "Croiser les doigts", "Test unique"], 0, "Méthodo"),
        q("Matrice de confusion ?", ["Tableau erreurs/succès", "Erreur code", "Données mélangées", "Graphique 3D"], 0, "Métriques"),
        q("Régression Logistique sert à ?", ["Classification", "Régression", "Clustering", "Réduction"], 0, "Modèles"),
        q("Random Forest ?", ["Ensemble d'arbres", "Un seul arbre", "Forêt aléatoire", "Graphique"], 0, "Modèles"),
        q("K-Means ?", ["Clustering (Centroids)", "Classification", "Régression", "Supervisé"], 0, "Modèles"),
        q("SVM Marge ?", ["Distance séparation max", "Erreur min", "Temps calcul", "Nombre vecteurs"], 0, "Modèles"),
        q("PCA sert à ?", ["Réduire dimensions", "Augmenter données", "Classif", "Nettoyage"], 0, "DimRed"),
        q("GridSearchCV ?", ["Optimiser hyperparamètres", "Chercher données", "Visualiser grille", "Nettoyer"], 0, "Tuning"),
        q("Biais vs Variance ?", ["Underfit vs Overfit", "Erreur vs Temps", "Train vs Test", "Haut vs Bas"], 0, "Théorie"),
        q("Ensemble Learning ?", ["Combiner modèles", "Apprendre seul", "Apprendre tout", "Rien"], 0, "Théorie")
    ],
    10: [
        q("Réseau de neurones ?", ["Inspiré cerveau bio", "Arbre décision", "Base de données", "Script"], 0, "Base"),
        q("Poids (Weights) ?", ["Force connexion", "Importance donnée", "Taille neurone", "Vitesse"], 0, "Base"),
        q("Fonction d'activation ?", ["Non-linéarité", "Addition", "Stockage", "Suppression"], 0, "Base"),
        q("Backpropagation ?", ["Mise à jour poids (Gradient)", "Avancer données", "Sauvegarder", "Initialiser"], 0, "Entraînement"),
        q("Fonction de perte (Loss) ?", ["Erreur à minimiser", "Score à maximiser", "Donnée perdue", "Temps"], 0, "Entraînement"),
        q("Epoch ?", ["Passe complète données", "Une itération", "Une seconde", "Un batch"], 0, "Entraînement"),
        q("Dropout ?", ["Éviter overfitting", "Accélérer", "Ajouter neurones", "Supprimer données"], 0, "Régularisation"),
        q("CNN spécialisé pour ?", ["Images", "Texte", "Son", "Tableaux"], 0, "Architectures"),
        q("Convolution ?", ["Extraction features", "Réduction taille", "Classification", "Tri"], 0, "Opérations"),
        q("Pooling ?", ["Réduire dimension", "Augmenter taille", "Colorier", "Inverser"], 0, "Opérations"),
        q("RNN spécialisé pour ?", ["Séquences", "Images", "Tableaux", "Statique"], 0, "Architectures"),
        q("Vanishing Gradient ?", ["Gradients trop petits", "Gradients trop grands", "Erreur calcul", "Perte données"], 0, "Problèmes"),
        q("Transfer Learning ?", ["Réutiliser modèle pré-entrainé", "Copier données", "Transférer fichiers", "Apprendre zéro"], 0, "Techniques"),
        q("Transformer ?", ["Attention Mechanism", "CNN avancé", "RNN lent", "Robot"], 0, "Architectures"),
        q("GPU utile pour ?", ["Calcul parallèle matriciel", "Stocker fichiers", "Afficher écran", "Rien"], 0, "Hardware")
    ],
    11: [
        q("NLP signifie ?", ["Natural Language Processing", "New Learning Process", "No Language Problem", "Neural Link Protocol"], 0, "Base"),
        q("Tokenization ?", ["Découper texte en unités", "Crypter texte", "Traduire", "Compter mots"], 0, "Preprocessing"),
        q("Stopwords ?", ["Mots courants (le, la...)", "Mots d'arrêt", "Mots clés", "Erreurs"], 0, "Preprocessing"),
        q("Lemmatization ?", ["Ramener forme canonique", "Couper fin mot", "Traduire", "Corriger"], 0, "Preprocessing"),
        q("TF-IDF ?", ["Importance mot/document", "Fréquence simple", "Traduction", "Grammaire"], 0, "Métriques"),
        q("Word Embedding ?", ["Vecteur sémantique", "Mot unique", "Liste mots", "Dictionnaire"], 0, "Représentation"),
        q("Transformer architecture ?", ["Attention is all you need", "Recurrent only", "Convolutional", "Linear"], 0, "Modèles"),
        q("BERT est ?", ["Bidirectional Encoder", "Unidirectionnel", "Générateur", "Traducteur"], 0, "Modèles"),
        q("GPT est ?", ["Generative Pre-trained Transformer", "General Process Tool", "Graph Pre-training", "Google Protocol"], 0, "Modèles"),
        q("Self-Attention ?", ["Pondérer importance mots contexte", "Regarder soi-même", "Attention utilisateur", "Focus image"], 0, "Mécanisme"),
        q("Fine-tuning ?", ["Adapter modèle pré-entrainé", "Entraîner zéro", "Régler vitesse", "Nettoyer"], 0, "Entraînement"),
        q("Sentiment Analysis ?", ["Classifier émotion", "Traduire", "Résumer", "Corriger"], 0, "Tâches"),
        q("NER (Named Entity Recognition) ?", ["Identifier Noms/Lieux...", "Corriger fautes", "Compter verbes", "Traduire"], 0, "Tâches"),
        q("Sequence-to-Sequence ?", ["Traduction/Résumé", "Classification image", "Clustering", "Régression"], 0, "Architectures"),
        q("Biais en NLP ?", ["Stéréotypes dans données", "Erreur code", "Bruit micro", "Faute frappe"], 0, "Éthique")
    ],
    12: [
        q("Pixel ?", ["Point image (R,G,B)", "Vecteur", "Son", "Texte"], 0, "Base"),
        q("OpenCV sert à ?", ["Traitement d'images", "Traitement texte", "Base données", "Serveur web"], 0, "Outils"),
        q("CNN : Filtre/Kernel ?", ["Détecter features (bords...)", "Flouter", "Colorier", "Supprimer"], 0, "CNN"),
        q("ReLU dans CNN ?", ["Non-linéarité (max(0,x))", "Réduction", "Normalisation", "Probabilité"], 0, "Fonctions"),
        q("MaxPooling ?", ["Réduire taille (garder max)", "Moyenne", "Augmenter", "Inverser"], 0, "Pooling"),
        q("Classification vs Détection ?", ["Quoi vs Quoi + Où", "Pareil", "Image vs Vidéo", "Simple vs Dur"], 0, "Tâches"),
        q("YOLO signifie ?", ["You Only Look Once", "Yellow Object Locator", "Yearly Object Log", "Young Online Learner"], 0, "Modèles"),
        q("Bounding Box ?", ["Rectangle autour objet", "Boîte noire", "Erreur", "Pixel"], 0, "Détection"),
        q("Data Augmentation ?", ["Créer variations images", "Acheter données", "Supprimer", "Compresser"], 0, "Preprocessing"),
        q("Transfer Learning Vision ?", ["Utiliser ResNet/VGG pré-entrainé", "Dessiner", "Prendre photos", "Scanner"], 0, "Techniques"),
        q("Segmentation ?", ["Classer chaque pixel", "Classer image", "Trouver boîte", "Rogner"], 0, "Tâches"),
        q("Intersection over Union (IoU) ?", ["Mesure précision boîte", "Union images", "Intersection routes", "Erreur"], 0, "Métriques"),
        q("Image en niveaux de gris ?", ["1 canal", "3 canaux", "4 canaux", "0 canal"], 0, "Images"),
        q("Normalisation pixels ?", ["Diviser par 255 (0-1)", "Multiplier par 100", "Rien", "Mettre au carré"], 0, "Preprocessing"),
        q("Non-Max Suppression ?", ["Garder meilleure boîte", "Supprimer tout", "Garder tout", "Inverser"], 0, "YOLO")
    ],
    13: [
        q("Big Data 3V ?", ["Volume, Vélocité, Variété", "Vitesse, Valeur, Vue", "Virtuel, Vital, Vrai", "Video, Voice, View"], 0, "Base"),
        q("Hadoop vs Spark ?", ["Disque (MapReduce) vs Mémoire (RAM)", "Même chose", "Spark est une DB", "Hadoop est un langage"], 0, "Outils"),
        q("Spark RDD ?", ["Resilient Distributed Dataset", "Raw Data Disk", "Real Distributed Data", "Rapid Data Drive"], 0, "Spark Core"),
        q("Spark DataFrame ?", ["Organisé en colonnes", "Texte brut", "Image", "Liste"], 0, "Spark SQL"),
        q("Lazy Evaluation ?", ["Exécution au moment de l'action", "Lenteur", "Paresse développeur", "Erreur"], 0, "Spark Concept"),
        q("Transformation vs Action ?", ["Planifier vs Exécuter", "Pareil", "Lire vs Écrire", "Entrée vs Sortie"], 0, "Spark Concept"),
        q("Cluster ?", ["Groupe ordinateurs (Noeuds)", "Un seul PC", "Base données", "Disque dur"], 0, "Architecture"),
        q("Shuffling ?", ["Redistribution données réseau (Coûteux)", "Mélanger cartes", "Trier", "Supprimer"], 0, "Performance"),
        q("Parquet format ?", ["Colonnaire compressé", "Texte brut", "Image", "Audio"], 0, "Stockage"),
        q("MapReduce ?", ["Diviser et Conquérir", "Ajouter et Soustraire", "Lire et Écrire", "Copier Coller"], 0, "Paradigme"),
        q("HDFS ?", ["Hadoop Distributed File System", "High Data File", "Hard Disk File", "Hyper Data System"], 0, "Stockage"),
        q("Spark Streaming ?", ["Traitement temps réel", "Vidéo", "Audio", "Batch"], 0, "Streaming"),
        q("Broadcast Variable ?", ["Envoyer copie à tous noeuds", "Variable globale", "Variable locale", "Erreur"], 0, "Optimisation"),
        q("Executor ?", ["Processus worker", "Chef projet", "Utilisateur", "Disque"], 0, "Architecture"),
        q("Partitioning ?", ["Diviser données pour parallélisme", "Formater disque", "Supprimer", "Visualiser"], 0, "Optimisation")
    ],
    14: [
        q("IaaS vs PaaS vs SaaS ?", ["Infra vs Plateforme vs Logiciel", "Internet vs PC vs Serveur", "Intra vs Pro vs Super", "Aucune"], 0, "Modèles"),
        q("AWS EC2 ?", ["Serveur Virtuel (Compute)", "Stockage", "Base données", "Réseau"], 0, "Services"),
        q("AWS S3 ?", ["Stockage Objet", "Serveur", "Base données", "DNS"], 0, "Services"),
        q("Scalability ?", ["Adapter ressources demande", "Aller vite", "Pesez lourd", "Être grand"], 0, "Concepts"),
        q("Serverless (Lambda) ?", ["Code sans gérer serveur", "Pas de code", "Serveur gratuit", "Site web"], 0, "Concepts"),
        q("Docker Container ?", ["App isolée avec dépendances", "Machine virtuelle lourde", "Fichier texte", "Image disque"], 0, "Conteneurs"),
        q("Kubernetes ?", ["Orchestrateur conteneurs", "Base données", "Langage", "OS"], 0, "Orchestration"),
        q("Region vs Availability Zone ?", ["Géo vs Data Center", "Ville vs Pays", "Grand vs Petit", "Même chose"], 0, "Architecture"),
        q("Load Balancer ?", ["Répartir trafic", "Stocker données", "Calculer", "Sécuriser"], 0, "Réseau"),
        q("IaC (Terraform) ?", ["Infrastructure as Code", "Internet a Code", "Intel and Core", "Image and Container"], 0, "DevOps"),
        q("VPC ?", ["Réseau Privé Virtuel", "PC Virtuel", "Very Personal Computer", "Visual Private Cloud"], 0, "Réseau"),
        q("CI/CD ?", ["Intégration/Déploiement Continus", "Code In Code Down", "Cloud I/O", "Copy Insert"], 0, "DevOps"),
        q("RDS ?", ["Base données relationnelle gérée", "Stockage", "Calcul", "DNS"], 0, "Services"),
        q("High Availability ?", ["Système toujours dispo", "Haute vitesse", "Haute altitude", "Cher"], 0, "Concepts"),
        q("Cost Management ?", ["Gérer dépenses Cloud", "Gagner argent", "Payer moins impôts", "Vendre"], 0, "FinOps")
    ],
    15: [
        q("MLOps but ?", ["Unifier ML et Ops (Prod)", "Faire du ML", "Faire des Ops", "Coder vite"], 0, "Base"),
        q("Data Drift ?", ["Changement distribution données", "Perte données", "Erreur disque", "Vitesse"], 0, "Monitoring"),
        q("Model Registry ?", ["Stocker/Versionner modèles", "Liste utilisateurs", "Registre Windows", "Log"], 0, "Outils"),
        q("Feature Store ?", ["Partager features (Train/Serve)", "Magasin", "Stockage fichiers", "Base données"], 0, "Outils"),
        q("Reproducibility ?", ["Refaire même résultat", "Copier coller", "Imprimer", "Produire"], 0, "Principe"),
        q("CI/CD pour ML ?", ["Automatiser train/deploy", "Juste code", "Installer Python", "Rien"], 0, "DevOps"),
        q("Canary Deployment ?", ["Déploiement progressif", "Déploiement oiseau", "Tout casser", "Test local"], 0, "Stratégie"),
        q("Monitoring Modèle ?", ["Surveiller Perf/Drift", "Regarder écran", "Vérifier électricité", "Rien"], 0, "Ops"),
        q("Docker en ML ?", ["Env isolement reproductible", "Lourd", "Inutile", "Compliqué"], 0, "Outils"),
        q("Kubeflow ?", ["Toolkit ML sur K8s", "Flux cube", "Outil Amazon", "Langage"], 0, "Outils"),
        q("A/B Testing ?", ["Comparer 2 versions", "Test alphabet", "Test sanguin", "Aucun"], 0, "Expérimentation"),
        q("Pipeline ML ?", ["Workflow automatisé", "Tuyau gaz", "Ligne code", "Câble"], 0, "Automation"),
        q("Serving ?", ["Mettre modèle dispo (API)", "Servir café", "Attendre", "Stocker"], 0, "Inférence"),
        q("Experiment Tracking (MLflow) ?", ["Suivre paramètres/métriques", "Suivre souris", "Espionner", "Rien"], 0, "Outils"),
        q("Retraining ?", ["Ré-entraîner sur nouvelles données", "Apprendre à lire", "Entraînement sportif", "Arrêter"], 0, "Cycle vie")
    ],
    16: [
        q("Série Temporelle ?", ["Données indexées par temps", "Série télé", "Liste aléatoire", "Image"], 0, "Base"),
        q("Stationnarité ?", ["Propriétés stats constantes", "Immobile", "Gare", "Tendance forte"], 0, "Concept"),
        q("Saisonnalité ?", ["Motif répétitif périodique", "Été/Hiver", "Épices", "Bruit"], 0, "Composante"),
        q("ARIMA ?", ["AutoRegressive Integrated Moving Avg", "Algorithm R", "Area Image", "Aucun"], 0, "Modèles"),
        q("Lag (Retard) ?", ["Valeur passée (t-k)", "Bug ordi", "Futur", "Erreur"], 0, "Concept"),
        q("Autocorrélation ?", ["Corrélation avec soi-même (décalé)", "Corrélation auto", "Voiture", "Rien"], 0, "Stats"),
        q("Prophet (Facebook) ?", ["Modèle additif robuste", "Devin", "Religion", "Réseau social"], 0, "Modèles"),
        q("LSTM pour Time Series ?", ["Mémoire long terme (Séquences)", "Image", "Tableau", "Son"], 0, "Deep Learning"),
        q("Forecasting ?", ["Prédiction futur", "Météo", "Casting", "Passé"], 0, "Tâche"),
        q("Tendance (Trend) ?", ["Direction long terme", "Mode", "Bruit", "Cycle"], 0, "Composante"),
        q("Bruit (Noise) ?", ["Variation aléatoire", "Son fort", "Erreur", "Musique"], 0, "Composante"),
        q("Differencing ?", ["Rendre stationnaire (t - t-1)", "Différence", "Diviser", "Multiplier"], 0, "Preprocessing"),
        q("Rolling Window ?", ["Fenêtre glissante (Moyenne...)", "Fenêtre Windows", "Rouler", "Fermer"], 0, "Technique"),
        q("Metric MAE ?", ["Mean Absolute Error", "Max Area", "Min Avg", "Most Accurate"], 0, "Métriques"),
        q("Multivarié ?", ["Plusieurs variables dépendantes", "Une seule", "Variable texte", "Aucune"], 0, "Données")
    ],
    17: [
        q("Agent ?", ["Entité qui agit", "Espion", "Robot", "Utilisateur"], 0, "Base"),
        q("Environnement ?", ["Monde avec lequel interagit agent", "Nature", "Bureau", "Code"], 0, "Base"),
        q("Récompense (Reward) ?", ["Signal retour (Bien/Mal)", "Argent", "Cadeau", "Score final"], 0, "Base"),
        q("Exploration vs Exploitation ?", ["Découvrir vs Utiliser acquis", "Voyager vs Travailler", "Chercher vs Trouver", "Rien"], 0, "Dilemme"),
        q("Q-Learning ?", ["Apprendre valeur action-état", "Question Learning", "Quick Learning", "Queue"], 0, "Algorithme"),
        q("Policy (Politique) ?", ["Stratégie choix action", "Loi", "Police", "Contrat"], 0, "Base"),
        q("Markov Decision Process ?", ["Cadre mathématique RL", "Marque", "Processus lent", "Aucun"], 0, "Théorie"),
        q("Deep Q-Network (DQN) ?", ["Q-Learning avec Réseau Neurones", "Deep Question", "Donut", "Dark Net"], 0, "Deep RL"),
        q("Discount Factor (Gamma) ?", ["Importance futur vs présent", "Promotion", "Maths", "Lumière"], 0, "Paramètre"),
        q("Episode ?", ["Séquence début à fin", "Série TV", "Chapitre", "Erreur"], 0, "Concept"),
        q("State (État) ?", ["Situation courante", "Pays", "Solide/Liquide", "Statistique"], 0, "Base"),
        q("Action Space ?", ["Actions possibles", "Espace", "Clavier", "Souris"], 0, "Base"),
        q("AlphaGo ?", ["IA Go (DeepMind)", "Jeu", "Alphabet", "Robot"], 0, "Histoire"),
        q("Reward Shaping ?", ["Guider apprentissage via récompenses", "Forme", "Tricher", "Dessiner"], 0, "Technique"),
        q("Off-policy vs On-policy ?", ["Apprendre d'autres vs propre exp", "Politique", "Interrupteur", "Rien"], 0, "Théorie")
    ],
    18: [
        q("Graphe ?", ["Noeuds et Arêtes", "Graphique Excel", "Dessin", "Courbe"], 0, "Base"),
        q("Node (Sommet) ?", ["Entité", "Point", "Ordi", "Tout"], 3, "Base"),
        q("Edge (Arête) ?", ["Relation/Lien", "Bord", "Coin", "Ligne"], 0, "Base"),
        q("Graphe Orienté ?", ["Sens défini (A->B)", "Boussole", "Perdu", "Carte"], 0, "Type"),
        q("NetworkX ?", ["Lib Python Graphes", "Réseau X", "Internet", "Connexion"], 0, "Outil"),
        q("Plus court chemin (Dijkstra) ?", ["Chemin coût min", "Ligne droite", "Vol oiseau", "Rapide"], 0, "Algo"),
        q("Centralité ?", ["Importance noeud", "Centre ville", "Moyenne", "Milieu"], 0, "Analyse"),
        q("PageRank ?", ["Algo Google (Importance)", "Rang page", "Livre", "Site"], 0, "Algo"),
        q("Connexe ?", ["Tout relié", "Internet", "Wifi", "Prise"], 0, "Propriété"),
        q("Cycle ?", ["Chemin revient départ", "Vélo", "Rond", "Tourne"], 0, "Propriété"),
        q("Matrice Adjacence ?", ["Représentation matricielle liens", "Tableau", "Excel", "Liste"], 0, "Représentation"),
        q("Graphe Biparti ?", ["2 ensembles disjoints", "Politique", "Deux parties", "Double"], 0, "Type"),
        q("Community Detection ?", ["Trouver groupes denses", "Police", "Voisins", "Social"], 0, "Analyse"),
        q("Degree (Degré) ?", ["Nombre connexions", "Température", "Angle", "Diplôme"], 0, "Mesure"),
        q("BFS vs DFS ?", ["Largeur vs Profondeur", "Rapide vs Lent", "Haut vs Bas", "Gauche vs Droite"], 0, "Parcours")
    ],
    19: [
        q("Anomalie ?", ["Déviation norme", "Erreur code", "Virus", "Panne"], 0, "Base"),
        q("Intrusion Detection System ?", ["Détecter accès non autorisé", "Antivirus", "Caméra", "Porte"], 0, "Sécurité"),
        q("False Positive ?", ["Alerte fausse (Normal marqué Anormal)", "Vrai positif", "Erreur", "Bon"], 0, "Métriques"),
        q("Log Analysis ?", ["Analyser traces événements", "Couper bois", "Maths", "Blog"], 0, "Technique"),
        q("Zero-day attack ?", ["Faille inconnue exploitée", "Attaque nulle", "Jour zéro", "Vieux virus"], 0, "Menace"),
        q("Isolation Forest ?", ["Algo détection anomalies", "Forêt", "Seul", "Arbre"], 0, "ML Algo"),
        q("DDoS ?", ["Déni service distribué", "Dos", "Disque", "Données"], 0, "Attaque"),
        q("Phishing ?", ["Hameçonnage (Email)", "Pêche", "Sport", "Virus"], 0, "Attaque"),
        q("Encryption ?", ["Chiffrer données", "Cacher", "Supprimer", "Compter"], 0, "Protection"),
        q("SIEM ?", ["Gestion info/événements sécu", "Sims", "Semence", "Système"], 0, "Outil"),
        q("Malware ?", ["Logiciel malveillant", "Materiel", "Bien", "Mauvais"], 0, "Menace"),
        q("Autoencoder pour Anomalies ?", ["Reconstruction error élevée = Anomalie", "Compresser", "Encoder", "Rien"], 0, "Deep Learning"),
        q("Network Traffic ?", ["Flux données réseau", "Voitures", "Gens", "Embouteillage"], 0, "Données"),
        q("User Behavior Analytics ?", ["Analyser comportement utilisateur", "Espionner", "Vendre", "Rien"], 0, "Analyse"),
        q("Threat Intelligence ?", ["Renseignement menaces", "QI", "Espion", "Police"], 0, "Domaine")
    ],
    20: [
        q("Biais Algorithmique ?", ["Erreur systématique (Discrimination)", "Bug", "Biais cognitif", "Rien"], 0, "Problème"),
        q("Fairness (Équité) ?", ["Traiter groupes équitablement", "Juste", "Fête", "Égalité stricte"], 0, "Principe"),
        q("GDPR (RGPD) ?", ["Protection données perso UE", "Loi USA", "Règlement foot", "Code"], 0, "Loi"),
        q("Explainability (XAI) ?", ["Comprendre décisions modèle", "Expliquer code", "Parler", "Rien"], 0, "Principe"),
        q("Black Box ?", ["Modèle opaque", "Boîte noire avion", "Magie", "Nuit"], 0, "Concept"),
        q("Data Privacy ?", ["Confidentialité données", "Données privées", "Secret", "Cacher"], 0, "Principe"),
        q("Accountability ?", ["Responsabilité", "Comptabilité", "Compte", "Nombre"], 0, "Principe"),
        q("Deepfake ?", ["Média synthétique réaliste", "Faux profond", "Mensonge", "Vidéo"], 0, "Menace"),
        q("Biais de sélection ?", ["Données non représentatives", "Choisir", "Élection", "Erreur"], 0, "Source Biais"),
        q("Right to explanation ?", ["Droit explication (GDPR)", "Droit parler", "Droit vote", "Rien"], 0, "Droit"),
        q("Automation Bias ?", ["Trop confiance machine", "Automate", "Robot", "Biais auto"], 0, "Biais Humain"),
        q("Ethical AI ?", ["IA respectant valeurs/droits", "IA gentille", "Robot", "Philosophie"], 0, "Domaine"),
        q("Surveillance ?", ["Suivi masse", "Caméra", "Regarder", "Sécurité"], 0, "Risque"),
        q("Transparence ?", ["Clarté fonctionnement/données", "Invisible", "Verre", "Rien"], 0, "Principe"),
        q("Impact social ?", ["Conséquences société (Emploi...)", "Réseaux sociaux", "Amis", "Rien"], 0, "Analyse")
    ]
}

LEVELS = {
    1: {"title": "Python: Les Fondations", "icon": "🐍", "desc": "Syntaxe, Variables, Boucles"},
    2: {"title": "Structures de Données", "icon": "📚", "desc": "Listes, Dicos, Sets"},
    3: {"title": "Algorithmique", "icon": "⚙️", "desc": "Tri, Recherche, Complexité"},
    4: {"title": "NumPy & Calcul", "icon": "🔢", "desc": "Matrices, Broadcasting"},
    5: {"title": "Pandas: DataFrames", "icon": "🐼", "desc": "Manipulation & Analyse"},
    6: {"title": "Data Visualization", "icon": "📊", "desc": "Matplotlib, Seaborn"},
    7: {"title": "SQL Databases", "icon": "🗄️", "desc": "Queries, Joins, Aggregations"},
    8: {"title": "Probabilités & Stats", "icon": "🎲", "desc": "Lois, Tests, Inférence"},
    9: {"title": "Machine Learning", "icon": "🤖", "desc": "Scikit-Learn, Modèles"},
    10: {"title": "Deep Learning", "icon": "🧠", "desc": "Neural Networks, Backprop"},
    11: {"title": "NLP Processing", "icon": "🗣️", "desc": "Tokenization, Transformers"},
    12: {"title": "Computer Vision", "icon": "👁️", "desc": "CNN, OpenCV, Yolo"},
    13: {"title": "Big Data (Spark)", "icon": "🐘", "desc": "RDD, Dataframes, Cluster"},
    14: {"title": "Cloud (AWS/Azure)", "icon": "☁️", "desc": "Services, Deployment"},
    15: {"title": "MLOps", "icon": "🚢", "desc": "Docker, Kubernetes, CI/CD"},
    16: {"title": "Time Series", "icon": "📈", "desc": "ARIMA, LSTM, Prophet"},
    17: {"title": "Reinforcement", "icon": "🎮", "desc": "Agents, Q-Learning"},
    18: {"title": "Graph Theory", "icon": "🕸️", "desc": "NetworkX, Nodes, Edges"},
    19: {"title": "Cybersecurity Data", "icon": "🔒", "desc": "Anomaly Detection"},
    20: {"title": "Ethics in AI", "icon": "⚖️", "desc": "Bias, Fairness, GDPR"}
}

# ==============================================================================
# 3. MOTEUR KPI
# ==============================================================================

@dataclass
class PedagogicalKPIs:
    retention_rate: float
    engagement_rate: float
    learning_velocity: float
    difficulty_index: Dict[str, float]
    dropout_risk: str
    dropout_probability: float
    meta_data: Dict[str, any] = field(default_factory=dict)

class KPICalculator:
    def compute(self, df: pd.DataFrame) -> PedagogicalKPIs:
        if df.empty: return PedagogicalKPIs(0, 0, 0, {}, "N/A", 0)
        
        # Vélocité
        df['adjusted_time'] = df.apply(lambda x: x['time'] if x['correct'] else x['time'] * 1.5, axis=1)
        avg_time = df['adjusted_time'].mean()
        velocity = (10 / avg_time) if avg_time > 0 else 0
        
        # Engagement & Difficulté
        accuracy = df['correct'].mean()
        difficulty = df.groupby('concept')['correct'].mean().apply(lambda x: 1 - x).to_dict()
        
        # Tendance
        y = df['correct'].astype(int).values
        trend = 0
        if len(y) > 1:
            trend = np.polyfit(np.arange(len(y)), y, 1)[0]
            
        # Risque
        risk_score = (1 - accuracy) * 0.6 + (1 / (velocity + 1)) * 0.4
        if trend < -0.1: risk_score += 0.2
        risk_label = "CRITIQUE 🚨" if risk_score > 0.7 else ("ATTENTION ⚠️" if risk_score > 0.4 else "OPTIMAL")
        
        # Streak
        streak, max_streak = 0, 0
        for res in df['correct']:
            if res: streak += 1
            else:
                max_streak = max(max_streak, streak)
                streak = 0
        max_streak = max(max_streak, streak)

        return PedagogicalKPIs(0.95, accuracy, round(velocity, 2), difficulty, risk_label, round(risk_score, 2),
                               {"trend": trend, "max_streak": max_streak, "fastest": df['time'].min()})

# ==============================================================================
# 4. FONCTIONS UTILITAIRES (PDF & GRAPHICS)
# ==============================================================================

def clean_text_for_pdf(text):
    if not isinstance(text, str): return str(text)
    # Nettoyage des emojis pour éviter les erreurs d'encodage Latin-1
    text = text.replace("🚨", "").replace("⚠️", "").replace("✅", "").replace("❌", "").replace("📈", "").replace("📉", "")
    return text.encode('latin-1', 'replace').decode('latin-1')

def generate_pdf_report(user, level_name, df, kpi):
    if not FPDF_AVAILABLE: return None
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- 1. EN-TÊTE PRO ---
    pdf.set_fill_color(14, 17, 23)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 24)
    pdf.set_xy(0, 10)
    pdf.cell(210, 10, clean_text_for_pdf("RAPPORT D'ANALYSE DE COMPETENCES"), 0, 1, 'C')
    pdf.set_text_color(0, 201, 255)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(210, 10, clean_text_for_pdf(f"MODULE : {level_name.upper()}"), 0, 1, 'C')
    
    pdf.ln(20)
    
    # --- 2. INFO CANDIDAT ---
    pdf.set_text_color(50, 50, 50)
    pdf.set_font("Arial", '', 11)
    pdf.set_fill_color(240, 240, 245)
    pdf.set_draw_color(200, 200, 200)
    date_str = datetime.now().strftime("%d/%m/%Y")
    
    pdf.rect(10, 45, 190, 20, 'FD')
    pdf.set_xy(15, 50)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(30, 10, "CANDIDAT :", 0, 0)
    pdf.set_font("Arial", '', 11)
    pdf.cell(60, 10, clean_text_for_pdf(user), 0, 0)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(20, 10, "DATE :", 0, 0)
    pdf.set_font("Arial", '', 11)
    pdf.cell(40, 10, date_str, 0, 1)
    
    pdf.ln(10)
    
    # --- 3. CITATION INSPIRANTE ---
    pdf.set_text_color(100, 100, 100)
    pdf.set_font("Times", 'I', 12)
    pdf.multi_cell(0, 8, clean_text_for_pdf('"Sans donnees, vous n\'etes qu\'une autre personne avec une opinion." - W. Edwards Deming'), 0, 'C')
    pdf.ln(5)

    # --- 4. RECOMMANDATIONS PÉDAGOGIQUES ---
    # Logique de recommandation
    score_pct = kpi.engagement_rate * 100
    velocity = kpi.learning_velocity
    advice = ""
    
    if score_pct >= 80:
        advice += "Niveau Expert atteint. Recommandation : Approfondir les cas limites et l'optimisation avancee pour viser la perfection. Vous etes pret pour des projets complexes."
    elif score_pct >= 50:
        advice += "Niveau Intermediaire valide. Recommandation : Revoyez les concepts marques en erreur dans le tableau ci-dessous. Pratiquez regulierement pour ancrer les connaissances."
    else:
        advice += "Niveau Debutant. Recommandation : Il est conseille de reprendre le cours theorique associe a ce module. Concentrez-vous sur les fondamentaux avant de reessayer."
    
    if velocity > 0.5:
        advice += " Votre rapidite est un atout, mais attention a la precipitation qui peut induire des erreurs d'inattention."
    else:
        advice += " Prenez confiance en vous pour accelerer progressivement votre prise de decision."

    pdf.set_fill_color(245, 245, 255)
    pdf.set_draw_color(0, 201, 255)
    pdf.rect(10, pdf.get_y(), 190, 25, 'FD')
    pdf.set_xy(15, pdf.get_y()+2)
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(0, 50, 100)
    pdf.cell(0, 6, "CONSEILS PERSONNALISES :", 0, 1)
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(180, 5, clean_text_for_pdf(advice))
    pdf.ln(10)

    # --- 5. KPIs (Encadrés) ---
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "PERFORMANCE ANALYTIQUE", 0, 1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font("Courier", '', 11)
    risk_txt = clean_text_for_pdf(kpi.dropout_risk)
    
    col_width = 95
    pdf.cell(col_width, 8, f"Score Global: {int(kpi.engagement_rate * 100)}%", 1, 0)
    pdf.cell(col_width, 8, f"Vitesse Cognitive: {kpi.learning_velocity} pts", 1, 1)
    pdf.cell(col_width, 8, f"Indice Risque: {risk_txt}", 1, 0)
    pdf.cell(col_width, 8, f"Max Streak: {kpi.meta_data.get('max_streak',0)}", 1, 1)
    pdf.ln(10)
    
    # --- 6. TABLEAU DE RÉPONSES ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "DETAIL SEQUENTIEL", 0, 1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(15, 8, "#", 1, 0, 'C', 1)
    pdf.cell(115, 8, "CONCEPT EVALUE", 1, 0, 'L', 1)
    pdf.cell(30, 8, "STATUT", 1, 0, 'C', 1)
    pdf.cell(30, 8, "TEMPS (s)", 1, 1, 'C', 1)
    
    pdf.set_font("Arial", '', 10)
    for i, row in df.iterrows():
        status = "VALIDE" if row['correct'] else "ERREUR"
        if row['correct']:
            pdf.set_text_color(0, 100, 0)
        else:
            pdf.set_text_color(150, 0, 0)
        concept = clean_text_for_pdf(row['concept'])
        time_s = f"{row['time']:.1f}"
        pdf.cell(15, 8, str(i+1), 1, 0, 'C')
        pdf.cell(115, 8, concept, 1, 0, 'L')
        pdf.cell(30, 8, status, 1, 0, 'C')
        pdf.cell(30, 8, time_s, 1, 1, 'C')
        
    pdf.set_text_color(0, 0, 0)
    pdf.set_y(-20)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, clean_text_for_pdf("Genere par DataMaster Pro - Plateforme d'Evaluation IA"), 0, 0, 'C')
    
    return bytes(pdf.output())

def show_dashboard_final():
    st.balloons()
    history = st.session_state.quiz_history
    lvl_id = st.session_state.current_level
    lvl_info = LEVELS[lvl_id]
    df = pd.DataFrame(history)
    df['status'] = df['correct'].apply(lambda x: 'Valide' if x else 'Erreur')
    df['xp'] = df['correct'].astype(int).cumsum() * 10
    df['q_num'] = range(1, len(df) + 1)
    
    # Rolling Accuracy (Moyenne Glissante sur 3 questions)
    df['rolling_acc'] = df['correct'].astype(int).rolling(window=3, min_periods=1).mean()
    
    engine = KPICalculator()
    kpi = engine.compute(df)
    
    st.markdown(f"## 🏆 Rapport : {lvl_info['title']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Précision", f"{int(kpi.engagement_rate*100)}%", f"{df['correct'].sum()}/{len(df)}")
    c2.metric("Vélocité", f"{kpi.learning_velocity}", "pts/sec")
    c3.metric("Record Série", f"{kpi.meta_data.get('max_streak',0)}", "réps")
    c4.metric("Risque", kpi.dropout_risk, delta_color="inverse")
    
    st.markdown("---")
    
    # --- 18 GRAPHIQUES (6 par Tab) ---
    tab1, tab2, tab3 = st.tabs(["🧠 Vue Globale", "⏱️ Chronométrie", "📊 Conversion"])
    
    # TAB 1 : VUE GLOBALE (6 Graphs)
    with tab1:
        ca, cb = st.columns(2)
        with ca:
            # 1. Gauge
            fig_g = go.Figure(go.Indicator(mode="gauge+number", value=kpi.engagement_rate*100, title={'text':"Score Global"}, gauge={'axis':{'range':[0,100]}, 'bar':{'color':"#00C9FF"}}))
            fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", height=300)
            st.plotly_chart(fig_g, use_container_width=True)
            
            # 2. Radar
            grp = df.groupby('concept')['correct'].mean().reset_index()
            fig_r = px.line_polar(grp, r='correct', theta='concept', line_close=True, range_r=[0,1], title="Maîtrise Concept")
            fig_r.update_traces(fill='toself', line_color='#00C9FF')
            fig_r.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig_r, use_container_width=True)
            
            # 3. Sunburst
            fig_sun = px.sunburst(df, path=['status', 'concept'], title="Hiérarchie Résultats")
            fig_sun.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig_sun, use_container_width=True)

        with cb:
            # 4. Area XP
            fig_a = px.area(df, x='q_num', y='xp', title="Progression XP")
            fig_a.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_a, use_container_width=True)
            
            # 5. Indicators
            fig_ind = go.Figure()
            fig_ind.add_trace(go.Indicator(mode="number+delta", value=kpi.meta_data.get('fastest',0), title={"text":"Record Vitesse"}, domain={'row':0, 'column':0}))
            fig_ind.add_trace(go.Indicator(mode="number", value=df['time'].mean(), title={"text":"Moyenne (s)"}, domain={'row':0, 'column':1}))
            fig_ind.update_layout(grid={'rows':1, 'columns':2}, paper_bgcolor='rgba(0,0,0,0)', font_color="white", height=300)
            st.plotly_chart(fig_ind, use_container_width=True)

            # 6. Treemap (Nouveau) - Poids temporel des concepts
            fig_tree = px.treemap(df, path=['concept'], values='time', color='correct', title="Impact Temps par Concept")
            fig_tree.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig_tree, use_container_width=True)

    # TAB 2 : CHRONOMÉTRIE (6 Graphs)
    with tab2:
        cc, cd = st.columns(2)
        with cc:
            # 7. Scatter Time vs Status
            fig_s = px.scatter(df, x='time', y='concept', color='status', title="Temps vs Statut", size='time')
            fig_s.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_s, use_container_width=True)
            
            # 8. Line Velocity
            fig_l = px.line(df, x='q_num', y='time', title="Vélocité", markers=True)
            fig_l.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_l, use_container_width=True)
            
            # 9. Bar Time Concept
            atc = df.groupby('concept')['time'].mean().reset_index()
            fig_tb = px.bar(atc, x='time', y='concept', orientation='h', title="Temps Moyen/Concept", color='time')
            fig_tb.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_tb, use_container_width=True)
        with cd:
            # 10. Box Plot
            fig_b = px.box(df, y="time", x="status", color="status", title="Distribution Temps")
            fig_b.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_b, use_container_width=True)
            
            # 11. Strip Plot
            fig_st = px.strip(df, x='time', y='concept', color='status', title="Densité Réponses")
            fig_st.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_st, use_container_width=True)

            # 12. Violin Plot (Nouveau) - Densité Temporelle
            fig_vio = px.violin(df, y="time", x="status", color="status", box=True, points="all", title="Violin Plot (Densité Temps)")
            fig_vio.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_vio, use_container_width=True)

    # TAB 3 : CONVERSION (6 Graphs)
    with tab3:
        ce, cf = st.columns(2)
        with ce:
            # 13. Pie
            fig_p = px.pie(df, names='concept', title="Répartition Concepts", hole=0.4)
            fig_p.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig_p, use_container_width=True)
            
            # 14. Histogram Time
            fig_h = px.histogram(df, x="time", nbins=10, title="Histogramme Temps")
            fig_h.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_h, use_container_width=True)
            
            # 15. Bar Success Rate
            suc = df.groupby('concept')['correct'].mean().reset_index()
            fig_su = px.bar(suc, x='concept', y='correct', title="Succès/Concept", color='correct', range_y=[0,1])
            fig_su.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_su, use_container_width=True)
        with cf:
            # 16. Funnel
            fd = dict(number=[len(df), len(df), df['correct'].sum()], stage=["Vues", "Répondues", "Justes"])
            fig_f = px.funnel(fd, x='number', y='stage', title="Conversion")
            fig_f.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_f, use_container_width=True)
            
            # 17. Step Line (Cumulative)
            df['cs'] = df['correct'].astype(int).cumsum()
            fig_step = px.line(df, x='q_num', y='cs', title="Trajectoire (Step)", line_shape='hv')
            fig_step.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_step, use_container_width=True)

            # 18. Rolling Accuracy (Nouveau) - Stabilité de performance
            fig_roll = px.line(df, x='q_num', y='rolling_acc', title="Précision Glissante (Stabilité)", markers=True)
            fig_roll.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", plot_bgcolor='rgba(0,0,0,0)', yaxis_range=[0,1.1])
            st.plotly_chart(fig_roll, use_container_width=True)

    col_pdf, col_quit = st.columns([1, 1])
    with col_pdf:
        if FPDF_AVAILABLE:
            user_name = st.session_state.get("user_name", "Candidat Anonyme")
            pdf_data = generate_pdf_report(user_name, lvl_info['title'], df, kpi)
            if pdf_data: st.download_button("📥 Télécharger Rapport PDF", data=pdf_data, file_name="audit_report.pdf", mime="application/pdf", type="primary")
        else: st.warning("PDF non dispo")
    with col_quit:
        if st.button("🔄 Retour", use_container_width=True):
            st.session_state.current_level = None
            st.session_state.quiz_submitted = False
            st.rerun()

# ==============================================================================
# 5. MAIN
# ==============================================================================
import streamlit as st
import time

# Assurez-vous d'avoir vos dictionnaires QUESTION_BANK et LEVELS définis ici ou importés
# ex: from data import QUESTION_BANK, LEVELS 

def main():
    defaults = {
        "page": "exercices", 
        "current_level": None, 
        "quiz_submitted": False, 
        "q_index": 0, 
        "score": 0, 
        "answer_validated": False, 
        "quiz_history": [], 
        "start_time": time.time(), 
        "user_name": "Dr. Data"
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    # --- PARTIE SUPPRIMÉE (SIDEBAR) ---
    # La variable search est nécessaire pour la suite, on la définit vide par défaut
    search = "" 

    if st.session_state.page == "exercices":
        if st.session_state.quiz_submitted:
            # Assurez-vous que la fonction show_dashboard_final() est définie
            show_dashboard_final() 
            
        elif st.session_state.current_level is None:
            st.markdown("<h1>🚀 Centre de Commande</h1>", unsafe_allow_html=True)
            st.markdown("---")
            cols = st.columns(3)
            levels = list(QUESTION_BANK.keys())
            
            # Filtre de recherche (optionnel maintenant que l'input est retiré)
            if search: 
                levels = [k for k in levels if search.lower() in LEVELS.get(k,{}).get('title','').lower()]
            
            for idx, i in enumerate(levels):
                lvl = LEVELS.get(i, {"title":f"Module {i}", "icon":"❓", "desc":"..."})
                with cols[idx%3]:
                    st.markdown(f"<div class='level-card'><div style='font-size:40px;text-align:center;'>{lvl['icon']}</div><h3 style='text-align:center;color:#fff;'>{lvl['title']}</h3><p style='text-align:center;color:#aaa;'>{lvl['desc']}</p></div>", unsafe_allow_html=True)
                    if st.button(f"Lancer", key=f"btn_{i}", use_container_width=True):
                        st.session_state.current_level = i
                        st.session_state.q_index = 0
                        st.session_state.score = 0
                        st.session_state.quiz_history = []
                        st.session_state.answer_validated = False
                        st.session_state.quiz_submitted = False
                        st.session_state.start_time = time.time()
                        st.rerun()
        else:
            lvl_id = st.session_state.current_level
            questions = QUESTION_BANK.get(lvl_id, [])
            idx = st.session_state.q_index
            total = len(questions)
            
            c1, c2, c3 = st.columns([1, 6, 2])
            with c1: 
                # Ce bouton permet de revenir au menu des niveaux (Centre de commande)
                if st.button("❌"): 
                    st.session_state.current_level = None
                    st.rerun()
            with c2: st.progress((idx+1)/total)
            with c3: st.markdown(f"**Score: {st.session_state.score}**")
            
            if idx < total:
                q_data = questions[idx]
                st.markdown(f"<div class='question-box'><span style='color:#00C9FF'>Q{idx+1}/{total}</span><br>{q_data['question']}</div>", unsafe_allow_html=True)
                rep = st.radio("Réponse :", q_data['options'], key=f"rad_{idx}", disabled=st.session_state.answer_validated)
                st.write("")
                
                if not st.session_state.answer_validated:
                    if st.button("VALIDER ➤", type="primary"):
                        if rep:
                            dur = time.time() - st.session_state.start_time
                            st.session_state.answer_validated = True
                            corr = (rep == q_data['options'][q_data['correct_index']])
                            if corr: st.session_state.score += 1
                            st.session_state.last_res = "ok" if corr else "ko"
                            st.session_state.quiz_history.append({'question': idx, 'correct': corr, 'time': dur, 'concept': q_data['concept']})
                            st.rerun()
                        else: st.toast("Choix requis")
                else:
                    if st.session_state.last_res == "ok": st.markdown("<div class='feedback-ok'>✅ Correct !</div>", unsafe_allow_html=True)
                    else: st.markdown(f"<div class='feedback-ko'>❌ Faux. Réponse : {q_data['options'][q_data['correct_index']]}</div>", unsafe_allow_html=True)
                    st.info(f"💡 {q_data['explanation']}")
                    btn_txt = "DASHBOARD 📊" if idx == total-1 else "SUIVANT ➡"
                    if st.button(btn_txt, type="primary"):
                        if idx < total-1:
                            st.session_state.q_index += 1
                            st.session_state.answer_validated = False
                            st.session_state.start_time = time.time()
                            st.rerun()
                        else:
                            st.session_state.quiz_submitted = True
                            st.rerun()

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import ast
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# =====================================================
# 1. CONFIGURATION STREAMLIT & STYLE
# =====================================================
st.set_page_config(page_title="SamaLearn Singularity", layout="wide", page_icon="🧬")

# Initialisation de la session
if 'page' not in st.session_state:
    st.session_state.page = "enseignant"

THEME = {
    "primary": "#00f2ff",    # Cyan néon
    "secondary": "#7000ff",  # Violet profond
    "bg": "#050505",
    "card_bg": "rgba(255,255,255,0.03)"
}

st.markdown(f"""
<style>
    /* Fond sombre tech */
    .stApp {{ background-color: {THEME['bg']}; }}
    
    /* Cartes KPI Holographiques */
    .kpi-panel {{
        background: linear-gradient(145deg, {THEME['card_bg']} 0%, rgba(255,255,255,0.01) 100%);
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
        box-shadow: 0 0 25px rgba(0, 242, 255, 0.15);
    }}
    .kpi-value {{
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
        text-shadow: 0 0 10px rgba(0,0,0,0.5);
    }}
    .kpi-label {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #aaa;
        margin-bottom: 5px;
    }}
    
    /* Titres Sections */
    .section-header {{
        font-size: 1.2rem;
        font-weight: 700;
        margin: 30px 0 15px 0;
        padding-left: 15px;
        border-left: 3px solid {THEME['secondary']};
        background: linear-gradient(90deg, rgba(112,0,255,0.1), transparent);
        color: white;
        padding: 8px;
        border-radius: 0 8px 8px 0;
    }}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. CLASSES UTILITAIRES (SIMULATION SI MANQUANTES)
# =====================================================
# Ces classes sont nécessaires pour éviter que le code ne plante
class PedagogicalKPIs:
    def __init__(self, retention, engagement, velocity, difficulty, dropout, score):
        self.retention_rate = retention
        self.engagement_rate = engagement
        self.learning_velocity = velocity
        self.difficulty_index = difficulty
        self.dropout_risk = dropout
        self.score = score

class KPICalculator:
    def compute(self, df):
        # Simulation de calculs pour l'exemple
        if df.empty: return PedagogicalKPIs(0,0,0,{},"N/A",0)
        return PedagogicalKPIs(
            retention=0.85, 
            engagement=0.72, 
            velocity=14.5, 
            difficulty={"Maths": 0.6, "Python": 0.4}, 
            dropout="NORMAL", 
            score=78
        )

# =====================================================
# 3. CHARGEMENT ET TRAITEMENT DES DONNÉES
# =====================================================
@st.cache_data
def get_singularity_data():
    """Charge les traces, nettoie et prépare les données pour les graphiques."""
    
    def safe_literal_eval(x):
        if pd.isna(x) or x is None: return {}
        try: return ast.literal_eval(str(x).replace("'", '"')) 
        except: return {} 

    try:
        df_history = pd.read_csv("learning_traces_cleaned.csv", sep=',')
    except FileNotFoundError:
        # Génération de fausses données pour tester si le fichier n'existe pas
        dates = pd.date_range(end=datetime.now(), periods=100)
        data = {
            'user': [f"User_{np.random.randint(1,20)}" for _ in range(100)],
            'timestamp': dates,
            'action': np.random.choice(['quiz_submit', 'code_run'], 100),
            'module': np.random.choice(['Maths', 'Python', 'DeepLearning'], 100),
            'success': np.random.choice([True, False], 100),
            'time_spent': np.random.randint(5, 60, 100),
            'details': "{'xp': 10}"
        }
        df_history = pd.DataFrame(data)
        st.warning("⚠️ Mode démo : Fichier CSV non trouvé, données générées.")

    # Nettoyage de base
    if 'details' in df_history.columns:
        df_history['details_dict'] = df_history['details'].apply(safe_literal_eval)
        df_history['xp'] = df_history['details_dict'].apply(lambda x: x.get('xp', 10))
    else:
        df_history['xp'] = 10

    df_history['timestamp'] = pd.to_datetime(df_history['timestamp'], errors='coerce')
    df_history.dropna(subset=['timestamp'], inplace=True)
    df_history['student_id'] = df_history['user'].str.extract('(\d+)').astype(float).fillna(0).astype(int)
    
    # --- 🛠️ CORRECTION CRITIQUE POUR LE GRAPHIQUE ---
    # Création de la colonne 'week' pour le groupement temporel
    df_history['week'] = df_history['timestamp'].dt.to_period('W').apply(lambda r: r.start_time)
    # Création de la colonne 'score' (basée sur XP ou Success) pour l'axe Y
    df_history['score'] = df_history['xp']
    # ------------------------------------------------

    # --- Résumé par étudiant (df_summary) ---
    unique_users = df_history[['student_id', 'user']].drop_duplicates()
    n_students = len(unique_users)
    
    df_summary = pd.DataFrame({
        "ID": unique_users['student_id'].tolist(),
        "Nom": unique_users['user'].tolist(),
        "Groupe": np.random.choice(["Alpha", "Beta"], n_students), # Simulation groupe
    })

    # Calcul des scores simulés ou réels par module pour l'UI
    for mod in ['Maths', 'Python', 'DeepLearning']:
        # On met des valeurs aléatoires cohérentes pour l'exemple
        df_summary[mod] = np.random.randint(40, 100, n_students)

    df_summary['Score_Global'] = df_summary[['Maths', 'Python', 'DeepLearning']].mean(axis=1)
    df_summary['Assiduité'] = np.random.randint(20, 100, n_students)
    df_summary['Heures_Code'] = np.random.randint(10, 200, n_students) # Nécessaire pour le graph 3D
    
    # Classification
    df_summary["Statut"] = np.where(df_summary["Score_Global"] < 50, "Critique", 
                           np.where(df_summary["Score_Global"] < 70, "Surveillance", "Optimal"))
    
    df_summary["Cluster"] = np.where(df_summary["Score_Global"] > 80, "Elite", 
                            np.where(df_summary["Score_Global"] > 60, "Standard", "Risque"))

    return df_summary, df_history

# =====================================================
# 4. INTERFACE UTILISATEUR
# =====================================================
if st.session_state.page == "enseignant":
    
    df_raw, df_hist_raw = get_singularity_data()

    if df_raw.empty:
        st.error("Aucune donnée disponible.")
        st.stop()

    with st.sidebar:
        st.header("🎛️ Commandes")
        # Gestion des filtres sécurisée (si une seule valeur existe)
        all_groups = df_raw["Groupe"].unique()
        sel_grp = st.multiselect("Cohorte", all_groups, default=all_groups, key="ens_multi_grp")
        
        sel_sts = st.multiselect("Statut", df_raw["Statut"].unique(), default=df_raw["Statut"].unique(), key="ens_multi_sts")
        
        # Filtrage
        df = df_raw[df_raw["Groupe"].isin(sel_grp) & df_raw["Statut"].isin(sel_sts)]
        valid_ids = df["ID"].unique()
        df_hist = df_hist_raw[df_hist_raw["student_id"].isin(valid_ids)].copy()

        st.divider()
        st.metric("Agents Actifs", len(df))

    # En-tête
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f'<h1 style="font-size:2.5rem; margin:0;">SamaLearn <span style="color:{THEME["primary"]}">SINGULARITY</span></h1>', unsafe_allow_html=True)
    with c2:
        current_avg = df['Score_Global'].mean() if not df.empty else 0
        st.markdown(f"""
        <div style="text-align:right; padding:10px; background:rgba(0,242,255,0.05); border-radius:8px; border:1px solid {THEME['primary']};">
            <div style="color:{THEME['primary']}; font-weight:bold;">● LIVE MONITOR</div>
            <div style="color:white; font-size:1.2rem; font-weight:bold;">{current_avg:.1f} <span style="font-size:0.6rem;">MOYENNE</span></div>
        </div>
        """, unsafe_allow_html=True)

    # KPIs
    calculator = KPICalculator()
    kpis = calculator.compute(df_hist)

    k1, k2, k3, k4 = st.columns(4)
    metrics_ui = [
        {"l": "Taux Rétention", "v": f"{kpis.retention_rate*100:.1f}%", "c": "#ffffff"},
        {"l": "Taux Engagement", "v": f"{kpis.engagement_rate*100:.1f}%", "c": THEME['secondary']},
        {"l": "Vélocité (Pts/h)", "v": f"+{kpis.learning_velocity:.2f}", "c": "#00ff94"},
        {"l": "Risque Décrochage", "v": kpis.dropout_risk, "c": "#ff2b2b" if kpis.dropout_risk == "CRITICAL" else "#00ff94"},
    ]
    for col, m in zip([k1, k2, k3, k4], metrics_ui):
        with col:
            st.markdown(f"""
            <div class="kpi-panel">
                <div class="kpi-label">{m['l']}</div>
                <div class="kpi-value" style="color:{m['c']}">{m['v']}</div>
            </div>
            """, unsafe_allow_html=True)

    # --- ZONE INTELLIGENCE ---
    st.markdown('<div class="section-header">🧠 CORE INTELLIGENCE ARTIFICIELLE</div>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["🧬 DEEP LEARNING", "🌌 CLUSTERING 3D", "🤖 ANALYSE DE TENDANCE", "🔮 PRÉDICTIONS"])

    # --- ONGLET 1 : DEEP LEARNING (CORRIGÉ) ---
    with t1:
        c_dl1, c_dl2 = st.columns([3, 1])
        with c_dl1:
            st.markdown("##### 📉 Dynamique d'Apprentissage Globale")
            
            # Vérification et tracé
            if not df_hist.empty and 'week' in df_hist.columns and 'score' in df_hist.columns:
                # Group by week and calculate mean score
                weekly_stats = df_hist.groupby("week")["score"].mean().reset_index()
                
                fig_dl = go.Figure()
                fig_dl.add_trace(go.Scatter(
                    x=weekly_stats["week"], 
                    y=weekly_stats["score"], 
                    mode='lines+markers', 
                    name='Score Moyen', 
                    line=dict(color='#00ff94', width=3, shape='spline'), 
                    fill='tozeroy', 
                    fillcolor='rgba(0, 255, 148, 0.1)'
                ))
                fig_dl.update_layout(
                    xaxis_title="Semaines", 
                    yaxis_title="XP / Score", 
                    height=300, 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    font=dict(color='white'), 
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig_dl, use_container_width=True)
            else:
                st.warning("Données insuffisantes pour le graphique temporel (Colonnes 'week' ou 'score' manquantes).")
                
        with c_dl2:
            st.markdown("##### ⚙️ Index Difficulté")
            if kpis.difficulty_index:
                for mod, score in kpis.difficulty_index.items():
                    st.progress(score, text=f"{mod}: {int(score*100)}%")
            else:
                st.caption("Données non disponibles")

    # --- ONGLET 2 : 3D ---
    with t2:
        st.markdown("##### 🌌 Cartographie 3D des Profils")
        if not df.empty:
            # Assurons-nous que les colonnes existent
            cols_ok = all(col in df.columns for col in ['Maths', 'DeepLearning', 'Heures_Code'])
            if cols_ok:
                fig_3d = px.scatter_3d(df, x='Maths', y='DeepLearning', z='Heures_Code', color='Cluster', size='Assiduité', 
                                     color_discrete_map={'Elite': '#00ff94', 'Standard': '#00f2ff', 'Risque': '#ff2b2b'}, opacity=0.8)
                fig_3d.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), 
                                   scene=dict(bgcolor='rgba(0,0,0,0)', xaxis=dict(backgroundcolor='rgba(0,0,0,0)')))
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.error("Colonnes manquantes pour la 3D")

    # --- ONGLET 3 : ANALYSE ---
    with t3:
        st.markdown("##### 🤖 Corrélation")
        if not df.empty and len(df) > 1 and 'Heures_Code' in df.columns:
            try:
                m, b = np.polyfit(df["Heures_Code"], df["Score_Global"], 1)
                fig_ml = px.scatter(df, x="Heures_Code", y="Score_Global", color="Cluster", 
                                  color_discrete_map={'Elite': '#00ff94', 'Standard': '#00f2ff', 'Risque': '#ff2b2b'})
                
                # Ajout ligne régression
                x_line = np.linspace(df["Heures_Code"].min(), df["Heures_Code"].max(), 100)
                fig_ml.add_trace(go.Scatter(x=x_line, y=m*x_line+b, mode='lines', name='Régression', line=dict(color='white', dash='dash')))
                
                fig_ml.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.05)', font=dict(color='white'))
                st.plotly_chart(fig_ml, use_container_width=True)
            except Exception as e:
                st.warning(f"Erreur calcul régression: {e}")

    # --- ONGLET 4 : PROJECTION ---
    with t4:
        st.markdown("##### 🔮 Projection IA")
        dates_proj = pd.date_range(start=datetime.now(), periods=5)
        base = df['Score_Global'].mean() if not df.empty else 50
        y_avg = [base + (kpis.learning_velocity * 0.5 * i) for i in range(5)] # Facteur ajusté
        
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=dates_proj, y=y_avg, name='Tendance', line=dict(color='#00f2ff', width=4)))
        fig_p.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_p, use_container_width=True)

    # =====================================================
    # PARALLEL COORDINATES (DATA MATRIX)
    # =====================================================
    st.markdown('<div class="section-header">🌀 ANALYSE MULTIDIMENSIONNELLE</div>', unsafe_allow_html=True)
    if not df.empty:
        # Transformation Groupe en numérique pour le plot
        df['Groupe_Num'] = df['Groupe'].astype('category').cat.codes
        
        fig_par = go.Figure(data=go.Parcoords(
            line=dict(color=df['Score_Global'], colorscale='Viridis', showscale=True, cmin=0, cmax=100),
            labelfont=dict(size=14, color="white"),
            tickfont=dict(size=12, color="#ccc"),
            dimensions=[
                dict(range=[0, 100], label='Assiduité', values=df['Assiduité']),
                dict(range=[0, 100], label='Maths', values=df['Maths']),
                dict(range=[0, 100], label='Python', values=df['Python']),
                dict(range=[0, 100], label='DL', values=df['DeepLearning']),
                dict(range=[0, 100], label='Global', values=df['Score_Global']),
            ]
        ))
        fig_par.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), margin=dict(l=40, r=40, t=40, b=40))
        st.plotly_chart(fig_par, use_container_width=True)

    # TABLEAU FINAL
    st.markdown('<div class="section-header">📋 REGISTRE DES AGENTS</div>', unsafe_allow_html=True)
    st.dataframe(
        df[["Nom", "Groupe", "Score_Global", "Statut", "Cluster"]],
        use_container_width=True,
        hide_index=True
    )
    # =====================================================
    # ZONE 5 : TABLEAU FINAL (PRO & COMPLET)
    # =====================================================
    st.markdown('<div class="section-header">📋 REGISTRE DÉTAILLÉ DES AGENTS</div>', unsafe_allow_html=True)
    
    if not df.empty:
        # On sélectionne les colonnes pertinentes à afficher
        cols_to_show = ["Nom", "Groupe", "Score_Global", "Assiduité", "Maths", "Python", "DeepLearning", "Statut", "Cluster"]
        
        st.dataframe(
            df[cols_to_show],
            column_config={
                "Nom": st.column_config.TextColumn(
                    "Agent",
                    width="medium",
                    help="Identifiant de l'apprenant"
                ),
                "Groupe": st.column_config.TextColumn(
                    "Cohorte",
                    width="small"
                ),
                "Score_Global": st.column_config.ProgressColumn(
                    "Performance",
                    format="%.1f/100",
                    min_value=0,
                    max_value=100,
                    width="medium",
                    help="Score calculé par l'IA sur l'ensemble des modules"
                ),
                "Assiduité": st.column_config.ProgressColumn(
                    "Assiduité",
                    format="%d%%",
                    min_value=0,
                    max_value=100,
                    width="medium",
                ),
                "Maths": st.column_config.NumberColumn(
                    "Maths",
                    format="%d",
                    width="small"
                ),
                "Python": st.column_config.NumberColumn(
                    "Python",
                    format="%d",
                    width="small"
                ),
                "DeepLearning": st.column_config.NumberColumn(
                    "DL",
                    format="%d",
                    width="small"
                ),
                "Statut": st.column_config.TextColumn(
                    "État",
                    width="small"
                ),
                "Cluster": st.column_config.TextColumn(
                    "Profil IA",
                    width="small"
                ),
            },
            hide_index=True,
            use_container_width=True,
            height=600
        )
    else:
        st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")