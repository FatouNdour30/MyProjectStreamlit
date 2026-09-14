# -*- coding: utf-8 -*-
import logging
from datetime import datetime, timedelta
from statistics import mean
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AdaptiveRecommender")

# --- CONTRATS DE DONNÉES (Dataclass) ---

@dataclass
class UserProfileMinimal:
    """Contrat de données pour le profil utilisateur."""
    id: str
    competences: Dict[str, float] = field(default_factory=dict) # Module -> Score [0.0, 1.0]
    last_review_dates: Dict[str, Union[datetime, str]] = field(default_factory=dict)

@dataclass
class LearningTrace:
    """Contrat de données pour une interaction historique."""
    module: str
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)

# --- HYPERPARAMÈTRES ET PONDÉRATIONS (Pour configuration facile) ---

class ScoringWeights:
    """Constantes de pondération des critères de recommandation."""
    DEFICIT_COMPETENCE = 0.40 # Poids pour le manque de maîtrise
    SPACED_REPETITION = 0.30  # Poids pour l'oubli et le retard de révision
    ZPD_RELEVANCE = 0.20      # Poids pour la proximité de la Zone Proximale de Développement
    RECENT_ERRORS = 0.10      # Poids pour le feedback immédiat sur les erreurs

# Seuil de maîtrise requis pour un prérequis (60%)
PREREQ_MASTERY_THRESHOLD = 0.60
# Seuil d'écarts de difficulté pour la ZPD (0.1)
ZPD_TOLERANCE = 0.10


# --- MÉTADONNÉES DU CURRICULUM (RÉFÉRENTIEL CURRICULAIRE) ---
CURRICULUM_METADATA = {
    "Python: Les Fondations": {"difficulty": 0.1, "prerequisite": None},
    "Python Avancé": {"difficulty": 0.3, "prerequisite": "Python: Les Fondations"},
    "Pandas: Manipulation": {"difficulty": 0.5, "prerequisite": "Python Avancé"},
    "ML: Supervisé": {"difficulty": 0.7, "prerequisite": "Pandas: Manipulation"},
    "Deep Learning (ANN)": {"difficulty": 0.9, "prerequisite": "ML: Supervisé"}
}


class AdaptiveRecommender:
    """
    Moteur de recommandation adaptatif utilisant une approche
    multi-critères basée sur les sciences cognitives.
    """

    def __init__(self, curriculum: Dict = CURRICULUM_METADATA):
        self.curriculum = curriculum
        # Configuration des constantes
        self.weights = ScoringWeights() 
        self.prereq_threshold = PREREQ_MASTERY_THRESHOLD

    def get_recommendations(
        self, 
        user_profile: UserProfileMinimal, 
        historical_traces: List[LearningTrace]
    ) -> List[Dict]:
        """Génère le Top 3 des modules recommandés."""
        
        if not user_profile.competences:
            return self._get_cold_start_recommendation()

        # 1. Calcul des Contextes Pédagogiques Globaux
        avg_competence = self._calculate_average_competence(user_profile.competences)
        target_difficulty = avg_competence + 0.15 
        cognitive_load_state = self._analyze_cognitive_load(historical_traces[-5:])
        
        if cognitive_load_state == "high":
            logger.warning("Charge cognitive élevée détectée. Réduction de la ZPD cible.")
            target_difficulty -= 0.10 

        # 2. Scoring par Module
        scored_modules = []
        all_available_modules = set(self.curriculum.keys())

        for module_name in all_available_modules:
            score, justification = self._score_module(
                module_name, 
                user_profile.competences, 
                user_profile.last_review_dates, 
                historical_traces, 
                target_difficulty, 
                cognitive_load_state
            )
            
            if score is None: 
                # Le module a été ignoré (ex: prérequis manquant)
                continue
            
            scored_modules.append({
                "module": module_name,
                "score": round(score, 3),
                "justification": justification,
                "cognitive_load_context": cognitive_load_state
            })

        # 3. Tri et retour du Top
        return sorted(scored_modules, key=lambda x: x["score"], reverse=True)[:3]

    # --------------------------------------------------------------------------
    # --- MÉTHODE PRINCIPALE DE SCORING (Modularisation) ---

    def _score_module(
        self, 
        module_name: str, 
        competences: Dict[str, float],
        last_reviews: Dict[str, Union[datetime, str]],
        historical_traces: List[LearningTrace],
        target_difficulty: float,
        cognitive_load_state: str
    ) -> tuple[Optional[float], Optional[str]]:
        """Calcule le score et la justification pour un seul module."""
        
        score = 0
        current_mastery = competences.get(module_name, 0.0)
        module_data = self.curriculum.get(module_name, {})
        
        # 0. Vérification des Prérequis (Critère bloquant)
        prereq = module_data.get("prerequisite")
        if prereq and competences.get(prereq, 0.0) < self.prereq_threshold: 
            return None, None 

        # A. Score du Déficit de Compétence
        score += (1.0 - current_mastery) * self.weights.DEFICIT_COMPETENCE
        
        # B. Score de Répétition Espacée (SRS)
        days_elapsed = self._get_days_since_review(module_name, last_reviews)
        if days_elapsed > 3: 
            urgency = min(days_elapsed / 10, 1.0) 
            score += self.weights.SPACED_REPETITION * urgency
        
        # C. Score de Pertinence ZPD
        module_difficulty = module_data.get("difficulty", 0.5)
        # Calcul de la distance entre le niveau du module et la ZPD cible
        distance_to_zpd = abs(module_difficulty - target_difficulty)
        # La pertinence décroît avec la distance, bornée par la tolérance ZPD_TOLERANCE
        relevance = max(0, 1 - distance_to_zpd / ZPD_TOLERANCE)
        score += self.weights.ZPD_RELEVANCE * relevance 
        
        # D. Score des Erreurs Récentes
        recent_errors = [t for t in historical_traces[-5:] 
                         if t.module == module_name and not t.success]
        score += min(len(recent_errors) * 0.05, self.weights.RECENT_ERRORS)

        # Génération de la justification
        justification = self._generate_justification(
            module_name, current_mastery, days_elapsed, len(recent_errors), cognitive_load_state
        )
        
        return score, justification

    # --------------------------------------------------------------------------
    # --- FONCTIONS UTILITAIRES PRIVÉES (Calcul du Contexte) ---

    def _calculate_average_competence(self, competences: Dict[str, float]) -> float:
        """Calcule la compétence moyenne de l'utilisateur."""
        competence_values = [v for k, v in competences.items() if v > 0]
        return mean(competence_values) if competence_values else 0.0

    def _get_days_since_review(self, module_name: str, last_reviews: Dict[str, Union[datetime, str]]) -> int:
        """Calcule le nombre de jours écoulés depuis la dernière révision (SRS)."""
        date_val = last_reviews.get(module_name)
        if not date_val:
            return 999 

        if isinstance(date_val, str):
            try:
                last_date = datetime.strptime(date_val, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return 0 
        else:
            last_date = date_val
            
        return (datetime.now() - last_date).days

    def _analyze_cognitive_load(self, recent_traces: List[LearningTrace]) -> str:
        """Détermine l'état de la charge cognitive (Sweller)."""
        if not recent_traces or len(recent_traces) < 3:
            return "optimal"
        
        failures = sum(1 for t in recent_traces if not t.success)
        
        if failures >= 3:
            return "high" # 3 échecs ou plus sur 5
        return "optimal"

    def _generate_justification(
        self, 
        module: str, 
        mastery: float, 
        days_since: int, 
        recent_errors: int,
        load_state: str
    ) -> str:
        """Génère la justification en langage naturel."""
        
        if load_state == "high":
            return f"Pause cognitive ! Ce module consolide vos acquis ({int(mastery*100)}%) sans surcharger."
        if recent_errors >= 2:
            return f"Action urgente ! Vous avez échoué {recent_errors} fois récemment sur ce concept."
        if days_since > 7:
            return f"Révision espacée : Cela fait {days_since} jours que vous n'avez pas pratiqué ce module."
        if days_since > 3 and mastery < 0.7:
             return f"Courbe d'oubli : Renforcez vos bases pour éviter de perdre les acquis."
        if mastery < self.prereq_threshold:
            return f"Niveau Débutant : Module essentiel pour construire les fondations (Maîtrise {int(mastery*100)}%)."
        if mastery >= 0.8 and self.curriculum.get(module, {}).get("prerequisite"):
            return f"Transition : Maîtrise élevée. Ce module prépare bien au prochain niveau."
        
        return "Niveau Optimal : Module idéalement placé dans votre Zone Proximale de Développement."

    def _get_cold_start_recommendation(self) -> List[Dict]:
        """Recommandations par défaut pour un nouvel utilisateur."""
        first_module = next(iter(self.curriculum), "Introduction")
        return [{
            "module": first_module,
            "score": 1.0,
            "justification": "Bienvenue ! Commencez par ce module fondamental.",
            "cognitive_load_context": "optimal"
        }]

# --- TEST UNITAIRE AJUSTÉ (Non modifié, mais utilise la nouvelle structure) ---
if __name__ == "__main__":
    from dataclasses import asdict
    
    # Données simulées utilisant les dataclass pour l'input
    user_profile_data = UserProfileMinimal(
        id="Agent-PRO-007",
        competences={"Python: Les Fondations": 0.9, "Python Avancé": 0.4, "Deep Learning (ANN)": 0.1},
        last_review_dates={
            "Python: Les Fondations": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S"), # Forte urgence
            "Python Avancé": (datetime.now() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S")
        }
    )
    
    history_data = [
        LearningTrace(module="Deep Learning (ANN)", success=False),
        LearningTrace(module="Deep Learning (ANN)", success=False),
        LearningTrace(module="Python Avancé", success=True)
    ]

    engine = AdaptiveRecommender()
    recs = engine.get_recommendations(user_profile_data, history_data)

    print("\n--- RÉSULTATS DU MOTEUR DE RECOMMANDATION ADAPTATIVE (MODULAIRE) ---")
    import json
    # Affichage plus professionnel des résultats
    print(json.dumps(recs, indent=2, ensure_ascii=False))

    # Test Cold Start
    cold_start_profile = UserProfileMinimal(id="Newbie-001")
    cold_recs = engine.get_recommendations(cold_start_profile, [])
    print("\n--- RÉSULTATS COLD START ---")
    print(json.dumps(cold_recs, indent=2, ensure_ascii=False))