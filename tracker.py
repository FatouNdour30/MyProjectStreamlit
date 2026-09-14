# -*- coding: utf-8 -*-
import sqlite3
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List, Any
from contextlib import contextmanager
from dataclasses import dataclass, asdict

# 1. Configuration des Logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [LAE] [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("LearningAnalyticsEngine")

# --- 2. STRUCTURE DE DONNÉES (Typage Fort) ---

@dataclass
class LearningTraceData:
    """Structure de données standardisée pour une trace d'apprentissage."""
    user_id: str
    action: str
    module: Optional[str] = "General"
    success: Optional[bool] = None
    time_spent: Optional[float] = 0.0 # Temps passé en secondes
    details: Optional[Dict[str, Any]] = field(default_factory=dict) # Métadonnées additionnelles

# --- 3. CONTEXT MANAGER (Pour la gestion de connexion) ---

@contextmanager
def db_connection_manager(db_name: str):
    """Gère la connexion SQLite (ouverture, commit, fermeture) de manière Thread-safe."""
    conn = None
    try:
        # check_same_thread=False est nécessaire pour les applications Web (Streamlit/Flask)
        conn = sqlite3.connect(db_name, check_same_thread=False)
        conn.row_factory = sqlite3.Row # Permet de récupérer les colonnes par nom
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Erreur SQLite : {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


class LearningAnalyticsEngine:
    """
    Moteur de traçage et d'analyse d'apprentissage basé sur SQLite.
    Garantit l'intégrité des données et fournit des méthodes d'analyse rapides.
    """

    def __init__(self, db_path: str = "learning_data.db"):
        self.db_path = db_path
        self._init_db()

    # Propriété pour l'accès facile à la connexion gérée
    @property
    def _conn(self):
        return db_connection_manager(self.db_path)

    def _init_db(self):
        """Initialise la structure de la base de données (idempotente)."""
        query = """
        CREATE TABLE IF NOT EXISTS traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            action TEXT NOT NULL,
            module TEXT,
            success BOOLEAN,
            time_spent REAL,
            metadata TEXT
        );
        """
        try:
            with self._conn as conn:
                conn.execute(query)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON traces(user_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_mod_succ ON traces(module, success);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON traces(timestamp);")
            logger.info(f"Base de données '{self.db_path}' prête et indexée.")
        except Exception as e:
            logger.critical(f"Erreur fatale d'initialisation de la DB : {e}")

    # --- MÉTHODES D'INGESTION ---

    def log_interaction(self, data: LearningTraceData) -> bool:
        """
        Enregistre une interaction en utilisant la structure de données standardisée.
        """
        try:
            # Sérialisation des métadonnées (details)
            meta_json = json.dumps(data.details, default=str)
            
            sql = """
            INSERT INTO traces (user_id, action, module, success, time_spent, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """
            
            # Utilisation du context manager géré par self._conn
            with self._conn as conn:
                conn.execute(sql, (
                    data.user_id, 
                    data.action, 
                    data.module, 
                    data.success, 
                    data.time_spent, 
                    meta_json
                ))
            
            logger.debug(f"Trace enregistrée: {data.user_id}/{data.action}")
            return True

        except Exception as e:
            logger.error(f"Échec de l'enregistrement de la trace pour {data.user_id}: {e}")
            return False

    # --- MÉTHODES D'ANALYSE ET D'EXTRACTION ---
    
    def get_dataframe(self, user_id: Optional[str] = None) -> pd.DataFrame:
        """Récupère les traces, optionnellement filtrées par utilisateur, en DataFrame."""
        try:
            with self._conn as conn:
                if user_id:
                    query = "SELECT * FROM traces WHERE user_id = ? ORDER BY timestamp DESC"
                    df = pd.read_sql_query(query, conn, params=(user_id,))
                else:
                    query = "SELECT * FROM traces ORDER BY timestamp DESC"
                    df = pd.read_sql_query(query, conn)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Conversion du JSON en Dict si nécessaire pour l'analyse approfondie (non fait ici pour la vitesse)
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur de lecture du DataFrame (user: {user_id}): {e}")
            return pd.DataFrame()

    def get_advanced_stats(self, user_id: Optional[str] = None) -> Dict:
        """Calcule des statistiques agrégées via SQL (très rapide)."""
        stats = {}
        where_clause = f"WHERE user_id = '{user_id}'" if user_id else ""
        
        try:
            with self._conn as conn:
                cursor = conn.cursor()
                
                # Nombre total d'actions
                cursor.execute(f"SELECT COUNT(*) FROM traces {where_clause}")
                stats['total_actions'] = cursor.fetchone()[0]

                # Taux de succès
                cursor.execute(f"SELECT AVG(success) FROM traces {where_clause} AND success IS NOT NULL")
                stats['success_rate_pct'] = round((cursor.fetchone()[0] or 0) * 100, 2)

                # Temps total passé (en heures)
                cursor.execute(f"SELECT SUM(time_spent) FROM traces {where_clause}")
                stats['total_hours'] = round((cursor.fetchone()[0] or 0) / 3600, 2)

                # Module le plus actif
                cursor.execute(f"SELECT module, COUNT(*) as c FROM traces {where_clause} GROUP BY module ORDER BY c DESC LIMIT 1")
                res = cursor.fetchone()
                stats['top_module'] = res[0] if res else "N/A"

        except Exception as e:
            logger.error(f"Erreur calcul stats pour {user_id or 'Global'} : {e}")
            return {"error": "Calcul impossible"}
        
        return stats

    def export_to_csv(self, filename: str = "export_analytics.csv"):
        """Exporte toutes les données en CSV."""
        df = self.get_dataframe()
        if not df.empty:
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"Données exportées vers {filename}")
        else:
            logger.warning("Aucune donnée à exporter.")
            
    # --- MÉTHODE DE DÉMO ---

    def generate_demo_data(self):
        """Génère des données si la base est vide (pour le test)."""
        if self.get_advanced_stats()['total_actions'] == 0:
            logger.info("Génération de données de démo...")
            self.log_interaction(LearningTraceData("Fatou", "login", "System"))
            self.log_interaction(LearningTraceData("Fatou", "quiz_submit", "Python Intro", True, 120, {"score": 10}))
            self.log_interaction(LearningTraceData("Moussa", "video_watched", "Data Science", success=True, time_spent=600, details={"percent": 90}))
            self.log_interaction(LearningTraceData("Jean", "code_error", "Python Intro", success=False, time_spent=30, details={"error_type": "SyntaxError"}))
            self.log_interaction(LearningTraceData("Fatou", "logout", "System"))
            self.log_interaction(LearningTraceData("Fatou", "quiz_submit", "Python Intro", success=False, time_spent=80, details={"score": 5}))

# ==========================================
# TEST DE ROBUSTESSE ET DÉMO
# ==========================================

if __name__ == "__main__":
    print("--- Démarrage du Test du Moteur d'Analyse ---")
    
    tracker = LearningAnalyticsEngine(db_path="test_lae.db") # Utiliser une DB dédiée au test
    
    # 1. Préparation des données
    tracker.generate_demo_data()

    # 2. Ajout d'une nouvelle interaction
    new_trace = LearningTraceData(
        user_id="Etudiant_Test",
        action="exercise_submit",
        module="Chapitre 5",
        success=True,
        time_spent=345.5,
        details={"browser": "Chrome", "retry_count": 2}
    )
    tracker.log_interaction(new_trace)

    # 3. Affichage des stats (Globales)
    global_stats = tracker.get_advanced_stats()
    print(f"\n--- STATISTIQUES GLOBALES ---")
    print(f"Total Interactions : {global_stats['total_actions']}")
    print(f"Taux de Succès     : {global_stats['success_rate_pct']}%")
    print(f"Heures d'apprentissage : {global_stats['total_hours']} h")
    print(f"Module le plus actif   : {global_stats['top_module']}")

    # 4. Affichage des stats (Par Utilisateur)
    fatou_stats = tracker.get_advanced_stats(user_id="Fatou")
    print(f"\n--- STATISTIQUES POUR FATOU ---")
    print(f"Total Actions : {fatou_stats['total_actions']}")
    print(f"Taux de Succès : {fatou_stats['success_rate_pct']}%")
    
    # 5. Récupération des données pour l'analyse dans Pandas
    df_fatou = tracker.get_dataframe(user_id="Fatou")
    print("\n--- DATAFRAME DES TRACES DE FATOU ---")
    print(df_fatou[['timestamp', 'action', 'module', 'success']].head())
    
    # 6. Export
    tracker.export_to_csv("rapport_analytics_pro.csv")
    print("\nFichier 'rapport_analytics_pro.csv' généré avec succès.")