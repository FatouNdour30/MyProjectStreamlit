# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib
import os
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

np.random.seed(42)
n = 200
taux_reussite = np.concatenate([
    np.random.normal(0.85, 0.08, 60),
    np.random.normal(0.35, 0.12, 80),
    np.random.normal(0.60, 0.15, 60)
])

temps_moyen = np.concatenate([
    np.random.normal(15, 5, 60),
    np.random.normal(65, 20, 80),
    np.random.normal(40, 12, 60)
])

nb_erreurs_consecutives = np.concatenate([
    np.random.poisson(0.2, 60),
    np.random.poisson(2.8, 80),
    np.random.poisson(1.3, 60)
])

persistance = np.concatenate([
    np.random.normal(0.92, 0.08, 60),
    np.random.normal(0.25, 0.15, 80),
    np.random.normal(0.60, 0.20, 60)
])

taux_reussite = np.clip(taux_reussite, 0, 1)
persistance = np.clip(persistance, 0, 1)
temps_moyen = np.clip(temps_moyen, 5, 120)
nb_erreurs_consecutives = np.clip(nb_erreurs_consecutives, 0, 5)

df = pd.DataFrame({
    "taux_reussite": taux_reussite,
    "temps_moyen": temps_moyen,
    "nb_erreurs_consecutives": nb_erreurs_consecutives,
    "persistance": persistance
})

X = df[["taux_reussite", "temps_moyen", "nb_erreurs_consecutives", "persistance"]]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X)

df.to_csv("data/synthetic_profiles.csv", index=False)
joblib.dump(kmeans, "model/clustering_model.pkl")

print("? Donn�es synth�tiques et mod�le sauvegard�s.")
