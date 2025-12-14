import os, sys
import pandas as pd

# ==========================================================
# RENDRE LE DOSSIER src IMPORTABLE
# ==========================================================
# Ajoute le dossier parent au PYTHONPATH afin de pouvoir
# importer les modules du projet (src/)
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features


# ==========================================================
# CHEMINS DES DONNÉES
# ==========================================================
# Dataset brut (tel que fourni)
RAW = "data/raw/Telco-Customer-Churn.csv"

# Dataset final prêt pour l'entraînement
OUT = "data/processed/telco_churn_processed.csv"


# ==========================================================
# 1) CHARGEMENT DES DONNÉES BRUTES
# ==========================================================
# Lecture du fichier CSV source
df = pd.read_csv(RAW)


# ==========================================================
# 2) PRÉTRAITEMENT DES DONNÉES
# ==========================================================
# - suppression des identifiants
# - correction des types (TotalCharges)
# - nettoyage des valeurs manquantes
df = preprocess_data(df, target_col="Churn")


# ==========================================================
# 3) SÉCURISATION DE LA VARIABLE CIBLE
# ==========================================================
# Vérification finale que la cible Churn est bien encodée en 0/1
# (sécurité supplémentaire pour l'industrialisation)
if "Churn" in df.columns and df["Churn"].dtype == "object":
    df["Churn"] = (
        df["Churn"]
        .str.strip()
        .map({"No": 0, "Yes": 1})
        .astype("Int64")
    )

# Vérifications de cohérence (sanity checks)
assert df["Churn"].isna().sum() == 0, "Churn contient des valeurs manquantes après le preprocess"
assert set(df["Churn"].unique()) <= {0, 1}, "Churn n'est pas strictement encodée en 0/1"


# ==========================================================
# 4) FEATURE ENGINEERING
# ==========================================================
# Transformation des données en variables exploitables
# par le modèle de Machine Learning
df_processed = build_features(df, target_col="Churn")


# ==========================================================
# 5) SAUVEGARDE DU DATASET FINAL
# ==========================================================
# Création du dossier de sortie s'il n'existe pas
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Sauvegarde du dataset transformé
df_processed.to_csv(OUT, index=False)

print(f"✅ Dataset traité sauvegardé dans {OUT} | Shape: {df_processed.shape}")


# =====================================================================
# EXPLICATION GLOBALE – PIPELINE DE PRÉPARATION DES DONNÉES (MLOps)
# =====================================================================
#
# Rôle de ce script :
# - Orchestrer la préparation complète des données
# - Transformer un CSV brut en dataset prêt pour l'entraînement
#
# Chaîne de traitement :
#
#   Données brutes (CSV)
#          ↓
#   preprocess_data
#     - nettoyage
#     - correction des types
#     - gestion des NA
#          ↓
#   Sécurisation de la cible (Churn)
#          ↓
#   build_features
#     - encodage binaire
#     - one-hot encoding
#     - préparation finale
#          ↓
#   Dataset final (data/processed/)
#
# Pourquoi ce script est important en industrialisation :
# - Centralise la logique de préparation
# - Garantit une transformation reproductible
# - Sépare clairement :
#     - données brutes
#     - données transformées
# - Facilite le versioning et le déploiement
#
# Bonnes pratiques MLOps respectées :
# - Paths relatifs (compatibles Docker / AWS)
# - Sanity checks explicites
# - Étapes clairement identifiées
# - Script exécutable de bout en bout
#
# Ce script est typiquement exécuté :
# - avant l'entraînement
# - dans un pipeline CI/CD
# - ou dans une étape automatisée Airflow / GitHub Actions
