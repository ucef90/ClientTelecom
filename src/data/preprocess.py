import pandas as pd


def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Nettoyage de base des données pour le cas d'usage Telco Churn.

    Cette fonction applique un prétraitement générique et robuste
    afin de préparer les données pour l'entraînement ou l'inférence
    d'un modèle de Machine Learning.

    Étapes principales :
    - Nettoyage des noms de colonnes
    - Suppression des identifiants non prédictifs
    - Encodage de la variable cible
    - Correction des types de données
    - Gestion simple des valeurs manquantes
    """

    # ------------------------------------------------------------
    # 1. Nettoyage des noms de colonnes
    # ------------------------------------------------------------
    # Supprime les espaces en début/fin de nom de colonne
    # (évite des erreurs silencieuses lors des sélections)
    df.columns = df.columns.str.strip()

    # ------------------------------------------------------------
    # 2. Suppression des colonnes d'identifiants
    # ------------------------------------------------------------
    # Les identifiants n'apportent aucune information prédictive
    # et peuvent nuire à l'apprentissage du modèle
    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # ------------------------------------------------------------
    # 3. Encodage de la variable cible (Churn)
    # ------------------------------------------------------------
    # Si la cible est de type texte (Yes/No),
    # elle est transformée en variable binaire 0/1
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = (
            df[target_col]
            .str.strip()
            .map({"No": 0, "Yes": 1})
        )

    # ------------------------------------------------------------
    # 4. Correction du type de la variable TotalCharges
    # ------------------------------------------------------------
    # Cette colonne contient parfois des valeurs vides ou invalides
    # Conversion forcée en numérique (les erreurs deviennent NaN)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    # ------------------------------------------------------------
    # 5. Normalisation de la variable SeniorCitizen
    # ------------------------------------------------------------
    # Cette variable doit être binaire (0/1)
    # On remplace les valeurs manquantes par 0
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = (
            df["SeniorCitizen"]
            .fillna(0)
            .astype(int)
        )

    # ------------------------------------------------------------
    # 6. Gestion simple des valeurs manquantes
    # ------------------------------------------------------------
    # Stratégie volontairement simple et robuste :
    # - Colonnes numériques : remplacement par 0
    # - Colonnes catégorielles : laissées telles quelles
    #   (les encodeurs comme get_dummies gèrent correctement les NaN)
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # ------------------------------------------------------------
    # 7. Retour du DataFrame nettoyé
    # ------------------------------------------------------------
    return df


# ==================================================================
# EXPLICATION GLOBALE (PÉDAGOGIQUE / INDUSTRIALISATION)
# ==================================================================
#
# Cette fonction s'inscrit dans un pipeline MLOps classique :
#
#   Chargement des données (load_data)
#           ↓
#   Prétraitement générique (preprocess_data)  ← CE FICHIER
#           ↓
#   Feature engineering
#           ↓
#   Entraînement ou inférence
#
# Points clés d'industrialisation :
# - La fonction est pure : entrée (DataFrame) → sortie (DataFrame)
# - Aucune dépendance au système de fichiers
# - Réutilisable pour le training ET le serving
# - Compatible Docker, AWS et CI/CD
# - Lisible et maintenable pour un contexte professionnel
#
# Ce choix de prétraitement volontairement simple permet :
# - de limiter la complexité
# - d'assurer la robustesse du pipeline
# - de faciliter les démonstrations pédagogiques
#
# Les transformations plus avancées (encodage, scaling, sélection
# de variables) sont volontairement déléguées aux étapes suivantes
# du pipeline.
