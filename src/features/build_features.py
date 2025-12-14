import pandas as pd


def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Applique un encodage binaire déterministe aux variables
    catégorielles contenant exactement deux modalités.

    Cette fonction implémente la logique centrale d'encodage binaire
    utilisée pour transformer certaines variables catégorielles en
    entiers 0/1. Les mappings sont déterministes et doivent être
    strictement identiques entre l'entraînement et le serving.
    """

    # Récupération des valeurs uniques (hors NaN) et conversion en chaînes
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    # ==========================================================
    # MAPPINGS BINAIRES DÉTERMINISTES
    # ==========================================================
    # IMPORTANT : ces mappings doivent être identiques
    # dans le pipeline de serving (API / UI)

    # Cas Yes / No (pattern le plus courant dans les données Telco)
    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")

    # Cas Gender (variable démographique)
    if valset == {"Male", "Female"}:
        return s.map({"Female": 0, "Male": 1}).astype("Int64")

    # ==========================================================
    # MAPPING BINAIRE GÉNÉRIQUE
    # ==========================================================
    # Pour toute autre variable à 2 modalités,
    # on utilise un ordre alphabétique stable
    if len(vals) == 2:
        # Tri alphabétique pour garantir un mapping stable
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")

    # ==========================================================
    # VARIABLES NON BINAIRES
    # ==========================================================
    # Les variables avec plus de 2 modalités
    # seront traitées par un encodage one-hot
    return s


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Applique l'ensemble du pipeline de feature engineering
    sur les données clients Telco.

    Cette fonction transforme les données nettoyées en
    variables prêtes pour l'entraînement ou l'inférence
    d'un modèle de Machine Learning.

    IMPORTANT :
    Les transformations appliquées ici doivent être
    rigoureusement répliquées dans le pipeline de serving
    afin de garantir la cohérence des prédictions.
    """

    # Copie défensive pour éviter toute modification en place
    df = df.copy()
    print(f"🔧 Démarrage du feature engineering sur {df.shape[1]} colonnes...")

    # ==========================================================
    # ÉTAPE 1 : Identification des types de variables
    # ==========================================================
    # Variables catégorielles (type object), hors variable cible
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]

    # Variables numériques
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"   📊 {len(obj_cols)} variables catégorielles | {len(numeric_cols)} variables numériques")

    # ==========================================================
    # ÉTAPE 2 : Séparation par cardinalité
    # ==========================================================
    # - variables binaires : exactement 2 modalités
    # - variables multi-catégories : plus de 2 modalités
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]

    print(f"   🔢 Variables binaires : {len(binary_cols)} | Variables multi-catégories : {len(multi_cols)}")
    if binary_cols:
        print(f"      Binaires : {binary_cols}")
    if multi_cols:
        print(f"      Multi-catégories : {multi_cols}")

    # ==========================================================
    # ÉTAPE 3 : Encodage binaire
    # ==========================================================
    # Transformation des variables à 2 modalités en 0/1
    # à l’aide de mappings déterministes
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str))
        print(f"      ✅ {c} : {original_dtype} → binaire (0/1)")

    # ==========================================================
    # ÉTAPE 4 : Conversion des booléens
    # ==========================================================
    # Les modèles comme XGBoost nécessitent des entiers
    # et non des booléens
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"   🔄 Conversion de {len(bool_cols)} colonnes booléennes en int : {bool_cols}")

    # ==========================================================
    # ÉTAPE 5 : Encodage One-Hot
    # ==========================================================
    # Utilisé pour les variables multi-catégories
    # drop_first=True permet d'éviter la multicolinéarité
    if multi_cols:
        print(f"   🌟 Application du one-hot encoding sur {len(multi_cols)} colonnes...")
        original_shape = df.shape

        df = pd.get_dummies(
            df,
            columns=multi_cols,
            drop_first=True
        )

        new_features = df.shape[1] - original_shape[1] + len(multi_cols)
        print(f"      ✅ {new_features} nouvelles variables créées")

    # ==========================================================
    # ÉTAPE 6 : Nettoyage final des types
    # ==========================================================
    # Conversion des entiers nullable (Int64) vers int standard
    # requis par XGBoost
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].fillna(0).astype(int)

    print(f"✅ Feature engineering terminé : {df.shape[1]} variables finales")
    return df


# =====================================================================
# EXPLICATION GLOBALE – FEATURE ENGINEERING & INDUSTRIALISATION
# =====================================================================
#
# Ce fichier implémente la phase de feature engineering du pipeline ML.
#
# Rôle clé :
# - Transformer les données nettoyées en variables numériques exploitables
# - Garantir une transformation STRICTEMENT identique entre :
#     - l'entraînement
#     - le serving (API / Gradio)
#
# Principes d’industrialisation respectés :
# - Fonctions pures (entrée → sortie)
# - Mappings déterministes (stabilité des prédictions)
# - Aucune dépendance au système de fichiers
# - Compatibilité Docker / AWS / CI-CD
#
# Choix techniques assumés :
# - Encodage binaire pour les variables à 2 modalités
# - One-hot encoding pour les variables multi-catégories
# - drop_first=True pour éviter la multicolinéarité
#
# Ce design permet :
# - une meilleure robustesse du modèle
# - une reproductibilité totale
# - une lecture claire pour un contexte professionnel,
#   pédagogique ou jury
