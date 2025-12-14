import great_expectations as ge
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Validation complète des données du dataset Telco Customer Churn avec Great Expectations.

    Cette fonction exécute des contrôles qualité critiques qui doivent être validés
    avant l'entraînement du modèle. Elle vérifie :
    - l'intégrité du schéma (colonnes obligatoires)
    - des contraintes métier (valeurs autorisées)
    - des contraintes numériques (bornes)
    - des propriétés statistiques raisonnables attendues par le modèle
    - des règles de cohérence entre colonnes
    """
    print("🔍 Démarrage de la validation des données avec Great Expectations...")

    # Conversion du DataFrame pandas en objet Great Expectations (Dataset)
    ge_df = ge.dataset.PandasDataset(df)

    # ==========================================================
    # VALIDATION DU SCHÉMA – COLONNES ESSENTIELLES
    # ==========================================================
    print("   📋 Validation du schéma et des colonnes requises...")

    # Identifiant client : doit exister (utile métier) et ne pas être vide
    ge_df.expect_column_to_exist("customerID")
    ge_df.expect_column_values_to_not_be_null("customerID")

    # Variables démographiques principales
    ge_df.expect_column_to_exist("gender")
    ge_df.expect_column_to_exist("Partner")
    ge_df.expect_column_to_exist("Dependents")

    # Variables de services (importantes pour l'analyse churn)
    ge_df.expect_column_to_exist("PhoneService")
    ge_df.expect_column_to_exist("InternetService")
    ge_df.expect_column_to_exist("Contract")

    # Variables financières (forts prédicteurs de churn)
    ge_df.expect_column_to_exist("tenure")
    ge_df.expect_column_to_exist("MonthlyCharges")
    ge_df.expect_column_to_exist("TotalCharges")

    # ==========================================================
    # VALIDATION MÉTIER – VALEURS AUTORISÉES
    # ==========================================================
    print("   💼 Validation des contraintes métier (valeurs possibles)...")

    # Genre : valeurs attendues
    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])

    # Champs Yes/No : valeurs attendues
    ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])

    # Types de contrat : contrainte métier
    ge_df.expect_column_values_to_be_in_set(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    # Types d'Internet : contrainte métier
    ge_df.expect_column_values_to_be_in_set(
        "InternetService",
        ["DSL", "Fiber optic", "No"]
    )

    # ==========================================================
    # VALIDATION DES PLAGES NUMÉRIQUES – CONTRAINTES DE BASE
    # ==========================================================
    print("   📊 Validation des bornes numériques et des contraintes métier...")

    # Tenure (ancienneté) ne peut pas être négatif
    ge_df.expect_column_values_to_be_between("tenure", min_value=0)

    # MonthlyCharges doit être >= 0 (pas de montant négatif)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0)

    # TotalCharges doit être >= 0
    ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)

    # ==========================================================
    # VALIDATION STATISTIQUE – BORNES RAISONNABLES
    # ==========================================================
    print("   📈 Validation des propriétés statistiques (valeurs raisonnables)...")

    # Tenure raisonnable : en télécom, on borne souvent à ~10 ans = 120 mois
    ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)

    # MonthlyCharges dans une plage réaliste
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)

    # Pas de valeurs manquantes sur des features numériques critiques
    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges")

    # ==========================================================
    # COHÉRENCE DES DONNÉES – RÈGLES ENTRE COLONNES
    # ==========================================================
    print("   🔗 Validation de la cohérence entre colonnes...")

    # En général : TotalCharges >= MonthlyCharges
    # (sauf cas limites comme clients très récents / anomalies)
    # mostly=0.95 autorise jusqu'à 5% d'exceptions
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95
    )

    # ==========================================================
    # EXÉCUTION DE LA VALIDATION
    # ==========================================================
    print("   ⚙️  Exécution de la suite complète de validations...")
    results = ge_df.validate()

    # ==========================================================
    # TRAITEMENT DES RÉSULTATS
    # ==========================================================
    # Extraction des expectations échouées pour remonter des erreurs exploitables
    failed_expectations = []
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            failed_expectations.append(expectation_type)

    # Résumé
    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks

    if results["success"]:
        print(f"✅ Validation OK : {passed_checks}/{total_checks} contrôles réussis")
    else:
        print(f"❌ Validation KO : {failed_checks}/{total_checks} contrôles en échec")
        print(f"   Expectations échouées : {failed_expectations}")

    return results["success"], failed_expectations


# =====================================================================
# EXPLICATION GLOBALE – DATA VALIDATION & INDUSTRIALISATION (MLOps)
# =====================================================================
#
# Objectif de ce module :
# - Bloquer l'entraînement / le déploiement si la qualité des données est insuffisante
# - Détecter tôt les erreurs de schéma, de valeurs, de types ou de cohérence
#
# Pourquoi c'est critique en MLOps :
# - Un modèle ML est très sensible aux variations de schéma (colonne manquante)
# - Des valeurs inattendues peuvent casser un pipeline (ex: nouvelles catégories)
# - Des anomalies numériques (valeurs négatives) peuvent fausser la prédiction
# - Les règles de cohérence évitent des incohérences métier invisibles
#
# Résultat renvoyé :
# - success (bool) : True si toutes les validations passent
# - failed_expectations (List[str]) : liste des contrôles échoués
#
# Intégration recommandée :
# - À exécuter juste après le chargement des données (load_data)
# - Et avant preprocess_data / build_features / entraînement
#
# Exemple de pipeline :
# df = load_data(PATH)
# ok, failures = validate_telco_data(df)
# if not ok:
#     raise ValueError(f"Data validation failed: {failures}")
# df = preprocess_data(df)
# df = build_features(df)
