import requests

# ==========================================================
# SCRIPT DE TEST DE L’API DE PRÉDICTION (FASTAPI)
# ==========================================================
# Ce script permet de tester manuellement l’endpoint /predict
# exposé par l’API FastAPI, en simulant une requête client.

# URL de l’API locale (FastAPI doit être lancée au préalable)
url = "http://127.0.0.1:8000/predict"

# ==========================================================
# DONNÉES DE TEST (PAYLOAD JSON)
# ==========================================================
# Exemple représentatif d’un client Telco.
# Les champs correspondent exactement au schéma CustomerData
# attendu par l’API (Pydantic).
sample_data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 350.75
}

# ==========================================================
# ENVOI DE LA REQUÊTE HTTP POST
# ==========================================================
# requests.post :
# - envoie une requête HTTP POST
# - sérialise automatiquement le dictionnaire Python en JSON
# - transmet les données à l’API FastAPI
response = requests.post(url, json=sample_data)

# ==========================================================
# AFFICHAGE DES RÉSULTATS
# ==========================================================
# Status Code :
# - 200 → succès
# - 422 → erreur de validation Pydantic
# - 500 → erreur serveur
print("Status Code:", response.status_code)

# Réponse JSON retournée par l’API
# Exemple :
# { "prediction": "Likely to churn" }
print("Response:", response.json())


# =====================================================================
# EXPLICATION GLOBALE – TEST DE L’API (SERVING ML)
# =====================================================================
#
# Objectif de ce script :
# - Vérifier que l’API FastAPI fonctionne correctement
# - Tester le pipeline d’inférence de bout en bout
#
# Chaîne complète testée ici :
#
#   Client (requests)
#          ↓
#   API FastAPI (/predict)
#          ↓
#   Validation Pydantic
#          ↓
#   Prétraitement + feature engineering
#          ↓
#   Modèle ML entraîné
#          ↓
#   Réponse JSON
#
# Pourquoi ce script est important en MLOps :
# - Validation fonctionnelle avant déploiement
# - Test rapide après un changement de modèle
# - Base pour des tests automatisés (CI/CD)
#
# Bonnes pratiques respectées :
# - Payload conforme au schéma de l’API
# - Utilisation du format JSON
# - Test indépendant de l’UI (Gradio)
#
# Évolution possible :
# - Transformer ce script en test automatisé (pytest)
# - Tester plusieurs profils clients
# - Intégrer ce test dans un pipeline CI/CD
