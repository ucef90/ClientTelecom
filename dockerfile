# 1. Utiliser une image Python officielle légère (base)
FROM python:3.11-slim

# 2. Définir le répertoire de travail dans le conteneur
WORKDIR /app

# 3. Copier d'abord le fichier des dépendances (optimisation du cache Docker)
COPY requirements.txt .

# 4. Installer les dépendances Python
# (et nettoyer le cache pour garder une image plus légère)
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 5. Copier tout le projet dans l'image
COPY . .

# Copie explicite du modèle
# Objectif : éviter les surprises si .dockerignore exclut certains fichiers (ex: mlruns)
# NOTE : destination définie sur /app/src/serving/model pour correspondre au path utilisé par inference.py
COPY src/serving/model /app/src/serving/model

# Copie du run MLflow (artefacts + metadata) vers un chemin "plat" de convenance : /app/model
# Objectif : simplifier le chargement en serving (chemin stable et direct)
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/model /app/model
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/feature_columns.txt /app/model/feature_columns.txt
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/preprocessing.pkl /app/model/preprocessing.pkl

# Rendre "serving" et "app" importables sans le préfixe "src."
# - PYTHONUNBUFFERED=1 : logs non bufferisés (utile en Docker, logs en temps réel)
# - PYTHONPATH=/app/src : permet des imports comme "from app..." plutôt que "from src.app..."
ENV PYTHONUNBUFFERED=1 \ 
    PYTHONPATH=/app/src

# 6. Exposer le port FastAPI
EXPOSE 8000

# 7. Lancer l'application FastAPI avec uvicorn
# (à adapter si ton point d'entrée change)
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# =====================================================================
# EXPLICATION GLOBALE – DOCKERFILE (SERVING FASTAPI + MODÈLE ML)
# =====================================================================
#
# Objectif :
# - Construire une image Docker exécutable qui embarque :
#   - ton code (API + pipeline inference)
#   - les dépendances (requirements.txt)
#   - le modèle entraîné (artefacts MLflow)
#   - les métadonnées nécessaires au serving (feature_columns + preprocessing.pkl)
#
# Logique globale :
#
# 1) Base image : python:3.11-slim
#    → image légère, suffisante pour FastAPI + dépendances ML
#
# 2) WORKDIR /app
#    → standardise les chemins et évite les chemins relatifs fragiles
#
# 3) COPY requirements.txt puis pip install
#    → optimise le cache Docker : si requirements ne change pas,
#      Docker réutilise la couche d’installation (build plus rapide)
#
# 4) COPY . .
#    → copie le projet complet (code, src, scripts, etc.)
#
# 5) COPY du modèle / artefacts
#    → garantit que les artefacts nécessaires à l’inférence sont présents
#    → /app/model sert de chemin “stable” (pratique pour inference.py)
#
# 6) ENV PYTHONPATH=/app/src
#    → simplifie les imports Python dans l'app
#
# 7) EXPOSE 8000 + CMD uvicorn
#    → expose l'API sur le port 8000 et lance le serveur web
#
# Résultat :
# - une image Docker "production-ready" que tu peux :
#   - exécuter localement
#   - pousser sur un registry (ECR)
#   - déployer sur AWS (ECS / EKS / EC2)
#
# Point clé MLOps :
# - Le modèle et ses métadonnées sont versionnés/embarqués
#   dans l’image : tu déploies un "package complet" (code + modèle).
