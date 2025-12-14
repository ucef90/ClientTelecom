# Industrialisation complète d’un modèle de Machine Learning – Cas Client Telecom Churn

## Objectif du projet

Ce projet a pour objectif de **concevoir, industrialiser et déployer une solution complète de Machine Learning** destinée à prédire le churn (résiliation) des clients dans un contexte télécom.

Contrairement à un simple prototype développé dans un notebook, ce projet illustre **le passage à l’échelle industrielle** d’un modèle ML : structuration du code, traçabilité des expériences, exposition via API, interface utilisateur, automatisation du déploiement et mise en production sur le cloud **AWS**.

---

## Problématique adressée & valeur ajoutée

- **Anticipation du churn client**  
  Le modèle permet d’identifier les clients à fort risque de résiliation afin de mettre en place des actions préventives (offres ciblées, fidélisation, support proactif).

- **Machine Learning réellement opérationnel**  
  Le modèle n’est pas réservé aux data scientists : il est accessible via une **API REST** et une **interface web simple**, utilisable par des équipes métiers ou techniques sans passer par un notebook.

- **Industrialisation et reproductibilité**  
  Grâce à la conteneurisation et à l’automatisation CI/CD, chaque évolution du code ou du modèle est **reconstruite, testée et redéployée de manière fiable et cohérente**.

- **Traçabilité, audit et gouvernance des modèles**  
  Les entraînements sont suivis avec **MLflow**, garantissant la reproductibilité, la transparence et la capacité à expliquer quel modèle est en production, avec quels paramètres et quelles performances.

---

## Réalisations techniques

### Données & Modélisation

- Pipeline de feature engineering structuré
- Modèle de classification **XGBoost**
- Enregistrement systématique des expérimentations dans **MLflow**

### Suivi et versioning des modèles (MLflow)

- Tracking des runs, paramètres, métriques et artefacts
- Modèles sérialisés associés à un **experiment MLflow nommé**
- Identification claire du modèle déployé en production

### Service d’inférence

- Application **FastAPI** exposant :
  - `POST /predict` pour effectuer des prédictions
  - `GET /` pour vérifier l’état de santé de l’application (health check)

### Interface utilisateur

- Interface **Gradio** montée sur `/ui`
- Tests manuels rapides, démonstrations et validation fonctionnelle du modèle via navigateur

### Conteneurisation

- Application packagée dans une **image Docker**
- Serveur `uvicorn` comme point d’entrée
- Exposition sur le **port 8000**
- Environnement d’exécution entièrement reproductible

### CI/CD (Intégration & Déploiement Continus)

- Pipeline **GitHub Actions** :
  - Build automatique de l’image Docker
  - Push vers Docker Hub
  - Déclenchement optionnel du redéploiement sur ECS
- Réduction des erreurs humaines et accélération des mises en production

### Déploiement Cloud & Orchestration

- Déploiement sur **AWS ECS Fargate** (containers serverless)
- Scalabilité et disponibilité sans gestion directe des serveurs

### Réseau & Sécurité

- **Application Load Balancer (ALB)** exposé sur HTTP:80
- Redirection vers un Target Group (IP) sur HTTP:8000
- Groupes de sécurité configurés selon le principe du moindre privilège

### Observabilité & supervision

- Centralisation des logs applicatifs via **CloudWatch Logs**
- Suivi des événements ECS (déploiements, redémarrages, erreurs)

---

## Flux de déploiement (vue d’ensemble)

1. Push du code sur la branche `main`
2. Exécution du pipeline GitHub Actions :
   - Build de l’image Docker
   - Publication sur Docker Hub
3. Mise à jour du service ECS (automatique ou manuelle)
4. L’ALB exécute les health checks sur `GET /`
5. Une fois le service déclaré sain, le trafic est redirigé vers la nouvelle version
6. Les utilisateurs peuvent :
   - Appeler l’endpoint `POST /predict`
   - Accéder à l’interface Gradio `/ui` via le DNS de l’ALB

---

## Problèmes rencontrés & solutions apportées

### Cibles ALB en état « unhealthy »

**Cause**

- Endpoint de health check absent ou incorrect
- Mauvais mapping des ports entre ALB et container

**Correctifs**

- Ajout d’un endpoint `GET /`
- Vérification du routage ALB (80 → 8000)
- Configuration du health check du Target Group sur `/`

---

### Erreur d’import Python dans le container (`ModuleNotFoundError`)

**Cause**

- Le dossier `src/` n’était pas inclus dans le `PYTHONPATH`

**Correctifs**

- Définition de `PYTHONPATH=/app/src` dans le Dockerfile
- Correction du chemin Uvicorn : `src.app.main:app`

---

### Timeout lors de l’accès au DNS de l’ALB

**Cause**

- Règles de sécurité non alignées avec le flux réseau réel

**Correctifs**

- ALB SG : inbound 80 depuis Internet
- ECS Task SG : inbound 8000 depuis le SG de l’ALB
- Sortant autorisé

---

### Nouveau modèle non pris en compte après déploiement

**Cause**

- Service ECS toujours lié à l’ancienne task definition

**Correctifs**

- Forcer un **nouveau déploiement ECS**
- Ajout d’une étape optionnelle de redeploy dans le pipeline CI/CD

---

### Erreur Gradio : “No runs found in experiment”

**Cause**

- Incohérence entre l’entraînement MLflow et le chargement du modèle en inférence

**Correctifs**

- Standardisation du nom de l’experiment MLflow
- Chargement cohérent du modèle loggé
- Fallback local pour le développement

---

### Différences entre environnement local et production

**Cause**

- Chemins MLflow différents selon l’environnement

**Correctifs**

- Local : chargement depuis `./mlruns/.../artifacts/model`
- Production : modèle packagé dans l’image Docker lors du build

---

## Conclusion

Ce projet démontre qu’un modèle de Machine Learning n’a de valeur en entreprise que s’il est **industrialisé, traçable, supervisé et maintenable**.  
Il illustre la transformation d’un notebook expérimental en une **application ML robuste, déployée sur le cloud et prête pour un usage réel en production**.
