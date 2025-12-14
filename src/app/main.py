"""
APPLICATION FASTAPI + GRADIO – SERVING ML PRÊT POUR LA PRODUCTION
=================================================================

Cette application fournit une solution complète de mise en production
d’un modèle de Machine Learning pour la prédiction du churn client
dans le secteur des télécommunications.

Elle expose :
- une API REST performante via FastAPI (accès programmatique)
- une interface Web interactive via Gradio (tests manuels et démonstrations)
- une validation stricte des données grâce à Pydantic

Architecture technique :
- FastAPI : API REST haute performance avec documentation OpenAPI automatique
- Gradio : Interface utilisateur simple et professionnelle pour la démonstration
- Pydantic : Validation des entrées et cohérence du schéma de données
- Pipeline ML : Logique d’inférence centralisée et réutilisable (API + UI)

Objectif principal :
Assurer une industrialisation propre du modèle ML en garantissant
la réutilisabilité du pipeline, la robustesse des entrées,
et la compatibilité avec un déploiement cloud (AWS).
"""


from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict  # Core ML inference logic


# Initialize FastAPI application
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0.0"
)

# === HEALTH CHECK ENDPOINT ===
@app.get("/")
def root():
    return {"status": "ok"}


# === REQUEST DATA SCHEMA ===
class CustomerData(BaseModel):
    gender: str
    Partner: str
    Dependents: str

    PhoneService: str
    MultipleLines: str

    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    tenure: int
    MonthlyCharges: float
    TotalCharges: float


# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: CustomerData):
    try:
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}


# === GRADIO WEB INTERFACE ===
def gradio_interface(
    gender, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    data = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }

    result = predict(data)
    return str(result)


# ===================================================
# === NEW PRO UI (replace ONLY the old gr.Interface) ===
# ===================================================

css = """
:root{
  --brand-blue:#1f2a67;
  --accent:#f08a1a;
  --muted:#6b7280;
  --bg:#ffffff;
  --card:#ffffff;
  --border:#e6e8ef;
}
.gradio-container{
  background: var(--bg) !important;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
}
.be-header{ text-align:center; padding: 18px 10px 6px 10px; }
.be-title{
  color: var(--brand-blue);
  font-size: 42px;
  font-weight: 800;
  margin: 0;
}
.be-subtitle{ color: var(--muted); font-size: 16px; margin-top: 8px; }
.be-divider{
  height: 4px; width: 100%;
  background: var(--accent);
  border-radius: 999px;
  margin: 12px 0 18px 0;
}
.be-card{
  border: 1px solid var(--border);
  background: var(--card);
  border-radius: 16px;
  padding: 16px 16px 10px 16px;
  box-shadow: 0 8px 18px rgba(15, 23, 42, .06);
}
.be-card-title{
  color: var(--brand-blue);
  font-weight: 800;
  font-size: 18px;
  margin: 0 0 8px 0;
}
.be-hours{
  color: var(--accent);
  font-weight: 800;
  font-size: 14px;
  margin: 0 0 10px 0;
}
.be-help{
  color: var(--muted);
  font-size: 13px;
  line-height: 1.35rem;
  margin: 0;
}
label span{ font-weight: 700 !important; color: #111827 !important; }
textarea{ border-radius: 14px !important; }
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill="#ffffff",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_radius="16px",
)

with gr.Blocks(theme=theme, css=css, title="Telco Customer Churn Predictor") as demo:

    gr.HTML(
        """
        <div class="be-header">
          <h1 class="be-title">Programmes Détaillés</h1>
          <div class="be-subtitle">Interface de démonstration – déploiement FastAPI + Gradio sur AWS</div>
        </div>
        <div class="be-divider"></div>
        """
    )

    with gr.Tabs():
        with gr.Tab("Data Analytics"):
            gr.Markdown("### Semestre 1 : Fondamentaux Data & Industrie")

            with gr.Row(equal_height=True):

                with gr.Column(scale=7):
                    with gr.Group(elem_classes=["be-card"]):
                        gr.HTML(
                            '<div class="be-card-title">Formulaire Client</div>'
                            '<div class="be-hours">Renseignez les 18 variables</div>'
                            '<p class="be-help">Les champs suivent exactement le schéma <b>CustomerData</b> '
                            'et utilisent le même pipeline <b>predict()</b> que l’API.</p>'
                        )

                        with gr.Accordion("Démographie", open=True):
                            gender = gr.Dropdown(["Male", "Female"], label="gender", value="Male")
                            Partner = gr.Dropdown(["Yes", "No"], label="Partner", value="No")
                            Dependents = gr.Dropdown(["Yes", "No"], label="Dependents", value="No")

                        with gr.Accordion("Téléphonie", open=False):
                            PhoneService = gr.Dropdown(["Yes", "No"], label="PhoneService", value="Yes")
                            MultipleLines = gr.Dropdown(["Yes", "No", "No phone service"], label="MultipleLines", value="No")

                        with gr.Accordion("Internet (services)", open=False):
                            InternetService = gr.Dropdown(["DSL", "Fiber optic", "No"], label="InternetService", value="Fiber optic")
                            OnlineSecurity = gr.Dropdown(["Yes", "No", "No internet service"], label="OnlineSecurity", value="No")
                            OnlineBackup = gr.Dropdown(["Yes", "No", "No internet service"], label="OnlineBackup", value="No")
                            DeviceProtection = gr.Dropdown(["Yes", "No", "No internet service"], label="DeviceProtection", value="No")
                            TechSupport = gr.Dropdown(["Yes", "No", "No internet service"], label="TechSupport", value="No")
                            StreamingTV = gr.Dropdown(["Yes", "No", "No internet service"], label="StreamingTV", value="Yes")
                            StreamingMovies = gr.Dropdown(["Yes", "No", "No internet service"], label="StreamingMovies", value="Yes")

                        with gr.Accordion("Contrat & Paiement", open=False):
                            Contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month")
                            PaperlessBilling = gr.Dropdown(["Yes", "No"], label="PaperlessBilling", value="Yes")
                            PaymentMethod = gr.Dropdown(
                                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                                label="PaymentMethod",
                                value="Electronic check",
                            )

                        with gr.Accordion("Numérique", open=False):
                            tenure = gr.Number(label="tenure", value=1, minimum=0, maximum=100)
                            MonthlyCharges = gr.Number(label="MonthlyCharges", value=85.0, minimum=0, maximum=200)
                            TotalCharges = gr.Number(label="TotalCharges", value=85.0, minimum=0, maximum=10000)

                        with gr.Row():
                            submit_btn = gr.Button("Lancer la prédiction", variant="primary")
                            clear_btn = gr.Button("Réinitialiser", variant="secondary")

                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["be-card"]):
                        gr.HTML(
                            '<div class="be-card-title">Résultat</div>'
                            '<div class="be-hours">Prédiction churn</div>'
                            '<p class="be-help">Résultat renvoyé par le modèle (même logique que l’endpoint <b>/predict</b>).</p>'
                        )
                        output = gr.Textbox(label="Churn Prediction", lines=2, interactive=False)

                        gr.Markdown("#### Exemples rapides")
                        gr.Examples(
                            examples=[
                                ["Female", "No", "No", "Yes", "No", "Fiber optic", "No", "No", "No",
                                 "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check",
                                 1, 85.0, 85.0],
                                ["Male", "Yes", "Yes", "Yes", "Yes", "DSL", "Yes", "Yes", "Yes",
                                 "Yes", "No", "No", "Two year", "No", "Credit card (automatic)",
                                 60, 45.0, 2700.0]
                            ],
                            inputs=[
                                gender, Partner, Dependents, PhoneService, MultipleLines,
                                InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                                TechSupport, StreamingTV, StreamingMovies, Contract,
                                PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
                            ],
                        )

            submit_btn.click(
                fn=gradio_interface,
                inputs=[
                    gender, Partner, Dependents, PhoneService, MultipleLines,
                    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                    TechSupport, StreamingTV, StreamingMovies, Contract,
                    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
                ],
                outputs=output
            )

            clear_btn.click(
                fn=lambda: (
                    "Male", "No", "No", "Yes", "No", "Fiber optic", "No", "No", "No",
                    "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check",
                    1, 85.0, 85.0, ""
                ),
                inputs=[],
                outputs=[
                    gender, Partner, Dependents, PhoneService, MultipleLines,
                    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                    TechSupport, StreamingTV, StreamingMovies, Contract,
                    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges,
                    output
                ]
            )

        with gr.Tab("Cybersecurity"):
            gr.Markdown("Contenu à venir.")
        with gr.Tab("Cloud & DevOps"):
            gr.Markdown("Contenu à venir.")
        with gr.Tab("AI & ML"):
            gr.Markdown("Contenu à venir.")


# === MOUNT GRADIO UI INTO FASTAPI ===
app = gr.mount_gradio_app(app, demo, path="/ui")
