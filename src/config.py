from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

for dir in [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    dir.mkdir(exist_ok=True)

MUTUAL_FUNDS_CSV = DATA_DIR / "MutualFunds.csv"

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

MODELS = {
    "ridge": {
        "name": "Ridge Regression",
        "description": "Baseline linéaire régularisé (L2). Rapide et interprétable.",
        "path": MODELS_DIR / "ridge.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Ensemble d'arbres de décision. Robuste aux outliers et non-linéaire.",
        "path": MODELS_DIR / "random_forest.joblib",
    },
    "gradient_boosting": {
        "name": "Hist Gradient Boosting",
        "description": "Boosting par histogrammes. Modèle de pointe, gère nativement les NaN.",
        "path": MODELS_DIR / "gradient_boosting.joblib",
    },
}
