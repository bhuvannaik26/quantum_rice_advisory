import os
from dotenv import load_dotenv
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
FLASK_SECRET_KEY    = os.getenv("FLASK_SECRET_KEY", "dev-secret")
MODEL_PATH          = os.path.join(os.path.dirname(__file__),
                                   "models", "stage1_rice.pth")

# Rice-specific agronomic thresholds (literature-backed)
RICE_THRESHOLDS = {
    "temp_blast_optimal_min": 24,    # °C — Magnaporthe oryzae thrives
    "temp_blast_optimal_max": 28,    # °C
    "temp_heat_stress":       35,    # °C — tillering damage
    "temp_cold_stress":       15,    # °C — germination failure
    "humidity_blight_trigger":85,    # % RH — bacterial blight threshold
    "humidity_blast_trigger": 90,    # % RH — blast spore germination
    "ph_min":                 5.5,   # optimal rice soil pH
    "ph_max":                 6.5,
    "nitrogen_low":           0.8,   # g/kg — deficiency threshold
    "nitrogen_optimal":       2.0,
    "rainfall_waterlog":      50,    # mm/day — waterlogging risk
}