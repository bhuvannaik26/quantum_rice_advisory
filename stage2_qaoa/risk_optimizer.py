import pennylane as qml
import numpy as np
from config import RICE_THRESHOLDS

# ── Quantum device ────────────────────────────────────────────
n_qubits = 4
dev      = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="numpy")
def qaoa_risk_circuit(features, weights):
    """
    4-qubit QAOA circuit for rice risk optimization.

    Qubit encoding:
      q[0] → blast_risk      (temp + humidity → fungal blast)
      q[1] → blight_risk     (humidity → bacterial blight)
      q[2] → soil_stress     (pH + nitrogen deviation)
      q[3] → disease_severity (from Stage 1 model output)

    Architecture:
      AngleEmbedding → QAOA cost unitary (ZZ) → mixer (RX) × 2 layers
    """
    # ── Encode features as qubit rotation angles ─────────────
    qml.AngleEmbedding(features * np.pi, wires=range(n_qubits),
                       rotation='Y')

    # ── QAOA layers: cost unitary + mixer ────────────────────
    n_layers = 2
    for layer in range(n_layers):
        # Cost unitary — ZZ interactions between risk factors
        # q0-q1: blast interacts with blight (both weather-driven)
        qml.CNOT(wires=[0, 1])
        qml.RZ(weights[layer, 0], wires=1)
        qml.CNOT(wires=[0, 1])

        # q1-q2: blight interacts with soil (soil amplifies blight)
        qml.CNOT(wires=[1, 2])
        qml.RZ(weights[layer, 1], wires=2)
        qml.CNOT(wires=[1, 2])

        # q2-q3: soil stress interacts with disease severity
        qml.CNOT(wires=[2, 3])
        qml.RZ(weights[layer, 2], wires=3)
        qml.CNOT(wires=[2, 3])

        # q0-q3: blast risk interacts with overall severity
        qml.CNOT(wires=[0, 3])
        qml.RZ(weights[layer, 3], wires=3)
        qml.CNOT(wires=[0, 3])

        # Mixer unitary — RX on all qubits
        for i in range(n_qubits):
            qml.RX(weights[layer, 4 + i], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


def normalize_inputs(weather: dict, soil: dict,
                     disease_severity: int) -> np.ndarray:
    """
    Convert real-world sensor values to [0,1] qubit encoding.
    Each output maps to one qubit.
    """
    T   = RICE_THRESHOLDS
    temp     = weather.get("temperature", 25.0)
    humidity = weather.get("humidity",    70.0)
    rainfall = weather.get("rainfall_mm", 0.0)
    ph       = soil.get("ph",             6.0)
    nitrogen = soil.get("nitrogen_g_kg",  1.0)

    # ── q[0]: blast_risk ─────────────────────────────────────
    # Peaks when temp=24-28°C AND humidity>90%
    t_mid        = (T["temp_blast_optimal_min"] +
                    T["temp_blast_optimal_max"]) / 2
    temp_factor  = max(0.0, 1.0 - abs(temp - t_mid) / 10.0)
    humid_factor = min(1.0, humidity / T["humidity_blast_trigger"])
    blast_risk   = temp_factor * humid_factor

    # ── q[1]: blight_risk ────────────────────────────────────
    # Purely humidity-driven — above 85% RH triggers blight
    if humidity >= T["humidity_blight_trigger"]:
        blight_risk = min(1.0, (humidity - T["humidity_blight_trigger"])
                          / (100 - T["humidity_blight_trigger"]))
        blight_risk = 0.5 + blight_risk * 0.5   # floor at 0.5
    else:
        blight_risk = humidity / (T["humidity_blight_trigger"] * 2)

    # Add rainfall component — waterlogging increases blight
    rain_factor  = min(0.3, rainfall / T["rainfall_waterlog"] * 0.3)
    blight_risk  = min(1.0, blight_risk + rain_factor)

    # ── q[2]: soil_stress ────────────────────────────────────
    ph_mid       = (T["ph_min"] + T["ph_max"]) / 2
    ph_stress    = min(1.0, abs(ph - ph_mid) / 2.0)
    n_stress     = max(0.0, 1.0 - nitrogen / T["nitrogen_optimal"])
    soil_stress  = (ph_stress * 0.5 + n_stress * 0.5)

    # ── q[3]: disease_severity from Stage 1 ──────────────────
    # Stage 1 returns 0=Low / 1=Medium / 2=High → normalize
    severity_norm = disease_severity / 2.0

    features = np.array(
        [blast_risk, blight_risk, soil_stress, severity_norm],
        dtype=float
    )
    # Clip to [0,1] safety
    return np.clip(features, 0.0, 1.0)


def compute_risk(weather: dict, soil: dict,
                 disease_severity: int = 0) -> dict:
    """
    Run QAOA circuit and return per-category risk scores.
    This is the main function called by routes.py.
    """
    features = normalize_inputs(weather, soil, disease_severity)

    # Pre-optimized weights for rice agronomic thresholds
    # Shape: (n_layers=2, cost_params=4 + mixer_params=4) = (2,8)
    weights = np.array([
        [0.42, 0.61, 0.38, 0.55, 0.48, 0.52, 0.44, 0.60],
        [0.35, 0.58, 0.41, 0.49, 0.53, 0.47, 0.39, 0.56],
    ])

    raw_expectations = qaoa_risk_circuit(features, weights)

    # Convert PauliZ expectations [-1,+1] → risk scores [0,1]
    # ⟨Z⟩ = +1 means low risk (qubit in |0⟩)
    # ⟨Z⟩ = -1 means high risk (qubit in |1⟩)
    scores = [(1.0 - float(r)) / 2.0 for r in raw_expectations]

    blast_risk    = round(scores[0], 3)
    blight_risk   = round(scores[1], 3)
    soil_stress   = round(scores[2], 3)
    sev_component = round(scores[3], 3)
    overall       = round(
        0.35 * blast_risk +
        0.30 * blight_risk +
        0.20 * soil_stress +
        0.15 * sev_component, 3
    )

    return {
        "blast_risk":         blast_risk,
        "blight_risk":        blight_risk,
        "soil_stress":        soil_stress,
        "severity_component": sev_component,
        "overall_risk":       overall,
        "risk_level":         _risk_label(overall),
        "features_encoded":   features.tolist(),
        "qaoa_expectations":  [round(float(r), 3)
                               for r in raw_expectations],
    }


def _risk_label(score: float) -> str:
    if score >= 0.7: return "Critical"
    if score >= 0.5: return "High"
    if score >= 0.3: return "Medium"
    return "Low"