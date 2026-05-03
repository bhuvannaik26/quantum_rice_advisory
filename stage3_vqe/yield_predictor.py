import pennylane as qml
import numpy as np

# ── VQE device ────────────────────────────────────────────────
n_qubits = 6
dev      = qml.device("default.qubit", wires=n_qubits)

# ── Rice yield constants (ICAR India data) ────────────────────
BASELINE_YIELD_T_HA  = 4.5    # India average t/ha
MSP_PER_TONNE_INR    = 23100  # MSP 2024-25 (₹/tonne common rice)

# ── Disease yield penalty table (literature-backed) ───────────
# Format: disease_label → {severity_idx: penalty_fraction}
DISEASE_PENALTIES = {
    "Rice Blast": {
        0: 0.08,   # Low
        1: 0.20,   # Medium
        2: 0.40,   # High
    },
    "Rice Tungro": {
        0: 0.15,
        1: 0.40,
        2: 0.70,
    },
    "Bacterial Leaf Blight": {
        0: 0.05,
        1: 0.18,
        2: 0.35,
    },
    "Sheath Blight": {
        0: 0.05,
        1: 0.15,
        2: 0.30,
    },
    "Brown Spot": {
        0: 0.04,
        1: 0.12,
        2: 0.25,
    },
    "Rice Hispa": {
        0: 0.03,
        1: 0.10,
        2: 0.20,
    },
    "Rice Leaffolder": {
        0: 0.03,
        1: 0.10,
        2: 0.20,
    },
    "Rice Stripes": {
        0: 0.10,
        1: 0.25,
        2: 0.50,
    },
    "Narrow Brown Leaf Spot": {
        0: 0.02,
        1: 0.08,
        2: 0.15,
    },
    "Leaf Scald": {
        0: 0.02,
        1: 0.07,
        2: 0.15,
    },
    "Healthy": {
        0: 0.00,
        1: 0.00,
        2: 0.00,
    },
}

# ── Seasonal forecast labels ──────────────────────────────────
def _seasonal_label(predicted: float) -> str:
    ratio = predicted / BASELINE_YIELD_T_HA
    if ratio >= 0.90: return "Good"
    if ratio >= 0.75: return "Moderate"
    if ratio >= 0.55: return "Poor"
    return "Critical"


# ── Hamiltonian defined OUTSIDE the circuit (PennyLane 0.36+) ─
# Weights reflect agronomic importance of each qubit factor
H_coeffs = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
H_obs    = [qml.PauliZ(i) for i in range(n_qubits)]
H        = qml.Hamiltonian(H_coeffs, H_obs)


# ── VQE Ansatz ───────────────────────────────────────────────
@qml.qnode(dev, interface="numpy")
def vqe_circuit(features, weights):
    """
    6-qubit VQE circuit for yield loss estimation.

    Qubit encoding:
      q[0] → disease_penalty    (from DISEASE_PENALTIES table)
      q[1] → blast_risk         (from Stage 2 QAOA)
      q[2] → soil_stress        (from Stage 2 QAOA)
      q[3] → temperature_factor (weather stress)
      q[4] → rainfall_factor    (waterlogging / drought)
      q[5] → severity_idx       (from Stage 1)

    Ansatz: StronglyEntanglingLayers
    Hamiltonian: weighted PauliZ sum — minimized ⟨H⟩
    maps to predicted yield loss
    """
    # ── Encode features as RY rotations ──────────────────────
    qml.AngleEmbedding(
        features * np.pi,
        wires=range(n_qubits),
        rotation='Y'
    )

    # ── StronglyEntanglingLayers ansatz ───────────────────────
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    # ── Single expval of the full Hamiltonian (PennyLane 0.36+) ─
    return qml.expval(H)


def _encode_features(disease_label: str,
                     severity_idx:   int,
                     risk:           dict,
                     weather:        dict,
                     soil:           dict) -> np.ndarray:
    """
    Convert all Stage 1 + Stage 2 outputs to
    6 normalized [0,1] features for VQE encoding.
    """
    # q[0] — disease penalty
    penalties   = DISEASE_PENALTIES.get(disease_label,
                  DISEASE_PENALTIES["Healthy"])
    disease_pen = float(penalties.get(severity_idx, 0.0))

    # q[1] — blast risk (already 0-1 from QAOA)
    blast_risk  = float(risk.get("blast_risk", 0.0))

    # q[2] — soil stress (already 0-1 from QAOA)
    soil_stress = float(risk.get("soil_stress", 0.0))

    # q[3] — temperature stress factor
    temp = weather.get("temperature", 25.0)
    # Optimal rice temp = 25-30°C
    # Stress ramps up outside this range
    if 25 <= temp <= 30:
        temp_stress = 0.0
    elif temp > 30:
        temp_stress = min(1.0, (temp - 30) / 15.0)
    else:
        temp_stress = min(1.0, (25 - temp) / 15.0)

    # q[4] — rainfall stress
    rainfall = weather.get("rainfall_mm", 0.0)
    humidity = weather.get("humidity", 70.0)
    # Waterlogging risk > 50mm OR drought < 20% humidity
    if rainfall > 50:
        rain_stress = min(1.0, (rainfall - 50) / 50.0)
    elif humidity < 30:
        rain_stress = min(1.0, (30 - humidity) / 30.0)
    else:
        rain_stress = 0.1  # baseline minimal stress

    # q[5] — severity from Stage 1 (0/1/2 → normalize)
    severity_norm = severity_idx / 2.0

    features = np.array([
        disease_pen,
        blast_risk,
        soil_stress,
        temp_stress,
        rain_stress,
        severity_norm
    ], dtype=float)

    return np.clip(features, 0.0, 1.0)


def predict_yield(disease_label: str,
                  severity_idx:   int,
                  severity_label: str,
                  category_label: str,
                  risk:           dict,
                  weather:        dict,
                  soil:           dict) -> dict:
    """
    Main Stage 3 function called by routes.py.
    Returns complete yield prediction with economic loss.
    """
    features = _encode_features(
        disease_label, severity_idx, risk, weather, soil
    )

    # Pre-optimized VQE weights
    # Shape: (n_layers=2, n_qubits=6, 3) for StronglyEntanglingLayers
    weights = np.array([
        [
            [0.42, 0.35, 0.28],
            [0.61, 0.44, 0.33],
            [0.38, 0.52, 0.41],
            [0.55, 0.39, 0.47],
            [0.48, 0.61, 0.35],
            [0.33, 0.45, 0.52],
        ],
        [
            [0.35, 0.48, 0.42],
            [0.58, 0.37, 0.55],
            [0.41, 0.63, 0.38],
            [0.49, 0.42, 0.61],
            [0.53, 0.35, 0.44],
            [0.39, 0.56, 0.48],
        ],
    ])

    # Run VQE circuit — returns scalar float
    vqe_energy = float(vqe_circuit(features, weights))

    # ── Convert VQE energy → yield loss ──────────────────────
    # ⟨H⟩ ∈ [-1, +1]
    # +1 = all qubits |0⟩ = minimal stress = low loss
    # -1 = all qubits |1⟩ = maximum stress = high loss
    # Map to [0, 1] loss fraction
    quantum_loss_fraction = (1.0 - vqe_energy) / 2.0

    # ── Disease penalty (from table — deterministic) ──────────
    penalties   = DISEASE_PENALTIES.get(disease_label,
                  DISEASE_PENALTIES["Healthy"])
    disease_pen = float(penalties.get(severity_idx, 0.0))

    # ── Environmental penalty (from VQE quantum component) ────
    # VQE captures interactions between weather+soil+blast risk
    env_pen = max(0.0, quantum_loss_fraction - disease_pen * 0.5)
    env_pen = min(env_pen, 0.30)  # cap env penalty at 30%

    # ── Total penalty ─────────────────────────────────────────
    total_penalty = min(0.95, disease_pen + env_pen)

    # ── Yield calculations ────────────────────────────────────
    predicted_yield  = round(BASELINE_YIELD_T_HA * (1.0 - total_penalty), 2)
    yield_loss_t_ha  = round(BASELINE_YIELD_T_HA - predicted_yield, 2)
    yield_loss_pct   = round(total_penalty * 100, 1)
    disease_loss_pct = round(disease_pen * 100, 1)
    env_loss_pct     = round(env_pen * 100, 1)

    # ── Economic loss in INR ──────────────────────────────────
    economic_loss_inr = round(yield_loss_t_ha * MSP_PER_TONNE_INR)
    potential_revenue = round(BASELINE_YIELD_T_HA * MSP_PER_TONNE_INR)
    expected_revenue  = round(predicted_yield * MSP_PER_TONNE_INR)

    # ── Loss reason string ────────────────────────────────────
    if disease_label == "Healthy":
        loss_reason = "No disease detected"
        if env_loss_pct > 5:
            loss_reason += f" — environmental stress {env_loss_pct}%"
    else:
        loss_reason = (
            f"{disease_label} ({severity_label} severity) "
            f"→ {disease_loss_pct}% disease loss"
        )
        if env_loss_pct > 3:
            loss_reason += (
                f" + {env_loss_pct}% environmental stress"
            )

    return {
        # ── Yield numbers ─────────────────────────────────────
        "baseline_yield_t_ha":  BASELINE_YIELD_T_HA,
        "predicted_yield_t_ha": predicted_yield,
        "yield_loss_t_ha":      yield_loss_t_ha,
        "yield_loss_pct":       yield_loss_pct,
        "disease_penalty_pct":  disease_loss_pct,
        "env_penalty_pct":      env_loss_pct,

        # ── Economic numbers ──────────────────────────────────
        "msp_per_tonne_inr":     MSP_PER_TONNE_INR,
        "potential_revenue_inr": potential_revenue,
        "expected_revenue_inr":  expected_revenue,
        "economic_loss_inr":     economic_loss_inr,

        # ── Labels ────────────────────────────────────────────
        "seasonal_forecast": _seasonal_label(predicted_yield),
        "loss_reason":       loss_reason,
        "category":          category_label,

        # ── Quantum diagnostics ───────────────────────────────
        "vqe_energy":       round(vqe_energy, 4),
        "features_encoded": features.tolist(),
    }