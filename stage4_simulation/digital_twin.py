"""
stage4_simulation/digital_twin.py
══════════════════════════════════════════════════════════════════════════════
Stage 4 — Quantum Farm Digital Twin
Integrates three simulation engines:
  A) PennyLane  — Farm Health Score (5 dimensions, colour-coded)
  B) Qiskit     — Single best intervention with confidence %
  C) Digital Twin — deterministic farm state machine across crop growth stages

Called by routes.py:
    from stage4_simulation.digital_twin import run_stage4
    sim = run_stage4(stage1, risk, yield_pred, weather, soil)
══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pennylane as qml
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_YIELD_T_HA = 4.5
MSP_PER_TONNE_INR   = 23100

INTERVENTION_COSTS = {
    "fungicide":    1800,
    "bactericide":  1500,
    "pesticide":    1200,
    "fertilizer_N": 900,
    "irrigation":   600,
    "drainage":     700,
    "no_action":    0,
}

INTERVENTION_RECOVERY = {
    "Rice Blast": {
        "fungicide": 0.72, "fertilizer_N": 0.10,
        "irrigation": 0.05, "no_action": 0.0
    },
    "Bacterial Leaf Blight": {
        "bactericide": 0.65, "drainage": 0.20,
        "fertilizer_N": 0.08, "no_action": 0.0
    },
    "Brown Spot": {
        "fungicide": 0.60, "fertilizer_N": 0.25,
        "no_action": 0.0
    },
    "Sheath Blight": {
        "fungicide": 0.65, "irrigation": 0.10,
        "no_action": 0.0
    },
    "Rice Hispa": {
        "pesticide": 0.75, "no_action": 0.0
    },
    "Rice Leaffolder": {
        "pesticide": 0.70, "no_action": 0.0
    },
    "Rice Tungro": {
        "pesticide": 0.45, "no_action": 0.0
    },
    "Rice Stripes": {
        "pesticide": 0.50, "no_action": 0.0
    },
    "Narrow Brown Leaf Spot": {
        "fungicide": 0.55, "fertilizer_N": 0.20,
        "no_action": 0.0
    },
    "Leaf Scald": {
        "fungicide": 0.58, "irrigation": 0.08,
        "no_action": 0.0
    },
    "Healthy": {
        "no_action": 0.0, "fertilizer_N": 0.05
    },
}

GROWTH_STAGES = [
    "Germination", "Tillering", "Panicle Initiation",
    "Heading", "Grain Filling", "Maturity",
]

STAGE_VULNERABILITY = {
    "Germination":        0.40,
    "Tillering":          0.85,
    "Panicle Initiation": 1.00,
    "Heading":            0.95,
    "Grain Filling":      0.75,
    "Maturity":           0.30,
}

# Human-readable intervention labels for farmers
INTERVENTION_LABELS = {
    "fungicide":    "Spray Fungicide",
    "bactericide":  "Spray Bactericide",
    "pesticide":    "Spray Pesticide",
    "fertilizer_N": "Apply Nitrogen Fertilizer",
    "irrigation":   "Improve Irrigation",
    "drainage":     "Open Field Drainage",
    "no_action":    "No Treatment Needed",
}

# What to buy / how to apply (farmer instructions)
INTERVENTION_INSTRUCTIONS = {
    "fungicide": {
        "product":   "Tricyclazole 75% WP  or  Propiconazole 25% EC",
        "dose":      "1g per litre of water (Tricyclazole)  /  1ml per litre (Propiconazole)",
        "method":    "Spray evenly on leaves in the early morning or evening",
        "when":      "Apply immediately — do not delay beyond 2 days",
    },
    "bactericide": {
        "product":   "Copper Oxychloride 50% WP",
        "dose":      "3g per litre of water",
        "method":    "Spray on lower leaves and stems",
        "when":      "Apply within 1–2 days; drain excess water first",
    },
    "pesticide": {
        "product":   "Chlorpyrifos 20% EC  or  Cartap Hydrochloride 50% SP",
        "dose":      "2ml per litre (Chlorpyrifos)  /  1g per litre (Cartap)",
        "method":    "Spray on affected tillers; target leaf-feeding insects",
        "when":      "Apply in the evening to protect pollinators",
    },
    "fertilizer_N": {
        "product":   "Urea (46% Nitrogen)",
        "dose":      "25–30 kg per acre as top dressing",
        "method":    "Broadcast evenly in standing water; keep field flooded 3 days",
        "when":      "Apply within this week for best uptake",
    },
    "irrigation": {
        "product":   "Water — maintain 5cm standing water",
        "dose":      "Flood to 5cm depth",
        "method":    "Ensure uniform water coverage across the field",
        "when":      "Start today; monitor daily",
    },
    "drainage": {
        "product":   "Open drainage channels",
        "dose":      "Drain fully for 2–3 days",
        "method":    "Open outlet channels; allow field to dry partially",
        "when":      "Begin today — waterlogging worsens bacterial spread",
    },
    "no_action": {
        "product":   "No treatment required",
        "dose":      "—",
        "method":    "Continue regular monitoring every 3–4 days",
        "when":      "Monitor and re-assess if symptoms appear",
    },
}

# Urgency based on severity
URGENCY_LABELS = {
    0: {"label": "Act This Week",  "days": 7,  "color": "green"},
    1: {"label": "Act Within 3 Days", "days": 3, "color": "orange"},
    2: {"label": "Act Today",      "days": 1,  "color": "red"},
}


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION A — PennyLane Farm Health Score
# ══════════════════════════════════════════════════════════════════════════════
#
# 5 qubits → 5 farm health dimensions:
#   q[0] → Disease Health    (how badly is disease hurting the crop?)
#   q[1] → Environmental Health (weather + climate stress)
#   q[2] → Soil Health       (pH, nitrogen, organic carbon)
#   q[3] → Intervention Readiness (how well will treatment work?)
#   q[4] → Recovery Potential (can the crop bounce back?)
#
# ⟨Z⟩ = +1 → perfectly healthy dimension
# ⟨Z⟩ = -1 → critically stressed dimension
# Score = (⟨Z⟩ + 1) / 2 × 100  →  0 to 100 per dimension

_n_health_qubits = 5
_dev_health = qml.device("default.qubit", wires=_n_health_qubits)


@qml.qnode(_dev_health, interface="numpy")
def _farm_health_circuit(health_inputs: np.ndarray, weights: np.ndarray):
    """
    5-qubit circuit encoding farm health dimensions.
    AngleEmbedding maps each dimension's stress → qubit rotation.
    Entanglement layers model how dimensions affect each other
    (e.g. bad soil worsens disease impact).
    """
    # Encode: high stress = large rotation = more |1⟩ = lower ⟨Z⟩
    qml.AngleEmbedding(health_inputs * np.pi, wires=range(_n_health_qubits), rotation='Y')

    # Layer 1: cross-dimension entanglement
    qml.CNOT(wires=[0, 1])   # disease affects environment response
    qml.RZ(weights[0], wires=1)
    qml.CNOT(wires=[0, 1])

    qml.CNOT(wires=[2, 0])   # soil health affects disease severity
    qml.RZ(weights[1], wires=0)
    qml.CNOT(wires=[2, 0])

    qml.CNOT(wires=[1, 3])   # env stress affects intervention readiness
    qml.RZ(weights[2], wires=3)
    qml.CNOT(wires=[1, 3])

    qml.CNOT(wires=[3, 4])   # intervention readiness affects recovery
    qml.RZ(weights[3], wires=4)
    qml.CNOT(wires=[3, 4])

    qml.CNOT(wires=[0, 4])   # disease severity affects recovery potential
    qml.RZ(weights[4], wires=4)
    qml.CNOT(wires=[0, 4])

    # Layer 2: mixer
    for i in range(_n_health_qubits):
        qml.RX(weights[5 + i], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(_n_health_qubits)]


_HEALTH_WEIGHTS = np.array([
    0.55, 0.42, 0.61, 0.38, 0.49,   # RZ weights
    0.31, 0.47, 0.52, 0.44, 0.36,   # RX mixer weights
], dtype=float)


def _compute_health_inputs(disease_pen: float, env_pen: float,
                           soil: dict, severity_idx: int,
                           recovery_map: dict) -> np.ndarray:
    """
    Convert farm agronomic context into 5 health stress values [0, 1].
    0 = perfectly healthy, 1 = critically stressed.
    """
    # q[0] Disease stress
    disease_stress = np.clip(disease_pen, 0.0, 1.0)

    # q[1] Environmental stress (temperature, humidity)
    temp = soil.get("temperature", 25.0) if isinstance(soil, dict) else 25.0
    temp_stress = np.clip((temp - 25.0) / 15.0, 0.0, 1.0)  # stress above 25°C
    env_stress  = np.clip((env_pen * 0.6 + temp_stress * 0.4), 0.0, 1.0)

    # q[2] Soil health stress (bad pH and low nitrogen = high stress)
    ph  = soil.get("ph", 6.5)
    nit = soil.get("nitrogen_g_kg", 1.5)
    ph_stress  = np.clip(abs(ph - 6.5) / 2.0, 0.0, 1.0)
    nit_stress = np.clip(1.0 - nit / 2.5, 0.0, 1.0)
    soil_stress = (ph_stress * 0.5 + nit_stress * 0.5)

    # q[3] Intervention readiness (high = treatment will work well)
    # Inverted: low stress = intervention works well
    best_recovery = max(recovery_map.values()) if recovery_map else 0.0
    interv_stress = np.clip(1.0 - best_recovery, 0.0, 1.0)

    # q[4] Recovery potential
    # Lower disease + better soil = higher recovery potential
    recovery_stress = np.clip(
        disease_pen * 0.5 + soil_stress * 0.3 + (severity_idx / 2.0) * 0.2,
        0.0, 1.0
    )

    return np.array([
        disease_stress,
        env_stress,
        soil_stress,
        interv_stress,
        recovery_stress,
    ], dtype=float)


def run_pennylane_health(disease_label: str,
                         disease_pen:   float,
                         env_pen:       float,
                         severity_idx:  int,
                         soil:          dict) -> dict:
    """
    Run 5-qubit PennyLane health circuit.
    Returns a farmer-friendly Farm Health Score with colour and 5 dimension bars.
    """
    recovery_map = INTERVENTION_RECOVERY.get(
        disease_label, INTERVENTION_RECOVERY["Healthy"]
    )

    health_inputs = _compute_health_inputs(
        disease_pen, env_pen, soil, severity_idx, recovery_map
    )

    expectations = _farm_health_circuit(health_inputs, _HEALTH_WEIGHTS)

    # Convert ⟨Z⟩ ∈ [-1, +1] → health score ∈ [0, 100]
    # +1 = healthy (score 100), -1 = critical (score 0)
    dimension_scores = [
        round((float(e) + 1.0) / 2.0 * 100, 1)
        for e in expectations
    ]

    dimension_names = [
        "Disease Control",
        "Environmental Health",
        "Soil Health",
        "Treatment Effectiveness",
        "Recovery Potential",
    ]

    # Weighted overall score
    weights = [0.35, 0.20, 0.20, 0.15, 0.10]
    overall = round(sum(s * w for s, w in zip(dimension_scores, weights)), 1)

    # Colour band
    if overall >= 70:
        color  = "green"
        label  = "Healthy"
        advice = "Your crop is in good condition. Continue regular monitoring."
    elif overall >= 45:
        color  = "orange"
        label  = "Moderate Stress"
        advice = "Your crop shows stress. Act soon to prevent yield loss."
    else:
        color  = "red"
        label  = "Critical"
        advice = "Your crop is under severe stress. Immediate action required."

    dimensions = [
        {
            "name":  name,
            "score": score,
            "color": (
                "green"  if score >= 70 else
                "orange" if score >= 45 else
                "red"
            ),
        }
        for name, score in zip(dimension_names, dimension_scores)
    ]

    return {
        "overall_score":  overall,
        "color":          color,
        "label":          label,
        "advice":         advice,
        "dimensions":     dimensions,
        "raw_expectations": [round(float(e), 4) for e in expectations],
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION B — Qiskit Single Best Intervention with Confidence
# ══════════════════════════════════════════════════════════════════════════════

QISKIT_INTERVENTION_MAP = {
    0: "no_action",
    1: "fungicide",
    2: "bactericide",
    3: "pesticide",
    4: "fertilizer_N",
    5: "irrigation",
    6: "drainage",
    7: "fungicide+fertilizer_N",
}


def _build_grover_oracle(target_indices: list, n_qubits: int = 3) -> QuantumCircuit:
    oracle = QuantumCircuit(n_qubits, name="Oracle")
    for idx in target_indices:
        bits = format(idx, f'0{n_qubits}b')
        for i, bit in enumerate(reversed(bits)):
            if bit == '0':
                oracle.x(i)
        oracle.h(n_qubits - 1)
        if n_qubits == 3:
            oracle.ccx(0, 1, 2)
        elif n_qubits == 2:
            oracle.cx(0, 1)
        oracle.h(n_qubits - 1)
        for i, bit in enumerate(reversed(bits)):
            if bit == '0':
                oracle.x(i)
    return oracle


def _build_grover_diffuser(n_qubits: int) -> QuantumCircuit:
    diffuser = QuantumCircuit(n_qubits, name="Diffuser")
    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))
    diffuser.h(n_qubits - 1)
    if n_qubits == 3:
        diffuser.ccx(0, 1, 2)
    elif n_qubits == 2:
        diffuser.cx(0, 1)
    diffuser.h(n_qubits - 1)
    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))
    return diffuser


def _select_target_interventions(disease_label: str,
                                 severity_idx:   int,
                                 overall_risk:   float) -> list:
    targets = [0]
    d = disease_label
    if d in ("Rice Blast", "Sheath Blight", "Brown Spot",
             "Narrow Brown Leaf Spot", "Leaf Scald"):
        targets = [1]
        if severity_idx >= 1:
            targets = [7]
    elif d == "Bacterial Leaf Blight":
        targets = [2]
        if overall_risk > 0.5:
            targets = [2, 6]
    elif d in ("Rice Hispa", "Rice Leaffolder",
               "Rice Tungro", "Rice Stripes"):
        targets = [3]
    elif d == "Healthy":
        targets = [4] if overall_risk > 0.35 else [0]
    else:
        targets = [1] if severity_idx >= 1 else [0]
    return targets


def run_qiskit_recommendation(disease_label: str,
                              severity_idx:   int,
                              risk:           dict,
                              yield_pred:     dict) -> dict:
    """
    Grover search over 8 interventions.
    Returns ONE clear winner with:
      - Confidence % (amplified probability vs random baseline)
      - Farmer-friendly action label
      - What to buy, dose, method, urgency
      - Expected yield recovery
    """
    n_qubits     = 3
    n_states     = 2 ** n_qubits
    simulator    = AerSimulator(method="statevector")
    overall_risk = risk.get("overall_risk", 0.3)

    target_indices = _select_target_interventions(
        disease_label, severity_idx, overall_risk
    )

    k      = len(target_indices)
    n_iter = max(1, round((np.pi / 4) * np.sqrt(n_states / k)))

    # Build and run Grover circuit
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle   = _build_grover_oracle(target_indices, n_qubits)
    diffuser = _build_grover_diffuser(n_qubits)
    for _ in range(n_iter):
        qc.compose(oracle,   inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    compiled = transpile(qc, simulator)
    counts   = simulator.run(compiled, shots=1024).result().get_counts()

    # Statevector for amplitude analysis
    sv_qc = QuantumCircuit(n_qubits)
    sv_qc.h(range(n_qubits))
    for _ in range(n_iter):
        sv_qc.compose(oracle,   inplace=True)
        sv_qc.compose(diffuser, inplace=True)

    amplitudes = np.abs(Statevector(sv_qc).data) ** 2

    # Pick winner
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top_state     = sorted_counts[0][0] if sorted_counts else "000"
    top_idx       = int(top_state, 2)
    top_interv    = QISKIT_INTERVENTION_MAP.get(top_idx, "no_action")
    base_interv   = top_interv.split("+")[0]

    # Confidence: shots share of winning state (clear farmer-readable %)
    total_shots    = sum(counts.values())
    winner_shots   = counts.get(top_state, 0)
    confidence_pct = round((winner_shots / total_shots) * 100, 1)

    # Yield recovery estimate
    recovery_map     = INTERVENTION_RECOVERY.get(
        disease_label, INTERVENTION_RECOVERY["Healthy"]
    )
    base_recovery    = recovery_map.get(base_interv, 0.0)
    yield_loss       = yield_pred.get("yield_loss_t_ha", 0.0)
    recovered_t      = round(yield_loss * base_recovery, 3)
    cost_inr         = INTERVENTION_COSTS.get(base_interv, 0)
    revenue_gain     = round(recovered_t * MSP_PER_TONNE_INR)
    net_benefit      = revenue_gain - cost_inr

    # Urgency
    urgency = URGENCY_LABELS.get(severity_idx, URGENCY_LABELS[0])

    # Instructions
    instructions = INTERVENTION_INSTRUCTIONS.get(
        base_interv, INTERVENTION_INSTRUCTIONS["no_action"]
    )

    # All options ranked by shot count (for reference bar chart)
    all_options = []
    for i in range(n_states):
        interv_name = QISKIT_INTERVENTION_MAP.get(i, "unknown")
        base_name   = interv_name.split("+")[0]
        shot_count  = counts.get(format(i, f'0{n_qubits}b'), 0)
        conf        = round((shot_count / total_shots) * 100, 1)
        all_options.append({
            "intervention":    interv_name,
            "label":           INTERVENTION_LABELS.get(base_name, interv_name),
            "confidence_pct":  conf,
            "shots":           shot_count,
        })
    all_options.sort(key=lambda x: x["confidence_pct"], reverse=True)

    return {
        # ── Winner (what the farmer sees) ──────────────────────
        "winner": {
            "intervention":    base_interv,
            "label":           INTERVENTION_LABELS.get(base_interv, base_interv),
            "confidence_pct":  confidence_pct,
            "urgency_label":   urgency["label"],
            "urgency_days":    urgency["days"],
            "urgency_color":   urgency["color"],
            "recovered_yield_t_ha": recovered_t,
            "cost_inr":        cost_inr,
            "revenue_gain_inr": revenue_gain,
            "net_benefit_inr": net_benefit,
            "instructions":    instructions,
        },
        # ── All options for bar chart ──────────────────────────
        "all_options":    all_options,
        "grover_iterations": n_iter,
        "shots":          total_shots,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION C — Digital Twin Farm State Machine (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _stage_yield_fraction(stage, disease_pen, env_pen,
                          intervened, recovery_frac):
    vuln   = STAGE_VULNERABILITY[stage]
    stress = (disease_pen + env_pen * 0.5) * vuln
    if intervened:
        stress *= (1.0 - recovery_frac)
    stage_weight   = 1.0 / len(GROWTH_STAGES)
    yield_fraction = 1.0 - (stress * stage_weight)
    return max(0.0, min(1.0, yield_fraction))


def run_digital_twin(stage1, risk, yield_pred, weather, soil):
    disease_label  = stage1.get("disease_label", "Healthy")
    severity_label = stage1.get("severity_label", "Low")
    disease_pen    = yield_pred.get("disease_penalty_pct", 0.0) / 100.0
    env_pen        = yield_pred.get("env_penalty_pct",     0.0) / 100.0
    overall_risk   = risk.get("overall_risk", 0.0)

    recovery_map     = INTERVENTION_RECOVERY.get(
        disease_label, INTERVENTION_RECOVERY["Healthy"]
    )
    best_interv      = max(recovery_map, key=lambda k: recovery_map[k])
    best_recovery    = recovery_map[best_interv]
    delayed_recovery = best_recovery * 0.65

    cumulative = {
        "no_intervention":        1.0,
        "immediate_intervention": 1.0,
        "delayed_intervention":   1.0,
    }
    stage_details = []

    for i, stage in enumerate(GROWTH_STAGES):
        row = {"stage": stage, "stage_index": i}
        for track, rec_frac, intervened in [
            ("no_intervention",        0.0,             False),
            ("immediate_intervention", best_recovery,   True),
            ("delayed_intervention",   delayed_recovery, i >= 1),
        ]:
            frac = _stage_yield_fraction(
                stage, disease_pen, env_pen, intervened, rec_frac
            )
            cumulative[track] *= frac
            row[track] = {
                "yield_fraction": round(frac, 4),
                "cumulative":     round(cumulative[track], 4),
                "predicted_t_ha": round(
                    cumulative[track] * BASELINE_YIELD_T_HA, 3
                ),
            }
        row["env_state"] = {
            "temperature":  weather.get("temperature", 25.0),
            "humidity":     weather.get("humidity", 70.0),
            "soil_ph":      soil.get("ph", 6.0),
            "disease_risk": round(
                disease_pen * STAGE_VULNERABILITY[stage], 3
            ),
            "vulnerability": STAGE_VULNERABILITY[stage],
        }
        stage_details.append(row)

    def _final(track):
        fc  = cumulative[track]
        yt  = round(fc * BASELINE_YIELD_T_HA, 3)
        lt  = round(BASELINE_YIELD_T_HA - yt, 3)
        rev = round(yt * MSP_PER_TONNE_INR)
        lrev= round(lt * MSP_PER_TONNE_INR)
        return {
            "final_yield_t_ha": yt,
            "yield_loss_t_ha":  lt,
            "yield_loss_pct":   round((1.0 - fc) * 100, 1),
            "expected_revenue": rev,
            "revenue_loss":     lrev,
        }

    no_tr  = _final("no_intervention")
    imm_tr = _final("immediate_intervention")
    del_tr = _final("delayed_intervention")

    cost_inr      = INTERVENTION_COSTS.get(best_interv, 0)
    yield_gain    = round(
        imm_tr["final_yield_t_ha"] - no_tr["final_yield_t_ha"], 3
    )
    rev_gain      = round(yield_gain * MSP_PER_TONNE_INR)
    net_benefit   = rev_gain - cost_inr
    roi_pct       = (round((net_benefit / cost_inr) * 100, 1)
                     if cost_inr > 0 else 0.0)

    health_score  = round(
        (1.0 - disease_pen * 0.5 - env_pen * 0.3 - overall_risk * 0.2) * 100, 1
    )
    health_score  = max(0.0, min(100.0, health_score))

    def _hl(s):
        if s >= 80: return "Healthy"
        if s >= 60: return "Moderate Stress"
        if s >= 40: return "High Stress"
        return "Critical"

    risk_trajectory = []
    cr = overall_risk
    for stage in GROWTH_STAGES:
        vuln = STAGE_VULNERABILITY[stage]
        sr   = min(1.0, cr * (1.0 + disease_pen * vuln))
        risk_trajectory.append({"stage": stage, "risk": round(sr, 3)})

    return {
        "engine": "Digital Twin Farm State Machine",
        "farm_state": {
            "disease":      disease_label,
            "severity":     severity_label,
            "overall_risk": overall_risk,
            "health_score": health_score,
            "health_label": _hl(health_score),
            "disease_pen":  round(disease_pen * 100, 1),
            "env_pen":      round(env_pen * 100, 1),
        },
        "growth_stages":   stage_details,
        "risk_trajectory": risk_trajectory,
        "tracks": {
            "no_intervention": no_tr,
            "immediate_intervention": {
                **imm_tr,
                "intervention": best_interv,
                "cost_inr":     cost_inr,
            },
            "delayed_intervention": {
                **del_tr,
                "intervention":  best_interv,
                "delay_penalty": "35% efficacy loss vs immediate",
            },
        },
        "cost_benefit": {
            "best_intervention":  best_interv,
            "cost_inr":           cost_inr,
            "yield_gain_t_ha":    yield_gain,
            "revenue_gain_inr":   rev_gain,
            "net_benefit_inr":    net_benefit,
            "roi_pct":            roi_pct,
            "break_even_yield_t": round(cost_inr / MSP_PER_TONNE_INR, 3),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# MASTER ENTRY POINT — called by routes.py
# ══════════════════════════════════════════════════════════════════════════════

def run_stage4(stage1, risk, yield_pred, weather, soil):
    """
    Orchestrates all three Stage 4 engines.
    Returns unified dict shaped for index.html renderResults().
    """
    disease_label = stage1.get("disease_label", "Healthy")
    severity_idx  = stage1.get("severity_idx",  0)
    disease_pen   = yield_pred.get("disease_penalty_pct", 0.0) / 100.0
    env_pen       = yield_pred.get("env_penalty_pct",     0.0) / 100.0

    errors = {}

    # ── A: PennyLane Farm Health Score ───────────────────────
    try:
        health_result = run_pennylane_health(
            disease_label, disease_pen, env_pen, severity_idx, soil
        )
    except Exception as e:
        health_result = {
            "overall_score": 50.0, "color": "orange",
            "label": "Unknown", "advice": "",
            "dimensions": [], "error": str(e),
        }
        errors["pennylane"] = str(e)

    # ── B: Qiskit Recommendation ──────────────────────────────
    try:
        qiskit_result = run_qiskit_recommendation(
            disease_label, severity_idx, risk, yield_pred
        )
    except Exception as e:
        qiskit_result = {
            "winner": {
                "intervention": "—", "label": "Unknown",
                "confidence_pct": 0, "urgency_label": "—",
                "urgency_days": 0,   "urgency_color": "grey",
                "recovered_yield_t_ha": 0, "cost_inr": 0,
                "revenue_gain_inr": 0,     "net_benefit_inr": 0,
                "instructions": {},
            },
            "all_options": [], "error": str(e),
        }
        errors["qiskit"] = str(e)

    # ── C: Digital Twin ───────────────────────────────────────
    try:
        twin_result = run_digital_twin(
            stage1, risk, yield_pred, weather, soil
        )
    except Exception as e:
        twin_result = {
            "error": str(e), "tracks": {}, "cost_benefit": {},
            "growth_stages": [],
        }
        errors["digital_twin"] = str(e)

    # ── Build frontend-ready scenario cards ───────────────────
    tracks      = twin_result.get("tracks", {})
    no_tr       = tracks.get("no_intervention",        {})
    imm_tr      = tracks.get("immediate_intervention", {})
    del_tr      = tracks.get("delayed_intervention",   {})
    cb          = twin_result.get("cost_benefit", {})
    best_interv = cb.get("best_intervention", "fungicide")

    def _vit(t_ha):
        return round((t_ha / BASELINE_YIELD_T_HA) * 100, 1)

    no_yield  = no_tr.get("final_yield_t_ha",  BASELINE_YIELD_T_HA * 0.7)
    imm_yield = imm_tr.get("final_yield_t_ha", BASELINE_YIELD_T_HA * 0.9)
    del_yield = del_tr.get("final_yield_t_ha", BASELINE_YIELD_T_HA * 0.8)

    scenarios = [
        {
            "label":   "A",
            "name":    "No Treatment",
            "description": "Disease progresses unchecked across all growth stages.",
            "predicted_yield_t_ha":        no_yield,
            "vitality_pct":                _vit(no_yield),
            "expected_revenue_inr":        no_tr.get("expected_revenue", 0),
            "treatment_cost_inr":          0,
            "economic_loss_inr":           no_tr.get("revenue_loss", 0),
            "yield_gain_vs_no_treat_t_ha": 0.0,
            "pennylane_energy":            0.0,
        },
        {
            "label":   "B",
            "name":    f"Delayed ({INTERVENTION_LABELS.get(best_interv, best_interv)})",
            "description": "Treatment applied 7 days late — partial efficacy.",
            "predicted_yield_t_ha":        del_yield,
            "vitality_pct":                _vit(del_yield),
            "expected_revenue_inr":        del_tr.get("expected_revenue", 0),
            "treatment_cost_inr":          cb.get("cost_inr", 0),
            "economic_loss_inr":           del_tr.get("revenue_loss", 0),
            "yield_gain_vs_no_treat_t_ha": round(del_yield - no_yield, 3),
            "pennylane_energy":            0.0,
        },
        {
            "label":   "C",
            "name":    f"Optimal ({INTERVENTION_LABELS.get(best_interv, best_interv)})",
            "description": "Immediate treatment — maximum yield recovery.",
            "predicted_yield_t_ha":        imm_yield,
            "vitality_pct":                _vit(imm_yield),
            "expected_revenue_inr":        imm_tr.get("expected_revenue", 0),
            "treatment_cost_inr":          cb.get("cost_inr", 0),
            "economic_loss_inr":           imm_tr.get("revenue_loss", 0),
            "yield_gain_vs_no_treat_t_ha": cb.get("yield_gain_t_ha", 0.0),
            "pennylane_energy":            0.0,
        },
    ]

    # 30-day trajectory
    growth      = twin_result.get("growth_stages", [])
    checkpoints = [0, 1, 3, 5]

    def _traj(track_key, idx):
        if idx >= len(growth):
            return {"vitality_pct": 0.0, "yield_t_ha": 0.0}
        e = growth[idx].get(track_key, {})
        return {
            "vitality_pct": round(e.get("cumulative", 0) * 100, 1),
            "yield_t_ha":   e.get("predicted_t_ha", 0.0),
        }

    trajectory = {
        "A": [_traj("no_intervention",        i) for i in checkpoints],
        "B": [_traj("delayed_intervention",   i) for i in checkpoints],
        "C": [_traj("immediate_intervention", i) for i in checkpoints],
    }

    winner = qiskit_result.get("winner", {})
    roi    = cb.get("roi_pct", 0.0)

    twin_summary = (
        f"Digital Twin projects {imm_yield} t/ha with immediate "
        f"{INTERVENTION_LABELS.get(best_interv, best_interv)} "
        f"vs {no_yield} t/ha untreated. "
        f"Net benefit ₹{cb.get('net_benefit_inr', 0):,} · ROI {roi}%."
    )

    return {
        "stage":  4,
        "errors": errors,

        # ── PennyLane: Farm Health Score ──────────────────────
        "farm_health": health_result,

        # ── Qiskit: Single best recommendation ────────────────
        "recommendation": qiskit_result,

        # ── Digital Twin scenario cards (for existing UI) ─────
        "scenarios":          scenarios,
        "best_scenario_label": "C",
        "trajectory":         trajectory,
        "days_simulated":     30,
        "twin_summary":       twin_summary,
        "intervention_roi":   roi,

        # Legacy fields (kept for backward compat)
        "qiskit_fidelity":    round(
            min(1.0, winner.get("confidence_pct", 0) / 100.0 / 0.125), 4
        ),
        "pennylane_energies": {
            "A": round(health_result.get("overall_score", 50) / 100, 4),
            "B": round(health_result.get("overall_score", 50) / 100 * 1.1, 4),
            "C": round(health_result.get("overall_score", 50) / 100 * 1.2, 4),
        },

        # Raw engine outputs
        "pennylane":    health_result,
        "qiskit":       qiskit_result,
        "digital_twin": twin_result,
    }