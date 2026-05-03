from config import RICE_THRESHOLDS

def get_recommendations(risk: dict, weather: dict,
                        soil: dict,
                        disease_label: str,
                        severity_label: str,
                        category_label: str) -> dict:
    """
    Rule-based recommendation engine driven by QAOA risk scores,
    weather, soil, and Stage 1 disease detection output.
    """
    T       = RICE_THRESHOLDS
    advice  = []
    urgency = "Low"
    temp     = weather.get("temperature", 25)
    humidity = weather.get("humidity",    70)
    rainfall = weather.get("rainfall_mm",  0)
    ph       = soil.get("ph",            6.0)
    nitrogen = soil.get("nitrogen_g_kg", 1.0)

    # ── Disease-specific advice from Stage 1 ─────────────────
    if disease_label and disease_label != "Healthy":
        disease_actions = {
            "Rice Blast": {
                "action": "Apply tricyclazole (0.6g/L) or propiconazole immediately. Remove infected tillers.",
                "reason": "Confirmed Blast — fungal infection spreads rapidly above 24°C with high humidity."
            },
            "Bacterial Leaf Blight": {
                "action": "Apply copper oxychloride spray (3g/L). Drain standing water from field.",
                "reason": "Confirmed BLB — bacteria thrive in waterlogged, humid conditions."
            },
            "Brown Spot": {
                "action": "Apply mancozeb (2.5g/L). Check potassium levels — deficiency worsens brown spot.",
                "reason": "Brown spot linked to nutrient-stressed plants under humidity stress."
            },
            "Sheath Blight": {
                "action": "Apply hexaconazole (2mL/L). Reduce canopy density by thinning planting.",
                "reason": "Sheath blight spreads through dense canopy — reduce plant-to-plant contact."
            },
            "Rice Hispa": {
                "action": "Apply chlorpyrifos (2mL/L) or clip and destroy infested leaf tips.",
                "reason": "Hispa beetle larvae mine inside leaves — early intervention prevents spread."
            },
            "Rice Leaffolder": {
                "action": "Apply cartap hydrochloride (1g/L). Scout at tillering stage.",
                "reason": "Leaffolder caterpillars fold and feed on leaf lamina reducing photosynthesis."
            },
            "Rice Tungro": {
                "action": "Remove infected plants immediately. Control green leafhopper vector with imidacloprid.",
                "reason": "Tungro is virus-based — no cure, only vector control and rouging."
            },
            "Rice Stripes": {
                "action": "Control small brown planthopper with thiamethoxam. Remove symptomatic plants.",
                "reason": "Rice Stripe Virus transmitted by SBPH — vector management is critical."
            },
            "Leaf Scald": {
                "action": "Apply propiconazole (1mL/L). Avoid overhead irrigation.",
                "reason": "Leaf Scald fungus spreads via water splash — avoid wetting foliage."
            },
            "Narrow Brown Leaf Spot": {
                "action": "Apply carbendazim (1g/L). Check silicon and potassium levels.",
                "reason": "NBLS severity increases under nutrient deficiency and high humidity."
            },
        }
        if disease_label in disease_actions:
            d = disease_actions[disease_label]
            advice.append({
                "category": "disease",
                "priority": "URGENT" if severity_label == "High" else "HIGH",
                "action":   d["action"],
                "reason":   d["reason"]
            })
            urgency = "Critical" if severity_label == "High" else "High"

    # ── QAOA blast risk advice ────────────────────────────────
    if risk["blast_risk"] > 0.65:
        advice.append({
            "category": "fungicide",
            "priority": "URGENT",
            "action":   "Preventive blast spray — tricyclazole 75WP at 0.6g/L within 24hrs.",
            "reason":   f"Blast risk {risk['blast_risk']:.0%} — temp {temp}°C + humidity {humidity}% are ideal for Magnaporthe oryzae spore germination."
        })
        urgency = "Critical"
    elif risk["blast_risk"] > 0.40:
        advice.append({
            "category": "monitoring",
            "priority": "HIGH",
            "action":   "Scout fields daily for blast lesions. Prepare fungicide.",
            "reason":   f"Moderate blast risk {risk['blast_risk']:.0%} — monitor leaf tips and nodes."
        })
        if urgency == "Low": urgency = "Medium"

    # ── QAOA blight risk advice ───────────────────────────────
    if risk["blight_risk"] > 0.60:
        advice.append({
            "category": "bactericide",
            "priority": "HIGH",
            "action":   "Apply copper-based bactericide. Improve field drainage immediately.",
            "reason":   f"Blight risk {risk['blight_risk']:.0%} — humidity {humidity}% exceeds {T['humidity_blight_trigger']}% danger threshold."
        })
        if urgency == "Low": urgency = "High"

    # ── Soil pH advice ────────────────────────────────────────
    if ph < T["ph_min"]:
        advice.append({
            "category": "soil",
            "priority": "MEDIUM",
            "action":   f"Apply agricultural lime at 2–4 tonnes/ha to raise pH from {ph} toward 5.5–6.5.",
            "reason":   "Acidic soil locks out phosphorus and micronutrients — rice roots cannot absorb them below pH 5.5."
        })
    elif ph > T["ph_max"]:
        advice.append({
            "category": "soil",
            "priority": "MEDIUM",
            "action":   f"Apply elemental sulfur at 100–200 kg/ha to lower pH from {ph} toward 6.5.",
            "reason":   "Alkaline soil causes iron and manganese deficiency in rice — yellowing and stunted growth."
        })

    # ── Nitrogen advice ───────────────────────────────────────
    if nitrogen < T["nitrogen_low"]:
        advice.append({
            "category": "fertilizer",
            "priority": "HIGH",
            "action":   "Apply urea at 40–60 kg N/ha in split doses (50% basal + 50% at tillering).",
            "reason":   f"Soil nitrogen {nitrogen} g/kg is below {T['nitrogen_low']} g/kg threshold — rice requires 80–120 kg N/ha per season."
        })
        if urgency == "Low": urgency = "Medium"

    # ── Temperature stress advice ─────────────────────────────
    if temp > T["temp_heat_stress"]:
        advice.append({
            "category": "irrigation",
            "priority": "HIGH",
            "action":   "Flood irrigate to 5cm depth immediately to reduce canopy temperature.",
            "reason":   f"Temperature {temp}°C exceeds {T['temp_heat_stress']}°C — spikelet sterility risk at flowering stage."
        })
    elif temp < T["temp_cold_stress"]:
        advice.append({
            "category": "irrigation",
            "priority": "HIGH",
            "action":   "Maintain 15–20cm deep water layer to insulate roots from cold.",
            "reason":   f"Temperature {temp}°C is below {T['temp_cold_stress']}°C cold stress threshold."
        })

    # ── Rainfall / waterlogging advice ───────────────────────
    if rainfall > T["rainfall_waterlog"]:
        advice.append({
            "category": "drainage",
            "priority": "URGENT",
            "action":   "Open drainage channels immediately. Do not apply fertilizer during waterlogging.",
            "reason":   f"Rainfall {rainfall}mm exceeds waterlogging threshold — anaerobic soil conditions promote root rot and blight."
        })
        if urgency in ("Low", "Medium"): urgency = "High"

    # ── Healthy crop ─────────────────────────────────────────
    if disease_label == "Healthy" and risk["overall_risk"] < 0.3:
        advice.append({
            "category": "maintenance",
            "priority": "LOW",
            "action":   "Continue current management. Scout weekly for early symptoms.",
            "reason":   "Crop is healthy and environmental risk is low — maintain monitoring schedule."
        })

    # Sort by priority
    priority_order = {"URGENT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    advice.sort(key=lambda x: priority_order.get(x["priority"], 4))

    return {
        "urgency":         urgency,
        "total_actions":   len(advice),
        "recommendations": advice,
        "summary": (
            f"{len(advice)} action(s) recommended · "
            f"Overall risk: {risk['overall_risk']:.0%} "
            f"({risk['risk_level']}) · Urgency: {urgency}"
        )
    }