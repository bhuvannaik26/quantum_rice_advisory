import requests

def get_soil(lat: float, lon: float) -> dict:
    """
    Primary  : OpenLandMap API — free, no key, reliable
    Secondary: NASA POWER     — free, no key, always up
    Fallback : Telangana regional defaults for rice
    """

    # ── Primary: OpenLandMap ──────────────────────────────────
    try:
        # pH at 0-5cm
        ph_r = requests.get(
            "https://api.openlandmap.org/query/point",
            params={
                "lon":      lon,
                "lat":      lat,
                "coll":     "sol",
                "variable": "phh2o",
                "depth":    "0..5cm",
                "quantile": "p50"
            },
            timeout=15
        )
        # Nitrogen at 0-5cm
        n_r = requests.get(
            "https://api.openlandmap.org/query/point",
            params={
                "lon":      lon,
                "lat":      lat,
                "coll":     "sol",
                "variable": "nitrogen",
                "depth":    "0..5cm",
                "quantile": "p50"
            },
            timeout=15
        )
        # Clay at 0-5cm
        clay_r = requests.get(
            "https://api.openlandmap.org/query/point",
            params={
                "lon":      lon,
                "lat":      lat,
                "coll":     "sol",
                "variable": "clay",
                "depth":    "0..5cm",
                "quantile": "p50"
            },
            timeout=15
        )
        # Organic carbon at 0-5cm
        oc_r = requests.get(
            "https://api.openlandmap.org/query/point",
            params={
                "lon":      lon,
                "lat":      lat,
                "coll":     "sol",
                "variable": "oc",
                "depth":    "0..5cm",
                "quantile": "p50"
            },
            timeout=15
        )

        result = {}

        if ph_r.status_code == 200:
            val = ph_r.json().get("result", {}).get("value")
            if val:
                result["ph"] = round(float(val) / 10, 1)

        if n_r.status_code == 200:
            val = n_r.json().get("result", {}).get("value")
            if val:
                result["nitrogen_g_kg"] = round(float(val) / 100, 2)

        if clay_r.status_code == 200:
            val = clay_r.json().get("result", {}).get("value")
            if val:
                result["clay_percent"] = round(float(val) / 10, 1)

        if oc_r.status_code == 200:
            val = oc_r.json().get("result", {}).get("value")
            if val:
                result["organic_carbon"] = round(float(val) / 10, 1)

        if len(result) >= 2:
            result.setdefault("ph",             6.0)
            result.setdefault("nitrogen_g_kg",  1.0)
            result.setdefault("clay_percent",   30.0)
            result.setdefault("bulk_density",   1.2)
            result.setdefault("organic_carbon", 10.0)
            result["source"] = "openlandmap"
            print(f"✅ Soil: OpenLandMap | pH={result['ph']} "
                  f"N={result['nitrogen_g_kg']} g/kg")
            return result

    except Exception as e:
        print(f"⚠️  OpenLandMap failed: {e}")

    # ── Secondary: NASA POWER ─────────────────────────────────
    try:
        r = requests.get(
            "https://power.larc.nasa.gov/api/temporal/climatology/point",
            params={
                "parameters": "GWETROOT,GWETPROF,GWETTOP",
                "community":  "AG",
                "longitude":  lon,
                "latitude":   lat,
                "format":     "JSON"
            },
            timeout=20
        )
        if r.status_code == 200:
            props    = r.json().get("properties", {})
            params   = props.get("parameter", {})

            # GWETTOP = surface soil moisture (0-1)
            gwettop  = params.get("GWETTOP", {})
            avg_moist= round(sum(v for v in gwettop.values()
                                 if isinstance(v, (int, float))) /
                             max(len(gwettop), 1), 3)

            # Estimate pH from moisture profile
            # Wetter = slightly more acidic for Indian soils
            estimated_ph = round(6.8 - avg_moist * 1.2, 1)
            estimated_ph = max(4.5, min(8.5, estimated_ph))

            result = {
                "ph":             estimated_ph,
                "nitrogen_g_kg":  1.0,
                "clay_percent":   30.0,
                "bulk_density":   1.2,
                "organic_carbon": 10.0,
                "soil_moisture":  avg_moist,
                "source":         "nasa-power"
            }
            print(f"✅ Soil: NASA POWER | "
                  f"moisture={avg_moist} pH~{estimated_ph}")
            return result

    except Exception as e:
        print(f"⚠️  NASA POWER failed: {e}")

    # ── Final fallback: Telangana rice region defaults ─────────
    # Based on ICAR soil survey data for Telangana
    print("⚠️  Using Telangana regional soil defaults")
    return {
        "ph":             6.2,    # Slightly acidic — typical red soil
        "nitrogen_g_kg":  0.85,   # Low-medium N — common in Telangana
        "clay_percent":   38.0,   # Heavy clay — black cotton soil
        "bulk_density":   1.35,
        "organic_carbon": 7.5,
        "source":         "telangana-regional-default"
    }