import requests
from config import OPENWEATHER_API_KEY

def get_weather(lat: float, lon: float) -> dict:
    """Fetch real-time weather for exact farmer location."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat":   lat,
        "lon":   lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        d = r.json()
        return {
            "temperature":   round(d["main"]["temp"], 1),
            "humidity":      d["main"]["humidity"],
            "feels_like":    round(d["main"]["feels_like"], 1),
            "rainfall_mm":   round(d.get("rain", {}).get("1h", 0), 2),
            "wind_speed":    round(d["wind"]["speed"], 1),
            "description":   d["weather"][0]["description"].title(),
            "city":          d.get("name", ""),
            "country":       d.get("sys", {}).get("country", ""),
            "visibility_km": round(d.get("visibility", 10000) / 1000, 1),
        }
    except Exception as e:
        return {"error": str(e)}