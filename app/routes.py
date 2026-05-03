import os
import torch
import torch.nn as nn
import pennylane as qml
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Blueprint, request, jsonify, render_template
from PIL import Image

from stage2_qaoa.weather_service  import get_weather
from stage2_qaoa.soil_service     import get_soil
from stage2_qaoa.risk_optimizer   import compute_risk
from stage2_qaoa.recommendation   import get_recommendations
from stage3_vqe.yield_predictor   import predict_yield
from stage4_simulation.digital_twin import run_stage4       # ← Stage 4
from config import MODEL_PATH

bp = Blueprint('main', __name__)

# ── Stage 1 model config (must match Colab exactly) ──────────
N_QUBITS       = 4
N_LAYERS       = 2
NUM_CLASSES    = 11
NUM_CATEGORIES = 5
NUM_SEVERITY   = 3

CLASSES = [
    "Bacterial Leaf Blight", "Brown Spot", "Healthy",
    "Leaf Scald", "Narrow Brown Leaf Spot", "Rice Blast",
    "Rice Hispa", "Rice Leaffolder", "Rice Stripes",
    "Rice Tungro", "Sheath Blight"
]
CATEGORY_NAMES = {0:"Bacterial", 1:"Fungal", 2:"Healthy",
                  3:"Pest",      4:"Viral"}
SEVERITY_NAMES = {0:"Low", 1:"Medium", 2:"High"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Quantum circuit (identical to Colab) ──────────────────────
dev_q = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev_q, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

weight_shapes = {"weights": (N_LAYERS, N_QUBITS)}

# ── HybridQCNN (identical to Colab) ──────────────────────────
class HybridQCNN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet34(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.pre_quantum = nn.Sequential(
            nn.Linear(512, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 32),  nn.BatchNorm1d(32),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, N_QUBITS), nn.Tanh()
        )
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.post_quantum = nn.Sequential(
            nn.Linear(N_QUBITS, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 64),  nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),  nn.ReLU()
        )
        self.disease_head  = nn.Linear(32, NUM_CLASSES)
        self.category_head = nn.Linear(32, NUM_CATEGORIES)
        self.severity_head = nn.Linear(32, NUM_SEVERITY)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.pre_quantum(x)
        x = self.quantum_layer(x)
        x = self.post_quantum(x)
        return (self.disease_head(x),
                self.category_head(x),
                self.severity_head(x))


# ── Load model once at startup ────────────────────────────────
model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return
    print("Loading Stage 1 model...")
    ckpt  = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = HybridQCNN().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Stage 1 model loaded | Val acc: {ckpt.get('val_acc', 0)*100:.1f}%")

load_model()

# ── Val transform ─────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def run_stage1(image_path: str) -> dict:
    if model is None:
        return {
            "disease_label":  "Unknown", "confidence": 0.0,
            "severity_label": "Unknown", "severity_idx": 0,
            "category_label": "Unknown", "category_idx": 0,
            "top3": []
        }
    img    = Image.open(image_path).convert("RGB")
    tensor = val_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        d_out, c_out, s_out = model(tensor)
    d_probs = torch.softmax(d_out, dim=1)[0]
    s_probs = torch.softmax(s_out, dim=1)[0]
    c_probs = torch.softmax(c_out, dim=1)[0]
    d_idx   = d_probs.argmax().item()
    s_idx   = s_probs.argmax().item()
    c_idx   = c_probs.argmax().item()
    return {
        "disease_label":  CLASSES[d_idx],
        "confidence":     round(d_probs[d_idx].item() * 100, 2),
        "severity_label": SEVERITY_NAMES[s_idx],
        "severity_idx":   s_idx,
        "category_label": CATEGORY_NAMES[c_idx],
        "category_idx":   c_idx,
        "top3": [
            {"disease": CLASSES[i],
             "confidence": round(d_probs[i].item() * 100, 2)}
            for i in d_probs.topk(3).indices.tolist()
        ]
    }


# ── Routes ────────────────────────────────────────────────────
@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/analyze", methods=["POST"])
def analyze():
    try:
        lat   = float(request.form.get("lat"))
        lon   = float(request.form.get("lon"))
        image = request.files.get("image")

        if not image:
            return jsonify({"error": "No image uploaded"}), 400

        upload_dir = os.path.join(
            os.path.dirname(__file__), "..", "static", "uploads"
        )
        os.makedirs(upload_dir, exist_ok=True)
        image_path = os.path.join(upload_dir, image.filename)
        image.save(image_path)

        # Stage 1 — disease detection
        stage1  = run_stage1(image_path)

        # Stage 2 — location data + QAOA risk
        weather = get_weather(lat, lon)
        soil    = get_soil(lat, lon)
        risk    = compute_risk(weather, soil,
                               disease_severity=stage1["severity_idx"])
        recs    = get_recommendations(
            risk=risk, weather=weather, soil=soil,
            disease_label=stage1["disease_label"],
            severity_label=stage1["severity_label"],
            category_label=stage1["category_label"]
        )

        # Stage 3 — VQE yield prediction
        yield_pred = predict_yield(
            disease_label  = stage1["disease_label"],
            severity_idx   = stage1["severity_idx"],
            severity_label = stage1["severity_label"],
            category_label = stage1["category_label"],
            risk=risk, weather=weather, soil=soil
        )

        # Stage 4 — Digital Twin simulation
        simulation = run_stage4(
            stage1=stage1, risk=risk,
            yield_pred=yield_pred,
            weather=weather, soil=soil
        )

        return jsonify({
            "location": {
                "lat": lat, "lon": lon,
                "city":    weather.get("city", ""),
                "country": weather.get("country", "")
            },
            "weather":         weather,
            "soil":            soil,
            "stage1":          stage1,
            "risk":            risk,
            "recommendations": recs,
            "yield":           yield_pred,
            "simulation":      simulation,     # ← Stage 4 output
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e),
                        "trace": traceback.format_exc()}), 500