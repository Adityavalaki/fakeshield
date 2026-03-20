"""
FakeShield — Flask Web App
Run: python app.py
Open: http://localhost:5000
"""
import os, sys, json, time, logging
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        from src.predict import FakeNewsPredictor
        try:
            _predictor = FakeNewsPredictor()
            log.info("Model loaded successfully")
        except FileNotFoundError:
            log.warning("No trained model — using DEMO mode")
            _predictor = "DEMO"
    return _predictor


def demo_predict(text):
    import random, hashlib
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % 1000
    rng  = random.Random(seed)
    fake_kw = ["shocking","breaking","exposed","secret","won't believe","deep state","hoax"]
    real_kw = ["according to","study","research","official","confirmed","peer-reviewed"]
    tl = text.lower()
    base = max(0.1, min(0.9, 0.45 + sum(kw in tl for kw in fake_kw)*0.15
                                    - sum(kw in tl for kw in real_kw)*0.12
                                    + rng.uniform(-0.1,0.1)))
    label = "FAKE" if base > 0.5 else "REAL"
    conf  = base if label=="FAKE" else 1-base
    return {"label": label, "confidence": round(conf,4),
            "verdict": ("Fake News" if label=="FAKE" else "Real News") if conf>=config.CONFIDENCE_THRESHOLD else "Uncertain",
            "real_prob": round(1-base,4), "fake_prob": round(base,4), "demo_mode": True}


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    p = get_predictor()
    return jsonify({"status":"ok","model_ready":p!="DEMO","demo_mode":p=="DEMO"})

@app.route("/model-info")
def model_info():
    path = os.path.join(config.MODEL_DIR, "metrics.json")
    metrics = json.load(open(path)) if os.path.exists(path) else {"note":"Not trained yet"}
    return jsonify({"model_name":config.MODEL_NAME,"metrics":metrics})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error":"Missing 'text' field"}), 400
    text = data["text"].strip()
    if len(text) < 10:
        return jsonify({"error":"Text too short (min 10 chars)"}), 400

    p = get_predictor()
    start = time.time()
    try:
        result = demo_predict(text) if p=="DEMO" else p.predict(text)[0]
    except Exception as e:
        return jsonify({"error":str(e)}), 500

    result["latency_ms"]  = round((time.time()-start)*1000, 1)
    result["text_length"] = len(text)
    return jsonify(result)

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error":"Missing 'texts' list"}), 400
    texts = data["texts"]
    if len(texts) > 50:
        return jsonify({"error":"Max 50 texts per batch"}), 400
    p = get_predictor()
    start = time.time()
    results = [demo_predict(t) for t in texts] if p=="DEMO" else p.predict(texts)
    return jsonify({"results":results,"count":len(results),
                    "latency_ms":round((time.time()-start)*1000,1)})


if __name__ == "__main__":
    log.info(f"Starting FakeShield on http://localhost:{config.FLASK_PORT}")
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
