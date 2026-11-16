"""Optional Python scoring API used by the PHP dashboard (Auto engine mode). Run:  python api.py"""
from flask import Flask, jsonify, request
import model

app = Flask(__name__)
_cache = {"mtime": None, "bundle": None}


def bundle():
    model.load()  # trains a default model if none exists
    m = model.MODEL_PATH.stat().st_mtime
    if _cache["mtime"] != m:  # reload automatically after retraining in Streamlit
        _cache.update(mtime=m, bundle=model.load())
    return _cache["bundle"]


@app.get("/health")
def health():
    return jsonify(status="ok", features=model.FEATURES)


@app.get("/model-info")
def info():
    return jsonify(bundle()["metrics"])


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    f = data.get("features", {})
    missing = [k for k in model.FEATURES if k not in f]
    if missing:
        return jsonify(error=f"missing features: {missing}"), 400
    try:
        p = model.predict_proba(bundle(), f)
    except (ValueError, TypeError) as e:
        return jsonify(error=str(e)), 400
    return jsonify(probability=round(p, 4), label="attack" if p >= 0.5 else "normal", engine="scikit-learn RandomForest")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
