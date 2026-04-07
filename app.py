from __future__ import annotations

import argparse
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from brain_tumor_web.inference import BrainTumorWebService


BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "brain_tumor_web" / "templates"),
    static_folder=str(BASE_DIR / "brain_tumor_web" / "static"),
)
service = BrainTumorWebService(BASE_DIR)


@app.get("/")
def index():
    service.ensure_started()
    return render_template("index.html")


@app.get("/api/health")
def health():
    service.ensure_started()
    return jsonify(service.health_payload())


@app.get("/api/models")
def models():
    service.ensure_started()
    return jsonify(service.models_payload())


@app.post("/api/predict")
def predict():
    service.ensure_started()
    if not service.is_ready:
        return jsonify(service.health_payload()), 503

    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Please upload an image file."}), 400

    model_name = request.form.get("model")
    try:
        result = service.predict(upload.read(), model_name=model_name)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Brain tumor classifier web app")
    parser.add_argument("--prepare", action="store_true", help="Build cached inference assets and exit.")
    parser.add_argument("--host", default=os.environ.get("HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()

    if args.prepare:
        service.prepare_blocking()
        return

    service.ensure_started()
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
