from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.exceptions import RequestEntityTooLarge


BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "brain_tumor_web" / "templates"),
    static_folder=str(BASE_DIR / "brain_tumor_web" / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False

_service = None


def get_service():
    global _service
    if _service is None:
        from brain_tumor_web.inference import BrainTumorWebService

        _service = BrainTumorWebService(BASE_DIR)
    return _service


def is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def resolve_runtime_port(host: str, preferred_port: int) -> int:
    if preferred_port != 8000:
        return preferred_port
    if is_port_available(host, 8000):
        return 8000
    if is_port_available(host, 8001):
        print("Port 8000 is busy. Falling back to http://localhost:8001")
        return 8001
    return 8000


@app.get("/")
def home():
    latest_report = get_service().get_report()
    return render_template("home.html", page="home", latest_report=latest_report)


@app.get("/analyze")
def analyze():
    get_service().ensure_started()
    return render_template("analyze.html", page="analyze")


@app.get("/report")
@app.get("/report/<analysis_id>")
def report_redirect(analysis_id: str | None = None):
    del analysis_id
    return redirect(url_for("history_page"))


@app.get("/history")
def history_page():
    history = get_service().list_history(limit=12)
    return render_template("history.html", page="history", history=history)


@app.get("/artifacts/<path:asset_path>")
def serve_artifact(asset_path: str):
    return send_from_directory(get_service().media_root, asset_path)


@app.get("/api/health")
def health():
    service = get_service()
    service.ensure_started()
    return jsonify(service.health_payload())


@app.get("/api/models")
def models():
    service = get_service()
    service.ensure_started()
    return jsonify(service.models_payload())


@app.get("/api/history")
def history_api():
    return jsonify({"items": get_service().list_history(limit=HISTORY_LIMIT_API)})


@app.get("/api/reports/latest")
def latest_report_api():
    report = get_service().get_report()
    if report is None:
        return jsonify({"error": "No reports are available yet."}), 404
    return jsonify(report)


@app.get("/api/reports/<analysis_id>")
def report_api(analysis_id: str):
    report = get_service().get_report(analysis_id)
    if report is None:
        return jsonify({"error": f"Report '{analysis_id}' was not found."}), 404
    return jsonify(report)


@app.post("/api/predict")
def predict():
    service = get_service()
    service.ensure_started()
    if not service.is_ready:
        return jsonify(service.health_payload()), 503

    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Please upload an MRI image file before running prediction."}), 400

    model_name = request.form.get("model")
    try:
        result = service.predict(
            upload.read(),
            filename=upload.filename,
            mimetype=upload.mimetype,
            model_name=model_name,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result)


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(_exc):
    if request.path.startswith("/api/"):
        return jsonify({"error": "The uploaded file is too large. Please keep uploads under 10 MB."}), 413
    return (
        render_template(
            "simple_message.html",
            page="error",
            title="Upload Too Large",
            message="Please keep uploaded MRI images under 10 MB.",
        ),
        413,
    )


HISTORY_LIMIT_API = 100


def main() -> None:
    parser = argparse.ArgumentParser(description="Brain tumor classifier web app")
    parser.add_argument("--prepare", action="store_true", help="Build cached inference assets and exit.")
    parser.add_argument("--host", default=os.environ.get("HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()

    if args.prepare:
        get_service().prepare_blocking()
        return

    port = resolve_runtime_port(args.host, args.port)
    app.run(host=args.host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
