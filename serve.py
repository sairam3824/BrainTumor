from waitress import serve

from app import app, resolve_runtime_port


if __name__ == "__main__":
    serve(app, host="localhost", port=resolve_runtime_port("localhost", 8000))
