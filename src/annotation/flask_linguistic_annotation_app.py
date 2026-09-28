"""Flask interface for linguistic annotation of corrected gold-train tokens."""

from functools import lru_cache

from flask import Flask, jsonify, render_template, request

from src.annotation.linguistic_annotation_core import LinguisticAnnotationService


app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


@lru_cache(maxsize=1)
def get_service() -> LinguisticAnnotationService:
    return LinguisticAnnotationService()


@app.get("/")
def index():
    return render_template("linguistic_annotation.html")


@app.get("/api/progress")
def progress():
    return jsonify(get_service().progress())


@app.get("/api/records/<int:position>")
def record(position: int):
    try:
        return jsonify(get_service().record_at(position))
    except IndexError as exc:
        return jsonify({"error": str(exc)}), 404


@app.get("/api/next")
def next_record():
    try:
        after = int(request.args.get("after", -1))
    except ValueError:
        return jsonify({"error": "after must be an integer"}), 400
    service = get_service()
    position = service.next_unannotated_position(after)
    return jsonify(
        {
            "record": service.record_at(position) if position is not None else None,
            "progress": service.progress(),
        }
    )


@app.post("/api/annotations")
def save_annotation():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON object"}), 400
    service = get_service()
    try:
        result = service.save(payload)
        position = service.next_unannotated_position(int(payload.get("position", -1)))
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(
        {
            "ok": True,
            "next_position": position,
            **result,
        }
    )


if __name__ == "__main__":
    app.run(port=5001, debug=False, threaded=True)
