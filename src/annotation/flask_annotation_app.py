from functools import lru_cache

from flask import Flask, jsonify, render_template, request

try:
    from .annotation_core import AnnotationService
except ImportError:  # Support direct ``python src/annotation/...py`` execution.
    from annotation_core import AnnotationService


app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


@lru_cache(maxsize=1)
def get_service():
    return AnnotationService()


@app.get("/")
def index():
    return render_template("annotation.html")


@app.get("/api/progress")
def progress():
    return jsonify(get_service().progress())


@app.get("/api/records/<int:position>")
def record(position):
    try:
        result = get_service().record_at(position)
    except IndexError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(result)


@app.get("/api/next")
def next_record():
    try:
        after = int(request.args.get("after", -1))
    except ValueError:
        return jsonify({"error": "after must be an integer"}), 400

    service = get_service()
    position = service.next_unannotated_position(after)
    if position is None:
        return jsonify({"record": None, "progress": service.progress()})
    return jsonify({"record": service.record_at(position), "progress": service.progress()})


@app.post("/api/annotations")
def save_annotation():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected a JSON object"}), 400

    service = get_service()
    try:
        learned_updates = service.save(payload)
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    position = service.next_unannotated_position(int(payload.get("position", -1)))
    return jsonify(
        {
            "ok": True,
            "next_position": position,
            "learned_updates": learned_updates,
            "progress": service.progress(),
        }
    )


if __name__ == "__main__":
    app.run(debug=False, threaded=True)
