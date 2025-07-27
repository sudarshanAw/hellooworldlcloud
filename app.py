from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello from Flask on Render!"

@app.route("/embed", methods=["POST"])
def embed():
    data = request.json.get('text', '')
    if not data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    # Simulate embedding (replace with real logic later)
    embedding = [0.1, 0.2, 0.3]
    return jsonify({"embedding": embedding})

if __name__ == "__main__":
    app.run(debug=True)
