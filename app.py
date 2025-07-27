from flask import Flask, request, jsonify
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

app = Flask(__name__)

# Load embedding model once at startup
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route("/")
def home():
    return "Hello from Flask on Render!"

@app.route("/embed", methods=["POST"])
def embed():
    data = request.json['text']
    # Sample test embedding output
    return jsonify({"embedding": [0.1, 0.2, 0.3]})

# Function to load and combine all PDF text
def extract_text_from_all_pdfs(folder="data"):
    combined_text = ""
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            path = os.path.join(folder, filename)
            text = extract_text(path)
            combined_text += f"\n\n--- {filename} ---\n{text}"
    return combined_text

@app.route("/reduce-from-pdfs", methods=["POST"])
def reduce_from_pdfs():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Missing 'question' in request."}), 400

    # Load full text from PDFs
    full_text = extract_text_from_all_pdfs("data")
    sentences = full_text.split(". ")

    # Embed question and sentences
    question_embedding = model.encode([question])
    sentence_embeddings = model.encode(sentences)

    # Find most similar sentences
    similarities = cosine_similarity(question_embedding, sentence_embeddings)[0]
    top_k = np.argsort(similarities)[-5:][::-1]

    # Combine top sentences as reduced context
    reduced_context = ". ".join([sentences[i] for i in top_k])
    return jsonify({
        "question": question,
        "reduced_context": reduced_context
    })

if __name__ == "__main__":
    app.run(debug=True)
