import os
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and memory-efficient

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def get_reduced_context(question, folder_path, max_sentences=5):
    question_embedding = model.encode(question, convert_to_tensor=True)
    best_matches = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(file_path)
            sentences = text.split(". ")
            embeddings = model.encode(sentences, convert_to_tensor=True)

            cosine_scores = util.pytorch_cos_sim(question_embedding, embeddings)[0]
            top_results = cosine_scores.topk(k=min(max_sentences, len(sentences)))

            for score, idx in zip(top_results[0], top_results[1]):
                best_matches.append((float(score), sentences[int(idx)]))

    best_matches.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in best_matches[:max_sentences]]

@app.route("/reduce-from-pdfs", methods=["POST"])
def reduce_from_pdfs():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Missing 'question' in request."}), 400

    reduced_context = get_reduced_context(question, folder_path="data")
    return jsonify({
        "question": question,
        "reduced_context": " ".join(reduced_context)
    })

if __name__ == "__main__":
    app.run(debug=True)
