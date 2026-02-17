from flask import Flask, request, jsonify, render_template
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# ----------------------------
# LOAD EVERYTHING ONCE
# ----------------------------

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index("faiss_index.index")

print("Loading stored texts...")
with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)

print("Backend loaded successfully ✅")


# ----------------------------
# HOME ROUTE
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ----------------------------
# RETRIEVAL FUNCTION
# ----------------------------
def retrieve_answer(query, top_k=3):
    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        if i < len(documents):
            results.append(documents[i])

    return results


# ----------------------------
# API ROUTE
# ----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_query = data.get("question")

    if not user_query:
        return jsonify({"error": "No question provided"}), 400

    retrieved_docs = retrieve_answer(user_query)

    return jsonify({
        "question": user_query,
        "retrieved_results": retrieved_docs
    })


# ----------------------------
# RUN SERVER
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
