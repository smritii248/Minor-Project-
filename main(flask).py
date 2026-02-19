from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

app = Flask(__name__)

# ==============================
# LOAD EMBEDDING MODEL
# ==============================
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==============================
# LOAD FAISS VECTOR DATABASE
# ==============================
print("Loading FAISS index...")

vectorstore = FAISS.load_local(
    "faiss_index",  # folder created from notebook
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("Backend ready ✅")


# ==============================
# ROUTES
# ==============================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simplify', methods=['POST'])
def simplify():
    try:
        data = request.get_json()
        user_text = data.get("text", "")

        if not user_text:
            return jsonify({"error": "No text provided"}), 400

        # Retrieve relevant documents
        docs = vectorstore.similarity_search(user_text, k=3)

        if not docs:
            return jsonify({
                "simplified_text": "No relevant medical information found.",
                "sources": []
            })

        # Combine retrieved content
        combined_text = " ".join([doc.page_content for doc in docs])

        # Simple fallback simplification (for MID demo)
        simplified = f"In simple terms: {combined_text[:300]}..."

        # Extract sources
        sources = []
        for doc in docs:
            if "source" in doc.metadata:
                sources.append(doc.metadata["source"])

        return jsonify({
            "simplified_text": simplified,
            "sources": list(set(sources))
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


        # Combine retrieved content
        combined_text = "\n\n".join([doc.page_content for doc in docs])

        # Simple explanation logic (for defense demo)
        simplified = f"""
🔎 Extracted Medical Context:

{combined_text[:800]}

📘 Simplified Explanation:

This medical text refers to the above condition. It describes the concept in medical terminology. 
In simple words, it explains the condition and its meaning based on trusted medical sources.
"""

        # Retrieval confidence (based on number of docs found)
        confidence_score = round(len(docs) / 3, 2)

        return jsonify({
            "simplified_explanation": simplified,
            "confidence_score": confidence_score
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# RUN SERVER
# ==============================
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)

