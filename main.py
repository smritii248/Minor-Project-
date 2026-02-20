from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import re
import sqlite3
import pandas as pd
import traceback

# Optional — uncomment if you want readability feature
# from textstat import flesch_reading_ease, flesch_kincaid_grade

app = Flask(__name__)
CORS(app)

# ==============================
# LOAD EMBEDDING MODEL
# ==============================
print("Loading embedding model...")
embedding_model = None
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embedding model loaded successfully")
except Exception as e:
    print(f"Failed to load embedding model: {str(e)}")

# ==============================
# LOAD FAISS VECTOR DATABASE
# ==============================
vectorstore = None
print("Loading FAISS index...")
try:
    vectorstore = FAISS.load_local(
        "faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print("FAISS index loaded successfully")
except Exception as e:
    print(f"Error loading FAISS index: {str(e)}")
    print("→ Make sure the 'faiss_index' folder exists and was created by the notebook.")

# ==============================
# LOAD MEDICAL TERMS FROM DB
# ==============================
medical_terms = set()
print("Loading medical terms from DB...")
try:
    conn = sqlite3.connect("medical_jargon.db")
    df = pd.read_sql_query("SELECT term FROM medical_terms;", conn)
    medical_terms = set(df['term'].str.lower().str.strip())
    conn.close()
    print(f"Loaded {len(medical_terms)} terms from database")
except Exception as e:
    print(f"Error loading medical terms from DB: {str(e)}")
    print("→ Using fallback terms for demo")

# Always have some fallback
if not medical_terms:
    print("WARNING: No terms loaded — using fallback")
    medical_terms = {"hypertension", "diabetes", "acromegaly", "paracetamol poisoning"}

# Improved term extraction with word boundaries
def extract_medical_terms(text):
    text_lower = text.lower()
    found = []
    for term in medical_terms:
        term_lower = term.lower().strip()
        if re.search(r'\b' + re.escape(term_lower) + r'\b', text_lower):
            found.append(term)
    return list(set(found))

# ==============================
# ROUTES
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "faiss_loaded": vectorstore is not None,
        "terms_loaded": len(medical_terms),
        "embedding_model": embedding_model is not None
    })

@app.route('/simplify', methods=['POST'])
def simplify():
    print("\n=== NEW SIMPLIFY REQUEST ===")
    try:
        data = request.get_json()
        user_text = data.get("medical_text", "").strip()
        print("Input text:", user_text)

        if not user_text:
            return jsonify({"error": "No text provided"}), 400

        # Extract terms
        potential_terms = extract_medical_terms(user_text)
        print("Extracted terms:", potential_terms)

        if not potential_terms:
            return jsonify({
                "simplified_explanation": "No medical terms detected in the input.",
                "terms": [],
                "sources": [],
                "confidence": None,
                "confidence_label": "Not available yet – human validation pending"
            })

        # Retrieve documents
        terms_list = []
        sources_set = set()

        for term in potential_terms:
            print(f"Retrieving for: {term}")
            if vectorstore is None:
                print(" → FAISS not loaded, skipping retrieval")
                continue

            try:
                docs_with_scores = vectorstore.similarity_search_with_score(term, k=1)
                print(f" → Found {len(docs_with_scores)} document(s)")

                if docs_with_scores:
                    doc, score = docs_with_scores[0]
                    print(f" → Score: {score:.3f}")

                    if score < 0.8:  # adjust threshold if needed
                        if 'term' in doc.metadata:
                            original = doc.metadata['term']
                            simplified = doc.metadata.get('summary', original.lower())
                            explanation = doc.page_content.strip() or "Medical term simplified for clarity"
                            terms_list.append({
                                "original": original,
                                "simplified": simplified,
                                "explanation": explanation
                            })
                        if "source" in doc.metadata:
                            sources_set.add(doc.metadata["source"])
            except Exception as retrieval_err:
                print(f"Retrieval error for '{term}': {str(retrieval_err)}")

        # If no valid terms found after retrieval
        if not terms_list:
            return jsonify({
                "simplified_explanation": "Found terms but no relevant explanations in the knowledge base.",
                "terms": [],
                "sources": ["WHO Medical Dictionary", "Mayo Clinic", "NIH Glossary"],
                "confidence": None,
                "confidence_label": "Not available yet – human validation pending"
            })

        # Build simplified explanation
        simplified_explanation = user_text
        for t in terms_list:
            simplified_explanation = re.sub(
                r'\b' + re.escape(t['original']) + r'\b',
                t['simplified'],
                simplified_explanation,
                flags=re.IGNORECASE
            )
        if not simplified_explanation.endswith('.'):
            simplified_explanation += '.'

        sources = list(sources_set) if sources_set else ["WHO Medical Dictionary", "Mayo Clinic", "NIH Glossary"]

        # Optional: Add readability here if you want it back
        # readability_info = ... (see previous messages)

        return jsonify({
            "simplified_explanation": simplified_explanation,
            "terms": terms_list,
            "sources": sources,
            "confidence": None,
            "confidence_label": "Not available yet – human validation pending"
            # "readability": readability_info   # ← uncomment if you add it
        })

    except Exception as e:
        print("!!! CRASH IN /simplify ROUTE !!!")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ==============================
# RUN SERVER
# ==============================
if __name__ == '__main__':
    print("\nStarting server...")
    print("→ Open http://localhost:5500 in browser")
    print("→ Or use Live Server on index.html and test POST to /simplify")
    app.run(debug=True, host="0.0.0.0", port=5500)