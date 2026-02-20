from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import re
import sqlite3

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
try:
    vectorstore = FAISS.load_local(
        "faiss_index",  # Folder from notebook
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print("Backend ready ✅")
except Exception as e:
    print(f"Error loading FAISS: {str(e)}")

# ==============================
# LOAD MEDICAL TERMS FROM DB FOR EXTRACTION
# ==============================
print("Loading medical terms from DB...")
try:
    conn = sqlite3.connect("medical_jargon.db")
    df = pd.read_sql_query("SELECT term FROM medical_terms;", conn)
    medical_terms = set(df['term'].str.lower())
    conn.close()
    print(f"Loaded {len(medical_terms)} terms ✅")
except Exception as e:
    print(f"Error loading DB: {str(e)}")
    medical_terms = set()  # Fallback empty

# Function to extract terms (from your notebook)
def extract_medical_terms(text):
    text = text.lower()
    found_terms = []
    for term in medical_terms:
        if term in text:
            found_terms.append(term)
    return list(set(found_terms))

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
        user_text = data.get("medical_text", "")
        if not user_text:
            return jsonify({"error": "No text provided"}), 400

        # Extract terms
        potential_terms = extract_medical_terms(user_text)
        if not potential_terms:
            return jsonify({
                "simplified_explanation": "No medical terms detected.",
                "terms": [],
                "sources": [],
                "confidence": 0,
                "confidence_label": "Low"
            })

        # Retrieve for each term
        terms = []
        sources_set = set()
        for term in potential_terms:
            docs_with_scores = vectorstore.similarity_search_with_score(term, k=1)
            if docs_with_scores:
                doc, score = docs_with_scores[0]
                if score < 0.8:  # Threshold for relevance (adjust if needed)
                    if 'term' in doc.metadata:
                        original = doc.metadata['term']
                        simplified = doc.metadata.get('summary', original.lower())
                        explanation = doc.page_content or "Medical term simplified for clarity"
                        terms.append({
                            "original": original,
                            "simplified": simplified,
                            "explanation": explanation
                        })
                    if "source" in doc.metadata:
                        sources_set.add(doc.metadata["source"])

        if not terms:
            return jsonify({
                "simplified_explanation": "No relevant medical information found.",
                "terms": [],
                "sources": [],
                "confidence": 0,
                "confidence_label": "Low"
            })

        # Generate simplified explanation
        simplified_explanation = user_text
        for t in terms:
            simplified_explanation = re.sub(r'\b' + re.escape(t['original']) + r'\b', t['simplified'], simplified_explanation, flags=re.IGNORECASE)
        simplified_explanation += "." if not simplified_explanation.endswith('.') else ""

        # Placeholder sources (since not in DB; add to DB for real ones)
        sources = list(sources_set) if sources_set else ["WHO Medical Dictionary", "Mayo Clinic", "NIH Glossary"]

        # Placeholder confidence (w1=0.5 retrieval, w2=0.3 sources, w3=0.2 human=0 pending)
        s_retrieval = len(terms) / len(potential_terms)
        s_source = len(sources) / 3
        s_human = 0.0  # Pending validation
        confidence = int((0.5 * s_retrieval + 0.3 * s_source + 0.2 * s_human) * 100)
        if confidence >= 80:
            confidence_label = "High Confidence"
        elif confidence >= 50:
            confidence_label = "Medium Confidence"
        else:
            confidence_label = "Low Confidence"

        return jsonify({
            "simplified_explanation": simplified_explanation,
            "terms": terms,
            "sources": sources,
            "confidence": confidence,
            "confidence_label": confidence_label
        })
    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

# ==============================
# RUN SERVER
# ==============================
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5500)  # Match your terminal port