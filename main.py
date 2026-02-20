from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
import sqlite3
import pandas as pd
import traceback

app = Flask(__name__)
CORS(app)

# ==============================
# READABILITY FUNCTIONS (Pure Python)
# ==============================

def count_syllables(word):
    word = word.lower().strip(".,!?;:")
    if len(word) <= 3:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e"):
        count -= 1
    return max(1, count)

def flesch_reading_ease(text):
    if not text or not text.strip():
        return 0
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = max(len(sentences), 1)
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    num_words = max(len(words), 1)
    num_syllables = sum(count_syllables(w) for w in words)
    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    return round(max(0, min(100, score)), 1)

def flesch_kincaid_grade(text):
    if not text or not text.strip():
        return 0
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = max(len(sentences), 1)
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    num_words = max(len(words), 1)
    num_syllables = sum(count_syllables(w) for w in words)
    grade = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
    return round(max(0, grade), 1)

def avg_sentence_length(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    if not sentences:
        return 0
    return round(len(words) / len(sentences), 1)

def avg_word_length(text):
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    if not words:
        return 0
    return round(sum(len(w) for w in words) / len(words), 1)

def count_complex_words(text):
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return sum(1 for w in words if count_syllables(w) >= 3)

def get_readability_label(score):
    if score >= 90:   return "Very Easy"
    elif score >= 70: return "Easy"
    elif score >= 60: return "Standard"
    elif score >= 50: return "Fairly Difficult"
    elif score >= 30: return "Difficult"
    else:             return "Very Difficult"

def get_suggestions(orig_score, simp_score, avg_sent, avg_word, complex_count):
    tips = []
    if avg_sent > 20:
        tips.append("Break long sentences into shorter ones (aim for under 20 words per sentence).")
    if avg_word > 6:
        tips.append("Replace long words with simpler everyday alternatives.")
    if complex_count > 3:
        tips.append(f"Found {complex_count} complex words (3+ syllables) — consider simplifying them.")
    if orig_score < 50:
        tips.append("Original text is highly technical — good candidate for simplification.")
    if simp_score and simp_score > orig_score:
        tips.append("Simplification improved readability — the text is now easier to understand.")
    if not tips:
        tips.append("Text readability is within acceptable range.")
    return tips

def full_readability_assessment(text):
    score       = flesch_reading_ease(text)
    fk_grade    = flesch_kincaid_grade(text)
    label       = get_readability_label(score)
    avg_sent    = avg_sentence_length(text)
    avg_word    = avg_word_length(text)
    complex_cnt = count_complex_words(text)
    words       = re.findall(r'\b[a-zA-Z]+\b', text)
    sentences   = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    return {
        "score":            score,
        "label":            label,
        "fk_grade":         fk_grade,
        "word_count":       len(words),
        "sentence_count":   len(sentences),
        "avg_sentence_len": avg_sent,
        "avg_word_len":     avg_word,
        "complex_words":    complex_cnt,
    }

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

if not medical_terms:
    print("WARNING: No terms loaded — using fallback")
    medical_terms = {"hypertension", "diabetes", "acromegaly", "paracetamol poisoning"}

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
        "status":       "ok",
        "faiss_loaded": vectorstore is not None,
        "terms_loaded": len(medical_terms),
        "embedding_ok": embedding_model is not None,
    })

# ==============================
# /simplify ROUTE
# ==============================
@app.route('/simplify', methods=['POST'])
def simplify():
    print("\n=== NEW SIMPLIFY REQUEST ===")
    try:
        data = request.get_json()
        user_text = data.get("medical_text", "").strip()
        print("Input text:", user_text)

        if not user_text:
            return jsonify({"error": "No text provided"}), 400

        orig_assessment = full_readability_assessment(user_text)
        potential_terms = extract_medical_terms(user_text)
        print("Extracted terms:", potential_terms)

        if not potential_terms:
            suggestions = get_suggestions(
                orig_assessment["score"], None,
                orig_assessment["avg_sentence_len"],
                orig_assessment["avg_word_len"],
                orig_assessment["complex_words"]
            )
            return jsonify({
                "simplified_explanation": "No medical terms detected in the input.",
                "terms": [], "sources": [], "confidence": None,
                "confidence_label": "Not available yet, human validation pending",
                "readability": {
                    "original": orig_assessment, "simplified": None,
                    "improvement": None, "suggestions": suggestions
                }
            })

        terms_list  = []
        sources_set = set()

        for term in potential_terms:
            print(f"Retrieving for: {term}")
            if vectorstore is None:
                print(" → FAISS not loaded, skipping retrieval")
                continue
            try:
                docs_with_scores = vectorstore.similarity_search_with_score(term, k=1)
                if docs_with_scores:
                    doc, score = docs_with_scores[0]
                    print(f" → Score: {score:.3f}")
                    if score < 0.8:
                        if 'term' in doc.metadata:
                            original    = doc.metadata['term']
                            simplified  = doc.metadata.get('summary', original.lower())
                            explanation = doc.page_content.strip() or "Medical term simplified for clarity"
                            terms_list.append({
                                "original":    original,
                                "simplified":  simplified,
                                "explanation": explanation
                            })
                        if "source" in doc.metadata:
                            sources_set.add(doc.metadata["source"])
            except Exception as retrieval_err:
                print(f"Retrieval error for '{term}': {str(retrieval_err)}")

        if not terms_list:
            suggestions = get_suggestions(
                orig_assessment["score"], None,
                orig_assessment["avg_sentence_len"],
                orig_assessment["avg_word_len"],
                orig_assessment["complex_words"]
            )
            return jsonify({
                "simplified_explanation": "Found terms but no relevant explanations in the knowledge base.",
                "terms": [], "sources": ["WHO Medical Dictionary", "Mayo Clinic", "NIH Glossary"],
                "confidence": None, "confidence_label": "Not available yet – human validation pending",
                "readability": {
                    "original": orig_assessment, "simplified": None,
                    "improvement": None, "suggestions": suggestions
                }
            })

        simplified_explanation = user_text
        for t in terms_list:
            simplified_explanation = re.sub(
                r'\b' + re.escape(t['original']) + r'\b',
                t['simplified'], simplified_explanation, flags=re.IGNORECASE
            )
        if not simplified_explanation.endswith('.'):
            simplified_explanation += '.'

        simp_assessment = full_readability_assessment(simplified_explanation)
        improvement     = round(simp_assessment["score"] - orig_assessment["score"], 1)
        suggestions     = get_suggestions(
            orig_assessment["score"], simp_assessment["score"],
            orig_assessment["avg_sentence_len"],
            orig_assessment["avg_word_len"],
            orig_assessment["complex_words"]
        )
        sources = list(sources_set) if sources_set else ["WHO Medical Dictionary", "Mayo Clinic", "NIH Glossary"]

        return jsonify({
            "simplified_explanation": simplified_explanation,
            "terms":            terms_list,
            "sources":          sources,
            "confidence_label": "Not available yet, human validation pending",
            "readability": {
                "original":    orig_assessment,
                "simplified":  simp_assessment,
                "improvement": improvement,
                "suggestions": suggestions
            }
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
    app.run(debug=True, host="0.0.0.0", port=5500)