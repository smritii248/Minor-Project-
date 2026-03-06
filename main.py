from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import re
import os
import sqlite3
import numpy as np
import pandas as pd
import traceback

# ==============================
# T5 MODEL IMPORTS
# ==============================
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


app = Flask(__name__)
CORS(app)


# ==============================
# READABILITY FUNCTIONS
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
# LOAD T5 FINE-TUNED MODEL
# ==============================
# Expected location: ./t5-medical-finetuned/  (export from Kaggle/Colab notebook)
# Falls back gracefully if the folder is not present.

T5_MODEL_DIR = os.environ.get("T5_MODEL_DIR", "./t5-medical-finetuned")
T5_PREFIX    = "simplify medical text: "
T5_MAX_INPUT = 256
T5_MAX_OUT   = 128

t5_model     = None
t5_tokenizer = None
t5_device    = "cuda" if torch.cuda.is_available() else "cpu"

print(f"T5 device: {t5_device}")
print(f"Loading T5 model from: {T5_MODEL_DIR} ...")

try:
    if os.path.isdir(T5_MODEL_DIR):
        t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_DIR)
        t5_model     = T5ForConditionalGeneration.from_pretrained(
            T5_MODEL_DIR,
            torch_dtype=torch.float32,
        ).to(t5_device)
        t5_model.eval()
        print("✅ T5 fine-tuned model loaded successfully")
    else:
        print(f"⚠️  T5 model directory '{T5_MODEL_DIR}' not found — T5 simplification disabled.")
        print("   Export your trained model from Kaggle/Colab and place it at that path.")
except Exception as e:
    print(f"❌ Failed to load T5 model: {e}")


def t5_simplify(text: str) -> str:
    """Run the fine-tuned T5 model to simplify a medical text.

    Returns the simplified string, or raises RuntimeError if the model
    is not loaded.
    """
    if t5_model is None or t5_tokenizer is None:
        raise RuntimeError("T5 model is not loaded.")

    input_ids = t5_tokenizer(
        T5_PREFIX + text,
        return_tensors="pt",
        max_length=T5_MAX_INPUT,
        truncation=True,
    ).input_ids.to(t5_device)

    with torch.no_grad():
        output_ids = t5_model.generate(
            input_ids,
            max_length=T5_MAX_OUT,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    return t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)


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

if not medical_terms:
    medical_terms = {
        "hypertension", "diabetes", "acromegaly", "paracetamol poisoning",
        "edema", "tachycardia", "bradycardia", "arrhythmia",
        "myocardial infarction", "dyspnea", "hyperlipidemia",
        "thrombosis", "embolism", "ischemia", "sepsis", "pneumonia",
        "anemia", "renal failure", "hepatitis", "fibrillation",
    }

def extract_medical_terms(text):
    text_lower = text.lower()
    found = []
    for term in medical_terms:
        term_lower = term.lower().strip()
        if re.search(r'\b' + re.escape(term_lower) + r'\b', text_lower):
            found.append(term)
    return list(set(found))


# ==============================
# INITIALIZE OLLAMA LLM
# ==============================
print("Loading Ollama LLM...")
llm = None
retriever = None
try:
    llm = Ollama(model="llama3")
    if vectorstore is not None:
        retriever = vectorstore.as_retriever()
    print("Ollama LLM loaded successfully")
except Exception as e:
    print(f"Failed to load Ollama LLM: {str(e)}")


# ==============================
# LANGCHAIN RAG — MANUAL (no deprecated chain helpers)
# ==============================
def get_answer(llm, retriever, question, chat_history):
    # Step 1: Reformulate question as standalone using history
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, reformulate the "
         "question to be standalone so it can be understood without the chat history. "
         "Do NOT answer it, just reformulate if needed and return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    contextualize_chain = contextualize_prompt | llm | StrOutputParser()

    if chat_history:
        standalone_question = contextualize_chain.invoke({
            "input": question,
            "chat_history": chat_history
        })
    else:
        standalone_question = question

    # Step 2: Retrieve relevant docs
    docs = retriever.invoke(standalone_question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Step 3: Answer with context + history
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are MediClare, a clinical assistant that explains medical jargon in "
         "simple plain English for patients and their families. Be clear, concise, "
         "and friendly. After any medical term add a plain explanation in brackets. "
         "Keep to 3-5 sentences.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = qa_prompt | llm | StrOutputParser()

    return qa_chain.invoke({
        "input": question,
        "context": context,
        "chat_history": chat_history
    })


# Chat histories: { session_id: [ HumanMessage | AIMessage, ... ] }
chat_histories = {}


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
        "ollama_ok":    llm is not None,
        "t5_loaded":    t5_model is not None,
        "t5_device":    t5_device,
    })


# ==============================
# /simplify ROUTE
# ==============================
@app.route('/simplify', methods=['POST'])
def simplify():
    print("\n=== NEW SIMPLIFY REQUEST ===")
    try:
        data      = request.get_json()
        user_text = data.get("medical_text", "").strip()
        print("Input text:", user_text)

        if not user_text:
            return jsonify({"error": "No text provided"}), 400

        orig_assessment = full_readability_assessment(user_text)
        potential_terms = extract_medical_terms(user_text)
        print("Extracted terms:", potential_terms)

        # ------------------------------------------------------------------
        # PRIMARY PATH — T5 fine-tuned model
        # ------------------------------------------------------------------
        if t5_model is not None:
            print("Using T5 model for simplification...")
            try:
                simplified_explanation = t5_simplify(user_text)
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

                # Build a lightweight terms list from detected terms for
                # display purposes (T5 rewrites the whole sentence, so we
                # just surface which terms were spotted).
                terms_list = [{"original": t, "simplified": t, "explanation": ""} for t in potential_terms]

                return jsonify({
                    "simplified_explanation": simplified_explanation,
                    "terms":            terms_list,
                    "sources":          ["Fine-tuned T5 (flan-t5-base)"],
                    "simplification_method": "t5",
                    "confidence_label": "Not available yet, human validation pending",
                    "readability": {
                        "original":    orig_assessment,
                        "simplified":  simp_assessment,
                        "improvement": improvement,
                        "suggestions": suggestions,
                    }
                })

            except Exception as t5_err:
                # T5 failed mid-way — fall through to FAISS path
                print(f"T5 simplification error: {t5_err} — falling back to FAISS")

        # ------------------------------------------------------------------
        # FALLBACK PATH — FAISS term-by-term replacement (original logic)
        # ------------------------------------------------------------------
        print("Using FAISS fallback for simplification...")

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
                "simplification_method": "none",
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
                continue
            try:
                docs_with_scores = vectorstore.similarity_search_with_score(term, k=1)
                if docs_with_scores:
                    doc, score = docs_with_scores[0]
                    score = float(score)
                    print(f" → Score: {score:.3f}")
                    if score < 0.8:
                        metadata = doc.metadata
                        if 'term' in metadata:
                            original    = str(metadata['term'])
                            simplified  = str(metadata.get('summary', original.lower()))
                            explanation = str(doc.page_content).strip() or "Medical term simplified for clarity"
                            terms_list.append({
                                "original":    original,
                                "simplified":  simplified,
                                "explanation": explanation
                            })
                        if "source" in metadata:
                            sources_set.add(str(metadata["source"]))
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
                "confidence": None,
                "simplification_method": "faiss",
                "confidence_label": "Not available yet – human validation pending",
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
            "simplification_method": "faiss",
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
# /chat ROUTE  (Ollama + LangChain RAG)
# ==============================
@app.route('/chat', methods=['POST'])
def chat():
    try:
        if llm is None or retriever is None:
            return jsonify({"error": "Ollama LLM or retriever not initialized."}), 503

        data       = request.get_json()
        question   = data.get("question", "").strip()
        session_id = data.get("session_id", "default")

        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Get or init session history
        history = chat_histories.setdefault(session_id, [])

        # Get answer using manual RAG
        answer = get_answer(llm, retriever, question, history)

        # Update history (keep last 20 messages = 10 turns)
        history.extend([HumanMessage(content=question), AIMessage(content=answer)])
        chat_histories[session_id] = history[-20:]

        return jsonify({
            "answer":      answer,
            "terms_found": extract_medical_terms(question),
            "session_id":  session_id,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/chat/reset', methods=['POST'])
def chat_reset():
    data       = request.get_json() or {}
    session_id = data.get("session_id", "default")
    chat_histories[session_id] = []
    return jsonify({"status": "ok", "message": "Chat history cleared"})


# ==============================
# RUN SERVER
# ==============================
if __name__ == '__main__':
    print("\nStarting MediClare server...")
    print("→ Open http://localhost:5500 in browser")
    print("→ Ensure Ollama is running with llama3 model")
    print(f"→ T5 model: {'loaded ✅' if t5_model else 'not found ⚠️  (FAISS fallback active)'}")
    app.run(debug=True, host="0.0.0.0", port=5500)