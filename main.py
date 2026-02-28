from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re, sqlite3, traceback
import pandas as pd

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════
#  READABILITY  — returns ease 0-100, grade, label, pct
# ══════════════════════════════════════════════════════
def count_syllables(word):
    word = word.lower().strip(".,!?;:")
    if len(word) <= 3: return 1
    vowels, count, prev = "aeiouy", 0, False
    for ch in word:
        v = ch in vowels
        if v and not prev: count += 1
        prev = v
    if word.endswith("e"): count -= 1
    return max(1, count)

def flesch_ease(text):
    text = re.sub(r'<[^>]+>', '', text).strip()
    if not text: return 0
    sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    sc = max(1, len(sents))
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    wc = len(words)
    if not wc: return 0
    syl = sum(count_syllables(w) for w in words)
    return round(max(0, min(100, 206.835 - 1.015*(wc/sc) - 84.6*(syl/wc))), 1)

def flesch_grade(text):
    text = re.sub(r'<[^>]+>', '', text).strip()
    if not text: return 0
    sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    sc = max(1, len(sents))
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    wc = len(words)
    if not wc: return 0
    syl = sum(count_syllables(w) for w in words)
    return round(max(0, 0.39*(wc/sc) + 11.8*(syl/wc) - 15.59), 1)

def ease_label(score):
    # returns label + word-difficulty tag
    if score >= 80:   return "Very Easy",   "Easy Words",      "green"
    elif score >= 60: return "Easy",         "Mostly Easy",     "lime"
    elif score >= 50: return "Fairly Easy",  "Mixed Difficulty","yellow"
    elif score >= 40: return "Moderate",     "Some Hard Words", "orange"
    else:             return "Difficult",    "Hard Words",      "red"

def readability(text):
    """Return full readability dict for frontend."""
    clean = re.sub(r'<[^>]+>', '', text).strip()
    if not clean: return None
    ease  = flesch_ease(clean)
    grade = flesch_grade(clean)
    lbl, word_tag, colour = ease_label(ease)
    # percentage: ease score IS 0-100, use directly
    return {
        "ease":       ease,               # 0-100
        "ease_pct":   ease,               # same, shown as %
        "grade":      grade,
        "label":      lbl,
        "word_tag":   word_tag,
        "colour":     colour,
    }


# ══════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════
embedding_model = None
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("✓ Embeddings loaded")
except Exception as e:
    print(f"✗ Embeddings: {e}")

vectorstore = None
try:
    vectorstore = FAISS.load_local(
        "faiss_index", embedding_model, allow_dangerous_deserialization=True)
    print("✓ FAISS loaded")
except Exception as e:
    print(f"✗ FAISS: {e}")


# ══════════════════════════════════════════════════════
#  LOAD DB  — columns confirmed: term, content, summary
# ══════════════════════════════════════════════════════
db_terms = {}   # lower_term -> {term, summary, content}

try:
    conn = sqlite3.connect("medical_jargon.db")
    df   = pd.read_sql_query("SELECT term, content, summary FROM medical_terms;", conn)
    conn.close()
    df = df.dropna(subset=['term'])
    for _, row in df.iterrows():
        k = str(row['term']).strip().lower()
        if k:
            db_terms[k] = {
                "term":    str(row['term']).strip(),
                "summary": str(row.get('summary') or '').strip(),
                "content": str(row.get('content') or '').strip(),
            }
    print(f"✓ Loaded {len(db_terms)} DB terms")
except Exception as e:
    print(f"✗ DB: {e}")
    traceback.print_exc()

# Demo fallback so app works even without DB
if not db_terms:
    db_terms = {
        "hypertension": {
            "term": "hypertension", "summary": "high blood pressure",
            "content": "A condition where blood pushes too hard against artery walls, straining the heart."},
        "paracetamol poisoning": {
            "term": "paracetamol poisoning", "summary": "painkiller overdose damaging the liver",
            "content": "Occurs when too much paracetamol is taken, overwhelming the liver's ability to process it safely."},
        "diabetes": {
            "term": "diabetes", "summary": "high blood sugar condition",
            "content": "A chronic condition where the body cannot properly regulate glucose (sugar) in the blood."},
        "meningitis": {
            "term": "meningitis", "summary": "brain membrane infection",
            "content": "Inflammation of membranes surrounding the brain and spinal cord, usually from infection."},
    }
    print("⚠ Using demo terms")


def find_terms(text):
    """Return list of DB keys found in text, longest first."""
    tl = text.lower()
    return list({k for k in sorted(db_terms, key=len, reverse=True)
                 if re.search(r'\b' + re.escape(k) + r'\b', tl)})


# ══════════════════════════════════════════════════════
#  FAISS LOOKUP
#  Your store uses L2 distance → lower = better match.
#  Metadata keys confirmed: "term", "summary"
#  We return ONLY summary (not full page_content paragraph)
#  so the UI stays clean.
# ══════════════════════════════════════════════════════
def faiss_lookup(query, k=1):
    if vectorstore is None:
        return None
    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
        if not results:
            return None
        doc, dist = results[0]

        # Accept dist < 2.0  (your notebook showed ~0.58 for good match)
        if dist > 2.0:
            return None

        sim = round(1 / (1 + dist), 4)   # 0-1, higher = better
        return {
            "term":     doc.metadata.get("term", query),
            "summary":  doc.metadata.get("summary", ""),
            # Only return page_content as fallback if summary is empty
            "content":  doc.page_content.strip(),
            "sim":      sim,
            "dist":     dist,
        }
    except Exception as e:
        print(f"FAISS error: {e}")
        return None


# ══════════════════════════════════════════════════════
#  CONFIDENCE  C = w1*S_ret + w2*S_src + w3*S_human
#  S_human = 0 until experts validate → shows as pending
# ══════════════════════════════════════════════════════
W1, W2, W3 = 0.4, 0.4, 0.2

def confidence(s_ret, s_src=0.8, s_human=0.0):
    c = W1*s_ret + W2*s_src + W3*s_human
    return round(min(1.0, max(0.0, c)), 3)

def confidence_label(c, s_human=0.0):
    if s_human == 0:
        return f"{c:.0%} (retrieval+source · awaiting expert validation)"
    return f"{c:.0%} (fully validated)"


# ══════════════════════════════════════════════════════
#  MEDICAL ADVICE GUARD
# ══════════════════════════════════════════════════════
ADVICE_KW = [
    "treatment","treat","cure","medicine","medication","drug","dose",
    "therapy","what causes","cause of","caused by","symptom","symptoms",
    "prevent","prevention","prognosis","serious","dangerous","risk",
    "how to manage","should i take","can i take","is it safe","will it",
    "emergency","diagnose","diagnosis","how to treat","what should i do"
]
LINKS = [
    ("Mayo Clinic","https://www.mayoclinic.org"),
    ("NIH MedlinePlus","https://medlineplus.gov"),
    ("NHS","https://www.nhs.uk"),
]

def is_advice(q):
    return any(kw in q.lower() for kw in ADVICE_KW)

def disclaimer():
    ls = " · ".join(f'<a href="{u}" target="_blank" class="underline">{n}</a>' for n,u in LINKS)
    return (
        "⚠️ <strong>I cannot provide medical advice, treatment, or diagnosis.</strong><br><br>"
        f"Please consult a qualified healthcare professional. Trusted resources: {ls}<br><br>"
        "I can explain what a medical <em>term means</em> — "
        "e.g. <em>\"What does hypertension mean?\"</em>"
    )


# ══════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/simplify', methods=['POST'])
def simplify():
    try:
        user_text = request.get_json().get("medical_text","").strip()
        if not user_text:
            return jsonify({"error": "No text provided"}), 400

        keys = find_terms(user_text)
        print(f"[simplify] Terms: {keys}")

        terms_out   = []
        sources_set = set()
        conf_scores = []
        output_text = user_text          # we replace jargon in this copy

        for k in keys:
            db  = db_terms.get(k, {})
            fai = faiss_lookup(k)

            if fai:
                term_name  = fai["term"]
                # ── KEY FIX: use summary (short), NOT full content paragraph ──
                plain      = fai["summary"] or db.get("summary") or k
                # Explanation shown in table = summary + maybe first sentence of content
                content    = fai["content"] or db.get("content","")
                # Trim content to first sentence only for the table
                first_sent = re.split(r'(?<=[.!?])\s', content)[0] if content else ""
                explanation = first_sent or plain
                s_ret      = fai["sim"]
                sources_set.add("FAISS Vector Database")
            elif db:
                term_name  = db["term"]
                plain      = db["summary"] or k
                content    = db["content"] or ""
                first_sent = re.split(r'(?<=[.!?])\s', content)[0] if content else ""
                explanation = first_sent or plain
                s_ret      = 0.65
                sources_set.add("Medical Jargon Database")
            else:
                continue

            c = confidence(s_ret)
            conf_scores.append(c)

            terms_out.append({
                "original":    term_name,
                "simplified":  plain,           # short plain-English label
                "explanation": explanation,     # 1 sentence max
                "confidence":  c,
                "s_retrieval": round(s_ret, 3),
                "source":      "FAISS" if fai else "DB",
            })

            # Replace jargon in output text with plain summary
            output_text = re.sub(
                r'\b' + re.escape(k) + r'\b',
                plain, output_text, flags=re.IGNORECASE
            )

        if not output_text.endswith('.'): output_text += '.'

        overall_c = round(sum(conf_scores)/len(conf_scores), 3) if conf_scores else None

        return jsonify({
            "simplified_explanation": output_text,
            "terms":      terms_out,
            "sources":    list(sources_set) or ["WHO Medical Dictionary","Mayo Clinic","NIH"],
            "confidence": overall_c,
            "confidence_label": confidence_label(overall_c) if overall_c else
                                 "Not yet calculated — human validation pending",
            "readability": readability(output_text),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data     = request.get_json()
        question = data.get("question","").strip()
        history  = data.get("history", [])
        if not question:
            return jsonify({"error":"No question"}), 400

        print(f"[chat] '{question}'")

        # 1. Advice guard
        if is_advice(question):
            return jsonify({"reply": disclaimer(), "source":"Safety policy",
                            "readability": None, "confidence": None})

        # 2. Find terms in question
        keys = find_terms(question)
        print(f"[chat] Terms found: {keys}")

        # 3. DB direct hit
        db_hit = None
        for k in keys:
            e = db_terms.get(k,{})
            if e.get("summary") or e.get("content"):
                db_hit = e; break

        # 4. FAISS semantic hit
        fai_hit = faiss_lookup(question, k=1)
        if not fai_hit and keys:
            for k in keys:
                fai_hit = faiss_lookup(k, k=1)
                if fai_hit: break

        # 5. Build answer
        if db_hit or fai_hit:
            if fai_hit:
                term_name   = fai_hit["term"]
                plain       = fai_hit["summary"] or (db_hit or {}).get("summary","")
                content     = fai_hit["content"] or (db_hit or {}).get("content","")
                s_ret       = fai_hit["sim"]
                src_label   = f"FAISS Vector DB (similarity {fai_hit['sim']:.0%})"
            else:
                term_name   = db_hit["term"]
                plain       = db_hit["summary"]
                content     = db_hit["content"]
                s_ret       = 0.65
                src_label   = "Medical Jargon Database"

            # Trim content to max 2 sentences
            sentences = re.split(r'(?<=[.!?])\s', content.strip()) if content else []
            short_def = " ".join(sentences[:2]) if sentences else plain

            c = confidence(s_ret)

            reply = (
                f"<strong>{term_name}</strong><br>"
                f"<span class='text-blue-600 font-medium'>In plain English:</span> {plain}<br><br>"
                f"{short_def}"
            )
            return jsonify({
                "reply": reply, "source": src_label,
                "confidence": c,
                "confidence_label": confidence_label(c),
                "readability": readability(f"{plain}. {short_def}"),
            })

        # 6. Term NOT in database → clear message
        # Check if it looks like a medical term search
        looks_medical = any(w in question.lower() for w in
            ["what is","what does","define","explain","mean","meaning"])

        not_found_reply = (
            "🔍 <strong>That term isn't in my database yet.</strong><br><br>"
            "Here are trusted sources where you can find it:<br>"
            '<a href="https://medlineplus.gov/ency/encyclopedia.htm" target="_blank" '
            'class="text-blue-600 underline">NIH MedlinePlus Medical Encyclopedia</a><br>'
            '<a href="https://www.mayoclinic.org/diseases-conditions" target="_blank" '
            'class="text-blue-600 underline">Mayo Clinic Diseases &amp; Conditions</a><br>'
            '<a href="https://www.nhs.uk/conditions/" target="_blank" '
            'class="text-blue-600 underline">NHS Health A–Z</a>'
        )

        if OLLAMA_AVAILABLE and looks_medical:
            try:
                resp = ollama.chat(model="llama3", messages=[
                    {"role":"system","content":
                     "You explain medical term meanings in 1-2 plain English sentences only. "
                     "No advice, no diagnosis. State clearly this is a general definition."},
                    {"role":"user","content": question}
                ])
                ai_text = resp['message']['content'].strip()
                return jsonify({
                    "reply": (
                        f"{ai_text}<br><br>"
                        "<em class='text-xs text-gray-400'>⚠ This term is not in my database. "
                        "AI-generated definition — verify with a trusted source.</em>"
                    ),
                    "source": "AI-generated (Llama3) — not from database",
                    "confidence": None,
                    "confidence_label": "Not validated — AI generated",
                    "readability": readability(ai_text),
                })
            except:
                pass

        return jsonify({
            "reply": not_found_reply,
            "source": "Term not found in database",
            "confidence": None,
            "confidence_label": None,
            "readability": None,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print(f"\n✓ DB terms: {len(db_terms)}")
    print("Server: http://localhost:5500")
    app.run(debug=True, host="0.0.0.0", port=5500)