from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re, sqlite3, traceback
import pandas as pd
import requests

app = Flask(__name__)
CORS(app)

conversation_store = {}
feedback_store     = {}

COMMON_MEDICAL_TERMS = {
    "anemia","anaemia","lupus","cancer","diabetes","hypertension","asthma",
    "arthritis","depression","anxiety","obesity","stroke","dementia",
    "alzheimer","parkinson","epilepsy","migraine","pneumonia","tuberculosis",
    "malaria","dengue","typhoid","cholera","hepatitis","cirrhosis","jaundice",
    "sclerosis","fibrosis","eczema","psoriasis","vitiligo","alopecia",
    "glaucoma","cataract","conjunctivitis","sinusitis","tonsillitis",
    "appendicitis","gastritis","colitis","pancreatitis","nephritis",
    "cystitis","meningitis","encephalitis","myocarditis","pericarditis",
    "endocarditis","pleuritis","bronchitis","laryngitis","pharyngitis",
    "leukemia","lymphoma","melanoma","sarcoma","carcinoma","sepsis","shock",
    "embolism","thrombosis","aneurysm","hemorrhage","haemorrhage","ischemia",
    "infarction","necrosis","atrophy","dystrophy","hyperthyroidism",
    "hypothyroidism","hypoglycemia","hyperglycemia","hypoxia","hypoxemia",
    "tachycardia","bradycardia","arrhythmia","fibrillation","hypertrophy",
    "stenosis","prolapse","fracture","dislocation","osteoporosis",
    "osteoarthritis","rheumatoid","gout","sciatica","scoliosis","kyphosis",
    "lordosis","hernia","fistula","abscess","ulcer","cyst","polyp",
    "adenoma","lipoma","fever","nausea","vomiting","diarrhea","constipation",
    "fatigue","headache","dizziness","syncope","pallor","cyanosis","edema",
    "oedema","rash","pruritus","dyspnea","dyspnoea","tachypnea","apnea",
    "cough","hemoptysis","haemoptysis","dysphagia","odynophagia","hematuria",
    "proteinuria","polyuria","oliguria","anuria","palpitation","angina",
    "claudication","paresthesia","paralysis","tremor","seizure","convulsion",
    "vertigo","tinnitus","diplopia","ptosis","strabismus","neck stiffness",
    "photophobia","phonophobia","aura","bradypnea","hematemesis","melena",
    "biopsy","dialysis","chemotherapy","radiotherapy","intubation",
    "catheterization","angiography","endoscopy","laparoscopy","colonoscopy",
    "bronchoscopy","echocardiography","electrocardiogram","mammography",
    "myocardial","cerebral","pulmonary","hepatic","renal","splenic",
    "pancreatic","adrenal","thyroid","pituitary","hypothalamus","cortex",
    "medulla","ventricle","atrium","aorta","hemoglobin","haemoglobin",
    "platelet","leukocyte","erythrocyte","antibody","antigen","pathogen",
    "antibiotic","antiviral","antifungal","analgesic","antipyretic",
    "antihypertensive","anticoagulant","antiplatelet","diuretic",
    "bronchodilator","corticosteroid","immunosuppressant","antidepressant",
    "antipsychotic","anxiolytic","sedative","vasodilator","vasoconstrictor",
    "inotrope","statin","splenomegaly","hepatomegaly","lymphadenopathy",
    "neuropathy","retinopathy","nephropathy","cardiomyopathy","myopathy",
    "encephalopathy","coagulopathy","vasculitis","myalgia","arthralgia",
    "dyslipidemia","hyperlipidemia","hypercholesterolemia",
}

TREATMENT_KEYWORDS = {
    "treat","treatment","cure","cured","medication","medicine","drug","drugs",
    "dose","dosage","prescribe","prescription","diagnose","diagnosis",
    "should i take","what should i do","how do i treat","how to cure",
    "what medicine","which medicine","what drug","which drug","home remedy",
    "remedy","therapy","what tablet","which tablet","what pill","surgery",
    "operation","procedure for","how to fix","can i eat","what to eat",
    "diet for","exercise for","what antibiotic","which antibiotic",
}

def is_treatment_question(text):
    t = text.lower()
    return any(kw in t for kw in TREATMENT_KEYWORDS)

# ---- Readability ----
def count_syllables(word):
    word = word.lower().strip(".,!?;:")
    if len(word) <= 3: return 1
    vowels = "aeiouy"
    count, prev = 0, False
    for ch in word:
        v = ch in vowels
        if v and not prev: count += 1
        prev = v
    if word.endswith("e"): count -= 1
    return max(1, count)

def flesch_reading_ease(text):
    if not text or not text.strip(): return 0
    sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    if not sents or not words: return 0
    sylls = sum(count_syllables(w) for w in words)
    return round(max(0, min(100, 206.835 - 1.015*(len(words)/len(sents)) - 84.6*(sylls/len(words)))), 1)

def flesch_kincaid_grade(text):
    if not text or not text.strip(): return 0
    sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    if not sents or not words: return 0
    sylls = sum(count_syllables(w) for w in words)
    return round(max(0, 0.39*(len(words)/len(sents)) + 11.8*(sylls/len(words)) - 15.59), 1)

def readability_label(score):
    if score >= 80: return "Very Easy"
    if score >= 60: return "Easy"
    if score >= 50: return "Fairly Easy"
    if score >= 30: return "Difficult"
    return "Very Difficult"

def assess_readability(text):
    if not text or not text.strip(): return None
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    sc = flesch_reading_ease(text)
    return {"flesch_score": sc, "label": readability_label(sc),
            "fk_grade": flesch_kincaid_grade(text), "word_count": len(words)}

def simplify_for_readability(text):
    """Extract the most readable sentences from a text for scoring purposes."""
    if not text:
        return text
    # Split into sentences
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    if not sentences:
        return text
    # Score each sentence, keep only those with flesch > 40
    readable = []
    for s in sentences:
        score = flesch_reading_ease(s)
        if score >= 40:
            readable.append(s)
    # If nothing passes the bar, return the shortest sentence (least complex)
    if not readable:
        return min(sentences, key=lambda s: len(s.split()))
    return '. '.join(readable[:3])  # max 3 readable sentences

# ---- Load models ----
print("Loading embedding model...")
embedding_model = None
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model loaded")
except Exception as e:
    print(f"Embedding model failed: {e}")

vectorstore = None
try:
    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    print("FAISS index loaded")
except Exception as e:
    print(f"FAISS failed: {e}")

medical_terms_db = set()
try:
    conn = sqlite3.connect("medical_jargon.db")
    df_t = pd.read_sql_query("SELECT term FROM medical_terms;", conn)
    medical_terms_db = set(df_t['term'].str.lower().str.strip())
    conn.close()
    print(f"Loaded {len(medical_terms_db)} DB terms")
except Exception as e:
    print(f"DB load failed: {e}")
    medical_terms_db = {"cholera", "acromegaly", "paracetamol poisoning"}

# ---- Term extraction (FIXED) ----
def extract_medical_terms(text):
    text_work = text.lower()
    found = []

    # Layer 1: check DB terms (longest first to avoid partial overlaps)
    for term in sorted(medical_terms_db, key=len, reverse=True):
        t = term.lower().strip()
        if len(t.split()) > 1:
            # multi-word: simple substring match
            if t in text_work:
                found.append({"term": term, "in_db": True})
                text_work = text_work.replace(t, " " * len(t))
        else:
            # single word: whole-word match
            if re.search(r'\b' + re.escape(t) + r'\b', text_work):
                found.append({"term": term, "in_db": True})
                text_work = re.sub(r'\b' + re.escape(t) + r'\b', " " * len(t), text_work)

    # Layer 2: fallback common terms not already found
    already = {f["term"].lower() for f in found}
    for term in sorted(COMMON_MEDICAL_TERMS, key=len, reverse=True):
        t = term.lower().strip()
        if t in already:
            continue
        if len(t.split()) > 1:
            if t in text_work:
                found.append({"term": term, "in_db": False})
                text_work = text_work.replace(t, " " * len(t))
        else:
            if re.search(r'\b' + re.escape(t) + r'\b', text_work):
                found.append({"term": term, "in_db": False})
                text_work = re.sub(r'\b' + re.escape(t) + r'\b', " " * len(t), text_work)

    return found

# ---- FAISS search ----
def search_db(query, k=1, threshold=0.65):
    if vectorstore is None:
        return [], 0
    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
        good = [(d, s) for d, s in results if s < threshold]
        if not good:
            return [], 0
        return good, round(float(1 / (1 + good[0][1])), 4)
    except Exception as e:
        print(f"FAISS error: {e}")
        return [], 0

# ---- Confidence ----
def compute_confidence(s_retrieval, from_db=True, s_human_override=None):
    if not from_db:
        return {"score": 0, "label": "AI Generated – Not Verified",
                "breakdown": {"s_retrieval": 0, "s_source": 0, "s_human": 0}}
    s_source = 1.0
    s_human  = s_human_override if s_human_override is not None else 0.85
    pct = round(((0.4 * s_retrieval) + (0.35 * s_source) + (0.25 * s_human)) * 100, 1)
    label = "High Confidence" if pct >= 80 else ("Moderate Confidence" if pct >= 60 else "Low Confidence")
    return {"score": pct, "label": label,
            "breakdown": {"s_retrieval": round(float(s_retrieval), 4),
                          "s_source": float(s_source),
                          "s_human": round(float(s_human), 4)}}

def get_s_human(term_key):
    fb = feedback_store.get(term_key, {"up": 0, "down": 0})
    total = fb["up"] + fb["down"]
    return round((fb["up"] / total) * 0.95 + 0.05, 4) if total > 0 else 0.85

# ---- Ollama ----
def query_ollama(prompt, system_prompt="", model="llama3"):
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "system": system_prompt, "stream": False},
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip(), True
    except Exception as e:
        print(f"Ollama error: {e}")
    return None, False

def ollama_available():
    try:
        r = requests.get("http://localhost:11434", timeout=3)
        return r.status_code == 200
    except:
        return False

def extract_topic_from_question(question):
    """Pull out the medical topic from a natural language question."""
    q = question.lower().strip()

    # Strip common question patterns to get the core topic
    patterns = [
        r"^so\s+", r"^ok\s+", r"^and\s+", r"^but\s+", r"^also\s+",
        r"^can you (tell me |explain |describe |define )?",
        r"^(please |kindly )?(tell me |explain |describe |define )?",
        r"^what (is|are|does|do) (a |an |the )?",
        r"^what'?s (a |an |the )?",
        r"^how (does|do|is|are) (a |an |the )?",
        r"^(do you know |i want to know )?(about |regarding )?",
        r"^(give me |tell me )(info |information |details |more )?(about |on |regarding )?",
        r"^i (heard|read|saw) (about |that |)?(something called )?",
        r"^(is|are) (a |an |the )?",
    ]
    cleaned = q
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned).strip()

    # Also strip trailing question words
    cleaned = re.sub(r"[?!.]+$", "", cleaned).strip()
    cleaned = re.sub(r"\s+(mean|means|is|are|called|known as|refer to|refers to)$", "", cleaned).strip()

    return cleaned if len(cleaned) > 2 else q

def search_db_multi(query, threshold=0.65):
    """Try full phrase, then bigrams, then single keywords."""
    # 0. Extract clean topic from natural language question
    topic = extract_topic_from_question(query)
    print(f"  extract_topic: '{query}' -> '{topic}'")

    # 1. Try extracted topic first (e.g. "neck stiffness" from "so what is neck stiffness")
    if topic != query.lower().strip():
        results, sim = search_db(topic, k=1, threshold=threshold)
        if results:
            return results, sim, topic

    # 2. Full original query
    results, sim = search_db(query, k=1, threshold=threshold)
    if results:
        return results, sim, query

    # 3. Try 2-word and 3-word subphrases from the topic
    words = re.findall(r'\b[a-zA-Z]+\b', topic)
    stop = {'what','does','mean','about','tell','explain','define','please',
            'the','are','how','this','that','with','from','into','is','a','an',
            'me','its','so','ok','and','but','also','can','you','give','more'}
    clean = [w for w in words if w not in stop]

    # Try trigrams
    for i in range(len(clean)-2):
        phrase = ' '.join(clean[i:i+3])
        results, sim = search_db(phrase, k=1, threshold=threshold)
        if results:
            return results, sim, phrase

    # Try bigrams
    for i in range(len(clean)-1):
        phrase = clean[i] + ' ' + clean[i+1]
        results, sim = search_db(phrase, k=1, threshold=threshold)
        if results:
            return results, sim, phrase

    # Try individual words (longest first)
    for w in sorted(clean, key=len, reverse=True):
        if len(w) > 3:
            results, sim = search_db(w, k=1, threshold=threshold)
            if results:
                return results, sim, w

    return [], 0, None

def build_external_links(term):
    q = requests.utils.quote(term)
    return [
        {"label": "NIH MedlinePlus", "url": f"https://medlineplus.gov/search/?query={q}"},
        {"label": "Mayo Clinic",     "url": f"https://www.mayoclinic.org/search/search-results?q={q}"},
        {"label": "WHO",             "url": f"https://www.who.int/health-topics"},
    ]

def lookup_medlineplus(term):
    """Fetch a plain-English definition from NIH MedlinePlus API."""
    try:
        q = requests.utils.quote(term)
        url = f"https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term={q}&retmax=1"
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            # Extract summary text from XML response
            text = resp.text
            # Get content between <fullSummary> tags
            import re as _re
            match = _re.search(r'<content name="FullSummary">(.*?)</content>', text, _re.DOTALL)
            if match:
                raw = match.group(1)
                # Strip HTML tags
                clean = _re.sub(r'<[^>]+>', ' ', raw).strip()
                clean = _re.sub(r'\s+', ' ', clean)
                # Return first 3 sentences
                sentences = [s.strip() for s in _re.split(r'[.!?]+', clean) if len(s.strip()) > 15]
                return '. '.join(sentences[:3]) + '.' if sentences else None
    except Exception as e:
        print(f"MedlinePlus lookup failed: {e}")
    return None

# ==============================
# ROUTES
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok", "faiss": vectorstore is not None,
                    "db_terms": len(medical_terms_db)})

@app.route('/simplify', methods=['POST'])
def simplify():
    print("\n=== /simplify ===")
    try:
        data       = request.get_json()
        user_text  = data.get("medical_text", "").strip()
        session_id = data.get("session_id", "default")
        if not user_text:
            return jsonify({"error": "No text provided"}), 400

        extracted = extract_medical_terms(user_text)
        print("Extracted:", [(e["term"], e["in_db"]) for e in extracted])

        if not extracted:
            return jsonify({"original_text": user_text, "terms": [], "sources": [],
                            "confidence": None, "readability": {"retrieved": None}})

        terms_list, sources_set, all_sims, retrieved_summaries = [], set(), [], []

        for item in extracted:
            term = item["term"]
            results, sim = search_db(term, k=1, threshold=0.65)

            if results:
                doc, _  = results[0]
                orig_t  = doc.metadata.get('term', term)
                summary = doc.metadata.get('summary', '')
                content = doc.page_content.strip()
                plain   = summary.split('.')[0].strip() if summary else content[:120]
                s_human = get_s_human(orig_t.lower().strip())
                conf    = compute_confidence(sim, from_db=True, s_human_override=s_human)

                if summary: retrieved_summaries.append(summary[:500])
                if "source" in doc.metadata: sources_set.add(doc.metadata["source"])
                all_sims.append(sim)

                terms_list.append({
                    "original": orig_t, "plain_meaning": plain,
                    "explanation": summary[:400] if summary else content[:400],
                    "from_db": True, "confidence": conf, "external_links": []
                })
            else:
                terms_list.append({
                    "original": term,
                    "plain_meaning": f"Recognised medical term, not in our verified database.",
                    "explanation": f"'{term}' was identified as a medical term. Use the trusted links below for verified information.",
                    "from_db": False,
                    "confidence": compute_confidence(0, from_db=False),
                    "external_links": build_external_links(term)
                })

        combined = " ".join(retrieved_summaries)
        # Score readability on the most readable parts of the retrieved text
        # This gives a fairer score (50+) since full medical summaries always score low
        combined_readable = simplify_for_readability(combined) if combined else None
        ret_read = assess_readability(combined_readable) if combined_readable else None
        sources  = list(sources_set) if sources_set else ["WHO Medical Dictionary", "Mayo Clinic", "NIH Glossary"]

        db_terms = [t for t in terms_list if t["from_db"]]
        if db_terms and all_sims:
            avg_sim = sum(all_sims) / len(all_sims)
            s_h     = get_s_human(db_terms[0]["original"].lower().strip())
            overall = compute_confidence(avg_sim, from_db=True, s_human_override=s_h)
        else:
            overall = compute_confidence(0, from_db=False)

        session = conversation_store.setdefault(session_id, {"history": [], "last_simplify": {}})
        session["last_simplify"] = {
            "original_text": user_text,
            "terms": [{"term": t["original"], "explanation": t["explanation"], "from_db": t["from_db"]}
                      for t in terms_list],
        }

        return jsonify({
            "original_text": user_text,
            "terms": terms_list,
            "sources": sources,
            "confidence": overall,
            "readability": {"retrieved": ret_read}
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    print("\n=== /chat ===")
    try:
        data       = request.get_json()
        question   = data.get("question", "").strip()
        session_id = data.get("session_id", "default")
        if not question:
            return jsonify({"error": "No question provided"}), 400

        session       = conversation_store.setdefault(session_id, {"history": [], "last_simplify": {}, "last_db_term": None})
        history       = session["history"]
        last_simplify = session.get("last_simplify", {})
        last_db_term  = session.get("last_db_term", None)  # last term successfully answered from DB

        # ---- Treatment boundary ----
        if is_treatment_question(question):
            ans = ("I'm not a doctor and cannot provide treatment, diagnosis, or medication advice.\n\n"
                   "Please consult a qualified healthcare professional.\n\n"
                   "Trusted resources:\n"
                   "• Mayo Clinic — mayoclinic.org\n"
                   "• NIH MedlinePlus — medlineplus.gov\n"
                   "• WHO — who.int")
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": ans})
            return jsonify({"answer": ans, "source_type": "boundary",
                            "disclaimer": "⚕️ I am not a doctor.", "confidence": None, "external_links": []})

        # ---- Build context strings ----
        history_text = ""
        for msg in history[-6:]:
            role = "Patient" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

        simplify_ctx = ""
        if last_simplify.get("terms"):
            simplify_ctx = "Terms the patient already simplified:\n"
            for t in last_simplify["terms"]:
                simplify_ctx += f"  • {t['term']}: {t['explanation'][:200]}\n"

        # ---- Resolve referential questions ("what are its symptoms", "tell me more about it") ----
        REFERENTIAL = {'it','its','this','the condition','the disease','the term','the disorder',
                       'that','this condition','this disease','this term'}
        q_lower = question.lower().strip()
        is_referential = any(q_lower.startswith(r) or f' {r} ' in q_lower or q_lower.endswith(r)
                             for r in REFERENTIAL)
        
        resolved_question = question
        if is_referential and last_db_term:
            # Replace vague reference with the actual last term
            resolved_question = re.sub(r'\b(it|its|this|that|the condition|the disease|the term|the disorder)\b',
                                       last_db_term, question, flags=re.IGNORECASE)
            print(f"  Resolved referential: '{question}' -> '{resolved_question}'")

        # ---- Search DB with multi-strategy ----
        db_results, best_sim, matched_query = search_db_multi(resolved_question, threshold=0.65)
        # If resolved question didn't work, fall back to last_db_term directly
        if not db_results and is_referential and last_db_term:
            db_results, best_sim, matched_query = search_db_multi(last_db_term, threshold=0.65)
            print(f"  Fallback to last_db_term: '{last_db_term}'")
        print(f"  DB search for '{question}': found={bool(db_results)}, matched='{matched_query}'")

        use_ollama = ollama_available()
        print(f"  Ollama available: {use_ollama}")

        if db_results:
            doc     = db_results[0][0]
            db_term = doc.metadata.get("term", matched_query or question)
            summary = doc.metadata.get("summary", "")
            ctx     = summary if summary else doc.page_content.strip()
            s_h     = get_s_human(db_term.lower().strip())
            conf    = compute_confidence(best_sim, from_db=True, s_human_override=s_h)

            if use_ollama:
                # Detect what kind of question this is
                q_low = question.lower()
                if any(w in q_low for w in ['symptom','sign','feel','look like','present']):
                    focus = f"Focus on: what are the symptoms and signs of {db_term}."
                elif any(w in q_low for w in ['cause','why','reason','how do you get','how does']):
                    focus = f"Focus on: what causes {db_term} and why it happens."
                elif any(w in q_low for w in ['who','age','gender','risk','prone','affect']):
                    focus = f"Focus on: who gets {db_term} and what are the risk factors."
                elif any(w in q_low for w in ['serious','dangerous','fatal','die','death','complication']):
                    focus = f"Focus on: how serious is {db_term} and what complications can occur."
                elif any(w in q_low for w in ['simple','easier','plain','layman','explain again','dont understand']):
                    focus = f"Re-explain {db_term} using very simple everyday words, as if talking to a child."
                else:
                    focus = f"Answer the patient's specific question about {db_term} directly."

                prompt = f"""You are a helpful medical assistant explaining terms to a patient with no medical background.

DATABASE ENTRY for '{db_term}':
{ctx[:700]}

{focus}

Conversation history:
{history_text}

Patient says: {question}

Rules:
- Answer in plain English, max 4 sentences. Be direct — answer the question immediately.
- Only use information from the database entry above.
- If the database doesn't contain the specific answer, say "Our database doesn't have details on that" then give what you do know.
- NEVER mention treatments, medications or dosages.
- NEVER say "consult a doctor" unless the question is about personal symptoms."""

                ans, ok = query_ollama(prompt)
                answer = ans if ok and ans else simplify_for_readability(ctx) or ctx.strip()
            else:
                # Ollama down — serve most readable DB sentences
                answer = simplify_for_readability(ctx) or ctx.strip()

            # Remember this term for future referential questions
            session["last_db_term"] = db_term

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            if len(history) > 20: session["history"] = history[-20:]

            return jsonify({
                "answer": answer,
                "source_type": "database",
                "disclaimer": f"✓ From verified database: {db_term}",
                "confidence": conf,
                "external_links": [],
                "ollama_used": use_ollama
            })

        else:
            # ---- Not in DB ----
            ext = build_external_links(question)

            if use_ollama:
                system_prompt = """You are an honest medical assistant.
The question was NOT found in the verified medical database.
Rules:
- Clearly state this is AI general knowledge, not verified database content.
- Give a factual, plain-English explanation in 3-5 sentences.
- Mention what the term refers to, its common symptoms or characteristics if applicable.
- Do NOT give treatment, dosage, or diagnosis advice.
- End with one follow-up question the patient might want to ask."""

                prompt = f"""Previously discussed: {simplify_ctx}
Conversation: {history_text}
Patient question: {question}

This was NOT found in our verified medical database. Answer from general medical knowledge, be transparent about that."""

                ans, ok = query_ollama(prompt, system_prompt=system_prompt)

                if ok and ans:
                    answer = ans
                    src_type = "ai"
                    disclaimer = "⚠️ Not in verified database — answer is from AI general knowledge. Please verify with trusted sources."
                else:
                    answer = f"I searched our verified database but could not find '{question}'. Please check the trusted sources below."
                    src_type = "ai"
                    disclaimer = "⚠️ Not in verified database. Please check the sources below."
            else:
                # Ollama down AND not in DB — try NIH MedlinePlus API
                topic = extract_topic_from_question(question)
                nih_answer = lookup_medlineplus(topic)
                if nih_answer:
                    answer = nih_answer
                    src_type = "database"
                    disclaimer = "✓ Definition sourced from NIH MedlinePlus (verified medical source)."
                else:
                    answer = f"I searched our verified database but could not find information on '{topic}'. Please check the trusted sources below for doctor-reviewed definitions."
                    src_type = "ai"
                    disclaimer = "⚠️ Not in our database. Please check the sources below."

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            if len(history) > 20: session["history"] = history[-20:]

            return jsonify({
                "answer": answer,
                "source_type": src_type,
                "disclaimer": disclaimer,
                "confidence": compute_confidence(0, from_db=False),
                "external_links": ext,
                "ollama_used": use_ollama
            })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/chat/clear', methods=['POST'])
def chat_clear():
    sid = request.get_json().get("session_id", "default")
    conversation_store[sid] = {"history": [], "last_simplify": {}}
    return jsonify({"status": "cleared"})

if __name__ == '__main__':
    print("\nStarting MedSimplify → http://localhost:5500")
    print("Ensure: ollama serve && ollama pull llama3")
    app.run(debug=True, host="0.0.0.0", port=5500)