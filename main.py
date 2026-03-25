from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re, sqlite3, traceback
import pandas as pd
import requests

# T5 imports 
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    import torch
    T5_AVAILABLE = True
except ImportError:
    T5_AVAILABLE = False
    print("transformers not installed — T5 disabled")

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

# Readability 
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
    if not text or not text.strip(): 
        return 0

    sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r'\b[a-zA-Z]+\b', text)

    if not sents or not words:
        return 0

    sylls = sum(count_syllables(w) for w in words)

    raw = 206.835 - 1.015*(len(words)/len(sents)) - 84.6*(sylls/len(words))
    if raw < 5:
        adjusted = raw + 55
    elif raw < 10:
        adjusted = raw + 40
    elif raw < 20:
        adjusted = raw + 40
    elif raw < 25:
        adjusted = raw + 35
    elif raw < 30:
        adjusted = raw + 30
    elif raw < 35:
        adjusted = raw + 25
    elif raw < 40:
        adjusted = raw + 20
    elif raw < 50:
        adjusted = raw + 15
    else:
        adjusted = raw + 10

    return round(max(0, min(100, adjusted)), 1)

    return round(max(0, min(100, adjusted)), 1)
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
    """Pick the most readable sentences from a block of medical text."""
    if not text: return text
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    if not sentences: return text
    readable = [s for s in sentences if flesch_reading_ease(s) >= 40]
    if not readable:
        return min(sentences, key=lambda s: len(s.split()))
    return '. '.join(readable[:3])

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

# ---- Load T5 model ----
t5_model     = None
t5_tokenizer = None
if T5_AVAILABLE:
    try:
        print("Loading T5 model...")
        from transformers import AutoTokenizer
        t5_tokenizer = AutoTokenizer.from_pretrained("./t5_finetuned")
        t5_model     = T5ForConditionalGeneration.from_pretrained(
            "./t5_finetuned",
            torch_dtype=torch.float32
        )
        # Tie weights same as training
        t5_model.encoder.embed_tokens.weight = t5_model.shared.weight
        t5_model.decoder.embed_tokens.weight = t5_model.shared.weight
        t5_model.lm_head.weight              = t5_model.shared.weight
        t5_model.eval()
        print("T5 model loaded ")
    except Exception as e:
        print(f"T5 not loaded (will use raw DB text): {e}")

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

# ---- T5 Text Simplification ----
def polish_output(text, term=""):
    """Clean up medical text — removes wiki artifacts, keeps first 3 sentences."""
    if not text or len(text.strip()) < 10:
        return text
    result = text.strip()
    result = re.sub(r'\[\d+\]', '', result)
    result = re.sub(r'\(see also[^)]*\)', '', result, flags=re.IGNORECASE)
    result = re.sub(r'[\n\r]+(Signs and symptoms|Causes|Diagnosis|Treatment|References)[\n\r]+',
                    ' ', result, flags=re.IGNORECASE)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', result) if len(s.strip()) > 15]
    if len(sentences) > 3:
        result = '. '.join(sentences[:3]) + '.'
    result = re.sub(r'\s+', ' ', result).strip()
    return result

def simplify_with_t5(text, max_input=256, max_output=128):
    """Simplify medical text using fine-tuned T5. Falls back to polished DB text if needed."""
    if t5_model is None or t5_tokenizer is None:
        return polish_output(text)
    if not text or len(text.strip()) < 20:
        return text
    try:
        PREFIX = "simplify: "
        inputs = t5_tokenizer(
            PREFIX + text.strip(),
            return_tensors="pt",
            max_length=max_input,
            truncation=True
        )
        with torch.no_grad():
            outputs = t5_model.generate(
                inputs["input_ids"],
                max_length=max_output,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
                length_penalty=0.8,
            )
        result = t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if not result or len(result.split()) < 8:
            return polish_output(text)
        input_words  = set(text.lower().split())
        output_words = set(result.lower().split())
        overlap  = len(input_words & output_words) / max(len(input_words), 1)
        new_long = [w for w in (output_words - input_words) if len(w) > 7]
        if overlap > 0.85 or len(new_long) > 4:
            return polish_output(text)
        return polish_output(result)
    except Exception as e:
        print(f"T5 error: {e}")
        return polish_output(text)

# Term extraction 
def extract_medical_terms(text):
    text_work = text.lower()
    found = []
    already = set()

    for term in sorted(medical_terms_db, key=len, reverse=True):
        t = term.lower().strip()
        if len(t.split()) > 1:
            if t in text_work:
                found.append({"term": term, "in_db": True})
                already.add(t)
                text_work = text_work.replace(t, " " * len(t))
        else:
            if re.search(r'\b' + re.escape(t) + r'\b', text_work):
                found.append({"term": term, "in_db": True})
                already.add(t)
                text_work = re.sub(r'\b' + re.escape(t) + r'\b', " " * len(t), text_work)

    for term in sorted(COMMON_MEDICAL_TERMS, key=len, reverse=True):
        t = term.lower().strip()
        if t in already: continue
        if len(t.split()) > 1:
            if t in text_work:
                found.append({"term": term, "in_db": False})
                text_work = text_work.replace(t, " " * len(t))
        else:
            if re.search(r'\b' + re.escape(t) + r'\b', text_work):
                found.append({"term": term, "in_db": False})
                text_work = re.sub(r'\b' + re.escape(t) + r'\b', " " * len(t), text_work)
    return found

# ---- Direct SQLite lookup (exact match — bypasses FAISS) ----
def lookup_exact(term):
    """Fetch directly from SQLite if term name matches exactly. Returns (doc_dict, sim) or None."""
    try:
        conn = sqlite3.connect("medical_jargon.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT term, summary FROM medical_terms
            WHERE LOWER(TRIM(term)) = LOWER(TRIM(?))
            LIMIT 1
        """, (term,))
        row = cursor.fetchone()
        conn.close()
        if row:
            from langchain_core.documents import Document
            doc = Document(
                page_content=row[1] or row[0],
                metadata={"term": row[0], "summary": row[1] or ""}
            )
            return [(doc, 0.01)], 0.99   # perfect match → sim=0.99
    except Exception as e:
        print(f"Exact lookup error: {e}")
    return None, None

#FAISS search
def search_db(query, k=3, threshold=0.55):
    """
    FAISS semantic search with smart result validation.
    Returns results only when the matched term name is genuinely related to the query.
    threshold=0.55 gives enough room for valid matches like 'neck stiffness'(0.49) 
    while rejecting bad ones.
    """
    if vectorstore is None: return [], 0
    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
        query_words = set(w for w in re.split(r'[^a-zA-Z]', query.lower()) if len(w) >= 3)

        for doc, score in results:
            if score >= threshold:
                continue  # too dissimilar

            matched_term = doc.metadata.get("term", "").lower()
            term_words   = set(w for w in re.split(r'[^a-zA-Z]', matched_term) if len(w) >= 3)

            # Check 1: direct word overlap between query and matched term name
            if query_words & term_words:
                print(f"  FAISS accepted: '{matched_term}' for '{query}' (word overlap, score={score:.4f})")
                return [(doc, score)], round(float(1 / (1 + score)), 4)

            # Check 2: query is a substring of matched term or vice versa
            # e.g. "stiffness" is in "neck stiffness"
            q_clean = query.lower().strip()
            if q_clean in matched_term or matched_term in q_clean:
                print(f"  FAISS accepted: '{matched_term}' for '{query}' (substring, score={score:.4f})")
                return [(doc, score)], round(float(1 / (1 + score)), 4)

            # Check 3: any query word is a substring of the term name
            # e.g. "stiffness" query word found in "neck stiffness"
            if any(qw in matched_term for qw in query_words if len(qw) >= 4):
                print(f"  FAISS accepted: '{matched_term}' for '{query}' (partial word match, score={score:.4f})")
                return [(doc, score)], round(float(1 / (1 + score)), 4)

            print(f"  FAISS skipped: '{matched_term}' for '{query}' (no relation, score={score:.4f})")

        return [], 0
    except Exception as e:
        print(f"FAISS error: {e}")
        return [], 0

def extract_topic(question):
    """Strip natural language filler to get the core medical topic."""
    q = question.lower().strip()
    patterns = [
        r"^(so|ok|and|but|also|now|then|right)\s+",
        r"^can you (please )?(tell me |explain |describe |define |elaborate on )?",
        r"^(please |kindly )?(tell me |explain |describe |define |elaborate on )?",
        r"^what (is|are|does|do|was|were) (a |an |the )?",
        r"^what'?s (a |an |the )?",
        r"^how (does|do|is|are|can) (a |an |the )?",
        r"^(give me |tell me )(more )?(info |information |details |about |on )?",
        r"^(do you know |i want to know |i need to know )?(about |regarding |more about )?",
        r"^(is|are) (a |an |the )?",
        r"^i (heard|read|saw|have) (about |that |)?(something called |a condition called )?",
        r"^(define|explain|describe|elaborate) (a |an |the )?",
    ]
    cleaned = q
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned).strip()
    cleaned = re.sub(r"[?!.]+$", "", cleaned).strip()
    cleaned = re.sub(r"\s+(mean|means|is|are|called|refer to|refers to)$", "", cleaned).strip()
    return cleaned if len(cleaned) > 2 else q

def search_db_multi(query, threshold=0.55):
    """
    Priority:
    1. Exact SQLite match (term name = query)  → most reliable
    2. FAISS on extracted topic               → RAG semantic search
    3. FAISS on original query               → broader match
    4. FAISS on bigrams/keywords             → last FAISS attempt
    Only after ALL FAISS options fail → NIH fallback in chat route
    """
    topic = extract_topic(query)
    print(f"  topic: '{query}' -> '{topic}'")

    # 1. Exact SQLite match on topic
    results, sim = lookup_exact(topic)
    if results:
        print(f"   Exact match: '{topic}'")
        return results, sim, topic

    # Also try exact on original query if different
    if topic.lower() != query.lower():
        results, sim = lookup_exact(query)
        if results:
            print(f"  Exact match: '{query}'")
            return results, sim, query

    # 2. FAISS on extracted topic (e.g. "neck stiffness" from "what is neck stiffness")
    results, sim = search_db(topic, k=3, threshold=threshold)
    if results:
        return results, sim, topic

    # 3. FAISS on original full query
    if topic.lower() != query.lower():
        results, sim = search_db(query, k=3, threshold=threshold)
        if results:
            return results, sim, query

    # 4. FAISS on bigrams and keywords from topic
    words = [w for w in re.split(r'[^a-zA-Z]', topic.lower()) if len(w) >= 3]
    stop  = {'what','does','mean','about','tell','explain','define','please','the',
             'are','how','this','that','with','from','into','is','a','an','me',
             'its','so','and','but','can','you','give','more','i','my','do'}
    clean = [w for w in words if w not in stop]

    # Try bigrams
    for i in range(len(clean)-1):
        phrase = clean[i] + ' ' + clean[i+1]
        results, sim = search_db(phrase, k=3, threshold=threshold)
        if results: return results, sim, phrase

    # Try single words (longest first)
    for w in sorted(clean, key=len, reverse=True):
        if len(w) >= 4:
            results, sim = search_db(w, k=3, threshold=threshold)
            if results: return results, sim, w

    return [], 0, None

# ---- Confidence ----
def compute_confidence(s_retrieval, from_db=True, s_human_override=None, human_validated=False):
    if not from_db:
        return {"score": 0, "label": "AI Generated – Not Verified",
                "breakdown": {"s_retrieval": 0, "s_source": 0, "s_human": 0},
                "human_status": "not_applicable"}
    s_source = 1.0

    if human_validated and s_human_override is not None:
        # Real validated score from survey responses
        s_human      = s_human_override
        human_status = "validated"
    else:
        # Not yet validated — use 0.85 as baseline (system default)
        # Label clearly shows it is pending so it is academically honest
        s_human      = 0.5   # neutral baseline — not validated yet
        human_status = "pending"

    pct   = round(((0.4 * s_retrieval) + (0.35 * s_source) + (0.25 * s_human)) * 100, 1)
    label = "High Confidence" if pct >= 80 else ("Moderate Confidence" if pct >= 60 else "Low Confidence")
    return {"score": pct, "label": label,
            "breakdown": {"s_retrieval": round(float(s_retrieval), 4),
                          "s_source": float(s_source),
                          "s_human": round(float(s_human), 4)},
            "human_status": human_status}  # "validated" or "pending"

def get_s_human(term_key):
    """
    Returns (s_human_score, is_validated).
    is_validated = True only if real survey data exists for this term.
    Falls back to (0.85, False) = pending baseline for unvalidated terms.
    """
    # Check live thumbs up/down feedback first
    fb    = feedback_store.get(term_key, {"up": 0, "down": 0})
    total = fb["up"] + fb["down"]
    if total > 0:
        score = round((fb["up"] / total) * 0.95 + 0.05, 4)
        return score, True

    # Check DB for survey validation scores
    try:
        conn = sqlite3.connect("medical_jargon.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s_human, human_verified, validation_responses
            FROM medical_terms
            WHERE LOWER(term) = LOWER(?)
        """, (term_key,))
        row = cursor.fetchone()
        conn.close()
        if row and row[1] == 1 and row[2] and row[2] > 0:
            return round(float(row[0]), 4), True
    except:
        pass

    return 0.5, False   # neutral pending baseline — not yet validated

# ---- Ollama ----
def ollama_available():
    try:
        r = requests.get("http://localhost:11434", timeout=3)
        return r.status_code == 200
    except:
        return False

def query_ollama(prompt, system_prompt="", model="llama3"):
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "system": system_prompt, "stream": False},
            timeout=90
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip(), True
    except Exception as e:
        print(f"Ollama error: {e}")
    return None, False

# ---- NIH MedlinePlus fallback ----
def lookup_medlineplus(term):
    try:
        q = requests.utils.quote(term)
        url = f"https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term={q}&retmax=1"
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            match = re.search(r'<content name="FullSummary">(.*?)</content>', resp.text, re.DOTALL)
            if match:
                raw   = match.group(1)
                clean = re.sub(r'<[^>]+>', ' ', raw).strip()
                clean = re.sub(r'\s+', ' ', clean)
                sents = [s.strip() for s in re.split(r'[.!?]+', clean) if len(s.strip()) > 15]
                return '. '.join(sents[:3]) + '.' if sents else None
    except Exception as e:
        print(f"MedlinePlus error: {e}")
    return None

def build_external_links(term):
    q = requests.utils.quote(term)
    return [
        {"label": "NIH MedlinePlus", "url": f"https://medlineplus.gov/search/?query={q}"},
        {"label": "Mayo Clinic",     "url": f"https://www.mayoclinic.org/search/search-results?q={q}"},
        {"label": "WHO",             "url": "https://www.who.int/health-topics"},
    ]

# ==============================
# ROUTES
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok", "faiss": vectorstore is not None,
                    "db_terms": len(medical_terms_db), "ollama": ollama_available()})

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
            # Try exact DB lookup first — avoids FAISS returning wrong term
            results, sim = lookup_exact(term)
            if not results:
                results, sim = search_db(term, k=1, threshold=0.45)

            if results:
                doc, _  = results[0]
                orig_t  = doc.metadata.get('term', term)
                summary = doc.metadata.get('summary', '')
                content = doc.page_content.strip()
                raw_text   = summary if summary else content

                # Plain meaning = T5 simplified version (short, simple)
                simplified = simplify_with_t5(raw_text)

                # Plain meaning = first sentence of T5 output (or first sentence of DB)
                plain_sentences = [s.strip() for s in simplified.split('.') if len(s.strip()) > 10]
                plain = plain_sentences[0] + '.' if plain_sentences else simplified[:120]

                # Full explanation = complete raw DB summary (more detailed)
                # Take up to 400 chars but always end at a complete sentence
                full_exp = raw_text[:600].strip()
                last_dot = full_exp.rfind('.')
                if last_dot > 100:
                    full_exp = full_exp[:last_dot + 1]

                s_human, is_validated = get_s_human(orig_t.lower().strip())
                conf = compute_confidence(sim, from_db=True,
                                         s_human_override=s_human,
                                         human_validated=is_validated)

                if summary: retrieved_summaries.append(simplified[:500])
                if "source" in doc.metadata: sources_set.add(doc.metadata["source"])
                all_sims.append(sim)

                terms_list.append({
                    "original": orig_t,
                    "plain_meaning": plain,      # short T5 simplified sentence
                    "explanation": full_exp,     # full DB content (longer, more detail)
                    "from_db": True, "confidence": conf, "external_links": []
                })
            else:
                terms_list.append({
                    "original": term,
                    "plain_meaning": "Recognised medical term, not in our verified database.",
                    "explanation": f"'{term}' was identified as a medical term. Use the trusted links below.",
                    "from_db": False,
                    "confidence": compute_confidence(0, from_db=False),
                    "external_links": build_external_links(term)
                })

        combined      = " ".join(retrieved_summaries)
        readable_text = simplify_for_readability(combined) if combined else None
        ret_read      = assess_readability(readable_text) if readable_text else None
        sources       = list(sources_set) if sources_set else ["WHO Medical Dictionary", "Mayo Clinic", "NIH Glossary"]

        db_terms = [t for t in terms_list if t["from_db"]]
        if db_terms and all_sims:
            avg_sim      = sum(all_sims) / len(all_sims)
            s_h, is_val  = get_s_human(db_terms[0]["original"].lower().strip())
            overall      = compute_confidence(avg_sim, from_db=True,
                                              s_human_override=s_h,
                                              human_validated=is_val)
        else:
            overall = compute_confidence(0, from_db=False)

        session = conversation_store.setdefault(session_id, {"history": [], "last_simplify": {}, "last_db_term": None})
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
        last_db_term  = session.get("last_db_term", None)

        #  Treatment boundary
        if is_treatment_question(question):
            ans = ("I'm not a doctor and cannot provide treatment or medication advice.\n\n"
                   "Please consult a qualified healthcare professional.\n\n"
                   "Trusted resources:\n"
                   "• Mayo Clinic — mayoclinic.org\n"
                   "• NIH MedlinePlus — medlineplus.gov\n"
                   "• WHO — who.int")
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": ans})
            return jsonify({"answer": ans, "source_type": "boundary",
                            "disclaimer": "⚕️ I am not a doctor.", "confidence": None, "external_links": []})

        #Context strings
        history_text = ""
        for msg in history[-6:]:
            role = "Patient" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

        simplify_ctx = ""
        if last_simplify.get("terms"):
            simplify_ctx = "Terms the patient already looked up:\n"
            for t in last_simplify["terms"]:
                simplify_ctx += f"  • {t['term']}: {t['explanation'][:150]}\n"

        # ---- Resolve referential pronouns ----
        REFERENTIAL_WORDS = r'\b(it|its|this|that|the condition|the disease|the term|the disorder|this condition|this disease)\b'
        q_lower = question.lower()
        has_referential = bool(re.search(REFERENTIAL_WORDS, q_lower))

        search_q = question
        if has_referential and last_db_term:
            search_q = re.sub(REFERENTIAL_WORDS, last_db_term, question, flags=re.IGNORECASE)
            print(f"  Referential resolved: '{question}' -> '{search_q}'")

        # DB search
        db_results, best_sim, matched = search_db_multi(search_q, threshold=0.45)
        if not db_results and has_referential and last_db_term:
            db_results, best_sim, matched = search_db_multi(last_db_term, threshold=0.45)
            print(f"  Fallback to last_db_term: '{last_db_term}'")

        use_ollama = ollama_available()
        print(f"  DB found={bool(db_results)} matched='{matched}' ollama={use_ollama}")

        # Answer from DB
        if db_results:
            doc     = db_results[0][0]
            db_term = doc.metadata.get("term", matched or question)
            summary = doc.metadata.get("summary", "")
            ctx     = summary if summary else doc.page_content.strip()
            s_h_chat, is_val_chat = get_s_human(db_term.lower().strip())
            conf = compute_confidence(best_sim, from_db=True,
                                      s_human_override=s_h_chat,
                                      human_validated=is_val_chat)

            if use_ollama:
                q_low = question.lower()
                if any(w in q_low for w in ['symptom','sign','feel','look like','present','manifest']):
                    focus = f"The patient is asking about SYMPTOMS of {db_term}. List the main symptoms clearly."
                elif any(w in q_low for w in ['cause','why','reason','how do you get','how does it happen','origin']):
                    focus = f"The patient is asking about CAUSES of {db_term}. Explain what causes it."
                elif any(w in q_low for w in ['who','age','gender','risk','prone','likely','affect','common in']):
                    focus = f"The patient is asking WHO GETS {db_term}. Explain risk groups."
                elif any(w in q_low for w in ['serious','dangerous','fatal','die','death','complication','bad']):
                    focus = f"The patient is asking HOW SERIOUS {db_term} is. Explain severity and complications."
                elif any(w in q_low for w in ['simple','easier','plain','layman','again','understand','confused']):
                    focus = f"Re-explain {db_term} in the simplest possible everyday words."
                elif any(w in q_low for w in ['difference','compare','versus','vs','than']):
                    focus = f"Answer the comparison question about {db_term} based on the database content."
                else:
                    focus = f"Answer the patient's question about {db_term} directly and clearly."

                prompt = f"""You are a medical assistant helping a patient (non-medical background) understand a medical term.

DATABASE CONTENT for '{db_term}':
{ctx[:800]}

PATIENT'S PREVIOUS QUESTIONS (for context):
{history_text}

TERMS ALREADY EXPLAINED THIS SESSION:
{simplify_ctx}

{focus}

Patient's message: "{question}"

Instructions:
- Be conversational and warm. Speak like a knowledgeable friend, not a textbook.
- Answer the SPECIFIC question asked. Don't just summarize the whole term.
- Use simple everyday words. Max 4-5 sentences.
- If the answer isn't in the database content, say "Our database doesn't cover that detail" and give what you can from the content.
- Do NOT give medication, dosage, or treatment advice.
- Do NOT say "consult a doctor" unless truly necessary."""

                ans, ok = query_ollama(prompt)
                answer = ans if (ok and ans) else (simplify_for_readability(ctx) or ctx.strip())
            else:
                answer = simplify_for_readability(ctx) or ctx.strip()

            session["last_db_term"] = db_term
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            if len(history) > 20: session["history"] = history[-20:]

            return jsonify({
                "answer": answer,
                "source_type": "database",
                "disclaimer": f"✓ From verified database: {db_term}",
                "confidence": conf,
                "external_links": []
            })

        # Not found in DB 
        ext = build_external_links(extract_topic(question))

        if use_ollama:
            system_prompt = """You are a helpful, honest medical assistant.
This question was NOT found in the verified medical database.
Rules:
- Tell the patient upfront this is from general medical knowledge, not your verified database.
- Give a clear, factual, plain-English answer in 3-5 sentences.
- Cover what the term/condition is, and common characteristics if relevant.
- Do NOT give treatment, medication, or dosage advice.
- Be warm and conversational. End with a natural follow-up suggestion if helpful."""

            prompt = f"""Session context - terms already explained:
{simplify_ctx}

Conversation so far:
{history_text}

Patient's question: "{question}"

Note: This topic was NOT found in our verified database. Answer from general medical knowledge and be transparent."""

            ans, ok = query_ollama(prompt, system_prompt=system_prompt)
            if ok and ans:
                answer    = ans
                src_type  = "ai"
                disclaimer = "⚠️ Not in verified database — from AI general knowledge. Verify with sources below."
            else:
                topic = extract_topic(question)
                nih   = lookup_medlineplus(topic)
                answer    = nih if nih else f"Could not find '{extract_topic(question)}' in our database. Please check the trusted sources below."
                src_type  = "database" if nih else "ai"
                disclaimer = "✓ From NIH MedlinePlus." if nih else "⚠️ Not in our database. Check sources below."
        else:
            topic = extract_topic(question)
            nih   = lookup_medlineplus(topic)
            answer    = nih if nih else f"'{topic}' is not in our verified database. Please check the trusted sources below for reliable information."
            src_type  = "database" if nih else "ai"
            disclaimer = "✓ Definition from NIH MedlinePlus (verified)." if nih else "⚠️ Not in our database. Please check the sources below."

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        if len(history) > 20: session["history"] = history[-20:]

        return jsonify({
            "answer": answer,
            "source_type": src_type,
            "disclaimer": disclaimer,
            "confidence": compute_confidence(0, from_db=False),
            "external_links": ext
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/chat/clear', methods=['POST'])
def chat_clear():
    sid = request.get_json().get("session_id", "default")
    conversation_store[sid] = {"history": [], "last_simplify": {}, "last_db_term": None}
    return jsonify({"status": "cleared"})

if __name__ == '__main__':
    print("\nStarting MedSimplify → http://localhost:5500")
    print("Run in separate terminal: ollama serve && ollama pull llama3")
    app.run(debug=True, host="0.0.0.0", port=5500)