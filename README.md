# Minor-Project

# Medical Jargon Simplification System
 
A web-based application that simplifies complex medical terminology into plain language for patients and the general public. The system combines a verified medical database, FAISS semantic search, a fine-tuned T5 model, and human validation to deliver accurate, readable explanations.
 
---
 
## Features
 
- **Medical Term Detection** — automatically extracts medical terms from any input text
- **Plain Language Simplification** — uses a fine-tuned T5 model to rewrite clinical text in simple English
- **FAISS Semantic Search** — retrieves the most relevant explanations from a verified database
- **Readability Scoring** — Flesch Reading Ease and Flesch-Kincaid Grade for every explanation
- **Confidence Score** — computed as `C = 0.4×S_retrieval + 0.35×S_source + 0.25×S_human`
- **Human Validation** — thumbs up/down feedback stored in SQLite; survey scores imported from annotators
- **Contextual Chatbot** — ask follow-up questions about any term; powered by Ollama (Llama3) with NIH MedlinePlus fallback
- **Treatment Boundary** — refuses medication/dosage advice and redirects to trusted sources
 
---
 
## Project Structure
 
```
Minor project/
│
├── main.py                      # Flask app — main backend (port 5500)
├── create_medical_db.py         # ETL pipeline + FastAPI search API (port 8000)
├── faiss_index_builder.py       # Builds FAISS vector index from dataset
├── import_human_validation.py   # Imports survey scores into database
│
├── datasets/
│   └── wiki_medical_terms.csv   # Source medical terms dataset (1,000 rows)
│
├── combine_response.csv         # Human evaluation survey responses (59 annotators)
│
├── faiss_index/                 # FAISS vector index (generated)
├── t5_finetuned/                # Fine-tuned T5 model folder
├── medical_jargon.db            # SQLite database (generated)
│
├── templates/
│   └── index.html               # Frontend UI
│
└── requirements.txt             # Python dependencies
```
 
---
 
## Installation
 
### 1. Clone the repository
```bash
git clone https://github.com/your-username/medical-jargon-simplification.git
cd "Minor project"
```
 
### 2. Create and activate virtual environment
```bash
python -m venv .venv
 
# Windows
.venv\Scripts\Activate.ps1
 
# Mac/Linux
source .venv/bin/activate
```
 
### 3. Install dependencies
```bash
pip install flask flask-cors fastapi uvicorn pandas sqlite3 langchain langchain-community faiss-cpu sentence-transformers transformers torch requests
```
 
---
 
## Setup & Running
 
### Step 1 — Build the database
```bash
python create_medical_db.py
```
Then visit: `http://127.0.0.1:8000/build-database`
 
### Step 2 — Build FAISS index
```bash
python faiss_index_builder.py
```
 
### Step 3 — Import human validation scores
```bash
python import_human_validation.py
```
 
### Step 4 — Start the Flask app
```bash
python main.py
```
App runs at: `http://localhost:5500`
 
### Step 5 (Optional) — Enable Ollama for AI responses
```bash
# In a separate terminal
ollama serve
ollama pull llama3
```
 
---
 
## API Endpoints
 
### Flask App (port 5500)
| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main UI |
| `/health` | GET | System status |
| `/simplify` | POST | Simplify medical text |
| `/chat` | POST | Chatbot question answering |
| `/feedback` | POST | Thumbs up/down validation |
| `/validation-stats` | GET | Human validation statistics |
| `/chat/clear` | POST | Clear chat session |
 
### FastAPI ETL (port 8000)
| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Database status |
| `/build-database` | GET | Run ETL pipeline |
| `/search?q=term` | GET | Search medical terms |
| `/docs` | GET | Swagger UI |
 
---
 
## Human Validation
 
The system uses a two-layer human validation approach:
 
**1. Survey-based annotation** (`combine_response.csv`)
- 59 annotators rated 20 cardiology term simplifications on a 1–5 scale
- Scores normalized to 0.0–1.0 and stored as `s_human` in the database
- Terms rated: Ventricular tachycardia, Pulmonary hypertension, Stroke, Angina, and 16 others
 
**2. Live thumbs up/down feedback**
- Users can rate any explanation directly in the app
- Votes update `s_human`, `human_verified`, `thumbs_up`, `thumbs_down` in real time
 
**Confidence Formula:**
```
C = 0.4 × S_retrieval + 0.35 × S_source + 0.25 × S_human
```
- `S_retrieval` — FAISS similarity score (how well the term matched the database)
- `S_source` — source faithfulness (1.0 for verified database)
- `S_human` — human validation score (from survey or thumbs feedback)
 
---
 
## Database Schema
 
```sql
CREATE TABLE medical_terms (
    term                TEXT NOT NULL,
    term_lower          TEXT,
    content             TEXT,
    summary             TEXT,
    content_length      INTEGER,
    extracted_date      TEXT,
    s_human             REAL    DEFAULT 0.5,
    human_verified      INTEGER DEFAULT 0,
    validation_responses INTEGER DEFAULT 0,
    thumbs_up           INTEGER DEFAULT 0,
    thumbs_down         INTEGER DEFAULT 0,
    last_validated      TEXT,
    UNIQUE(term)
)
```
 
---
 
## Technologies Used
 
| Component | Technology |
|---|---|
| Backend | Flask, FastAPI |
| Database | SQLite with FTS5 |
| Vector Search | FAISS + sentence-transformers (all-MiniLM-L6-v2) |
| Simplification Model | Fine-tuned T5 |
| AI Chatbot | Ollama (Llama3) |
| Fallback | NIH MedlinePlus API |
| Frontend | HTML, Tailwind CSS, JavaScript |
| ETL | Pandas |
 
---

 
---
 
