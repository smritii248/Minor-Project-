"""
===========================================================
  FAISS INDEX BUILDER — Medical Jargon Simplification
===========================================================
Run this script (or paste as cells in your notebook)
AFTER building df_final to generate the faiss_index folder.

Requirements:
    pip install langchain langchain-community faiss-cpu sentence-transformers
===========================================================
"""

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# ── 1. Re-build df_final (skip if already in notebook memory) ──────────────
# If running standalone, load your saved CSV:
# df_final = pd.read_csv("final_dataset.csv")

# ── 2. Drop rows with nulls in either column ──────────────────────────────
df_final = df_final.dropna(subset=["medical", "simple"]).reset_index(drop=True)
print(f"Total rows after cleaning: {len(df_final)}")

# ── 3. Build LangChain Documents ──────────────────────────────────────────
#    Each document stores BOTH the medical text (as content to embed)
#    AND the simple translation (as metadata for retrieval).
documents = []
for _, row in df_final.iterrows():
    doc = Document(
        page_content=row["medical"],          # what gets embedded & searched
        metadata={"simple": row["simple"]}    # returned alongside results
    )
    documents.append(doc)

print(f"Created {len(documents)} documents.")

# ── 4. Load embedding model ───────────────────────────────────────────────
print("Loading embedding model (this may take ~1 min first time)...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ── 5. Build FAISS index ──────────────────────────────────────────────────
print("Building FAISS index (may take a few minutes)...")
vectorstore = FAISS.from_documents(documents, embedding_model)

# ── 6. Save to disk (creates ./faiss_index/ folder) ──────────────────────
vectorstore.save_local("faiss_index")
print("✅ faiss_index/ folder saved successfully!")
print("   → Copy this folder next to main_flask_.py before running Flask.")
