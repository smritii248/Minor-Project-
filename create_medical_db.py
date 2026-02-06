#ETL
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict, Any
import logging
import sys

CSV_PATH     = Path("datasets") / "wiki_medical_terms.csv"
DB_PATH      = Path("medical_jargon.db")
TABLE_NAME   = "medical_terms"
FTS_TABLE    = "medical_fts"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("medical_etl.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)



def extract():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV file not found → {CSV_PATH}")

    logger.info(f"Reading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows | columns: {list(df.columns)}")
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting data transformation...")

    # Remove junk columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Rename columns (flexible)
    rename_map = {
        'page_title': 'term', 'title': 'term',
        'page_text': 'content', 'text': 'content', 'definition': 'content'
    }
    df.rename(columns=rename_map, inplace=True)

    # Fallback naming
    if 'term' not in df.columns and len(df.columns) > 0:
        df.rename(columns={df.columns[0]: 'term'}, inplace=True)
    if 'content' not in df.columns and len(df.columns) > 1:
        df.rename(columns={df.columns[1]: 'content'}, inplace=True)

    # Clean text
    if 'term' in df.columns:
        df['term'] = df['term'].astype(str).str.strip()
    if 'content' in df.columns:
        df['content'] = df['content'].astype(str).str.strip()

    # Enrich
    if 'term' in df.columns:
        df['term_lower'] = df['term'].str.lower()
    if 'content' in df.columns:
        df['content_length'] = df['content'].str.len()
        df['extracted_date'] = datetime.now().strftime("%Y-%m-%d")

        def summary(text):
            if pd.isna(text) or not text.strip():
                return ""
            parts = str(text).split(". ", 2)
            return ". ".join(parts[:2]) + "." if len(parts) >= 2 else str(text)[:400] + "..."

        df['summary'] = df['content'].apply(summary)

    # Clean rows
    if 'content_length' in df.columns:
        df = df[df['content_length'] > 50]
    if 'term' in df.columns:
        df = df.drop_duplicates(subset=['term'])
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Transformed → {len(df):,} rows | columns: {list(df.columns)}")
    return df


def load(df: pd.DataFrame):
    logger.info(f"Creating/updating database: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

        # FTS5 index
        try:
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} USING fts5(
                    term, term_lower, content, summary
                )
            """)
            conn.execute(f"""
                INSERT INTO {FTS_TABLE}(rowid, term, term_lower, content, summary)
                SELECT rowid, term, term_lower, content, summary FROM {TABLE_NAME}
            """)
            logger.info("FTS5 index created")
        except Exception as e:
            logger.warning(f"FTS5 index creation skipped: {e}")

    logger.info(f"Database ready: {DB_PATH}")


def run_etl():
    try:
        logger.info("ETL pipeline started")
        df_raw = extract()
        df_clean = transform(df_raw)
        load(df_clean)
        logger.info("ETL completed successfully")
        print("\nDatabase created/updated successfully!")
        print(f"→ {DB_PATH}")
        print(f"→ Table: {TABLE_NAME} ({len(df_clean):,} rows)")
    except Exception as e:
        logger.error(f"ETL failed: {str(e)}", exc_info=True)
        print(f"\nERROR: {str(e)}")
        sys.exit(1)



app = FastAPI(
    title="MediClare - Medical Terms API",
    description="Simple API to query medical terms database",
    version="0.1"
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "database_exists": DB_PATH.exists(),
        "table_exists": False  # we'll check below
    }


@app.get("/build-database")
def build_database():
    run_etl()
    return {"message": "Database built / updated successfully"}


@app.get("/search", response_model=List[Dict[str, Any]])
def search(
    q: str = Query(..., min_length=2, description="Search term"),
    limit: int = Query(5, ge=1, le=20)
):
    if not DB_PATH.exists():
        raise HTTPException(503, "Database not found. Run /build-database first")

    conn = sqlite3.connect(DB_PATH)

    try:
        # Prefer FTS if exists
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        if FTS_TABLE in tables:
            query = f"""
                SELECT term, summary, content_length
                FROM {FTS_TABLE}
                WHERE {FTS_TABLE} MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            params = (q, limit)
        else:
            query = f"""
                SELECT term, summary, content_length
                FROM {TABLE_NAME}
                WHERE term_lower LIKE ? OR content LIKE ?
                LIMIT ?
            """
            params = (f"%{q.lower()}%", f"%{q}%", limit)

        df_result = pd.read_sql_query(query, conn, params=params)
        return df_result.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(500, f"Search error: {str(e)}")
    finally:
        conn.close()


@app.get("/")
def root():
    return {
        "message": "MediClare Medical Terms API",
        "endpoints": {
            "/health": "Check status",
            "/build-database": "Build or rebuild the database from CSV",
            "/search?q=hypertension": "Search medical terms",
            "/docs": "Interactive Swagger UI"
        }
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting FastAPI server...")
    print("First time? Visit: http://127.0.0.1:8000/build-database")
    print("Then try: http://127.0.0.1:8000/search?q=hypertension")
    print("Docs: http://127.0.0.1:8000/docs\n")

    uvicorn.run(
        "create_medical_db:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )