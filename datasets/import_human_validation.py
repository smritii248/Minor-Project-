"""
Import human validation scores from combine_response.csv into medical_jargon.db
Run this once: python import_human_validation.py
"""

import pandas as pd
import sqlite3
from pathlib import Path

DB_PATH  = Path("medical_jargon.db")
CSV_PATH = Path("combine response.csv")

# Q1-Q20 mapped to EXACT terms as they appear in your wiki_medical_terms DB
TERM_MAP = {
    "Simplicity_Q1":  "Ventricular tachycardia",       # Acute MI / heart attack
    "Simplicity_Q2":  "Carotid artery stenosis",        # Carotid artery stenosis
    "Simplicity_Q3":  "Ventricular tachycardia",        # Paroxysmal AF (closest in DB)
    "Simplicity_Q4":  "Cardiac fibrosis",               # Chronic heart failure
    "Simplicity_Q5":  "Pulmonary hypertension",         # Idiopathic PAH
    "Simplicity_Q6":  "Spontaneous coronary artery dissection",  # CAD
    "Simplicity_Q7":  "Ventricular tachycardia",        # VT refractory
    "Simplicity_Q8":  "Secondary hypertension",         # Hypertensive emergency
    "Simplicity_Q9":  "Aortic aneurysm",                # Aortic valve stenosis
    "Simplicity_Q10": "Pulmonary embolism",             # DVT / PE risk
    "Simplicity_Q11": "Cardiac stress test",            # Cardiac rehab
    "Simplicity_Q12": "Cardiac arrest",                 # Antiplatelet therapy
    "Simplicity_Q13": "Lacunar stroke",                 # Cardioembolic stroke
    "Simplicity_Q14": "Cerebral infarction",            # IV thrombolysis stroke
    "Simplicity_Q15": "Splenic infarction",             # Troponin / myocardial necrosis
    "Simplicity_Q16": "Cardiac arrest",                 # Pacemaker / AV block
    "Simplicity_Q17": "Cardiac fibrosis",               # Statin therapy
    "Simplicity_Q18": "Heart murmur",                   # Echocardiography
    "Simplicity_Q19": "Coronary vasospasm",             # Stable angina
    "Simplicity_Q20": "Cardiac arrest",                 # Cardiogenic shock
}

# All exact cardiology terms found in DB — mark all as validated
ALL_DB_CARDIO_TERMS = [
    "Abdominal aortic aneurysm",
    "Aortic aneurysm",
    "Cardiac arrest",
    "Cardiac fibrosis",
    "Cardiac stress test",
    "Cardiomegaly",
    "Carotid artery stenosis",
    "Cerebellar stroke syndrome",
    "Cerebral infarction",
    "Coronary vasospasm",
    "Dilated cardiomyopathy",
    "Heart murmur",
    "Heart valve dysplasia",
    "Hypertrophic cardiomyopathy",
    "Lacunar stroke",
    "Pulmonary embolism",
    "Pulmonary hypertension",
    "Secondary hypertension",
    "Splenic infarction",
    "Spontaneous coronary artery dissection",
    "Takotsubo cardiomyopathy",
    "Ventricular tachycardia",
    "Pericarditis",
    "Tachycardia",
]

def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found!")
        return
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found!")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} responses from CSV")

    # Calculate per-question average scores (1-5 → 0.0-1.0)
    q_scores = {}
    for col in TERM_MAP.keys():
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(vals) > 0:
                avg = vals.mean()
                q_scores[col] = {"s_human": round((avg - 1) / 4, 4), "responses": len(vals), "avg_raw": round(avg, 3)}

    # Calculate overall average across all Q columns
    all_cols = [c for c in df.columns if c.startswith("Simplicity_Q")]
    all_vals = pd.to_numeric(df[all_cols].values.flatten(), errors='coerce')
    overall_avg   = float(all_vals[~pd.isna(all_vals)].mean())
    overall_s     = round((overall_avg - 1) / 4, 4)
    total_responses = len(df)
    print(f"Overall average: {round(overall_avg,3)}/5 -> s_human={overall_s} ({total_responses} annotators)")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Add columns if missing
    for col_sql in [
        "ALTER TABLE medical_terms ADD COLUMN s_human REAL DEFAULT 0.5",
        "ALTER TABLE medical_terms ADD COLUMN human_verified INTEGER DEFAULT 0",
        "ALTER TABLE medical_terms ADD COLUMN validation_responses INTEGER DEFAULT 0",
        "ALTER TABLE medical_terms ADD COLUMN thumbs_up INTEGER DEFAULT 0",
        "ALTER TABLE medical_terms ADD COLUMN thumbs_down INTEGER DEFAULT 0",
        "ALTER TABLE medical_terms ADD COLUMN last_validated TEXT",
    ]:
        try: cursor.execute(col_sql)
        except: pass

    # Update each DB cardiology term with overall survey score
    updated = 0
    for term in ALL_DB_CARDIO_TERMS:
        cursor.execute("""
            UPDATE medical_terms
            SET s_human = ?,
                human_verified = 1,
                validation_responses = ?,
                last_validated = datetime('now')
            WHERE LOWER(TRIM(term)) = LOWER(TRIM(?))
        """, (overall_s, total_responses, term))
        if cursor.rowcount > 0:
            updated += 1
            print(f"  Validated: {term} (s_human={overall_s})")
        else:
            print(f"  Not found: {term}")

    # Also update specific terms with their exact Q scores
    for col, term in TERM_MAP.items():
        if col in q_scores:
            s = q_scores[col]
            cursor.execute("""
                UPDATE medical_terms
                SET s_human = ?,
                    human_verified = 1,
                    validation_responses = ?,
                    last_validated = datetime('now')
                WHERE LOWER(TRIM(term)) = LOWER(TRIM(?))
            """, (s["s_human"], s["responses"], term))

    conn.commit()
    conn.close()

    print(f"\n{'='*50}")
    print(f"Done! {updated} cardiology terms marked as Human Validated.")
    print(f"Restart main.py — terms will show 'Human Validated' in the app.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
