import pandas as pd
import sqlite3
from pathlib import Path

DB_PATH  = Path("medical_jargon.db")
CSV_PATH = Path("combine response.csv")

TERM_MAP = {
    "Simplicity_Q1":  "Myocardial infarction",              # Acute ST-elevation MI with LV dysfunction
    "Simplicity_Q2":  "Carotid artery stenosis",            # Extracranial carotid artery stenosis / endarterectomy
    "Simplicity_Q3":  "Atrial fibrillation",                # Paroxysmal AF with rapid ventricular response
    "Simplicity_Q4":  "Heart failure",                      # Chronic systolic heart failure HFrEF NYHA III
    "Simplicity_Q5":  "Coronary artery disease",            # Obstructive CAD multi-vessel preserved LVEF
    "Simplicity_Q6":  "Ventricular tachycardia",            # VT refractory to antiarrhythmic pharmacotherapy
    "Simplicity_Q7":  "Aortic stenosis",                    # Severe calcific aortic valve stenosis
    "Simplicity_Q8":  "Deep vein thrombosis",               # Proximal DVT with high PE risk
    "Simplicity_Q9":  "Hypertension",                       # Chronic elevated arterial blood pressure
    "Simplicity_Q10": "Atherosclerosis",                    # Plaque build-up narrowing arteries
    "Simplicity_Q11": "Stroke",                             # Loss of brain function / blood supply disturbance
    "Simplicity_Q12": "Angina pectoris",                    # Chest pain due to ischaemia during exertion
    "Simplicity_Q13": "Cardiac arrhythmia",                 # Irregular too fast or too slow heartbeat
    "Simplicity_Q14": "Pericarditis",                       # Inflammation of the pericardium / sharp chest pain
    "Simplicity_Q15": "Aortic aneurysm",                    # Enlargement of aorta life-threatening if ruptured
    "Simplicity_Q16": "Pulmonary embolism",                 # Blood clot blocking lung artery
    "Simplicity_Q17": "Hypertrophic cardiomyopathy",        # Abnormally thickened heart muscle
    "Simplicity_Q18": "Endocarditis",                       # Infection of inner heart lining / valves
    "Simplicity_Q19": "Cardiac rehabilitation",             # Supervised centre-based rehab post-revascularisation
    "Simplicity_Q20": "Percutaneous coronary intervention", # Dual antiplatelet therapy post-PCI
    "Simplicity_Q21": "Myocardial necrosis",                # Elevated high-sensitivity troponin I
    "Simplicity_Q22": "Atrioventricular block",             # Dual-chamber permanent pacemaker complete AV block
    "Simplicity_Q23": "Statin therapy",                     # High-intensity statin secondary prevention
    "Simplicity_Q24": "Mitral regurgitation",               # Echocardiography wall motion abnormalities
    "Simplicity_Q25": "Mitral valve prolapse",              # Mitral valve leaflets bulging into left atrium
    "Simplicity_Q26": "Cardiogenic shock",                  # Cardiogenic shock post-cardiac arrest vasopressor
}

# ── Verify no duplicate terms in TERM_MAP ────────────────────────────────────
_terms = list(TERM_MAP.values())
_dupes = [t for t in set(_terms) if _terms.count(t) > 1]
if _dupes:
    raise ValueError(f"Duplicate terms found in TERM_MAP: {_dupes}")

# ── All unique DB terms to mark as validated ─────────────────────────────────
ALL_DB_CARDIO_TERMS = [
    # From Cardiology Annotation Form
    "Myocardial infarction",
    "Carotid artery stenosis",
    "Atrial fibrillation",
    "Heart failure",
    "Pulmonary hypertension",
    "Coronary artery disease",
    "Ventricular tachycardia",
    "Secondary hypertension",
    "Aortic stenosis",
    "Deep vein thrombosis",
    "Cardiac rehabilitation",
    "Percutaneous coronary intervention",
    "Cardioembolic stroke",
    "Cerebral infarction",
    "Myocardial necrosis",
    "Atrioventricular block",
    "Statin therapy",
    "Mitral regurgitation",
    "Angina pectoris",
    "Cardiogenic shock",
    # From Heart Disease Simplified Pairs
    "Hypertension",
    "Atherosclerosis",
    "Stroke",
    "Cardiac arrhythmia",
    "Pericarditis",
    "Aortic aneurysm",
    "Pulmonary embolism",
    "Hypertrophic cardiomyopathy",
    "Endocarditis",
    "Mitral valve prolapse",
    "Bradycardia",
    "Tachycardia",
    "Cardiomyopathy",
    # From original DB
    "Abdominal aortic aneurysm",
    "Cardiac arrest",
    "Cardiac fibrosis",
    "Cardiac stress test",
    "Cardiomegaly",
    "Cerebellar stroke syndrome",
    "Coronary vasospasm",
    "Dilated cardiomyopathy",
    "Heart murmur",
    "Heart valve dysplasia",
    "Lacunar stroke",
    "Splenic infarction",
    "Spontaneous coronary artery dissection",
    "Takotsubo cardiomyopathy",
    "Ventricular fibrillation",
    "Tricuspid regurgitation",
    "Aortic regurgitation",
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

    # ── Per-question average scores (1–5 -> 0.0–1.0) ────────────────────────
    q_scores = {}
    for col in TERM_MAP.keys():
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(vals) > 0:
                avg = vals.mean()
                q_scores[col] = {
                    "s_human":   round((avg - 1) / 4, 4),
                    "responses": len(vals),
                    "avg_raw":   round(avg, 3)
                }

    # ── Overall average across all Q1–Q26 columns ────────────────────────────
    all_cols        = [c for c in df.columns if c.startswith("Simplicity_Q")]
    all_vals        = pd.to_numeric(df[all_cols].values.flatten(), errors='coerce')
    overall_avg     = float(all_vals[~pd.isna(all_vals)].mean())
    overall_s       = round((overall_avg - 1) / 4, 4)
    total_responses = len(df)
    print(f"Overall average: {round(overall_avg, 3)}/5  ->  s_human={overall_s}  ({total_responses} annotators)")

    conn   = sqlite3.connect(DB_PATH)
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
        try:
            cursor.execute(col_sql)
        except Exception:
            pass  # Column already exists

    # ── Step 1: Update all terms with overall survey score ───────────────────
    updated   = 0
    not_found = []
    for term in ALL_DB_CARDIO_TERMS:
        cursor.execute("""
            UPDATE medical_terms
            SET s_human              = ?,
                human_verified       = 1,
                validation_responses = ?,
                last_validated       = datetime('now')
            WHERE LOWER(TRIM(term)) = LOWER(TRIM(?))
        """, (overall_s, total_responses, term))

        if cursor.rowcount > 0:
            updated += 1
            print(f"  ✔ Validated : {term}  (s_human={overall_s})")
        else:
            not_found.append(term)
            print(f"  ✘ Not found : {term}")

    # ── Step 2: Override with exact per-question scores ──────────────────────
    print("\n── Applying per-question scores (Q1–Q26) ──")
    for col, term in TERM_MAP.items():
        if col in q_scores:
            s = q_scores[col]
            cursor.execute("""
                UPDATE medical_terms
                SET s_human              = ?,
                    human_verified       = 1,
                    validation_responses = ?,
                    last_validated       = datetime('now')
                WHERE LOWER(TRIM(term)) = LOWER(TRIM(?))
            """, (s["s_human"], s["responses"], term))

            status = "✔" if cursor.rowcount > 0 else "✘"
            print(f"  {status} {col} -> {term}  "
                  f"(raw={s['avg_raw']}/5, s_human={s['s_human']}, n={s['responses']})")

    conn.commit()
    conn.close()

    print(f"\n{'=' * 55}")
    print(f"Done!  {updated} / {len(ALL_DB_CARDIO_TERMS)} terms marked as Human Validated.")
    if not_found:
        print(f"Terms not found in DB ({len(not_found)}): {', '.join(not_found)}")
        print("-> Add these terms to your DB or check for spelling mismatches.")
    print(f"Restart main.py — validated terms will show 'Human Validated' in the app.")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()