
import sqlite3
import os
from datetime import datetime

DB_PATH = "medical_jargon.db"

# Common terms patients frequently search for
# Format: (term, summary, full_content)
COMMON_TERMS = [
    (
        "Myocardial infarction",
        "Myocardial infarction, commonly known as a heart attack, occurs when blood flow to part of the heart muscle is blocked, causing damage or death to that area of the heart. It is a medical emergency requiring immediate treatment.",
        "Myocardial infarction, commonly known as a heart attack, occurs when blood flow to part of the heart muscle is blocked, causing damage or death to that area of the heart. It is a medical emergency requiring immediate treatment. Symptoms include chest pain or pressure, shortness of breath, pain radiating to the arm or jaw, nausea, and sweating. Risk factors include high blood pressure, high cholesterol, smoking, diabetes, obesity, and family history. Treatment includes medications to dissolve clots, procedures to open blocked arteries, and lifestyle changes."
    ),
    (
        "Gastritis",
        "Gastritis is inflammation of the stomach lining. It can occur suddenly (acute gastritis) or gradually over time (chronic gastritis). Common causes include infection with Helicobacter pylori bacteria, regular use of pain relievers, and excessive alcohol consumption.",
        "Gastritis is inflammation of the stomach lining. It can occur suddenly (acute gastritis) or gradually over time (chronic gastritis). Common causes include infection with Helicobacter pylori bacteria, regular use of pain relievers such as aspirin or ibuprofen, and excessive alcohol consumption. Symptoms include burning pain in the upper abdomen, nausea, vomiting, and feeling full after eating. Treatment depends on the cause and may include antibiotics, antacids, and dietary changes."
    ),
    (
        "Edema",
        "Edema is swelling caused by excess fluid trapped in the body's tissues. It most commonly affects the feet, ankles, and legs but can also affect the face, hands, and other areas of the body.",
        "Edema is swelling caused by excess fluid trapped in the body's tissues. It most commonly affects the feet, ankles, and legs but can also affect the face, hands, and other areas. Edema can be a symptom of an underlying disease such as heart failure, kidney disease, or liver disease. It can also result from pregnancy, medications, or prolonged standing. Treatment depends on the cause and may include diuretics, compression stockings, and elevation of the affected limb."
    ),
    (
        "Hypertension",
        "Hypertension, also known as high blood pressure, is a condition in which the force of blood against artery walls is consistently too high. It is often called a silent killer because it usually has no symptoms but significantly increases the risk of heart disease and stroke.",
        "Hypertension, also known as high blood pressure, is a condition in which the force of blood against artery walls is consistently too high. It is often called a silent killer because it usually has no symptoms but significantly increases the risk of heart disease, stroke, and kidney disease. Blood pressure is measured in millimeters of mercury and recorded as two numbers — systolic pressure over diastolic pressure. Normal blood pressure is below 120/80 mmHg. Hypertension is diagnosed when readings are consistently 130/80 mmHg or higher. Treatment includes lifestyle changes and medications."
    ),
    (
        "Diabetes mellitus",
        "Diabetes mellitus is a group of metabolic diseases in which blood sugar levels are too high over a prolonged period. It occurs when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces.",
        "Diabetes mellitus is a group of metabolic diseases in which blood sugar levels are too high over a prolonged period. It occurs when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Type 1 diabetes is an autoimmune condition where the body attacks insulin-producing cells. Type 2 diabetes is more common and involves insulin resistance. Symptoms include frequent urination, increased thirst, unexplained weight loss, fatigue, and blurred vision. Long-term complications include heart disease, kidney damage, nerve damage, and eye problems."
    ),
    (
        "Pneumonia",
        "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm, fever, chills, and difficulty breathing.",
        "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm or pus, fever, chills, and difficulty breathing. It can range from mild to life-threatening and is most serious for infants, young children, people older than 65, and people with health problems or weakened immune systems. Causes include bacteria, viruses, and fungi. The most common bacterial cause is Streptococcus pneumoniae. Treatment depends on the cause and severity."
    ),
    (
        "Asthma",
        "Asthma is a condition in which the airways narrow and swell and may produce extra mucus, making breathing difficult and triggering coughing, wheezing, and shortness of breath.",
        "Asthma is a condition in which the airways narrow and swell and may produce extra mucus, making breathing difficult and triggering coughing, wheezing, and shortness of breath. For some people asthma is a minor nuisance but for others it can be a major problem that interferes with daily activities. Asthma cannot be cured but its symptoms can be controlled. Triggers include allergens, exercise, cold air, respiratory infections, and stress. Treatment includes long-term control medications and quick-relief inhalers."
    ),
    (
        "Anemia",
        "Anemia is a condition in which the blood does not have enough healthy red blood cells to carry adequate oxygen to the body's tissues. It can cause fatigue, weakness, pale skin, and shortness of breath.",
        "Anemia is a condition in which the blood does not have enough healthy red blood cells to carry adequate oxygen to the body's tissues. It is the most common blood disorder. Symptoms include fatigue, weakness, pale or yellowish skin, irregular heartbeats, shortness of breath, dizziness, chest pain, cold hands and feet, and headaches. The most common cause is iron deficiency. Other causes include vitamin B12 deficiency, chronic disease, bone marrow disorders, and inherited conditions. Treatment depends on the underlying cause."
    ),
    (
        "Appendicitis",
        "Appendicitis is an inflammation of the appendix, a finger-shaped pouch that projects from the colon on the lower right side of the abdomen. It causes pain that typically begins around the navel and shifts to the lower right abdomen.",
        "Appendicitis is an inflammation of the appendix, a finger-shaped pouch that projects from the colon on the lower right side of the abdomen. It causes pain that typically begins around the navel and shifts to the lower right abdomen. The pain usually worsens with movement, deep breathing, coughing, or sneezing. Other symptoms include nausea, vomiting, fever, and loss of appetite. Appendicitis requires prompt medical treatment. Standard treatment is surgical removal of the appendix called appendectomy. If untreated the appendix can rupture causing a life-threatening infection."
    ),
    (
        "Migraine",
        "A migraine is a headache of varying intensity, often accompanied by nausea and sensitivity to light and sound. Migraine headaches are sometimes preceded by warning symptoms such as flashes of light, blind spots, or tingling in the hands or face.",
        "A migraine is a headache of varying intensity, often accompanied by nausea and sensitivity to light and sound. Migraine headaches are sometimes preceded by warning symptoms called an aura, such as flashes of light, blind spots, or tingling in the hands or face. The exact cause is unknown but involves changes in brain chemistry and nerve pathways. Triggers include hormonal changes, certain foods and drinks, stress, sleep changes, sensory stimuli, and medications. Treatment includes pain-relieving medications taken during migraine attacks and preventive medications to reduce frequency."
    ),
    (
        "Sepsis",
        "Sepsis is a life-threatening medical emergency caused by the body's extreme response to an infection. It occurs when an infection triggers a chain reaction throughout the body, leading to organ damage and failure.",
        "Sepsis is a life-threatening medical emergency caused by the body's extreme response to an infection. It occurs when an infection triggers a chain reaction throughout the body, causing widespread inflammation that can lead to organ damage and failure. Symptoms include high or low temperature, rapid heart rate, rapid breathing, confusion, and extreme pain. Sepsis requires immediate hospital treatment including antibiotics and intravenous fluids. Without prompt treatment sepsis can lead to septic shock, multiple organ failure, and death. Any type of infection can lead to sepsis including bacterial, viral, and fungal infections."
    ),
    (
        "Stroke",
        "A stroke occurs when the blood supply to part of the brain is cut off. Without blood, brain cells begin to die within minutes. There are two main types ischemic stroke caused by a blocked artery and hemorrhagic stroke caused by a burst blood vessel.",
        "A stroke occurs when the blood supply to part of the brain is cut off. Without blood, brain cells begin to die within minutes. There are two main types ischemic stroke caused by a blocked artery, which accounts for about 87% of all strokes, and hemorrhagic stroke caused by a burst blood vessel. Symptoms include sudden numbness or weakness in the face, arm, or leg especially on one side, sudden confusion, trouble speaking, sudden vision problems, sudden trouble walking, and sudden severe headache. The acronym FAST — Face drooping, Arm weakness, Speech difficulty, Time to call emergency services — helps identify stroke symptoms."
    ),
    (
        "Tuberculosis",
        "Tuberculosis is a potentially serious infectious disease that mainly affects the lungs. It is caused by the bacterium Mycobacterium tuberculosis and spreads through the air when infected people cough, sneeze, or spit.",
        "Tuberculosis is a potentially serious infectious disease that mainly affects the lungs. It is caused by the bacterium Mycobacterium tuberculosis and spreads through the air when infected people cough, sneeze, or spit. Symptoms of active tuberculosis include a bad cough lasting 3 weeks or more, coughing up blood, chest pain, weakness, weight loss, fever, and night sweats. Not everyone infected develops active disease — latent tuberculosis has no symptoms but can become active. Treatment involves a course of antibiotics taken for 6 to 9 months."
    ),
    (
        "Fracture",
        "A fracture is a break in the continuity of a bone. It can range from a thin crack to a complete break. Fractures can occur in any bone in the body and are most commonly caused by trauma, overuse, or osteoporosis.",
        "A fracture is a break in the continuity of a bone. It can range from a thin crack to a complete break. Fractures can occur in any bone in the body and are most commonly caused by trauma such as a fall or accident, overuse injuries in athletes, or weakened bones due to osteoporosis. Symptoms include pain, swelling, bruising, deformity, and inability to use the affected area. Treatment depends on the type and location of the fracture and may include splinting, casting, traction, or surgery."
    ),
    (
        "Urinary tract infection",
        "A urinary tract infection is an infection in any part of the urinary system including the kidneys, ureters, bladder, and urethra. Most infections involve the lower urinary tract — the bladder and the urethra.",
        "A urinary tract infection is an infection in any part of the urinary system including the kidneys, ureters, bladder, and urethra. Most infections involve the lower urinary tract — the bladder and the urethra. Women are at greater risk of developing a UTI than men. Symptoms include a strong persistent urge to urinate, a burning sensation when urinating, passing frequent small amounts of urine, cloudy urine, strong-smelling urine, and pelvic pain. Treatment typically involves antibiotics. Drinking plenty of water and urinating frequently can help prevent UTIs."
    ),
    (
        "Hypothyroidism",
        "Hypothyroidism is a condition in which the thyroid gland does not produce enough thyroid hormone. It can cause many body functions to slow down. The most common cause is Hashimoto disease, an autoimmune disorder.",
        "Hypothyroidism is a condition in which the thyroid gland does not produce enough thyroid hormone. It can cause many body functions to slow down. The most common cause is Hashimoto disease, an autoimmune disorder in which the immune system attacks the thyroid gland. Symptoms include fatigue, increased sensitivity to cold, constipation, pale dry skin, a puffy face, brittle nails, hair loss, enlargement of the tongue, unexplained weight gain, muscle aches, depression, and slowed heart rate. Treatment involves daily use of the synthetic thyroid hormone levothyroxine."
    ),
    (
        "Osteoporosis",
        "Osteoporosis is a condition that weakens bones, making them fragile and more likely to break. It develops slowly over several years and is often only diagnosed when a fall or sudden impact causes a bone fracture.",
        "Osteoporosis is a condition that weakens bones, making them fragile and more likely to break. It develops slowly over several years and is often only diagnosed when a fall or sudden impact causes a bone fracture. The most common injuries in people with osteoporosis are wrist fractures, hip fractures, and spinal fractures. Risk factors include age, female sex, low body weight, low calcium intake, physical inactivity, smoking, and excessive alcohol. Treatment includes calcium and vitamin D supplements, medications to slow bone loss, and lifestyle changes including weight-bearing exercise."
    ),
    (
        "Anxiety disorder",
        "Anxiety disorder is a mental health condition characterized by persistent and excessive worry that interferes with daily activities. It is one of the most common mental health conditions, affecting millions of people worldwide.",
        "Anxiety disorder is a mental health condition characterized by persistent and excessive worry about a number of different things that is difficult to control and interferes with daily activities. It is one of the most common mental health conditions. Symptoms include feeling nervous, restless or tense, having a sense of impending danger or doom, increased heart rate, rapid breathing, sweating, trembling, feeling weak or tired, difficulty concentrating, trouble sleeping, and gastrointestinal problems. Treatment includes psychotherapy especially cognitive behavioral therapy, medications, and lifestyle changes."
    ),
    (
        "Depression",
        "Depression is a mood disorder that causes a persistent feeling of sadness and loss of interest. Also called major depressive disorder, it affects how a person feels, thinks, and behaves and can lead to various emotional and physical problems.",
        "Depression is a mood disorder that causes a persistent feeling of sadness and loss of interest in activities once enjoyed. Also called major depressive disorder or clinical depression, it affects how a person feels, thinks, and behaves and can lead to various emotional and physical problems. Symptoms include persistent sad or empty mood, loss of interest in activities, changes in appetite, sleep disturbances, fatigue, feelings of worthlessness, difficulty thinking or concentrating, and thoughts of death or suicide. Treatment includes medications such as antidepressants and psychotherapy."
    ),
    (
        "Dengue fever",
        "Dengue fever is a mosquito-borne viral infection causing a severe flu-like illness. It is transmitted by Aedes mosquitoes and is common in tropical and subtropical regions including Nepal and other South Asian countries.",
        "Dengue fever is a mosquito-borne viral infection causing a severe flu-like illness. It is transmitted by Aedes mosquitoes and is common in tropical and subtropical regions including Nepal, India, and other South Asian countries. Symptoms include sudden high fever, severe headache, pain behind the eyes, muscle and joint pains, nausea, vomiting, swollen glands, and rash. There is no specific treatment — management focuses on relieving symptoms with rest, fluids, and pain relievers. Severe dengue can be life-threatening and requires hospital care. Prevention involves eliminating mosquito breeding sites and using mosquito protection."
    ),     (
        "Carotid artery stenosis",
        "Carotid artery stenosis is narrowing of the major arteries in the neck that supply blood to the brain. This can reduce blood flow and increase the risk of stroke.",
        "Carotid artery stenosis is narrowing of the major arteries in the neck that supply blood to the brain. It is usually caused by fatty deposits called plaque. Symptoms may not appear until a stroke occurs. Treatment includes lifestyle changes, medications, and sometimes surgery to remove the blockage."
    ),
    (
        "Atrial fibrillation",
        "Atrial fibrillation is an irregular and often fast heartbeat. It can lead to poor blood flow and increase the risk of stroke.",
        "Atrial fibrillation is an irregular and often rapid heart rhythm. It occurs when the upper chambers of the heart beat chaotically. Symptoms include palpitations, fatigue, dizziness, and shortness of breath. Treatment includes medications, procedures, and lifestyle changes."
    ),
    (
        "Heart failure",
        "Heart failure is a condition where the heart cannot pump blood effectively to meet the body's needs.",
        "Heart failure occurs when the heart muscle becomes weak or stiff and cannot pump enough blood. Symptoms include shortness of breath, swelling in legs, fatigue, and rapid heartbeat. Treatment includes medications, lifestyle changes, and sometimes medical devices."
    ),
    (
        "Coronary artery disease",
        "Coronary artery disease occurs when the arteries supplying blood to the heart become narrowed or blocked.",
        "Coronary artery disease is caused by plaque buildup in the heart arteries. This reduces blood flow and can cause chest pain or heart attack. Risk factors include smoking, high cholesterol, diabetes, and high blood pressure."
    ),
    (
        "Ventricular tachycardia",
        "Ventricular tachycardia is a fast heart rhythm that starts in the lower chambers of the heart.",
        "Ventricular tachycardia is a dangerous rapid heartbeat originating in the ventricles. It can cause dizziness, fainting, or cardiac arrest. Immediate medical treatment may be required."
    ),
    (
        "Aortic stenosis",
        "Aortic stenosis is narrowing of the aortic valve in the heart, which restricts blood flow.",
        "Aortic stenosis occurs when the aortic valve becomes stiff or narrowed. This makes the heart work harder. Symptoms include chest pain, fainting, and shortness of breath."
    ),
    (
        "Deep vein thrombosis",
        "Deep vein thrombosis is a blood clot that forms in a deep vein, usually in the leg.",
        "Deep vein thrombosis occurs when a clot forms in a deep vein. Symptoms include swelling, pain, warmth, and redness. The clot can travel to the lungs causing pulmonary embolism."
    ),
    (
        "Atherosclerosis",
        "Atherosclerosis is buildup of fatty deposits inside arteries that narrows them.",
        "Atherosclerosis develops when cholesterol and fat accumulate in artery walls. This reduces blood flow and increases risk of heart attack and stroke."
    ),
    (
        "Angina pectoris",
        "Angina is chest pain caused by reduced blood flow to the heart muscle.",
        "Angina occurs when the heart muscle does not receive enough oxygen. It often happens during physical activity or stress and improves with rest."
    ),
    (
        "Cardiac arrhythmia",
        "Cardiac arrhythmia is an abnormal heart rhythm where the heart beats too fast, too slow, or irregularly.",
        "Cardiac arrhythmias occur when electrical signals in the heart malfunction. Symptoms include palpitations, dizziness, and fainting."
    ),
    (
        "Pericarditis",
        "Pericarditis is inflammation of the sac surrounding the heart.",
        "Pericarditis causes sharp chest pain that may worsen with breathing. It can be caused by infection, heart attack, or autoimmune conditions."
    ),
    (
        "Aortic aneurysm",
        "An aortic aneurysm is an abnormal bulging of the aorta, the main artery carrying blood from the heart.",
        "An aortic aneurysm occurs when part of the aorta weakens and expands. It may rupture if untreated. Large aneurysms may require surgery."
    ),
    (
        "Pulmonary embolism",
        "Pulmonary embolism is a blockage in the lung arteries caused by a blood clot.",
        "Pulmonary embolism usually occurs when a clot from the leg travels to the lungs. Symptoms include sudden shortness of breath, chest pain, and coughing blood."
    ),
    (
        "Hypertrophic cardiomyopathy",
        "Hypertrophic cardiomyopathy is a condition where the heart muscle becomes abnormally thick.",
        "This thickening makes it harder for the heart to pump blood. Some people may have no symptoms while others may experience fainting or chest pain."
    ),
    (
        "Endocarditis",
        "Endocarditis is infection of the inner lining of the heart.",
        "Endocarditis usually affects heart valves and is caused by bacteria entering the bloodstream. Symptoms include fever, fatigue, and heart murmurs."
    ),
    (
        "Cardiac rehabilitation",
        "Cardiac rehabilitation is a supervised program to help patients recover after heart problems.",
        "It includes exercise, education, and lifestyle counseling to improve heart health after heart attack or surgery."
    ),
    (
        "Percutaneous coronary intervention",
        "Percutaneous coronary intervention is a procedure used to open blocked heart arteries.",
        "A small balloon and often a stent are used to widen narrowed arteries. It improves blood flow to the heart."
    ),
    (
        "Myocardial necrosis",
        "Myocardial necrosis means death of heart muscle cells due to lack of blood supply.",
        "It usually occurs during a heart attack and is detected using blood tests such as troponin."
    ),
    (
        "Atrioventricular block",
        "Atrioventricular block is a condition where electrical signals between heart chambers are delayed or blocked.",
        "This can cause a slow heartbeat. Severe cases may require a pacemaker."
    ),
    (
        "Statin therapy",
        "Statin therapy involves medications used to lower cholesterol levels.",
        "Statins reduce risk of heart attack and stroke by lowering bad cholesterol and stabilizing plaque."
    ),
    (
        "Mitral regurgitation",
        "Mitral regurgitation is leakage of blood backward through the mitral valve.",
        "This occurs when the valve does not close properly, causing fatigue and shortness of breath."
    ),
    (
        "Mitral valve prolapse",
        "Mitral valve prolapse occurs when the mitral valve bulges into the upper chamber of the heart.",
        "Most cases are mild, but some may cause palpitations or chest discomfort."
    ),
    (
        "Cardiogenic shock",
        "Cardiogenic shock is a life-threatening condition where the heart suddenly cannot pump enough blood.",
        "It often occurs after a severe heart attack. Symptoms include low blood pressure, confusion, and cold skin."
    ),
]

def main():
    if not os.path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} not found!")
        print("Run build_new_db.py first to create the database.")
        return

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
        try:
            cursor.execute(col_sql)
        except:
            pass

    inserted = 0
    skipped  = 0
    today    = datetime.now().strftime('%Y-%m-%d')

    for term, summary, content in COMMON_TERMS:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO medical_terms
                (term, content, summary, term_lower, content_length, extracted_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                term,
                content,
                summary,
                term.lower().strip(),
                len(content),
                today
            ))
            if cursor.rowcount > 0:
                inserted += 1
                print(f"  Added: {term}")
            else:
                skipped += 1
                print(f"  Already exists: {term}")
        except Exception as e:
            print(f"  Error adding {term}: {e}")
            skipped += 1

    conn.commit()

    # Final count
    total = conn.execute("SELECT COUNT(*) FROM medical_terms").fetchone()[0]
    conn.close()

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Added:   {inserted} new terms")
    print(f"  Skipped: {skipped} already existed")
    print(f"  Total in DB: {total}")
    print(f"{'='*50}")
    print(f"\nNext step: Rebuild FAISS index!")
    print(f"  python3 faiss_index_builder.py")

if __name__ == "__main__":
    main()