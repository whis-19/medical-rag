import pandas as pd
import time
from medical_qa import *

queries = [
    # Medical Questions (20)
    "What technique was used for the laparoscopic cholecystectomy?",
    "Describe the incision made for the carpal tunnel release.",
    "What anesthesia was used for the colonoscopy procedure?",
    "What were the findings of the MRI of the lumbar spine?",
    "How was the patient positioned for the right total knee arthroplasty?",
    "What sutures were used to close the fascia in the hernia repair?",
    "Describe the findings during the cystoscopy.",
    "What complications occurred during the cataract surgery?",
    "What was the estimated blood loss for the lumbar fusion?",
    "What specific medication was injected during the epidural steroid injection?",
    "What are the presenting symptoms for the patient with allergic rhinitis?",
    "Describe the patient's history of chronic obstructive pulmonary disease (COPD).",
    "What symptoms did the patient with acute appendicitis exhibit?",
    "What were the vital signs recorded for the patient with chest pain?",
    "Describe the physical exam findings for the patient with otitis media.",
    "What previous surgeries did the patient with the hip fracture have?",
    "What allergies does the patient with the skin abscess list?",
    "Is there a family history of heart disease mentioned in the consultations?",
    "What neurological symptoms were reported by the patient with the headache?",
    "Describe the social history for the patient evaluated for depression.",
    
    # Non-Medical Questions (15)
    "Who invented the stethoscope?",
    "What is the capital of France?",
    "What is the current stock price of Pfizer?",
    "Summarize the plot of the movie 'The Doctor'.",
    "How do I bake a cake?",
    "What is the population of New York City?",
    "Who wrote the novel 'Pride and Prejudice'?",
    "What is the boiling point of water?",
    "How tall is Mount Everest?",
    "What year was the iPhone first released?",
    "What is the currency of Japan?",
    "Who painted the Mona Lisa?",
    "What is the largest planet in our solar system?",
    "How many countries are in the European Union?",
    "What is the square root of 144?",
    
    # Mixed Questions (5)
    "Compare the anesthesia types used in the orthopedic surgeries mentioned.",
    "List common risk factors mentioned for patients with cardiovascular issues.",
    "What represent the most frequent postoperative diagnoses in the dataset?",
    "How do medical procedures differ between emergency and elective surgeries?",
    "What are the economic impacts of chronic diseases on healthcare systems?"
]

results = []

print(f"Starting evaluation of {len(queries)} queries...")

for i, query in enumerate(queries):
    print(f"Running query {i+1}/{len(queries)}: {query}")
    try:
        response = rag_chain.invoke({"input": query})
        
        source_docs_combined = []
        if response.get('context'):
            for doc in response['context']:
                row_id = doc.metadata.get('row', 'Unknown')
                source_docs_combined.append(f"[SOURCE ROW {row_id}]:\n{doc.page_content}")
        
        final_source_string = "\n\n".join(source_docs_combined)

        results.append({
            "Query": query,
            "Answer": response["answer"],
            "Source_Documents": final_source_string # Store the new combined string
        })
    except Exception as e:
        print(f"Error on query {query}: {e}")
        results.append({"Query": query, "Answer": "ERROR", "Source_Documents": "Error during retrieval."})
    time.sleep(5) 

df_results = pd.DataFrame(results)
df_results.to_csv("rag_evaluation_results_with_content.csv", index=False)
print("Evaluation complete! Results saved to 'rag_evaluation_results_with_content.csv'")