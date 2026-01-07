import sys
import os
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ==========================================================
# 1. SETUP & CHARGEMENT DU MODÈLE
# ==========================================================
app = FastAPI(
    title="Student Recommender System API",
    description="API for recommending study programs based on interests and grades.",
    version="1.0"
)

# Chemin vers le dossier models (remonte d'un niveau par rapport à backend/)
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FEEDBACK_FILE = "feedback_log.csv"

# Variable globale pour stocker le modèle chargé
recommender_system = {}
students_df = None
programs_df = None

def get_latest_model_path():
    """Trouve le fichier .pkl le plus récent dans le dossier models/."""
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Le dossier models n'existe pas : {MODEL_DIR}")
    
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    if not files:
        raise FileNotFoundError("Aucun fichier .pkl trouvé dans le dossier models/")
    
    # Trie par date de modification (le plus récent en dernier)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)))
    return os.path.join(MODEL_DIR, files[-1])

@app.on_event("startup")
def load_model():
    """Charge le modèle au démarrage de l'API."""
    global recommender_system, students_df, programs_df
    try:
        # Load model
        model_path = get_latest_model_path()
        print(f"[OK] Chargement du modèle depuis : {model_path}")
        
        with open(model_path, "rb") as f:
            recommender_system = pickle.load(f)
        
        # Load data files
        students_path = os.path.join(DATA_DIR, "students.csv")
        programs_path = os.path.join(DATA_DIR, "programs.csv")
        
        students_df = pd.read_csv(students_path)
        programs_df = pd.read_csv(programs_path)
        
        print(f"[OK] Loaded {len(students_df)} students and {len(programs_df)} programs")
        print(f"[OK] Model keys: {list(recommender_system.keys())}")
        print("[OK] Modèle chargé avec succès !")
    except Exception as e:
        print(f"[ERROR] Erreur lors du chargement du modèle : {e}")
        import traceback
        traceback.print_exc()

# ==========================================================
# 2. DÉFINITION DES DONNÉES (SCHEMAS)
# ==========================================================
class StudentInput(BaseModel):
    interests: str  # ex: "coding ai software"
    math_grade: Optional[float] = 10.0
    art_grade: Optional[float] = 10.0
    history_grade: Optional[float] = 10.0
    technology_grade: Optional[float] = 10.0

class FeedbackInput(BaseModel):
    student_id: str
    program_recommended: str
    rating: int # 1 to 5
    comment: Optional[str] = None

# ==========================================================
# 3. ENDPOINTS (LES FONCTIONS DE L'API)
# ==========================================================

@app.get("/")
def home():
    return {"status": "active", "message": "L'API de Recommandation est en ligne."}

@app.post("/recommend")
def recommend_programs(student: StudentInput):
    """
    Génère des recommandations pour un étudiant basé sur ses intérêts et ses notes.
    Utilise le modèle hybride (Content-Based + Collaborative Filtering).
    """
    if not recommender_system:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    try:
        # 1. Récupérer les composants du modèle
        tfidf = recommender_system["tfidf_vectorizer"]
        programs = recommender_system["programs_df"]
        program_soup = recommender_system["program_soup"]
        config = recommender_system.get("config", {})
        
        # 2. Create student soup (same logic as train.py)
        HIGH_GRADE_THRESHOLD = config.get("HIGH_GRADE_THRESHOLD", 14)
        
        student_soup = student.interests
        
        # Add domain keywords based on high grades (same as train.py)
        if student.math_grade > HIGH_GRADE_THRESHOLD:
            student_soup += " math algebra statistics computation"
        
        if student.art_grade > HIGH_GRADE_THRESHOLD:
            student_soup += " art drawing design visual creativity"
        
        if student.history_grade > HIGH_GRADE_THRESHOLD:
            student_soup += " history law reading analysis literature"
        
        if student.technology_grade > HIGH_GRADE_THRESHOLD:
            student_soup += " technology coding programming ai software"
        
        # 3. Vectorize student soup with the trained TF-IDF
        student_vector = tfidf.transform([student_soup])
        
        # 4. Vectorize program soups
        program_vectors = tfidf.transform(program_soup)
        
        # 5. Calculate cosine similarity
        similarity_scores = cosine_similarity(student_vector, program_vectors).flatten()
        
        # 6. Get top 5 recommendations
        top_indices = similarity_scores.argsort()[::-1][:5]
        
        recommendations = []
        for idx in top_indices:
            score = similarity_scores[idx]
            
            # Filter out low scores
            if score < 0.01:
                continue
            
            program = programs.iloc[idx]
            
            # Build explanation
            matched_terms = []
            if "math" in student.interests.lower() or student.math_grade > HIGH_GRADE_THRESHOLD:
                if "math" in program["domain"].lower() or "math" in str(program.get("tags", "")).lower():
                    matched_terms.append("math")
            if "art" in student.interests.lower() or student.art_grade > HIGH_GRADE_THRESHOLD:
                if "art" in program["domain"].lower() or "art" in str(program.get("tags", "")).lower():
                    matched_terms.append("art")
            if "technology" in student.interests.lower() or "coding" in student.interests.lower() or student.technology_grade > HIGH_GRADE_THRESHOLD:
                if "technology" in program["domain"].lower() or "coding" in str(program.get("tags", "")).lower():
                    matched_terms.append("technology/coding")
            if "history" in student.interests.lower() or student.history_grade > HIGH_GRADE_THRESHOLD:
                if "history" in program["domain"].lower() or "history" in str(program.get("tags", "")).lower():
                    matched_terms.append("history")
            
            explanation = f"Matches your profile ({', '.join(matched_terms) if matched_terms else 'general interests'}) with {int(score*100)}% similarity"
            
            recommendations.append({
                "program_id": int(program["id"]),
                "program_name": program["name"],
                "domain": program["domain"],
                "tags": program["tags"],
                "score": round(float(score), 4),
                "explanation": explanation
            })
        
        return {
            "input_interests": student.interests,
            "input_grades": {
                "math": student.math_grade,
                "art": student.art_grade,
                "history": student.history_grade,
                "technology": student.technology_grade
            },
            "recommendations": recommendations
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing recommendation: {str(e)}")

@app.post("/feedback")
def log_feedback(feedback: FeedbackInput):
    """
    Enregistre le feedback utilisateur dans un fichier CSV (Pour retraining).
    """
    file_exists = os.path.isfile(FEEDBACK_FILE)
    
    try:
        with open(FEEDBACK_FILE, "a") as f:
            # Si le fichier est nouveau, on met les entêtes
            if not file_exists:
                f.write("timestamp,student_id,program,rating,comment\n")
            
            # On écrit la ligne
            timestamp = datetime.now().isoformat()
            clean_comment = feedback.comment.replace("\n", " ") if feedback.comment else ""
            f.write(f"{timestamp},{feedback.student_id},{feedback.program_recommended},{feedback.rating},{clean_comment}\n")
            
        return {"status": "success", "message": "Feedback enregistré."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Lance le serveur
    uvicorn.run(app, host="127.0.0.1", port=8000)