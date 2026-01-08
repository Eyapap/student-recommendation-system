import sys
import os
import pickle
import csv
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ==========================================================
# 1. CONFIGURATION & CHEMINS
# ==========================================================
app = FastAPI(
    title="Student Recommender System API",
    description="API for recommending study programs based on interests and grades.",
    version="2.0"
)

# Chemins dynamiques
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) # Remonte d'un niveau (racine du projet)
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
TARGET_MODEL_FILE = os.environ.get(
    "RECOMMENDER_MODEL_FILE",
    "ClassifierModel.pkl"
)

# Fichiers de Logs (Séparés des données d'entraînement !)
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback_log.csv")
INFERENCE_LOG_FILE = os.path.join(DATA_DIR, "inference_log.csv")

# Variables globales
recommender_system = {}
students_df = None
programs_df = None

# ==========================================================
# 2. FONCTIONS UTILITAIRES
# ==========================================================

def get_latest_model_path():
    """Trouve le fichier .pkl le plus récent."""
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Dossier models introuvable : {MODEL_DIR}")
    
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    if not files:
        raise FileNotFoundError("Aucun fichier .pkl trouvé.")
    
    # Trie par date (le plus récent en dernier)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)))
    return os.path.join(MODEL_DIR, files[-1])


def resolve_model_path():
    """Privilégie un modèle ciblé, sinon revient au plus récent."""
    target_path = os.path.join(MODEL_DIR, TARGET_MODEL_FILE)
    if os.path.exists(target_path):
        print(f"[INFO] Utilisation du modèle ciblé: {TARGET_MODEL_FILE}")
        return target_path
    print("[WARN] Modèle ciblé introuvable, fallback vers le plus récent disponible.")
    return get_latest_model_path()

def log_inference(student_data, recommendations):
    """
    Sauvegarde la requête de l'étudiant dans un fichier séparé.
    Ne touche PAS à students.csv (Training Data).
    """
    file_exists = os.path.isfile(INFERENCE_LOG_FILE)
    try:
        with open(INFERENCE_LOG_FILE, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Entête
            if not file_exists:
                writer.writerow(["timestamp", "interests", "math", "art", "history", "tech", "top_recommendation"])
            
            # Données
            top_rec = recommendations[0]['program_name'] if recommendations else "None"
            writer.writerow([
                datetime.now().isoformat(),
                student_data.interests,
                student_data.math_grade,
                student_data.art_grade,
                student_data.history_grade,
                student_data.technology_grade,
                top_rec
            ])
    except Exception as e:
        print(f"[WARNING] Impossible de logger l'inférence : {e}")

# ==========================================================
# 3. CHARGEMENT AU DÉMARRAGE
# ==========================================================
@app.on_event("startup")
def load_model():
    global recommender_system, students_df, programs_df
    try:
        # 1. Charger le modèle
        model_path = resolve_model_path()
        print(f"[INFO] Chargement du modèle : {model_path}")
        with open(model_path, "rb") as f:
            recommender_system = pickle.load(f)
        
        # 2. Charger les données (Lecture Seule)
        students_path = os.path.join(DATA_DIR, "students.csv")
        programs_path = os.path.join(DATA_DIR, "programs.csv")
        
        students_df = pd.read_csv(students_path)
        programs_df = pd.read_csv(programs_path)
        
        print(f"[OK] Système prêt. {len(programs_df)} programmes chargés.")
    except Exception as e:
        print(f"[CRITICAL] Erreur au démarrage : {e}")

# ==========================================================
# 4. MODÈLES DE DONNÉES (SCHEMAS)
# ==========================================================
class StudentInput(BaseModel):
    interests: str 
    math_grade: float = 10.0
    art_grade: float = 10.0
    history_grade: float = 10.0
    technology_grade: float = 10.0


GRADE_KEYWORD_BOOSTS = {
    "math": "math algebra statistics modeling finance economics environment sustainability biology research data-science ",
    "art": "art drawing design visual animation architecture music culinary creativity expression ",
    "history": "history law reading analysis literature journalism politics philosophy international-relations storytelling ",
    "technology": "technology coding ai programming robotics cybersecurity networking systems data-engineering "
}

# Modèle souple pour accepter ce que le Frontend envoie
class FeedbackInput(BaseModel):
    student_id: Union[str, int] # Peut être "web_user" ou un ID
    program_id: int             # ID du programme
    program_name: str           # Nom du programme
    rating: int                 # 1 à 5
    comment: Optional[str] = None

# ==========================================================
# 5. ENDPOINTS
# ==========================================================

@app.get("/")
def home():
    return {"status": "active", "version": "2.0"}

@app.post("/recommend")
def recommend_programs(student: StudentInput):
    """
    Endpoint principal de recommandation.
    """
    if not recommender_system:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    try:
        # --- 1. Logique Hybride (Notes + Mots-clés) ---
        tfidf = recommender_system["tfidf_vectorizer"]
        programs = recommender_system["programs_df"]
        program_soup = recommender_system["program_soup"]
        config = recommender_system.get("config", {})
        
        HIGH_GRADE_THRESHOLD = config.get("HIGH_GRADE_THRESHOLD", 14)
        
        # Construction du profil enrichi
        student_soup = student.interests
        
        # Boost automatique basé sur les notes
        grade_boosts = []
        if student.math_grade > HIGH_GRADE_THRESHOLD:
            student_soup += GRADE_KEYWORD_BOOSTS["math"]
            grade_boosts.append("math")
        if student.art_grade > HIGH_GRADE_THRESHOLD:
            student_soup += GRADE_KEYWORD_BOOSTS["art"]
            grade_boosts.append("art")
        if student.history_grade > HIGH_GRADE_THRESHOLD:
            student_soup += GRADE_KEYWORD_BOOSTS["history"]
            grade_boosts.append("history")
        if student.technology_grade > HIGH_GRADE_THRESHOLD:
            student_soup += GRADE_KEYWORD_BOOSTS["technology"]
            grade_boosts.append("technology")
            
        # --- 2. Calcul de similarité ---
        student_vector = tfidf.transform([student_soup])
        program_vectors = tfidf.transform(program_soup)
        similarity_scores = cosine_similarity(student_vector, program_vectors).flatten()
        
        # Top 5
        top_indices = similarity_scores.argsort()[::-1][:5]
        
        recommendations = []
        for idx in top_indices:
            score = similarity_scores[idx]
            if score < 0.01: continue # Filtrer le bruit
            
            program = programs.iloc[idx]
            
            # --- 3. Génération de l'explication (Netflix Style) ---
            matched_terms = []
            
            # On cherche pourquoi ça a matché
            interest_tokens = student.interests.lower().split()
            domain_lower = str(program['domain']).lower()
            tags_lower = str(program.get('tags', '')).lower()
            
            # A. Match par Mots-clés directs
            for term in interest_tokens:
                if len(term) > 3 and (term in domain_lower or term in tags_lower):
                    matched_terms.append(term)
            
            # B. Match par Notes Fortes
            for domain in grade_boosts:
                if domain in domain_lower or domain in tags_lower:
                    matched_terms.append(domain)
            
            # Création de la phrase
            reasons = ", ".join(list(set(matched_terms))) # Supprime les doublons
            if reasons:
                explanation = f"You like {reasons}, have you considered {program['name']}?"
            else:
                explanation = f"Matches your profile with {int(score*100)}% similarity."

            recommendations.append({
                "program_id": int(program["id"]),
                "program_name": program["name"],
                "domain": program["domain"],
                "tags": program.get("tags", ""),
                "score": round(float(score), 4),
                "explanation": explanation
            })
            
        # --- 4. Logging (Séparation Inference vs Training) ---
        log_inference(student, recommendations)
        
        return {
            "input_interests": student.interests,
            "recommendations": recommendations
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.post("/feedback")
def log_feedback(feedback: FeedbackInput):
    """
    Enregistre le feedback utilisateur dans feedback_log.csv
    """
    print(f"[DEBUG] Received feedback request: {feedback}")
    print(f"[DEBUG] FEEDBACK_FILE path: {FEEDBACK_FILE}")
    print(f"[DEBUG] File exists: {os.path.isfile(FEEDBACK_FILE)}")
    
    try:
        file_exists = os.path.isfile(FEEDBACK_FILE)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
        print(f"[DEBUG] Directory ensured: {os.path.dirname(FEEDBACK_FILE)}")
        
        with open(FEEDBACK_FILE, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            if not file_exists:
                print("[DEBUG] Writing header...")
                writer.writerow(["timestamp", "student_id", "program_id", "program_name", "rating", "comment"])
            
            # Nettoyage
            clean_comment = feedback.comment.replace("\n", " ") if feedback.comment else ""
            
            row_data = [
                datetime.now().isoformat(),
                feedback.student_id,
                feedback.program_id,
                feedback.program_name,
                feedback.rating,
                clean_comment
            ]
            print(f"[DEBUG] Writing row: {row_data}")
            writer.writerow(row_data)
            
        print(f"[INFO] Feedback saved: Student {feedback.student_id} rated '{feedback.program_name}' with {feedback.rating} stars")
        print(f"[DEBUG] File size after write: {os.path.getsize(FEEDBACK_FILE)} bytes")
        return {"status": "success", "message": "Feedback enregistré."}
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Feedback failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/register_student")
def register_student(student: StudentInput):
    """
    Generate a student ID for the session (doesn't persist to students.csv)
    """
    try:
        # Generate a unique temporary ID using timestamp
        student_id = int(datetime.now().timestamp() * 1000) % 100000
        print(f"[INFO] Generated student ID: {student_id} for interests: {student.interests}")
        return {"status": "success", "student_id": student_id}
    except Exception as e:
        print(f"[ERROR] Register student failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Lancement
    uvicorn.run(app, host="127.0.0.1", port=8000)