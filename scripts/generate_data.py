import pandas as pd
import random
import os

# S'assurer que le dossier data existe
os.makedirs('data', exist_ok=True)

# 1. LES PROGRAMMES (Catalogue)
programs_data = [
    {"id": 1, "name": "Computer Science", "domain": "technology", "tags": "coding ai software"},
    {"id": 2, "name": "Fine Arts", "domain": "art", "tags": "drawing design painting"},
    {"id": 3, "name": "History & Law", "domain": "history", "tags": "reading law logic"},
    {"id": 4, "name": "Applied Math", "domain": "math", "tags": "algebra stats logic"},
    {"id": 5, "name": "Architecture", "domain": "art", "tags": "design drawing buildings"},
    {"id": 6, "name": "Data Science", "domain": "math", "tags": "stats ai coding"},
]
df_programs = pd.DataFrame(programs_data)
df_programs.to_csv('data/programs.csv', index=False)

# 2. LES ÉTUDIANTS (Avec notes par matière)
students = []
interests_list = ["coding", "drawing", "reading", "logic", "biology"]

for i in range(200):
    # Notes aléatoires par matière
    math = round(random.uniform(5, 20), 1)
    art = round(random.uniform(5, 20), 1)
    history = round(random.uniform(5, 20), 1)
    tech = round(random.uniform(5, 20), 1)
    
    # Intérêts basés sur les notes
    student_interests = []
    if tech > 14: student_interests.append("coding")
    if art > 14: student_interests.append("drawing")
    if history > 14: student_interests.append("reading")
    if not student_interests: student_interests.append(random.choice(interests_list))
        
    # --- CORRECTION ICI : "technology_grade" au lieu de "tech_grade" ---
    students.append({
        "student_id": 1000 + i,
        "math_grade": math,
        "art_grade": art,
        "history_grade": history,
        "technology_grade": tech,  # C'est corrigé ici !
        "interests": " ".join(student_interests)
    })

df_students = pd.DataFrame(students)
df_students.to_csv('data/students.csv', index=False)

# 3. LES RATINGS (Interactions simulées pour l'IA)
ratings = []
for student in students:
    reviewed_programs = random.sample(programs_data, 3)
    for prog in reviewed_programs:
        # Si bonne note dans la matière du programme -> 5/5, sinon note basse
        domain = prog["domain"]
        
        # Maintenant ça va marcher car "technology_grade" existe
        grade = student[f"{domain}_grade"]
        
        if grade >= 14: rating = 5
        elif grade <= 8: rating = 1
        else: rating = 3
            
        ratings.append({
            "student_id": student["student_id"],
            "program_id": prog["id"],
            "rating": rating
        })

df_ratings = pd.DataFrame(ratings)
df_ratings.to_csv('data/ratings.csv', index=False)

print("SUCCÈS : 3 fichiers CSV créés dans le dossier /data !")