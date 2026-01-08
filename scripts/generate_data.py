import pandas as pd
import random
import os

# S'assurer que le dossier data existe
os.makedirs('data', exist_ok=True)

# 1. LES PROGRAMMES (Catalogue)
programs_data = [
    {"id": 1, "name": "Computer Science & AI", "domain": "technology", "tags": "coding ai software robotics"},
    {"id": 2, "name": "Data Science & Analytics", "domain": "math", "tags": "statistics data-visualization machine-learning"},
    {"id": 3, "name": "Cybersecurity & Networks", "domain": "technology", "tags": "security networking systems"},
    {"id": 4, "name": "Biomedical Engineering", "domain": "science", "tags": "biology medicine innovation"},
    {"id": 5, "name": "Environmental Science", "domain": "environment", "tags": "climate sustainability biology"},
    {"id": 6, "name": "Marine Biology", "domain": "biology", "tags": "ocean wildlife research"},
    {"id": 7, "name": "Literature & Creative Writing", "domain": "literature", "tags": "writing storytelling poetry"},
    {"id": 8, "name": "Journalism & Media", "domain": "humanities", "tags": "communication reporting storytelling"},
    {"id": 9, "name": "Visual Arts & Animation", "domain": "art", "tags": "drawing animation illustration"},
    {"id": 10, "name": "Architecture & Urban Design", "domain": "design", "tags": "design drawing buildings sustainability"},
    {"id": 11, "name": "Applied Mathematics & Statistics", "domain": "math", "tags": "algebra statistics modeling"},
    {"id": 12, "name": "Business & Entrepreneurship", "domain": "business", "tags": "finance startup leadership"},
    {"id": 13, "name": "Psychology & Cognitive Science", "domain": "humanities", "tags": "behavior research neuroscience"},
    {"id": 14, "name": "Sports Science & Kinesiology", "domain": "science", "tags": "health performance biomechanics"},
    {"id": 15, "name": "Music Production & Sound Design", "domain": "art", "tags": "music composition audio"},
    {"id": 16, "name": "Culinary Arts & Gastronomy", "domain": "culinary", "tags": "cooking nutrition creativity"},
    {"id": 17, "name": "Philosophy & Ethics", "domain": "humanities", "tags": "logic ethics debate"},
    {"id": 18, "name": "International Relations", "domain": "politics", "tags": "diplomacy economics law"}
]
df_programs = pd.DataFrame(programs_data)
df_programs.to_csv('data/programs.csv', index=False)

# 2. LES ÉTUDIANTS (Avec notes par matière)
STUDENT_COUNT = 300

students = []
interests_pool = [
    "coding", "robotics", "ai", "drawing", "animation", "design",
    "reading", "literature", "journalism", "history", "law",
    "biology", "environment", "marine", "psychology", "business",
    "entrepreneurship", "finance", "sustainability", "storytelling"
]

for i in range(STUDENT_COUNT):
    # Notes aléatoires par matière
    math = round(random.uniform(5, 20), 1)
    art = round(random.uniform(5, 20), 1)
    history = round(random.uniform(5, 20), 1)
    tech = round(random.uniform(5, 20), 1)
    
    # Intérêts basés sur les notes
    student_interests = set()
    if tech > 14:
        student_interests.update(["coding", "robotics", "ai"])
    if art > 14:
        student_interests.update(["drawing", "animation", "design"])
    if history > 14:
        student_interests.update(["reading", "literature", "journalism"])
    if math > 14:
        student_interests.update(["math", "statistics", "finance"])
    if not student_interests:
        student_interests.update(random.sample(interests_pool, k=2))
    else:
        # Ajouter un intérêt complémentaire pour varier
        student_interests.add(random.choice(interests_pool))
        
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
domain_grade_map = {
    "technology": "technology_grade",
    "art": "art_grade",
    "design": "art_grade",
    "literature": "history_grade",
    "humanities": "history_grade",
    "journalism": "history_grade",
    "history": "history_grade",
    "politics": "history_grade",
    "math": "math_grade",
    "science": "math_grade",
    "biology": "math_grade",
    "environment": "math_grade",
    "business": "math_grade",
    "culinary": "art_grade"
}

ratings = []
for student in students:
    reviewed_programs = random.sample(programs_data, 5)
    student_interest_tokens = set(student["interests"].split())
    for prog in reviewed_programs:
        domain = prog["domain"]
        grade_key = domain_grade_map.get(domain, "technology_grade")
        grade = student.get(grade_key, student["technology_grade"])
        base_rating = 3
        if grade >= 15:
            base_rating = 5
        elif grade <= 7:
            base_rating = 1
        
        # Bonus si les centres d'intérêt contiennent une balise du programme
        tags = set(prog["tags"].split())
        if student_interest_tokens & tags:
            base_rating = min(5, base_rating + 1)
        elif base_rating == 5 and not (student_interest_tokens & tags):
            base_rating = 4
        
        ratings.append({
            "student_id": student["student_id"],
            "program_id": prog["id"],
            "rating": base_rating
        })

df_ratings = pd.DataFrame(ratings)
df_ratings.to_csv('data/ratings.csv', index=False)

print("SUCCÈS : 3 fichiers CSV créés dans le dossier /data !")