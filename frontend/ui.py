import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime

# Configuration
API_URL = "http://127.0.0.1:8000"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback_log.csv")

st.set_page_config(
    page_title="Program Recommender",
    page_icon="🎓",
    layout="wide"
)

# CSS styling - Force dark text on light background
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top, #f2fbff 0%, #eef4ff 40%, #f8f9fb 100%);
        color: #212529;
        font-family: "Segoe UI", "Helvetica Neue", sans-serif;
        font-size: 1.05rem;
    }
    h1 {
        font-size: 2.4rem !important;
    }
    h2 {
        font-size: 1.8rem !important;
    }
    h3 {
        font-size: 1.35rem !important;
    }
    p, span, label, div {
        color: #1f2933 !important;
        font-size: 1.05rem !important;
    }
    .main > div {
        padding: 1.5rem 2rem 3rem 2rem;
    }
    .stTabs [role="tab"] {
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 999px;
        border: 1px solid transparent;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: linear-gradient(120deg, #6366f1, #8b5cf6);
        color: #fff !important;
        box-shadow: 0 15px 30px rgba(99, 102, 241, 0.35);
    }
    .stTabs [role="tab"]:hover {
        border-color: #cdd5ff;
    }
    .recommendation-card {
        padding: 1.25rem;
        border-radius: 22px;
        background: #fff;
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.08);
        margin-bottom: 1.5rem;
    }
    .stButton > button {
        border-radius: 999px;
        background: linear-gradient(120deg, #10b981, #22d3ee);
        border: none;
        color: #fff;
        font-weight: 600;
        box-shadow: 0 15px 35px rgba(16, 185, 129, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 20px 40px rgba(16, 185, 129, 0.4);
    }
    .star-button button {
        background: rgba(99, 102, 241, 0.08);
        color: #4f46e5;
        border-radius: 14px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        font-size: 0.9rem;
    }
    .star-button button:hover {
        background: rgba(99, 102, 241, 0.16);
        border-color: #7c3aed;
    }
    .metric-row div [data-testid="stMetricValue"] {
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def check_api_status():
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_recommendations(interests, math_grade, art_grade, history_grade, technology_grade):
    try:
        payload = {
            "interests": interests,
            "math_grade": math_grade,
            "art_grade": art_grade,
            "history_grade": history_grade,
            "technology_grade": technology_grade
        }
        response = requests.post(f"{API_URL}/recommend", json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except:
        st.error("Cannot connect to API. Start backend with: python backend/app.py")
        return None

def submit_feedback(student_id: int, program_id: int, program_name: str, rating: int):
    """Submit feedback to backend with proper types"""
    try:
        payload = {
            "student_id": student_id,
            "program_id": program_id,
            "program_name": program_name,
            "rating": rating
        }
        print(f"[DEBUG] Sending feedback: {payload}")
        response = requests.post(f"{API_URL}/feedback", json=payload, timeout=5)
        print(f"[DEBUG] Response status: {response.status_code}")
        print(f"[DEBUG] Response body: {response.text}")
        if response.status_code == 200:
            return True
        else:
            st.error(f"Backend error: {response.text}")
            return False
    except Exception as e:
        st.error(f"Feedback error: {str(e)}")
        print(f"[ERROR] Exception in submit_feedback: {e}")
        return False

def register_student(interests: str, math_grade: float, art_grade: float, history_grade: float, technology_grade: float):
    """Register a new student and get their ID"""
    try:
        payload = {
            "interests": interests,
            "math_grade": math_grade,
            "art_grade": art_grade,
            "history_grade": history_grade,
            "technology_grade": technology_grade
        }
        response = requests.post(f"{API_URL}/register_student", json=payload, timeout=5)
        if response.status_code == 200:
            return response.json().get("student_id")
        else:
            print(f"Register error: {response.text}")
        return None
    except Exception as e:
        print(f"Register exception: {e}")
        return None

def load_feedback_data():
    """Load feedback data from data/feedback_log.csv"""
    if os.path.exists(FEEDBACK_FILE):
        try:
            df = pd.read_csv(FEEDBACK_FILE)
            return df
        except:
            pass
    return pd.DataFrame()

# Main app with tabs
def main():
    st.title("🎓 Student Program Recommender System")
    if 'current_student_id' not in st.session_state:
        st.session_state['current_student_id'] = None
    if 'last_results' not in st.session_state:
        st.session_state['last_results'] = None
    if 'has_requested' not in st.session_state:
        st.session_state['has_requested'] = False
    
    # Create tabs
    tab1, tab2 = st.tabs(["📝 Get Recommendations", "📊 Admin Dashboard"])
    
    # TAB 1: Recommendations
    with tab1:
        # Check API
        if not check_api_status():
            st.error("⚠️ Backend API is offline. Run: python backend/app.py")
            return
        
        st.success("✅ Connected to API")
        st.markdown("---")
        
        # Input section with checkboxes for interests
        st.subheader("🎯 Select Your Interests")
        
        interest_options = {
            "coding": "💻 Coding",
            "math": "🔢 Mathematics", 
            "drawing": "🎨 Drawing",
            "art": "🖼️ Art",
            "reading": "📚 Reading",
            "logic": "🧠 Logic",
            "ai": "🤖 AI",
            "design": "✏️ Design",
            "stats": "📊 Statistics",
            "software": "⚙️ Software"
        }
        
        # Create 5 columns for checkboxes
        cols = st.columns(5)
        selected_interests = []
        
        for i, (key, label) in enumerate(interest_options.items()):
            with cols[i % 5]:
                if st.checkbox(label, key=f"interest_{key}"):
                    selected_interests.append(key)
        
        # Additional text input for custom interests
        custom = st.text_input("➕ Add more interests (space-separated)", placeholder="e.g., biology chemistry")
        
        if custom:
            selected_interests.extend(custom.split())
        
        interests_text = " ".join(selected_interests)
        
        st.markdown("---")
        st.subheader("📊 Your Grades (0-20)")
        
        col1, col2 = st.columns(2)
        with col1:
            math_grade = st.slider("Math", 0.0, 20.0, 10.0, 0.5)
            history_grade = st.slider("History", 0.0, 20.0, 10.0, 0.5)
        with col2:
            art_grade = st.slider("Art", 0.0, 20.0, 10.0, 0.5)
            technology_grade = st.slider("Technology", 0.0, 20.0, 10.0, 0.5)
        
        st.markdown("---")
        
        # Get recommendations
        if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
            if not interests_text.strip():
                st.warning("⚠️ Please select at least one interest")
                return
            
            with st.spinner("🔍 Finding best programs for you..."):
                # First register the student to get an ID
                student_id = register_student(
                    interests_text,
                    math_grade,
                    art_grade,
                    history_grade,
                    technology_grade
                )
                
                if student_id:
                    st.session_state['current_student_id'] = student_id
                else:
                    st.session_state['current_student_id'] = None
                
                results = get_recommendations(
                    interests_text,
                    math_grade,
                    art_grade,
                    history_grade,
                    technology_grade
                )
                st.session_state['last_results'] = results if results else None
                st.session_state['has_requested'] = True
        
        student_id = st.session_state.get('current_student_id')
        results_data = st.session_state.get('last_results')
        
        if student_id:
            st.info(f"📋 Your Student ID: {student_id}")
        
        if results_data and results_data.get("recommendations"):
            st.markdown("---")
            st.subheader("📚 Recommended Programs")
            
            for i, rec in enumerate(results_data["recommendations"], 1):
                st.markdown("<div class='recommendation-card'>", unsafe_allow_html=True)
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"### {i}. {rec['program_name']}")
                        st.write(f"**Domain:** {rec['domain']} | **Tags:** {rec['tags']}")
                        st.info(f"💡 {rec['explanation']}")
                    
                    with col2:
                        st.metric("Match", f"{int(rec['score']*100)}%")
                    
                    st.write("**Rate this recommendation:**")
                    
                    if not student_id:
                        st.warning("⚠️ Student ID not found. Feedback disabled.")
                        print(f"[ERROR] student_id is None!")
                    else:
                        print(f"[DEBUG] student_id = {student_id}")
                    
                    rating_cols = st.columns(5)
                    
                    for star_num in range(1, 6):
                        with rating_cols[star_num - 1]:
                            st.markdown("<div class='star-button'>", unsafe_allow_html=True)
                            if st.button("⭐" * star_num, key=f"star_{i}_{star_num}", use_container_width=True, disabled=(student_id is None)):
                                if student_id:
                                    if submit_feedback(student_id, rec['program_id'], rec['program_name'], star_num):
                                        st.success(f"✅ Rated {star_num}/5 stars!")
                                        st.balloons()
                                else:
                                    st.error("Cannot submit feedback without student ID")
                            st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        elif st.session_state.get('has_requested'):
            st.warning("No recommendations found")
    
    # TAB 2: Admin Dashboard
    with tab2:
        st.header("📊 Admin Dashboard - Feedback Metrics")
        
        feedback_df = load_feedback_data()
        
        if feedback_df.empty:
            st.info("No feedback data yet. Users need to provide feedback first.")
        else:
            st.success(f"✅ Loaded {len(feedback_df)} feedback entries")
            
            # Calculate metrics
            total_feedback = len(feedback_df)
            positive_feedback = len(feedback_df[feedback_df['rating'] >= 4])
            negative_feedback = len(feedback_df[feedback_df['rating'] <= 2])
            acceptance_rate = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Feedback", total_feedback)
            with col2:
                st.metric("👍 Positive", positive_feedback)
            with col3:
                st.metric("👎 Negative", negative_feedback)
            with col4:
                st.metric("Acceptance Rate", f"{acceptance_rate:.1f}%")
            
            st.markdown("---")
            
            # Show recent feedback
            st.subheader("Recent Feedback")
            st.dataframe(feedback_df.tail(20), use_container_width=True)
            
            # Program recommendations stats
            if 'program_id' in feedback_df.columns:
                st.markdown("---")
                st.subheader("📚 Programs Performance")
                
                # Load programs to get names
                programs_path = os.path.join(DATA_DIR, "programs.csv")
                if os.path.exists(programs_path):
                    programs_df = pd.read_csv(programs_path)
                    # Merge to get program names
                    merged = feedback_df.merge(programs_df[['id', 'name']], left_on='program_id', right_on='id', how='left')
                    
                    program_stats = merged.groupby('name').agg({
                        'rating': ['count', 'mean']
                    }).round(2)
                    program_stats.columns = ['Count', 'Avg Rating']
                    program_stats = program_stats.sort_values('Avg Rating', ascending=False)
                    
                    st.dataframe(program_stats, use_container_width=True)

if __name__ == "__main__":
    main()
