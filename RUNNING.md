# 🎓 Student Program Recommender - Quick Start Guide

## Running the Application

### 1. Start the Backend API (Terminal 1)
```bash
cd c:\Mlops
python backend/app.py
```
**Expected output:** `Uvicorn running on http://127.0.0.1:8000`

### 2. Start the Frontend UI (Terminal 2)
```bash
cd c:\Mlops
streamlit run frontend/ui.py
```
**Expected output:** Opens browser at `http://localhost:8501`

### 3. Use the Application
1. Fill in your Student ID in the sidebar
2. Select your interests (coding, math, art, etc.)
3. Adjust your grade sliders (0-20 scale)
4. Click "🚀 Get Recommendations"
5. View your personalized program matches!
6. Provide feedback to improve recommendations

---

## Troubleshooting

**Backend not starting?**
- Make sure port 8000 is available
- Check that all packages are installed: `pip install -r requirements.txt`

**Frontend shows "API is offline"?**
- Ensure backend is running on port 8000
- Try: `curl http://127.0.0.1:8000` to test connection

**No recommendations?**
- Check that models/ folder contains .pkl file
- Re-run training: `python scripts/train.py`

---

## Architecture

```
Frontend (Streamlit) → Backend (FastAPI) → ML Model (scikit-learn)
   Port 8501              Port 8000           models/*.pkl
```
