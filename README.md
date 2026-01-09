# Student Program Recommender System

An end-to-end hybrid recommendation platform for students combining FastAPI, Streamlit, and MLflow. The backend serves personalized program recommendations powered by TF-IDF and TruncatedSVD, while the frontend provides an interactive web interface with feedback collection and admin insights.

## Project Structure

```
.
├── backend/               # FastAPI service exposing recommendation & feedback APIs
├── frontend/              # Streamlit UI for students and admins
├── scripts/               # Data generation and training utilities
├── models/                # Trained recommender pickle artifacts
├── data/                  # CSV datasets + feedback logs
├── Dockerfile.backend     # Backend image recipe
├── Dockerfile.frontend    # Frontend image recipe
├── docker-compose.yml     # Multi-container orchestrator
├── requirements.txt       # Python dependencies
└── README.md
```

## Prerequisites

- Python 3.10+
- pip / virtualenv
- Docker Desktop (optional but recommended for deployment)

## Local Development Setup

1. **Clone & Enter Project**
   ```bash
   git clone <your-repo-url>
   cd Mlops
   ```

2. **Create & Activate Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate     # macOS/Linux
   venv\Scripts\activate       # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Backend (FastAPI)**
   ```bash
   uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
   ```

5. **Run Frontend (Streamlit)**
   ```bash
   export API_URL=http://127.0.0.1:8000  # PowerShell: $env:API_URL="http://127.0.0.1:8000"
   streamlit run frontend/ui.py
   ```

6. **Access UI**
   - Frontend: http://localhost:8501
   - Backend docs: http://localhost:8000/docs

## Docker & Docker Compose

1. **Build & Run containers**
   ```bash
   docker-compose up --build        # first time / when dependencies change
   docker-compose up                # subsequent runs
   ```

2. **Services**
   - `mon-api-reco`: FastAPI backend (http://localhost:8000)
   - `mon-ui-reco`: Streamlit frontend (http://localhost:8501)

3. **Environment Wiring**
   - Frontend receives `API_URL=http://backend:8000` automatically (Compose service name).
   - Models and data directories are mounted for hot-swapping artifacts without rebuilding images.

4. **Stop Stack**
   ```bash
   docker-compose down
   ```

## Training the Recommender

1. **Prerequisites**
   - Ensure `data/students.csv`, `data/programs.csv`, and `data/ratings.csv` exist (use `scripts/generate_data.py` if needed).
   - Optional: track experiments with MLflow (default local `mlruns/`).

2. **Run Training Pipeline**
   ```bash
   python scripts/train.py
   ```

3. **Outputs**
   - New model saved under `models/model-<run_id>.pkl`
   - `model_config.json` and `metrics_summary.json`
   - Metrics logged to MLflow (`mlruns/` or DB if configured)

4. **Updating Backend**
   - Backend loads `ClassifierModel.pkl` by default. Replace/rename the latest trained artifact or set `RECOMMENDER_MODEL_FILE` env var before starting backend.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_URL` | Streamlit -> FastAPI base URL | `http://127.0.0.1:8000` locally, `http://backend:8000` in Docker |
| `RECOMMENDER_MODEL_FILE` | Backend target pickle filename | `ClassifierModel.pkl` |
| `MLFLOW_TRACKING_URI` | Optional custom tracking URI | `mlruns` |

Set env vars via shell, `.env`, or `streamlit secrets` as needed.

## Feedback & Logging

- User ratings are stored in `data/feedback_log.csv` (mounted volume in Docker).
- Inference requests logged to `data/inference_log.csv` for diagnostics.
- Admin tab in Streamlit shows basic KPIs and latest feedback.

## Troubleshooting

- **UI shows “Cannot connect to API”**: ensure backend is running and `API_URL` points to it (localhost outside Docker, `backend` inside Compose).
- **Docker version warning**: Compose `version` field is deprecated; can remove it without impact.
- **Scikit-learn InconsistentVersionWarning**: align scikit-learn version inside Docker with the one used to pickle the model.
- **Ports in use**: stop local services (`Ctrl+C` or `docker-compose down`) before restarting to free ports 8000/8501.

## Useful Commands

```bash
# Lint data / inspect logs
tail -f data/feedback_log.csv

# View MLflow UI
mlflow ui --host 127.0.0.1 --port 5000

# Regenerate synthetic data
python scripts/generate_data.py
```

## License

MIT (update if different). Feel free to adapt and extend the system for your own educational recommendation scenarios.
