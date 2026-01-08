"""
===============================================================================
HYBRID RECOMMENDATION SYSTEM - TRAINING PIPELINE
===============================================================================
Senior MLOps Engineering - Production-Grade Implementation

This script implements a Hybrid Recommender System combining:
1. Content-Based Filtering (TF-IDF + Cosine Similarity)
2. Collaborative Filtering (SVD Matrix Factorization)
3. Hybrid Ensemble (Weighted Combination)

Features:
- MLflow experiment tracking and model versioning
- Comprehensive error handling and logging
- Data validation and preprocessing
- Evaluation metrics (Precision@K, NDCG, Recall)
- Artifact management
===============================================================================
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(log_dir: str = "./logs") -> logging.Logger:
    """Configure logging with both file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("RecommenderSystem")
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(
        os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    )
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for the recommender system."""
    
    # Data paths
    DATA_DIR = Path(__file__).parent.parent / "data"
    MODELS_DIR = Path(__file__).parent.parent / "models"
    
    # File names
    STUDENTS_CSV = DATA_DIR / "students.csv"
    PROGRAMS_CSV = DATA_DIR / "programs.csv"
    RATINGS_CSV = DATA_DIR / "ratings.csv"
    
    # Model parameters
    TFIDF_MAX_FEATURES = 200
    TFIDF_NGRAM_RANGE = (1, 2)
    SVD_N_COMPONENTS = 6
    SVD_N_ITER = 60
    SVD_MEAN_CENTER = True
    CF_RIDGE_EPS = 1e-3
    
    # Hybrid weights
    CONTENT_WEIGHT = 0.75
    CF_WEIGHT = 0.25
    
    # Evaluation
    PRECISION_K = 5
    RELEVANCE_THRESHOLD = 4  # rating >= 4 is positive
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    MIN_RATINGS_PER_STUDENT = 2
    
    # MLflow
    EXPERIMENT_NAME = "5_Final_K5"
    TRACKING_URI = "mlruns"
    
    # Grade threshold for interest keywords
    HIGH_GRADE_THRESHOLD = 15


# ============================================================================
# DATA LOADING & VALIDATION
# ============================================================================

def load_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and validate input CSV files.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (students_df, programs_df, ratings_df)
        
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If required columns are missing
    """
    logger.info("=" * 70)
    logger.info("LOADING DATA")
    logger.info("=" * 70)
    
    # Load CSV files
    try:
        students = pd.read_csv(config.STUDENTS_CSV)
        programs = pd.read_csv(config.PROGRAMS_CSV)
        ratings = pd.read_csv(config.RATINGS_CSV)
        logger.info("[OK] Data files loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"[ERROR] Data file not found: {e}")
        raise
    
    # Validate required columns
    required_students_cols = {'student_id', 'interests', 'math_grade', 'art_grade', 
                             'history_grade', 'technology_grade'}
    required_programs_cols = {'id', 'name', 'domain', 'tags'}
    required_ratings_cols = {'student_id', 'program_id', 'rating'}
    
    if not required_students_cols.issubset(set(students.columns)):
        missing = required_students_cols - set(students.columns)
        raise ValueError(f"Missing columns in students.csv: {missing}")
    
    if not required_programs_cols.issubset(set(programs.columns)):
        missing = required_programs_cols - set(programs.columns)
        raise ValueError(f"Missing columns in programs.csv: {missing}")
    
    if not required_ratings_cols.issubset(set(ratings.columns)):
        missing = required_ratings_cols - set(ratings.columns)
        raise ValueError(f"Missing columns in ratings.csv: {missing}")
    
    logger.info(f"[STATS] Students:  {students.shape[0]:,} records | {students.shape[1]} features")
    logger.info(f"[STATS] Programs:  {programs.shape[0]:,} records | {programs.shape[1]} features")
    logger.info(f"[STATS] Ratings:   {ratings.shape[0]:,} records | {ratings.shape[1]} features")
    
    # Check for missing values
    missing_students = students.isnull().sum().sum()
    missing_programs = programs.isnull().sum().sum()
    missing_ratings = ratings.isnull().sum().sum()
    
    if missing_students > 0:
        logger.warning(f"[WARN] Found {missing_students} missing values in students dataset")
    if missing_programs > 0:
        logger.warning(f"[WARN] Found {missing_programs} missing values in programs dataset")
    if missing_ratings > 0:
        logger.warning(f"[WARN] Found {missing_ratings} missing values in ratings dataset")
    
    return students, programs, ratings


def filter_ratings_by_min_interactions(
    ratings: pd.DataFrame,
    min_ratings: int
) -> pd.DataFrame:
    """Keep only students who have at least `min_ratings` interactions."""
    user_counts = ratings['student_id'].value_counts()
    valid_users = user_counts[user_counts >= min_ratings].index
    filtered = ratings[ratings['student_id'].isin(valid_users)].copy()
    logger.info("\n[FILTER] Minimum ratings per student: %d", min_ratings)
    logger.info("[FILTER] Students kept: %d", len(valid_users))
    logger.info("[FILTER] Removed interactions: %d", len(ratings) - len(filtered))
    if filtered.empty:
        raise ValueError(
            "Filtering removed all ratings. Reduce MIN_RATINGS_PER_STUDENT or collect more data."
        )
    return filtered


def split_ratings_per_student(
    ratings: pd.DataFrame,
    test_size: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform per-student hold-out using deterministic shuffling."""
    rng = np.random.default_rng(random_state)
    train_indices = []
    test_indices = []
    for student_id, group in ratings.groupby('student_id'):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        n = len(indices)
        if n == 1:
            train_indices.extend(indices.tolist())
            continue
        n_test = max(1, int(round(n * test_size)))
        n_test = min(n - 1, n_test)
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])
    train_df = ratings.loc[train_indices].sort_index().reset_index(drop=True)
    test_df = ratings.loc[test_indices].sort_index().reset_index(drop=True)
    return train_df, test_df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_program_soup(programs: pd.DataFrame) -> pd.Series:
    """
    Create text features for programs by combining domain, tags, and description.
    
    Args:
        programs: Programs dataframe
        
    Returns:
        Series with concatenated text (soup)
    """
    logger.info("\n[FEATURE] Creating program feature vectors (soup)...")
    
    programs['soup'] = (
        programs['domain'].fillna("") + " " +
        programs['tags'].fillna("")
    )
    
    logger.info(f"[OK] Created soup for {len(programs)} programs")
    logger.debug(f"   Sample: {programs['soup'].iloc[0][:80]}...")
    
    return programs['soup']


def create_student_soup(students: pd.DataFrame, config: Config) -> pd.Series:
    """
    Create text features for students by combining interests with grade-based keywords.
    
    Higher grades (>threshold) in specific subjects are treated as implicit interests.
    
    Args:
        students: Students dataframe
        config: Configuration object
        
    Returns:
        Series with student features (soup)
    """
    logger.info("\n[FEATURE] Creating student feature vectors (soup)...")
    
    def student_to_soup(row: pd.Series) -> str:
        """Convert a student record to text representation."""
        soup = str(row['interests']).replace("[", "").replace("]", "").replace("'", "")
        
        # Add domain keywords based on high grades
        if pd.notna(row['math_grade']) and row['math_grade'] > config.HIGH_GRADE_THRESHOLD:
            soup += " math algebra statistics computation"
        
        if pd.notna(row['art_grade']) and row['art_grade'] > config.HIGH_GRADE_THRESHOLD:
            soup += " art drawing design visual creativity"
        
        if pd.notna(row['history_grade']) and row['history_grade'] > config.HIGH_GRADE_THRESHOLD:
            soup += " history law reading analysis literature"
        
        if pd.notna(row['technology_grade']) and row['technology_grade'] > config.HIGH_GRADE_THRESHOLD:
            soup += " technology coding programming ai software"

        grade_token_map = {
            'math_grade': 'math_score',
            'art_grade': 'art_score',
            'history_grade': 'history_score',
            'technology_grade': 'tech_score'
        }
        grade_tokens = []
        for col, token_prefix in grade_token_map.items():
            grade_val = row[col] if col in row else None
            if pd.notna(grade_val):
                bucket = int(round(float(grade_val)))
                grade_tokens.append(f"{token_prefix}_{bucket}")
        if grade_tokens:
            soup += " " + " ".join(grade_tokens)
        
        return soup
    
    students['soup'] = students.apply(student_to_soup, axis=1)
    
    logger.info(f"[OK] Created soup for {len(students)} students")
    logger.debug(f"   Sample: {students['soup'].iloc[0][:80]}...")
    
    return students['soup']


# ============================================================================
# CONTENT-BASED FILTERING
# ============================================================================

def build_content_based_model(
    program_soup: pd.Series,
    student_soup: pd.Series,
    config: Config
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Build content-based similarity matrix using TF-IDF vectorization.
    
    Args:
        program_soup: Program text features
        student_soup: Student text features
        config: Configuration object
        
    Returns:
        Tuple of (similarity_matrix, vectorizer)
    """
    logger.info("\n" + "=" * 70)
    logger.info("BUILDING CONTENT-BASED MODEL")
    logger.info("=" * 70)
    
    logger.info(f"\n[TFIDF] TF-IDF Vectorization:")
    logger.info(f"   Max features: {config.TFIDF_MAX_FEATURES}")
    logger.info(f"   N-gram range: {config.TFIDF_NGRAM_RANGE}")
    
    # Initialize and fit vectorizer
    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        stop_words='english'
    )
    
    # Fit on all text (programs + students)
    all_text = pd.concat([program_soup, student_soup])
    vectorizer.fit(all_text)
    
    # Transform separately
    program_features = vectorizer.transform(program_soup)
    student_features = vectorizer.transform(student_soup)
    
    logger.info(f"\n[OK] Vectorization complete:")
    logger.info(f"   Program feature matrix: {program_features.shape}")
    logger.info(f"   Student feature matrix: {student_features.shape}")
    logger.info(f"   Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    
    # Compute cosine similarity
    logger.info(f"\n[SIMILARITY] Computing cosine similarity...")
    similarity = cosine_similarity(student_features, program_features)
    
    logger.info(f"[OK] Content-based similarity matrix: {similarity.shape}")
    logger.info(f"   Range: [{similarity.min():.3f}, {similarity.max():.3f}]")
    logger.info(f"   Mean: {similarity.mean():.3f}")
    
    return similarity, vectorizer


# ============================================================================
# COLLABORATIVE FILTERING
# ============================================================================

def build_collaborative_model(
    ratings: pd.DataFrame,
    students: pd.DataFrame,
    programs: pd.DataFrame,
    config: Config
) -> Tuple[np.ndarray, TruncatedSVD, Dict[int, int], Dict[int, int]]:
    """
    Build collaborative filtering model using SVD matrix factorization.
    
    Args:
        ratings: Ratings dataframe
        students: Students dataframe
        programs: Programs dataframe
        config: Configuration object
        
    Returns:
        Tuple of (similarity_matrix, svd, student_to_idx, program_to_idx)
    """
    logger.info("\n" + "=" * 70)
    logger.info("BUILDING COLLABORATIVE FILTERING MODEL")
    logger.info("=" * 70)
    
    # Create index mappings
    student_ids = sorted(students['student_id'].unique())
    program_ids = sorted(programs['id'].unique())
    
    student_to_idx = {sid: idx for idx, sid in enumerate(student_ids)}
    program_to_idx = {pid: idx for idx, pid in enumerate(program_ids)}
    
    logger.info(f"\n[MATRIX] Creating interaction matrix:")
    logger.info(f"   Students: {len(student_ids)}")
    logger.info(f"   Programs: {len(program_ids)}")
    logger.info(f"   Max possible interactions: {len(student_ids) * len(program_ids):,}")
    
    # Build interaction matrix
    interaction_matrix = np.zeros((len(student_ids), len(program_ids)))
    
    for _, row in ratings.iterrows():
        if row['student_id'] in student_to_idx and row['program_id'] in program_to_idx:
            student_idx = student_to_idx[row['student_id']]
            program_idx = program_to_idx[row['program_id']]
            interaction_matrix[student_idx, program_idx] = row['rating']
    
    # Convert to sparse matrix
    mask = interaction_matrix > 0
    user_interaction_counts = mask.sum(axis=1)
    user_rating_sums = interaction_matrix.sum(axis=1)
    user_means = np.divide(
        user_rating_sums,
        user_interaction_counts,
        out=np.zeros_like(user_rating_sums),
        where=user_interaction_counts > 0
    )

    if config.SVD_MEAN_CENTER:
        logger.info("[CF] Applying per-student mean centering before SVD.")
        centered_matrix = interaction_matrix - (user_means[:, None] * mask)
    else:
        centered_matrix = interaction_matrix

    interaction_sparse = csr_matrix(centered_matrix)
    sparsity = (1 - len(interaction_sparse.data) / interaction_sparse.size) * 100
    
    logger.info(f"\n[OK] Interaction matrix created:")
    logger.info(f"   Shape: {interaction_sparse.shape}")
    logger.info(f"   Sparsity: {sparsity:.2f}%")
    logger.info(f"   Actual interactions: {len(interaction_sparse.data):,}")
    
    # Apply SVD
    logger.info(f"\n[SVD] Applying TruncatedSVD:")
    logger.info(f"   N components: {config.SVD_N_COMPONENTS}")
    logger.info(f"   N iterations: {config.SVD_N_ITER}")
    
    n_components = min(config.SVD_N_COMPONENTS, min(interaction_sparse.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, n_iter=config.SVD_N_ITER, random_state=42)
    user_factors = svd.fit_transform(interaction_sparse)
    program_factors = svd.components_.T
    
    explained_var = svd.explained_variance_ratio_.sum()
    logger.info(f"\n[OK] SVD decomposition complete:")
    logger.info(f"   User latent factors: {user_factors.shape}")
    logger.info(f"   Program latent factors: {program_factors.shape}")
    logger.info(f"   Explained variance: {explained_var:.2%}")
    
    # Compute similarity
    logger.info(f"\n[CF] Computing CF similarity...")
    cf_similarity_raw = np.dot(user_factors, program_factors.T)
    
    # Normalize to [0, 1]
    cf_min, cf_max = cf_similarity_raw.min(), cf_similarity_raw.max()
    cf_similarity = (cf_similarity_raw - cf_min) / (cf_max - cf_min + config.CF_RIDGE_EPS)
    
    logger.info(f"[OK] CF similarity matrix: {cf_similarity.shape}")
    logger.info(f"   Range: [{cf_similarity.min():.3f}, {cf_similarity.max():.3f}]")
    logger.info(f"   Mean: {cf_similarity.mean():.3f}")
    
    return cf_similarity, svd, student_to_idx, program_to_idx


# ============================================================================
# HYBRID MODEL
# ============================================================================

def build_hybrid_model(
    content_sim: np.ndarray,
    cf_sim: np.ndarray,
    config: Config
) -> np.ndarray:
    """
    Combine content-based and collaborative filtering using weighted average.
    
    Args:
        content_sim: Content-based similarity matrix
        cf_sim: Collaborative filtering similarity matrix
        config: Configuration object
        
    Returns:
        Hybrid similarity matrix
    """
    logger.info("\n" + "=" * 70)
    logger.info("BUILDING HYBRID MODEL")
    logger.info("=" * 70)
    
    # Normalize content similarity
    content_norm = (content_sim - content_sim.min()) / (content_sim.max() - content_sim.min() + 1e-10)
    
    # Weighted combination
    hybrid = config.CONTENT_WEIGHT * content_norm + config.CF_WEIGHT * cf_sim
    
    logger.info(f"\n[OK] Hybrid model created:")
    logger.info(f"   Content weight: {config.CONTENT_WEIGHT} (semantic relevance)")
    logger.info(f"   CF weight: {config.CF_WEIGHT} (latent preferences)")
    logger.info(f"   Hybrid similarity: {hybrid.shape}")
    logger.info(f"   Range: [{hybrid.min():.3f}, {hybrid.max():.3f}]")
    logger.info(f"   Mean: {hybrid.mean():.3f}")
    
    return hybrid


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_precision_at_k(
    rankings: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 3
) -> float:
    """
    Compute Precision@K for a single user.
    
    Precision@K = (# relevant items in top-k) / k
    
    Args:
        rankings: User's ranking scores for all items
        ground_truth: Binary relevance labels (1 = relevant, 0 = not relevant)
        k: Cutoff value
        
    Returns:
        Precision@K score (0 to 1)
    """
    # Get top-k indices
    top_k_indices = np.argsort(rankings)[::-1][:k]
    
    # Count relevant items in top-k
    relevant_in_topk = np.sum(ground_truth[top_k_indices])
    
    # Precision@K
    return relevant_in_topk / k


def compute_ndcg_at_k(
    rankings: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 3
) -> float:
    """Compute NDCG@K for a single user using binary relevance."""
    top_k_indices = np.argsort(rankings)[::-1][:k]
    gains = ground_truth[top_k_indices]
    if gains.sum() == 0:
        return 0.0
    discounts = np.log2(np.arange(1, len(gains) + 1) + 1)
    dcg = np.sum((2 ** gains - 1) / discounts)
    ideal_gains = np.sort(ground_truth)[::-1][:k]
    ideal_discounts = np.log2(np.arange(1, len(ideal_gains) + 1) + 1)
    idcg = np.sum((2 ** ideal_gains - 1) / ideal_discounts)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_recommender(
    similarity_matrix: np.ndarray,
    ratings: pd.DataFrame,
    students: pd.DataFrame,
    programs: pd.DataFrame,
    student_to_idx: Dict[int, int],
    program_to_idx: Dict[int, int],
    config: Config,
    approach_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate recommender system using Precision@K and other metrics.
    
    Args:
        similarity_matrix: Similarity scores matrix
        ratings: Ratings dataframe
        students: Students dataframe
        programs: Programs dataframe
        student_to_idx: Student ID to index mapping
        program_to_idx: Program ID to index mapping
        config: Configuration object
        approach_name: Name of the approach for logging
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"\n[EVAL] Evaluating {approach_name}...")
    
    # Create ground truth matrix
    ground_truth = np.zeros((len(students), len(programs)))
    
    for _, row in ratings.iterrows():
        if row['student_id'] in student_to_idx and row['program_id'] in program_to_idx:
            student_idx = student_to_idx[row['student_id']]
            program_idx = program_to_idx[row['program_id']]
            # Binary relevance: rating >= threshold
            if row['rating'] >= config.RELEVANCE_THRESHOLD:
                ground_truth[student_idx, program_idx] = 1
    
    # Compute Precision@K for each user
    precisions = []
    recalls = []
    ndcgs = []
    
    for user_idx in range(len(similarity_matrix)):
        scores = similarity_matrix[user_idx]
        gt = ground_truth[user_idx]
        
        total_relevant = np.sum(gt)
        if total_relevant == 0:
            continue  # Skip users with no relevant items
        
        precision = compute_precision_at_k(scores, gt, config.PRECISION_K)
        
        # Recall@K
        top_k_indices = np.argsort(scores)[::-1][:config.PRECISION_K]
        recall = np.sum(gt[top_k_indices]) / total_relevant
        ndcg = compute_ndcg_at_k(scores, gt, config.PRECISION_K)
        
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
    
    mean_precision = np.mean(precisions) if precisions else 0.0
    mean_recall = np.mean(recalls) if recalls else 0.0
    mean_ndcg = np.mean(ndcgs) if ndcgs else 0.0
    
    logger.info(f" Evaluation metrics ({approach_name}):")
    logger.info(f"   Precision@{config.PRECISION_K}: {mean_precision:.4f}")
    logger.info(f"   Recall@{config.PRECISION_K}: {mean_recall:.4f}")
    logger.info(f"   NDCG@{config.PRECISION_K}: {mean_ndcg:.4f}")
    logger.info(f"   Users evaluated: {len(precisions)}")
    
    return {
        f"precision_at_{config.PRECISION_K}": mean_precision,
        f"recall_at_{config.PRECISION_K}": mean_recall,
        f"ndcg_at_{config.PRECISION_K}": mean_ndcg,
        "users_evaluated": len(precisions)
    }


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline with MLflow integration."""
    
    try:
        logger.info("\n" + "=" * 70)
        logger.info("STARTING HYBRID RECOMMENDER SYSTEM TRAINING")
        logger.info("=" * 70)
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        config = Config()
        
        # ====================================================================
        # STEP 1: LOAD DATA
        # ====================================================================
        students, programs, ratings = load_data(config)
        ratings = filter_ratings_by_min_interactions(
            ratings,
            config.MIN_RATINGS_PER_STUDENT
        )
        logger.info("\n" + "=" * 70)
        logger.info("SPLITTING RATINGS INTO TRAIN/TEST (per student)")
        logger.info("=" * 70)
        logger.info(
            "[SPLIT] Deterministic per-student split across %d students (test_size=%.2f)",
            ratings['student_id'].nunique(),
            config.TEST_SIZE
        )

        train_ratings, test_ratings = split_ratings_per_student(
            ratings,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE
        )
        logger.info("[SPLIT] Split completed successfully.")
        held_out_students = test_ratings['student_id'].nunique()
        logger.info("   Students with hold-out interactions: %d", held_out_students)

        logger.info(f"   Train ratings: {len(train_ratings):,}")
        logger.info(f"   Test ratings:  {len(test_ratings):,}")
        logger.info(f"   Hold-out ratio: {config.TEST_SIZE:.2%}")
        
        # ====================================================================
        # STEP 2: FEATURE ENGINEERING
        # ====================================================================
        program_soup = create_program_soup(programs)
        student_soup = create_student_soup(students, config)
        
        # ====================================================================
        # STEP 3: BUILD MODELS
        # ====================================================================
        
        # Content-based model
        content_similarity, tfidf_vectorizer = build_content_based_model(
            program_soup, student_soup, config
        )
        
        # Collaborative filtering model
        cf_similarity, svd_model, student_to_idx, program_to_idx = build_collaborative_model(
            train_ratings, students, programs, config
        )
        
        # Hybrid model
        hybrid_similarity = build_hybrid_model(content_similarity, cf_similarity, config)
        
        # ====================================================================
        # STEP 4: EVALUATION
        # ====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("EVALUATING ALL MODELS")
        logger.info("=" * 70)
        
        content_train_metrics = evaluate_recommender(
            content_similarity, train_ratings, students, programs,
            student_to_idx, program_to_idx, config, "Content-Based [train]"
        )
        content_test_metrics = evaluate_recommender(
            content_similarity, test_ratings, students, programs,
            student_to_idx, program_to_idx, config, "Content-Based [test]"
        )
        
        cf_train_metrics = evaluate_recommender(
            cf_similarity, train_ratings, students, programs,
            student_to_idx, program_to_idx, config, "Collaborative Filtering [train]"
        )
        cf_test_metrics = evaluate_recommender(
            cf_similarity, test_ratings, students, programs,
            student_to_idx, program_to_idx, config, "Collaborative Filtering [test]"
        )
        
        hybrid_train_metrics = evaluate_recommender(
            hybrid_similarity, train_ratings, students, programs,
            student_to_idx, program_to_idx, config, "Hybrid [train]"
        )
        hybrid_test_metrics = evaluate_recommender(
            hybrid_similarity, test_ratings, students, programs,
            student_to_idx, program_to_idx, config, "Hybrid [test]"
        )
        
        # ====================================================================
        # STEP 5: MLflow LOGGING
        # ====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("LOGGING TO MLflow")
        logger.info("=" * 70)
        
        mlflow.set_tracking_uri(config.TRACKING_URI)
        mlflow.set_experiment(config.EXPERIMENT_NAME)
        
        # Variable to capture the run_id for the filename
        run_id = None
        
        with mlflow.start_run(run_name=f"hybrid_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            
            run_id = run.info.run_id
            
            # Log parameters
            logger.info("\n Logging parameters...")
            mlflow.log_param("content_weight", config.CONTENT_WEIGHT)
            mlflow.log_param("cf_weight", config.CF_WEIGHT)
            mlflow.log_param("svd_n_components", config.SVD_N_COMPONENTS)
            mlflow.log_param("tfidf_max_features", config.TFIDF_MAX_FEATURES)
            mlflow.log_param("precision_k", config.PRECISION_K)
            mlflow.log_param("relevance_threshold", config.RELEVANCE_THRESHOLD)
            mlflow.log_param("test_size", config.TEST_SIZE)
            mlflow.log_param("random_state", config.RANDOM_STATE)
            logger.info(" Parameters logged")
            
            # Log metrics
            logger.info("\n Logging evaluation metrics...")
            logger.info("   (MLflow: hybrid-only as requested)")
            
            metric_suffix = config.PRECISION_K
            # Hybrid metrics
            mlflow.log_metric(
                f"hybrid_precision_at_{metric_suffix}_train",
                hybrid_train_metrics[f"precision_at_{config.PRECISION_K}"]
            )
            mlflow.log_metric(
                f"hybrid_precision_at_{metric_suffix}_test",
                hybrid_test_metrics[f"precision_at_{config.PRECISION_K}"]
            )
            mlflow.log_metric(
                f"hybrid_recall_at_{metric_suffix}_train",
                hybrid_train_metrics[f"recall_at_{config.PRECISION_K}"]
            )
            mlflow.log_metric(
                f"hybrid_recall_at_{metric_suffix}_test",
                hybrid_test_metrics[f"recall_at_{config.PRECISION_K}"]
            )
            mlflow.log_metric(
                f"hybrid_ndcg_at_{metric_suffix}_train",
                hybrid_train_metrics[f"ndcg_at_{config.PRECISION_K}"]
            )
            mlflow.log_metric(
                f"hybrid_ndcg_at_{metric_suffix}_test",
                hybrid_test_metrics[f"ndcg_at_{config.PRECISION_K}"]
            )
            
            logger.info(" Metrics logged")
            
            # Log artifacts (JSON configs)
            logger.info("\n Logging artifacts...")
            
            config_dict = {
                "tfidf_max_features": config.TFIDF_MAX_FEATURES,
                "tfidf_ngram_range": config.TFIDF_NGRAM_RANGE,
                "svd_n_components": config.SVD_N_COMPONENTS,
                "svd_mean_center": config.SVD_MEAN_CENTER,
                "cf_ridge_eps": config.CF_RIDGE_EPS,
                "content_weight": config.CONTENT_WEIGHT,
                "cf_weight": config.CF_WEIGHT,
                "precision_k": config.PRECISION_K,
                "relevance_threshold": config.RELEVANCE_THRESHOLD
            }
            
            with open("model_config.json", "w") as f:
                json.dump(config_dict, f, indent=2)
            mlflow.log_artifact("model_config.json")
            
            metrics_summary = {
                "train": {
                    "hybrid": hybrid_train_metrics
                },
                "test": {
                    "hybrid": hybrid_test_metrics
                }
            }
            
            with open("metrics_summary.json", "w") as f:
                json.dump(metrics_summary, f, indent=2)
            mlflow.log_artifact("metrics_summary.json")
            
            # ====================================================================
            #  SAVING MODEL TO 'models/' WITH RUN_ID
            # ====================================================================
            import os
            import pickle
            
            # 1. Ensure 'models' directory exists
            output_dir = "models"
            os.makedirs(output_dir, exist_ok=True)
            
            # 2. Create the filename using the Run ID
            # Example: models/model-fee3bcccc76c43f4ae2a0780f67bdfbd.pkl
            model_filename = f"model-{run_id}.pkl"
            local_model_path = os.path.join(output_dir, model_filename)
            
            logger.info(f"\nSaving model to: {local_model_path}...")
            
            # 3. Prepare the object
            recommender = {
                "tfidf_vectorizer": tfidf_vectorizer,
                "svd_model": svd_model,
                "student_to_idx": student_to_idx,
                "program_to_idx": program_to_idx,
                "programs_df": programs, 
                "program_soup": program_soup,
                "config": config_dict
            }
            
            # 4. Save pickle file
            with open(local_model_path, "wb") as f:
                pickle.dump(recommender, f)
            
            # 5. Log it to MLflow as well
            mlflow.log_artifact(local_model_path)
            
            logger.info(f"Model saved successfully as: {model_filename}")
            # ====================================================================

        # ====================================================================
        # STEP 6: SUMMARY & RESULTS
        # ====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE - FINAL RESULTS")
        logger.info("=" * 70)
        
        logger.info(f"\n PRECISION@{config.PRECISION_K} Summary (Train set):")
        logger.info(f"   Content-Based: {content_train_metrics[f'precision_at_{config.PRECISION_K}']:.4f}")
        logger.info(f"   Collaborative:  {cf_train_metrics[f'precision_at_{config.PRECISION_K}']:.4f}")
        logger.info(f"   Hybrid:         {hybrid_train_metrics[f'precision_at_{config.PRECISION_K}']:.4f}")
        
        logger.info(f"\n PRECISION@{config.PRECISION_K} Summary (Test set):")
        logger.info(f"   Content-Based: {content_test_metrics[f'precision_at_{config.PRECISION_K}']:.4f}")
        logger.info(f"   Collaborative:  {cf_test_metrics[f'precision_at_{config.PRECISION_K}']:.4f}")
        logger.info(f"   Hybrid:         {hybrid_test_metrics[f'precision_at_{config.PRECISION_K}']:.4f}")

        logger.info(f"\n NDCG@{config.PRECISION_K} Summary (Train set):")
        logger.info(f"   Content-Based: {content_train_metrics[f'ndcg_at_{config.PRECISION_K}']:.4f}")
        logger.info(f"   Collaborative:  {cf_train_metrics[f'ndcg_at_{config.PRECISION_K}']:.4f}")
        logger.info(f"   Hybrid:         {hybrid_train_metrics[f'ndcg_at_{config.PRECISION_K}']:.4f}")
        
        logger.info(f"\n NDCG@{config.PRECISION_K} Summary (Test set):")
        logger.info(f"   Content-Based: {content_test_metrics[f'ndcg_at_{config.PRECISION_K}']:.4f}")
        logger.info(f"   Collaborative:  {cf_test_metrics[f'ndcg_at_{config.PRECISION_K}']:.4f}")
        logger.info(f"   Hybrid:         {hybrid_test_metrics[f'ndcg_at_{config.PRECISION_K}']:.4f}")
        
        best_train = max(
            [("Content-Based", content_train_metrics[f'precision_at_{config.PRECISION_K}']),
             ("Collaborative", cf_train_metrics[f'precision_at_{config.PRECISION_K}']),
             ("Hybrid", hybrid_train_metrics[f'precision_at_{config.PRECISION_K}'])],
            key=lambda x: x[1]
        )
        best_test = max(
            [("Content-Based", content_test_metrics[f'precision_at_{config.PRECISION_K}']),
             ("Collaborative", cf_test_metrics[f'precision_at_{config.PRECISION_K}']),
             ("Hybrid", hybrid_test_metrics[f'precision_at_{config.PRECISION_K}'])],
            key=lambda x: x[1]
        )
        
        logger.info(f"\n[BEST][TRAIN] {best_train[0]} (Precision@{config.PRECISION_K}: {best_train[1]:.4f})")
        logger.info(f"[BEST][TEST]  {best_test[0]} (Precision@{config.PRECISION_K}: {best_test[1]:.4f})")
        logger.info(f"[RUN_ID] MLflow Run ID: {run_id}")
        logger.info(f"[TRACKING] Tracking URI: {config.TRACKING_URI}")
        
        print("\n" + "=" * 70)
        print("[SUCCESS] TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nFinal Hybrid Precision@{config.PRECISION_K} (test): {hybrid_test_metrics[f'precision_at_{config.PRECISION_K}']:.4f}")
        print(f"Final Hybrid Precision@{config.PRECISION_K} (train): {hybrid_train_metrics[f'precision_at_{config.PRECISION_K}']:.4f}")
        print(f"Final Hybrid NDCG@{config.PRECISION_K} (test): {hybrid_test_metrics[f'ndcg_at_{config.PRECISION_K}']:.4f}")
        print(f"Final Hybrid NDCG@{config.PRECISION_K} (train): {hybrid_train_metrics[f'ndcg_at_{config.PRECISION_K}']:.4f}")
        print(f" MLflow Run ID: {run_id}")
        print(f" Model saved locally at: models/model-{run_id}.pkl")
        print("\nTo view results:")
        print("   $ mlflow ui --host 127.0.0.1 --port 5000")
        print("=" * 70 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n[ERROR] {str(e)}", exc_info=True)
        print(f"\n[FAILED] Training failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
