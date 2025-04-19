
# 🎬 IMDb Movie Analysis with GPU-Accelerated Machine Learning

This project explores the [Full IMDb Dataset](https://www.kaggle.com/datasets/octopusteam/full-imdb-dataset/data) using GPU-accelerated libraries from the RAPIDS ecosystem to perform scalable data preprocessing, feature engineering, regression modeling, and clustering on over 1 million movie records.

---

## 📦 Dataset

- **Source:** [Kaggle - Full IMDb Dataset](https://www.kaggle.com/datasets/octopusteam/full-imdb-dataset/data)
- **Size:** `1,056,870` rows × `7` columns
- **Content:** Structured information about movies, ratings, genres, and more, suitable for large-scale analysis.

---

## 🎯 Objectives

- Efficiently process and analyze large-scale movie data using GPU.
- Train multiple machine learning models for regression tasks.
- Evaluate feature importance using permutation techniques.
- Apply clustering to group similar movies.

---

## 🛠️ Technologies Used

- 🐍 Python 3.11+
- ⚡ RAPIDS (`cuDF`, `cuML`, `CuPy`)
- 📘 scikit-learn
- 📊 Matplotlib, Seaborn
- ☁️ Google Colab with NVIDIA L4 GPU

---

## 🧠 ML Models Implemented

| Model                        | R² Score (GPU) |
|-----------------------------|----------------|
| Gradient Boosted Regressor  | 0.1673         |
| SVM Regressor               | 0.0076         |
| XGBoost Regressor           | 0.0337         |
| Best Cross-Validated Model  | 0.0322         |

> 📉 *Note: The relatively low R² scores suggest further tuning, additional features, or more complex modeling could improve prediction accuracy.*

---

## 🔍 Feature Engineering & Importance

- Encoded categorical variables (e.g., genres)
- Log-transformed skewed features
- Used permutation importance with cuML's Random Forest to evaluate how each feature impacts model performance

---

## 📊 Visualizations

- Correlation Heatmaps
- Permutation Feature Importance Bar Charts
- Model Performance Comparisons
- KMeans Clustering (grouping movies)

---

## 📁 Project Structure

.
├── Mustafa_AlAli.ipynb                   # Main Jupyter notebook with data loading, preprocessing, modeling, and analysis
├── best_svr_model.joblib                 # Serialized Support Vector Regressor (cuML) for IMDb rating prediction
├── best_xgb_model_gridsearch.joblib      # XGBoost model optimized with GridSearchCV (GPU-supported tuning)
├── best_model_xgb.joblib                 # Final XGBoost model with selected features and tuned hyperparameters (GPU-based)
├── best_gbr_model.joblib                 # cuML Gradient Boosted Regressor with best performance (highest R²)
├── requirements.txt                      # Required packages and dependencies for project environment
└── README.md                             # Project overview and instructions
