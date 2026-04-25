# Sesi 6: Bangun Project AI (End-to-End) - Versi Google Colab
## Panduan Lengkap untuk Coding di Google Colab

---

# 📌 Persiapan: Mengatur Struktur di Google Colab

```python
# ============================================
# CELL 1: Mount Google Drive untuk menyimpan proyek
# ============================================
from google.colab import drive
drive.mount('/content/drive')

# Buat folder proyek di Google Drive
import os

PROJECT_PATH = '/content/drive/MyDrive/project_ai_sesi6'
os.makedirs(PROJECT_PATH, exist_ok=True)
os.makedirs(f"{PROJECT_PATH}/data/raw", exist_ok=True)
os.makedirs(f"{PROJECT_PATH}/data/processed", exist_ok=True)
os.makedirs(f"{PROJECT_PATH}/src", exist_ok=True)
os.makedirs(f"{PROJECT_PATH}/models", exist_ok=True)
os.makedirs(f"{PROJECT_PATH}/notebooks", exist_ok=True)
os.makedirs(f"{PROJECT_PATH}/logs", exist_ok=True)

print(f"✅ Project created at: {PROJECT_PATH}")
```

```python
# ============================================
# CELL 2: Install dependencies tambahan
# ============================================
!pip install -q xgboost optuna mlflow joblib

# Untuk Colab, MLflow perlu konfigurasi khusus
!pip install -q mlflow dbapi_sqlite
```

---

# 📁 Bagian 1: Struktur Proyek di Colab

## 1.1 Membuat File-File Python (.py) di Colab

Google Colab memungkinkan kita membuat file `.py` menggunakan `%%writefile` magic command.

```python
# ============================================
# CELL 3: Membuat file config.py
# ============================================
%%writefile {PROJECT_PATH}/src/config.py
"""
Konfigurasi sentral untuk seluruh proyek
"""

import os
from pathlib import Path

# Project root (akan di-update saat runtime)
PROJECT_ROOT = Path(os.getenv('PROJECT_PATH', '/content/drive/MyDrive/project_ai_sesi6'))

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = MODELS_DIR / "metrics"

# Log paths
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, METRICS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model Parameters - Oil Regression
OIL_RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'random_state': 42
}

OIL_XGB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 5,
    'random_state': 42
}

# Model Parameters - Diabetes Classification
DIABETES_XGB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'random_state': 42
}

# Model Parameters - Mall Clustering
MALL_KMEANS_PARAMS = {
    'n_clusters': 5,
    'random_state': 42,
    'n_init': 10
}

# Data split configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

```python
# ============================================
# CELL 4: Membuat file preprocess.py
# ============================================
%%writefile {PROJECT_PATH}/src/preprocess.py
"""
Modul preprocessing data: cleaning, feature engineering
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
from pathlib import Path

# Dynamic import for config
PROJECT_PATH = Path('/content/drive/MyDrive/project_ai_sesi6')
sys.path.append(str(PROJECT_PATH))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_STATE

def handle_missing_values(df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
    """Handle missing values"""
    df_clean = df.copy()
    missing_ratio = df_clean.isnull().sum() / len(df_clean)
    
    for col in df_clean.columns:
        if missing_ratio[col] > 0.5:
            df_clean = df_clean.drop(columns=[col])
            print(f"Dropped {col} ({missing_ratio[col]:.1%} missing)")
        elif missing_ratio[col] > 0:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    return df_clean

def remove_outliers_iqr(df: pd.DataFrame, cols: List[str], multiplier: float = 1.5) -> pd.DataFrame:
    """Remove outliers using IQR"""
    df_clean = df.copy()
    initial_len = len(df_clean)
    
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    
    print(f"Removed {initial_len - len(df_clean)} outliers")
    return df_clean

def scale_features(df: pd.DataFrame, scaler: Optional[StandardScaler] = None, 
                   fit: bool = True) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale numeric features"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if fit:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[numeric_cols])
    else:
        scaled_values = scaler.transform(df[numeric_cols])
    
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaled_values
    return df_scaled, scaler

def create_synthetic_oil_data(n_samples=1000) -> pd.DataFrame:
    """Membuat synthetic data untuk oil (karena dataset asli mungkin tidak tersedia)"""
    np.random.seed(42)
    
    data = {
        'temperature': np.random.normal(75, 10, n_samples),
        'pressure': np.random.normal(300, 50, n_samples),
        'flow_rate': np.random.normal(500, 100, n_samples),
        'viscosity': np.random.normal(30, 5, n_samples),
        'water_content': np.random.uniform(0, 20, n_samples),
        'depth': np.random.uniform(1000, 3000, n_samples),
        'operating_hours': np.random.randint(100, 1000, n_samples),
        # Target dengan hubungan non-linear
        'oil_output': np.random.normal(1000, 200, n_samples)
    }
    
    # Tambahkan hubungan realistis
    df = pd.DataFrame(data)
    df['oil_output'] = (
        500 + 
        0.5 * df['temperature'] + 
        1.2 * df['pressure'] - 
        0.3 * df['viscosity'] +
        0.1 * df['flow_rate'] +
        np.random.normal(0, 50, n_samples)
    )
    
    # Simpan ke raw
    df.to_csv(RAW_DATA_DIR / 'oil_production.csv', index=False)
    print(f"✅ Synthetic oil data saved: {n_samples} rows")
    return df

def create_synthetic_diabetes_data(n_samples=1000) -> pd.DataFrame:
    """Membuat synthetic data untuk diabetes"""
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(20, 80, n_samples),
        'bmi': np.random.normal(28, 6, n_samples),
        'glucose': np.random.normal(120, 30, n_samples),
        'blood_pressure': np.random.normal(120, 15, n_samples),
        'insulin': np.random.normal(80, 40, n_samples),
        'skin_thickness': np.random.normal(25, 10, n_samples),
        'pregnancies': np.random.randint(0, 15, n_samples),
        'diabetes_pedigree': np.random.uniform(0.05, 2.5, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Target: diabetes positif jika kombinasi tertentu
    df['diabetes'] = ((df['glucose'] > 140) & (df['bmi'] > 30)).astype(int)
    df.loc[df['glucose'] > 180, 'diabetes'] = 1
    df.loc[(df['glucose'] < 100) & (df['bmi'] < 25), 'diabetes'] = 0
    
    df.to_csv(RAW_DATA_DIR / 'diabetes.csv', index=False)
    print(f"✅ Synthetic diabetes data saved: {n_samples} rows")
    print(f"   Imbalance ratio: {df['diabetes'].value_counts(normalize=True)[1]:.1%} positive")
    return df

def create_synthetic_mall_data(n_samples=500) -> pd.DataFrame:
    """Membuat synthetic data untuk mall customers"""
    np.random.seed(42)
    
    # 5 segmen customer yang berbeda
    segments = {
        0: {'age': (18, 30), 'income': (15000, 35000), 'score': (1, 40)},   # Muda, income rendah
        1: {'age': (25, 40), 'income': (40000, 70000), 'score': (40, 60)},  # Professional muda
        2: {'age': (35, 50), 'income': (60000, 100000), 'score': (50, 70)}, # Middle management
        3: {'age': (45, 65), 'income': (80000, 150000), 'score': (60, 85)},  # Senior
        4: {'age': (50, 70), 'income': (50000, 90000), 'score': (20, 50)}     # Pensiun
    }
    
    data = []
    for segment, params in segments.items():
        n = n_samples // 5
        age = np.random.randint(params['age'][0], params['age'][1], n)
        income = np.random.uniform(params['income'][0], params['income'][1], n)
        score = np.random.uniform(params['score'][0], params['score'][1], n)
        data.append(pd.DataFrame({'age': age, 'annual_income': income, 
                                  'spending_score': score, 'segment': segment}))
    
    df = pd.concat(data, ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    
    df.to_csv(RAW_DATA_DIR / 'mall_customers.csv', index=False)
    print(f"✅ Synthetic mall data saved: {len(df)} rows, 5 segments")
    return df

if __name__ == "__main__":
    # Test sintetik data
    print("Creating synthetic datasets...")
    create_synthetic_oil_data(1000)
    create_synthetic_diabetes_data(1000)
    create_synthetic_mall_data(500)
```

```python
# ============================================
# CELL 5: Membuat file train.py
# ============================================
%%writefile {PROJECT_PATH}/src/train.py
"""
Modul training untuk berbagai model
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, silhouette_score
import xgboost as xgb

PROJECT_PATH = Path('/content/drive/MyDrive/project_ai_sesi6')
sys.path.append(str(PROJECT_PATH))

from src.config import MODELS_DIR, METRICS_DIR, RANDOM_STATE, TEST_SIZE
from src.preprocess import scale_features

def train_oil_regression(df_path=None, model_type='random_forest', tune=False):
    """Train model untuk oil regression"""
    
    # Load data
    if df_path is None:
        df = pd.read_csv(PROJECT_PATH / 'data/processed/oil_clean.csv')
    else:
        df = pd.read_csv(df_path)
    
    # Prepare features and target
    target_col = 'oil_output'
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Scale features
    X_train_scaled, scaler = scale_features(X_train, fit=True)
    X_test_scaled, _ = scale_features(X_test, scaler=scaler, fit=False)
    
    # Train model
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=200 if tune else 100,
            max_depth=15 if tune else 10,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model_name = f"oil_rf_{'tuned' if tune else 'baseline'}"
    else:
        model = xgb.XGBRegressor(
            n_estimators=300 if tune else 100,
            learning_rate=0.03 if tune else 0.1,
            max_depth=6 if tune else 4,
            random_state=RANDOM_STATE
        )
        model_name = f"oil_xgb_{'tuned' if tune else 'baseline'}"
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    metrics = {
        'model_name': model_name,
        'model_type': model_type,
        'tuned': tune,
        'r2_score': float(r2),
        'rmse': float(rmse),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'n_features': X.shape[1],
        'n_samples': len(df),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save model and scaler
    joblib.dump(model, MODELS_DIR / f"{model_name}.pkl")
    joblib.dump(scaler, MODELS_DIR / f"{model_name}_scaler.pkl")
    
    # Save metrics
    with open(METRICS_DIR / f"{model_name}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ {model_name} - R2: {r2:.4f}, RMSE: {rmse:.2f}")
    return model, metrics

def train_diabetes_classifier(df_path=None, use_smote=False):
    """Train classifier untuk diabetes"""
    
    if df_path is None:
        df = pd.read_csv(PROJECT_PATH / 'data/processed/diabetes_clean.csv')
    else:
        df = pd.read_csv(df_path)
    
    feature_cols = [col for col in df.columns if col != 'diabetes']
    X = df[feature_cols]
    y = df['diabetes']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # SMOTE jika diperlukan
    if use_smote:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"SMOTE applied - new training size: {len(X_train)}")
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        'model_name': 'diabetes_xgb',
        'use_smote': use_smote,
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'timestamp': datetime.now().isoformat()
    }
    
    joblib.dump(model, MODELS_DIR / "diabetes_xgb.pkl")
    
    with open(METRICS_DIR / "diabetes_xgb_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Diabetes Classifier - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
    return model, metrics

def train_mall_clustering(df_path=None, n_clusters=5):
    """Train clustering untuk mall customers"""
    
    if df_path is None:
        df = pd.read_csv(PROJECT_PATH / 'data/processed/mall_clean.csv')
    else:
        df = pd.read_csv(df_path)
    
    # Features untuk clustering
    feature_cols = ['age', 'annual_income', 'spending_score']
    X = df[feature_cols]
    
    # Scale
    X_scaled, scaler = scale_features(X, fit=True)
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Evaluate
    silhouette = silhouette_score(X_scaled, labels)
    
    # Inertia (WCSS)
    inertia = kmeans.inertia_
    
    metrics = {
        'model_name': 'mall_kmeans',
        'n_clusters': n_clusters,
        'silhouette_score': float(silhouette),
        'inertia': float(inertia),
        'n_samples': len(df),
        'timestamp': datetime.now().isoformat()
    }
    
    joblib.dump(kmeans, MODELS_DIR / "mall_kmeans.pkl")
    joblib.dump(scaler, MODELS_DIR / "mall_kmeans_scaler.pkl")
    
    # Save cluster labels to CSV
    df_result = df.copy()
    df_result['cluster'] = labels
    df_result.to_csv(PROJECT_PATH / 'data/processed/mall_with_clusters.csv', index=False)
    
    print(f"✅ Mall Clustering (k={n_clusters}) - Silhouette: {silhouette:.4f}")
    return kmeans, metrics

if __name__ == "__main__":
    print("="*50)
    print("Training Models")
    print("="*50)
    
    # Oil regression
    print("\n🔹 Oil Regression Models")
    train_oil_regression(model_type='random_forest', tune=False)
    train_oil_regression(model_type='random_forest', tune=True)
    train_oil_regression(model_type='xgboost', tune=False)
    train_oil_regression(model_type='xgboost', tune=True)
    
    # Diabetes classification
    print("\n🔹 Diabetes Classification")
    train_diabetes_classifier(use_smote=False)
    train_diabetes_classifier(use_smote=True)
    
    # Mall clustering
    print("\n🔹 Mall Clustering")
    train_mall_clustering(n_clusters=5)
    train_mall_clustering(n_clusters=4)
    train_mall_clustering(n_clusters=6)
    
    print("\n✅ All models trained and saved!")

---

# 🎯 6.4 TUGAS: MEMBANGUN MODULAR PROJECT (STEP-BY-STEP)

Sekarang saatnya kamu mempraktikkan semua yang telah dipelajari dengan membangun project terstruktur di Google Colab.

### 📋 Langkah-langkah Pengerjaan:

1. **Persiapan Folder**: 
   - Jalankan Cell 1 untuk membuat folder `/drive/MyDrive/project_ai_sesi6`.
   - Pastikan subfolder `data/`, `src/`, `models/`, dan `logs/` sudah muncul.

2. **Membuat Modul Utama**:
   - Tulis file `config.py` menggunakan `%%writefile`.
   - Tulis file `preprocess.py` (pastikan fungsi `handle_missing_values` dan `remove_outliers_iqr` sudah masuk).
   - Tulis file `train.py` (implementasikan training untuk Regresi, Klasifikasi, atau Klustering).

3. **Eksperimen Training**:
   - Lakukan training minimal 2 model berbeda untuk dataset yang sama (misal: Random Forest vs XGBoost).
   - Catat hasil akurasi/R2 score masing-masing.

4. **Visualisasi Hasil**:
   - Jalankan modul `evaluate.py` untuk melihat visualisasi perbandingan model (Actual vs Predicted atau Confusion Matrix).

5. **Simpan Model**:
   - Pastikan file model `.pkl` dan metrik `.json` tersimpan rapi di folder `models/`.

### 💡 Tips Sukses:
- Jika ada error `ModuleNotFoundError`, pastikan kamu sudah melakukan `sys.path.append(PROJECT_PATH)`.
- Gunakan `!ls -R {PROJECT_PATH}` untuk memverifikasi apakah file `.py` benar-benar sudah terbuat di folder yang benar.

---

> **"Modularitas adalah kunci dari scalability. Model yang rapi akan sangat mudah untuk di-deploy dan dikembangkan oleh tim lain."**
```

```python
# ============================================
# CELL 6: Membuat file evaluate.py
# ============================================
%%writefile {PROJECT_PATH}/src/evaluate.py
"""
Modul evaluasi model dengan visualisasi
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import json

PROJECT_PATH = Path('/content/drive/MyDrive/project_ai_sesi6')
sys.path.append(str(PROJECT_PATH))

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

def plot_regression_results(y_test, y_pred, model_name, save_path=None):
    """Plot regression evaluation results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'{model_name}\nActual vs Predicted')
    
    # 2. Residuals
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    
    # 3. Residual Distribution
    axes[2].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Residuals')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Residual Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_classification_results(y_test, y_pred, y_pred_proba, model_name, save_path=None):
    """Plot classification evaluation results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = np.trapz(tpr, fpr)
    axes[1].plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.3f}')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = np.trapz(precision, recall)
    axes[2].plot(recall, precision, label=f'PR-AUC = {pr_auc:.3f}')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('Precision-Recall Curve')
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_clustering_results(X, labels, centers, model_name, save_path=None):
    """Plot clustering results (2D projection)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Jika lebih dari 2 dimensi, gunakan 2 fitur pertama
    if X.shape[1] > 2:
        X_plot = X.iloc[:, :2] if hasattr(X, 'iloc') else X[:, :2]
        centers_plot = centers[:, :2] if centers.ndim == 2 else centers[:2]
    else:
        X_plot = X
        centers_plot = centers
    
    # 1. Scatter plot clusters
    scatter = axes[0].scatter(X_plot.iloc[:, 0] if hasattr(X_plot, 'iloc') else X_plot[:, 0],
                              X_plot.iloc[:, 1] if hasattr(X_plot, 'iloc') else X_plot[:, 1],
                              c=labels, cmap='viridis', alpha=0.6)
    axes[0].scatter(centers_plot[:, 0], centers_plot[:, 1], 
                    c='red', marker='X', s=200, edgecolors='black')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].set_title(f'{model_name}\nClusters')
    plt.colorbar(scatter, ax=axes[0])
    
    # 2. Distribution of cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    axes[1].bar(unique, counts, edgecolor='black')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title('Cluster Sizes')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def compare_experiments(metrics_list):
    """Bandingkan multiple experiments"""
    df_metrics = pd.DataFrame(metrics_list)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sort and plot
    if 'r2_score' in df_metrics.columns:
        df_sorted = df_metrics.sort_values('r2_score', ascending=False)
        axes[0].barh(df_sorted['model_name'], df_sorted['r2_score'], color='steelblue')
        axes[0].set_xlabel('R2 Score')
        axes[0].set_title('Model Comparison - R2 Score')
        
        # RMSE comparison
        axes[1].barh(df_sorted['model_name'], df_sorted['rmse'], color='coral')
        axes[1].set_xlabel('RMSE')
        axes[1].set_title('Model Comparison - RMSE')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Evaluation module ready")
```

```python
# ============================================
# CELL 7: Membuat file experiment_tracker.py (MLflow alternative)
# ============================================
%%writefile {PROJECT_PATH}/src/experiment_tracker.py
"""
Simple experiment tracker (MLflow alternative untuk Colab)
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path

PROJECT_PATH = Path('/content/drive/MyDrive/project_ai_sesi6')
EXPERIMENT_FILE = PROJECT_PATH / 'logs/experiments.json'

class ExperimentTracker:
    """Simple experiment tracker without MLflow dependency"""
    
    def __init__(self, experiment_name="default"):
        self.experiment_name = experiment_name
        self.runs = []
        self.load()
    
    def load(self):
        """Load existing experiments"""
        if EXPERIMENT_FILE.exists():
            with open(EXPERIMENT_FILE, 'r') as f:
                data = json.load(f)
                self.runs = data.get('runs', [])
    
    def save(self):
        """Save experiments to file"""
        data = {
            'experiment_name': self.experiment_name,
            'runs': self.runs,
            'last_updated': datetime.now().isoformat()
        }
        with open(EXPERIMENT_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    def log_run(self, model_name, params, metrics, tags=None):
        """Log a single run"""
        run = {
            'run_id': len(self.runs) + 1,
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'params': params,
            'metrics': metrics,
            'tags': tags or {}
        }
        self.runs.append(run)
        self.save()
        print(f"✅ Logged run {run['run_id']}: {model_name}")
    
    def get_runs_as_df(self):
        """Convert runs to DataFrame for analysis"""
        records = []
        for run in self.runs:
            record = {
                'run_id': run['run_id'],
                'model_name': run['model_name'],
                'timestamp': run['timestamp']
            }
            # Add metrics
            for k, v in run['metrics'].items():
                record[f'metric_{k}'] = v
            # Add params
            for k, v in run['params'].items():
                record[f'param_{k}'] = v
            records.append(record)
        return pd.DataFrame(records)
    
    def compare_best_model(self, metric='r2_score'):
        """Get best model based on metric"""
        df = self.get_runs_as_df()
        metric_col = f'metric_{metric}'
        if metric_col in df.columns:
            best = df.loc[df[metric_col].idxmax()]
            return best
        return None
    
    def plot_comparison(self):
        """Plot comparison of all runs"""
        import matplotlib.pyplot as plt
        
        df = self.get_runs_as_df()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Metrics comparison
        metric_cols = [c for c in df.columns if c.startswith('metric_')]
        if metric_cols:
            for col in metric_cols:
                axes[0].bar(df['run_id'].astype(str), df[col], label=col.replace('metric_', ''))
            axes[0].set_xlabel('Run ID')
            axes[0].set_ylabel('Score')
            axes[0].set_title('Metrics Comparison')
            axes[0].legend()
            axes[0].tick_params(axis='x', rotation=45)
        
        # Parameter comparison
        param_cols = [c for c in df.columns if c.startswith('param_')]
        if param_cols:
            df_params = df[['run_id'] + param_cols].set_index('run_id')
            df_params.T.plot(kind='bar', ax=axes[1])
            axes[1].set_title('Parameter Values')
            axes[1].legend(title='Run ID')
        
        plt.tight_layout()
        plt.show()

# Simple MLflow simulation
class SimpleMLflow:
    """Mock MLflow for compatibility"""
    
    def __init__(self):
        self.tracker = ExperimentTracker()
    
    def start_run(self, run_name):
        self.current_run_name = run_name
        self.current_params = {}
        self.current_metrics = {}
        print(f"📊 Started run: {run_name}")
        return self
    
    def log_param(self, key, value):
        self.current_params[key] = value
        return self
    
    def log_metric(self, key, value):
        self.current_metrics[key] = value
        return self
    
    def end_run(self):
        self.tracker.log_run(
            model_name=self.current_run_name,
            params=self.current_params,
            metrics=self.current_metrics
        )
        print(f"✅ Ended run: {self.current_run_name}")
        return self

if __name__ == "__main__":
    # Test experiment tracker
    tracker = ExperimentTracker("sesi6_demo")
    
    # Log dummy experiment
    tracker.log_run(
        model_name="RandomForest_Baseline",
        params={"n_estimators": 100, "max_depth": 10},
        metrics={"r2_score": 0.72, "rmse": 45.3}
    )
    
    tracker.log_run(
        model_name="RandomForest_Tuned",
        params={"n_estimators": 200, "max_depth": 20},
        metrics={"r2_score": 0.78, "rmse": 40.1}
    )
    
    print("\n" + "="*50)
    print("Experiment Tracker Demo")
    print("="*50)
    print(tracker.get_runs_as_df())
    
    best = tracker.compare_best_model('r2_score')
    print(f"\nBest model: {best['model_name']} with R2={best['metric_r2_score']:.4f}")
```

---

# 🚀 Bagian 2: Menjalankan Pipeline End-to-End

```python
# ============================================
# CELL 8: Generate synthetic data
# ============================================
import sys
from pathlib import Path

PROJECT_PATH = '/content/drive/MyDrive/project_ai_sesi6'
sys.path.append(PROJECT_PATH)

from src.preprocess import (
    create_synthetic_oil_data, 
    create_synthetic_diabetes_data, 
    create_synthetic_mall_data
)

print("="*60)
print("📊 GENERATING SYNTHETIC DATASETS")
print("="*60)

# Generate data
oil_df = create_synthetic_oil_data(2000)
diabetes_df = create_synthetic_diabetes_data(1500)
mall_df = create_synthetic_mall_data(500)

print("\n✅ All datasets created!")
print(f"   - Oil: {len(oil_df)} rows")
print(f"   - Diabetes: {len(diabetes_df)} rows, imbalance: {diabetes_df['diabetes'].mean():.1%}")
print(f"   - Mall: {len(mall_df)} rows")
```

```python
# ============================================
# CELL 9: Run preprocessing untuk semua dataset
# ============================================
from src.preprocess import handle_missing_values, remove_outliers_iqr

print("="*60)
print("🔧 PREPROCESSING DATA")
print("="*60)

# Process Oil data
print("\n1. Processing Oil Data...")
oil_df = pd.read_csv(f"{PROJECT_PATH}/data/raw/oil_production.csv")
oil_df = handle_missing_values(oil_df)
# No outlier removal for oil (keep all)
oil_df.to_csv(f"{PROJECT_PATH}/data/processed/oil_clean.csv", index=False)
print(f"   ✅ Saved to processed/oil_clean.csv ({len(oil_df)} rows)")

# Process Diabetes data
print("\n2. Processing Diabetes Data...")
diabetes_df = pd.read_csv(f"{PROJECT_PATH}/data/raw/diabetes.csv")
diabetes_df = handle_missing_values(diabetes_df)
diabetes_df.to_csv(f"{PROJECT_PATH}/data/processed/diabetes_clean.csv", index=False)
print(f"   ✅ Saved to processed/diabetes_clean.csv ({len(diabetes_df)} rows)")

# Process Mall data
print("\n3. Processing Mall Data...")
mall_df = pd.read_csv(f"{PROJECT_PATH}/data/raw/mall_customers.csv")
mall_df = handle_missing_values(mall_df)
mall_df.to_csv(f"{PROJECT_PATH}/data/processed/mall_clean.csv", index=False)
print(f"   ✅ Saved to processed/mall_clean.csv ({len(mall_df)} rows)")

print("\n✅ Preprocessing complete!")
```

```python
# ============================================
# CELL 10: Train all models
# ============================================
from src.train import *

print("="*60)
print("🤖 MODEL TRAINING")
print("="*60)

# Dictionary to store all metrics
all_metrics = {}

# Oil Regression Models
print("\n" + "─"*40)
print("OIL REGRESSION")
print("─"*40)

# Random Forest Baseline
model, metrics = train_oil_regression(model_type='random_forest', tune=False)
all_metrics['oil_rf_baseline'] = metrics

# Random Forest Tuned
model, metrics = train_oil_regression(model_type='random_forest', tune=True)
all_metrics['oil_rf_tuned'] = metrics

# XGBoost Baseline
model, metrics = train_oil_regression(model_type='xgboost', tune=False)
all_metrics['oil_xgb_baseline'] = metrics

# XGBoost Tuned
model, metrics = train_oil_regression(model_type='xgboost', tune=True)
all_metrics['oil_xgb_tuned'] = metrics

# Diabetes Classification
print("\n" + "─"*40)
print("DIABETES CLASSIFICATION")
print("─"*40)

model, metrics = train_diabetes_classifier(use_smote=False)
all_metrics['diabetes_no_smote'] = metrics

model, metrics = train_diabetes_classifier(use_smote=True)
all_metrics['diabetes_smote'] = metrics

# Mall Clustering
print("\n" + "─"*40)
print("MALL CLUSTERING")
print("─"*40)

model, metrics = train_mall_clustering(n_clusters=4)
all_metrics['mall_k4'] = metrics

model, metrics = train_mall_clustering(n_clusters=5)
all_metrics['mall_k5'] = metrics

model, metrics = train_mall_clustering(n_clusters=6)
all_metrics['mall_k6'] = metrics

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
```

```python
# ============================================
# CELL 11: Experiment Tracking dengan tracker
# ============================================
from src.experiment_tracker import ExperimentTracker

print("="*60)
print("📈 EXPERIMENT TRACKING")
print("="*60)

# Initialize tracker
tracker = ExperimentTracker("project_ai_sesi6")

# Log all experiments to tracker
for name, metrics in all_metrics.items():
    tracker.log_run(
        model_name=name,
        params={k: v for k, v in metrics.items() if 'param' not in k and k not in ['timestamp', 'model_name']},
        metrics={k: v for k, v in metrics.items() if k in ['r2_score', 'rmse', 'accuracy', 'roc_auc', 'silhouette_score']},
        tags={'dataset': name.split('_')[0], 'type': 'model'}
    )

# Show comparison
print("\n" + "─"*40)
print("MODEL COMPARISON")
print("─"*40)

df_runs = tracker.get_runs_as_df()
display(df_runs)

# Find best models
print("\n🏆 BEST MODELS:")

# Best oil model
oil_models = df_runs[df_runs['model_name'].str.contains('oil')]
if len(oil_models) > 0 and 'metric_r2_score' in oil_models.columns:
    best_oil = oil_models.loc[oil_models['metric_r2_score'].idxmax()]
    print(f"   Oil Regression: {best_oil['model_name']} (R2={best_oil['metric_r2_score']:.4f})")

# Best diabetes model
diabetes_models = df_runs[df_runs['model_name'].str.contains('diabetes')]
if len(diabetes_models) > 0 and 'metric_roc_auc' in diabetes_models.columns:
    best_diab = diabetes_models.loc[diabetes_models['metric_roc_auc'].idxmax()]
    print(f"   Diabetes: {best_diab['model_name']} (ROC-AUC={best_diab['metric_roc_auc']:.4f})")

# Best clustering
cluster_models = df_runs[df_runs['model_name'].str.contains('mall')]
if len(cluster_models) > 0 and 'metric_silhouette_score' in cluster_models.columns:
    best_cluster = cluster_models.loc[cluster_models['metric_silhouette_score'].idxmax()]
    print(f"   Clustering: {best_cluster['model_name']} (Silhouette={best_cluster['metric_silhouette_score']:.4f})")
```

```python
# ============================================
# CELL 12: Visualisasi Hasil
# ============================================
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("📊 VISUALIZATION RESULTS")
print("="*60)

# 1. Oil Model Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

oil_metrics = {k: v for k, v in all_metrics.items() if 'oil' in k}
model_names = list(oil_metrics.keys())
r2_scores = [oil_metrics[m]['r2_score'] for m in model_names]
rmse_scores = [oil_metrics[m]['rmse'] for m in model_names]

axes[0].barh(model_names, r2_scores, color='steelblue')
axes[0].set_xlabel('R2 Score')
axes[0].set_title('Oil Regression - R2 Score Comparison')
for i, v in enumerate(r2_scores):
    axes[0].text(v + 0.01, i, f'{v:.3f}', va='center')

axes[1].barh(model_names, rmse_scores, color='coral')
axes[1].set_xlabel('RMSE')
axes[1].set_title('Oil Regression - RMSE Comparison')
for i, v in enumerate(rmse_scores):
    axes[1].text(v + 1, i, f'{v:.1f}', va='center')

plt.tight_layout()
plt.show()

# 2. Diabetes Model Comparison
fig, ax = plt.subplots(figsize=(8, 5))

diabetes_metrics = {k: v for k, v in all_metrics.items() if 'diabetes' in k}
model_names = list(diabetes_metrics.keys())
roc_auc_scores = [diabetes_metrics[m]['roc_auc'] for m in model_names]

bars = ax.bar(model_names, roc_auc_scores, color=['lightcoral', 'lightblue'])
ax.set_ylim(0.5, 1.0)
ax.set_ylabel('ROC-AUC Score')
ax.set_title('Diabetes Classification - ROC-AUC Comparison')
for bar, score in zip(bars, roc_auc_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{score:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 3. Clustering Comparison
fig, ax = plt.subplots(figsize=(8, 5))

cluster_metrics = {k: v for k, v in all_metrics.items() if 'mall' in k}
model_names = list(cluster_metrics.keys())
silhouette_scores = [cluster_metrics[m]['silhouette_score'] for m in model_names]

bars = ax.bar(model_names, silhouette_scores, color='mediumseagreen')
ax.set_ylabel('Silhouette Score')
ax.set_title('Mall Clustering - Silhouette Score Comparison')
ax.axhline(y=0.5, color='orange', linestyle='--', label='Good threshold (0.5)')
ax.legend()

for bar, score in zip(bars, silhouette_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{score:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n✅ Visualizations complete!")
```

```python
# ============================================
# CELL 13: Create prediction function for inference
# ============================================
%%writefile {PROJECT_PATH}/src/predict.py
"""
Inference module untuk prediction
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_PATH = Path('/content/drive/MyDrive/project_ai_sesi6')
MODELS_DIR = PROJECT_PATH / 'models'

class OilPredictor:
    """Predictor for oil production"""
    
    def __init__(self, model_name='oil_xgb_tuned'):
        self.model = joblib.load(MODELS_DIR / f"{model_name}.pkl")
        self.scaler = joblib.load(MODELS_DIR / f"{model_name}_scaler.pkl")
        self.model_name = model_name
    
    def predict(self, features):
        """
        Predict oil output
        
        Parameters:
        -----------
        features: dict or DataFrame
            Should contain: temperature, pressure, flow_rate, viscosity, 
                           water_content, depth, operating_hours
        
        Returns:
        --------
        float: predicted oil output
        """
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features
        
        # Scale features
        df_scaled, _ = self.scaler.transform(df)
        
        # Predict
        prediction = self.model.predict(df_scaled)
        return prediction[0]

class DiabetesPredictor:
    """Predictor for diabetes"""
    
    def __init__(self):
        self.model = joblib.load(MODELS_DIR / "diabetes_xgb.pkl")
    
    def predict(self, features):
        """
        Predict diabetes
        
        Returns:
        --------
        tuple: (prediction, probability)
        """
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features
        
        pred_class = self.model.predict(df)[0]
        pred_proba = self.model.predict_proba(df)[0][1]
        
        return {
            'diabetes': bool(pred_class),
            'probability': float(pred_proba),
            'risk_level': 'High' if pred_proba > 0.7 else 'Medium' if pred_proba > 0.3 else 'Low'
        }

class MallClusterPredictor:
    """Predictor for customer clustering"""
    
    def __init__(self):
        self.model = joblib.load(MODELS_DIR / "mall_kmeans.pkl")
        self.scaler = joblib.load(MODELS_DIR / "mall_kmeans_scaler.pkl")
    
    def predict(self, features):
        """
        Predict customer cluster
        
        Returns:
        --------
        dict: cluster information
        """
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features
        
        df_scaled = self.scaler.transform(df)
        cluster = self.model.predict(df_scaled)[0]
        
        cluster_labels = {
            0: 'Young & Low Income',
            1: 'Young Professional',
            2: 'Middle Management',
            3: 'Senior Executive',
            4: 'Retirement'
        }
        
        return {
            'cluster_id': int(cluster),
            'cluster_label': cluster_labels.get(cluster, 'Unknown'),
            'segment': ['Standard', 'Premium', 'Luxury'][cluster % 3] if cluster < 5 else 'Standard'
        }

# Test function
if __name__ == "__main__":
    print("Testing predictors...")
    
    # Test oil predictor
    oil_pred = OilPredictor()
    test_features = {
        'temperature': 80,
        'pressure': 350,
        'flow_rate': 550,
        'viscosity': 28,
        'water_content': 5,
        'depth': 2000,
        'operating_hours': 500
    }
    result = oil_pred.predict(test_features)
    print(f"Oil Prediction: {result:.2f} barrels")
    
    # Test diabetes predictor
    diabetes_pred = DiabetesPredictor()
    test_diabetes = {
        'age': 45,
        'bmi': 32,
        'glucose': 160,
        'blood_pressure': 130,
        'insulin': 90,
        'skin_thickness': 28,
        'pregnancies': 2,
        'diabetes_pedigree': 0.85
    }
    result = diabetes_pred.predict(test_diabetes)
    print(f"Diabetes Prediction: {result}")
```

```python
# ============================================
# CELL 14: Test inference functions
# ============================================
from src.predict import OilPredictor, DiabetesPredictor, MallClusterPredictor

print("="*60)
print("🔮 INFERENCE TESTING")
print("="*60)

# Test Oil Predictor
print("\n1. Oil Production Predictor")
oil_predictor = OilPredictor('oil_xgb_tuned')

test_oil = {
    'temperature': 82,
    'pressure': 360,
    'flow_rate': 520,
    'viscosity': 29,
    'water_content': 4.5,
    'depth': 2100,
    'operating_hours': 480
}

oil_pred = oil_predictor.predict(test_oil)
print(f"   Input: {test_oil}")
print(f"   Predicted oil output: {oil_pred:.2f} barrels")

# Test Diabetes Predictor
print("\n2. Diabetes Predictor")
diabetes_predictor = DiabetesPredictor()

test_diabetes_high = {
    'age': 55, 'bmi': 35, 'glucose': 170, 'blood_pressure': 140,
    'insulin': 100, 'skin_thickness': 30, 'pregnancies': 3, 'diabetes_pedigree': 1.2
}

test_diabetes_low = {
    'age': 30, 'bmi': 22, 'glucose': 90, 'blood_pressure': 110,
    'insulin': 50, 'skin_thickness': 18, 'pregnancies': 0, 'diabetes_pedigree': 0.3
}

result_high = diabetes_predictor.predict(test_diabetes_high)
result_low = diabetes_predictor.predict(test_diabetes_low)

print(f"   High risk case: {result_high}")
print(f"   Low risk case: {result_low}")

# Test Mall Predictor
print("\n3. Mall Customer Cluster Predictor")
mall_predictor = MallClusterPredictor()

test_customers = [
    {'age': 25, 'annual_income': 30000, 'spending_score': 35},
    {'age': 45, 'annual_income': 85000, 'spending_score': 75},
    {'age': 60, 'annual_income': 60000, 'spending_score': 40}
]

for i, customer in enumerate(test_customers, 1):
    cluster = mall_predictor.predict(customer)
    print(f"   Customer {i}: Age={customer['age']}, Income={customer['annual_income']}")
    print(f"     → {cluster['cluster_label']} ({cluster['segment']})")

print("\n✅ Inference test complete!")
```

```python
# ============================================
# CELL 15: Create final report
# ============================================
print("="*60)
print("📋 FINAL PROJECT REPORT")
print("="*60)

report = f"""
PROJECT AI: END-TO-END PIPELINE
================================

📁 PROJECT STRUCTURE
-------------------
Location: {PROJECT_PATH}

Directories:
- data/raw/         : Raw synthetic datasets
- data/processed/   : Cleaned datasets
- src/              : Modular Python code
- models/           : Saved models (.pkl)
- logs/             : Experiment logs
- notebooks/        : (Can add Jupyter notebooks)

📊 DATASETS
-----------
1. Oil Production: {len(oil_df)} rows, 8 features
   Target: oil_output (regression)

2. Diabetes: {len(diabetes_df)} rows, 9 features
   Target: diabetes (classification)
   Imbalance: {diabetes_df['diabetes'].mean():.1%} positive

3. Mall Customers: {len(mall_df)} rows, 4 features
   Unsupervised clustering

🤖 MODELS TRAINED
----------------
Oil Regression (4 models):
- RandomForest (baseline & tuned)
- XGBoost (baseline & tuned)

Diabetes Classification (2 models):
- XGBoost (with & without SMOTE)

Mall Clustering (3 models):
- K-Means (k=4,5,6)

🏆 BEST PERFORMANCE
------------------
Oil Regression: {best_oil['model_name'] if 'best_oil' in dir() else 'N/A'}
   R2 Score: {best_oil['metric_r2_score']:.4f if 'best_oil' in dir() else 'N/A'}

Diabetes: {best_diab['model_name'] if 'best_diab' in dir() else 'N/A'}
   ROC-AUC: {best_diab['metric_roc_auc']:.4f if 'best_diab' in dir() else 'N/A'}

Clustering: {best_cluster['model_name'] if 'best_cluster' in dir() else 'N/A'}
   Silhouette: {best_cluster['metric_silhouette_score']:.4f if 'best_cluster' in dir() else 'N/A'}

📈 EXPERIMENT TRACKING
---------------------
All experiments logged to: {PROJECT_PATH}/logs/experiments.json
Total runs: {len(df_runs)}

🔮 INFERENCE READY
-----------------
3 predictor classes available:
- OilPredictor()
- DiabetesPredictor()
- MallClusterPredictor()

📦 DEPLOYMENT READY
------------------
For next session (Sesi 7): Build API with FastAPI or Streamlit
"""

print(report)

# Save report
with open(f"{PROJECT_PATH}/PROJECT_REPORT.txt", 'w') as f:
    f.write(report)

print(f"\n✅ Report saved to {PROJECT_PATH}/PROJECT_REPORT.txt")
```

```python
# ============================================
# CELL 16: Final verification - show all files
# ============================================
import os

print("="*60)
print("📁 PROJECT FILES VERIFICATION")
print("="*60)

def list_files(path, indent=0):
    for item in sorted(os.listdir(path)):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            print("  " * indent + f"📁 {item}/")
            list_files(item_path, indent + 1)
        else:
            print("  " * indent + f"📄 {item}")

list_files(PROJECT_PATH)

print("\n" + "="*60)
print("✅ PROJECT AI END-TO-END COMPLETE!")
print("="*60)
print(f"\nNext step: Sesi 7 - Deployment (API & Streamlit)")
print(f"Project location: {PROJECT_PATH}")
```

---

# 📝 Ringkasan dan Checklist

## Apa yang Sudah Dibangun:

| Komponen | Status | File Location |
|----------|--------|---------------|
| **Struktur Proyek** | ✅ | `/content/drive/MyDrive/project_ai_sesi6/` |
| **Config Module** | ✅ | `src/config.py` |
| **Preprocessing Module** | ✅ | `src/preprocess.py` |
| **Training Module** | ✅ | `src/train.py` |
| **Evaluation Module** | ✅ | `src/evaluate.py` |
| **Experiment Tracker** | ✅ | `src/experiment_tracker.py` |
| **Inference Module** | ✅ | `src/predict.py` |
| **Synthetic Data** | ✅ | `data/raw/*.csv` |
| **Processed Data** | ✅ | `data/processed/*.csv` |
| **Saved Models** | ✅ | `models/*.pkl` |
| **Experiment Logs** | ✅ | `logs/experiments.json` |
| **Project Report** | ✅ | `PROJECT_REPORT.txt` |

## Best Practices yang Diterapkan:

```python
# ✅ Separation of Concerns
src/preprocess.py  # Hanya preprocessing
src/train.py       # Hanya training
src/predict.py     # Hanya inference

# ✅ DRY Principle
# Fungsi preprocessing reusable untuk semua dataset

# ✅ Configuration Centralization
# Semua path dan parameter di config.py

# ✅ Experiment Tracking
# Semua run tercatat di JSON

# ✅ Modular Code
# Setiap fungsi bisa diimport dan di-test

# ✅ Version Control Ready
# Struktur folder siap untuk Git
```
