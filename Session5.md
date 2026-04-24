# Sesi 5: Optimasi dan Evaluasi Model AI

## Tujuan Pembelajaran
- Memahami perbedaan parameter dan hyperparameter.
- Mampu melakukan tuning hyperparameter dengan berbagai metode.
- Mengoptimasi model regresi, klasifikasi, dan clustering.
- Melakukan evaluasi mendalam untuk mendeteksi overfitting/underfitting.

---

## 5.1 Hyperparameter Tuning

### Perbedaan Parameter vs Hyperparameter

| Parameter | Hyperparameter |
|-----------|----------------|
| Dipelajari secara otomatis oleh model dari data | Ditetapkan oleh pengguna sebelum proses training |
| Contoh: koefisien regresi, bobot neuron | Contoh: learning rate, jumlah pohon (n_estimators) |
| Berubah selama training | Tetap selama training |

### Metode Hyperparameter Tuning

#### 1. GridSearchCV (Exhaustive)
- **Prinsip**: Mencoba semua kombinasi hyperparameter yang ditentukan.
- **Kelebihan**: Pasti menemukan kombinasi terbaik dalam grid.
- **Kekurangan**: Komputasi sangat berat untuk grid besar.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
```

#### 2. RandomizedSearchCV (Sampling)
- **Prinsip**: Mengambil sampel acak dari distribusi hyperparameter.
- **Kelebihan**: Lebih efisien untuk ruang hyperparameter besar.
- **Kekurangan**: Bisa melewatkan kombinasi optimal.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [10, 20, None],
    'min_samples_split': randint(2, 20)
}
random_search = RandomizedSearchCV(model, param_dist, n_iter=50, cv=5)
```

#### 3. Optuna (Bayesian Optimization) - Advanced
- **Prinsip**: Menggunakan history trial sebelumnya untuk memprediksi hyperparameter berikutnya.
- **Kelebihan**: Lebih efisien dari random search, bisa handling parameter conditional.

```python
import optuna

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split
    )
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

---

## 5.2 Optimasi untuk Regresi (Oil Dataset)

### Tuning RandomForestRegressor

**Dataset**: Oil production data (target: oil output)

**Hyperparameters utama**:
- `n_estimators`: jumlah pohon (default 100)
- `max_depth`: kedalaman maksimal pohon (default None)
- `min_samples_split`: minimal sampel untuk split node (default 2)

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load data oil
X_train, X_test, y_train, y_test = train_test_split(X_oil, y_oil, test_size=0.2)

# Before tuning
rf_before = RandomForestRegressor(random_state=42)
rf_before.fit(X_train, y_train)
y_pred_before = rf_before.predict(X_test)
print(f"Before tuning - R2: {r2_score(y_test, y_pred_before):.4f}")

# After tuning with GridSearch
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), 
                       param_grid, cv=5, scoring='r2')
grid_rf.fit(X_train, y_train)
rf_after = grid_rf.best_estimator_
y_pred_after = rf_after.predict(X_test)
print(f"After tuning - R2: {r2_score(y_test, y_pred_after):.4f}")
print(f"Best params: {grid_rf.best_params_}")
```

### XGBoost Regressor (Lebih Powerful)

```python
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print(f"XGBoost R2: {r2_score(y_test, y_pred_xgb):.4f}")
```

### Perbandingan Performa

| Model | R2 Score | RMSE |
|-------|----------|------|
| RandomForest (before tuning) | 0.72 | 45.3 |
| RandomForest (after tuning) | 0.78 | 40.1 |
| XGBoost (default) | 0.81 | 37.2 |
| XGBoost (tuned) | 0.84 | 34.5 |

---

## 5.3 Optimasi untuk Klasifikasi (Diabetes Dataset)

### Tuning XGBoost Classifier

**Hyperparameters utama**:
- `learning_rate`: kecepatan belajar (default 0.3)
- `max_depth`: kedalaman pohon (default 6)
- `subsample`: proporsi sampel per pohon (default 1.0)

### Handle Imbalance dengan SMOTE

```python
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, classification_report

# Handle imbalance
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Tuning dengan Optuna
def objective(trial):
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    
    model = XGBClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train_bal, y_train_bal, 
                           cv=5, scoring='roc_auc').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best model
best_xgb = XGBClassifier(**study.best_params, random_state=42)
best_xgb.fit(X_train_bal, y_train_bal)
y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

### ROC-AUC sebagai Metric Utama

ROC-AUC lebih baik dari akurasi untuk data imbalance karena mengukur kemampuan model membedakan kelas.

---

## 5.4 Optimasi untuk Clustering (Mall Customer Dataset)

### Mencari K Optimal dengan Silhouette Score

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np

# Mencari K terbaik untuk K-Means
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_mall)
    score = silhouette_score(X_mall, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette={score:.4f}")

best_k = K_range[np.argmax(silhouette_scores)]
print(f"Best K: {best_k}")

# K-Means with best K
kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_kmeans = kmeans_best.fit_predict(X_mall)
```

### K-Means vs DBSCAN (Tanpa Perlu Menentukan K)

```python
# DBSCAN - automatically determines number of clusters
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_mall)

n_clusters_db = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = list(labels_dbscan).count(-1)

print(f"DBSCAN - Clusters: {n_clusters_db}, Noise points: {n_noise}")

# Compare with K-Means
sil_kmeans = silhouette_score(X_mall, labels_kmeans)
sil_dbscan = silhouette_score(X_mall, labels_dbscan) if n_clusters_db > 1 else -1
print(f"K-Means Silhouette: {sil_kmeans:.4f}")
print(f"DBSCAN Silhouette: {sil_dbscan:.4f}")
```

### Evaluasi Stabilitas Cluster dengan Bootstrap

```python
def cluster_stability(X, n_bootstrap=50, k=5):
    """Ukur stabilitas clustering dengan bootstrap"""
    from sklearn.utils import resample
    from sklearn.metrics import adjusted_rand_score
    
    stabilities = []
    n_samples = len(X)
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        idx_boot = resample(range(n_samples), replace=True, n_samples=n_samples)
        X_boot = X[idx_boot]
        
        # Cluster on bootstrap
        kmeans_boot = KMeans(n_clusters=k, random_state=None, n_init=10)
        labels_boot = kmeans_boot.fit_predict(X_boot)
        
        # Cluster on original (subset same indices)
        labels_orig = kmeans_labels[idx_boot]
        
        # Compare cluster assignments
        ari = adjusted_rand_score(labels_orig, labels_boot)
        stabilities.append(ari)
    
    return np.mean(stabilities), np.std(stabilities)

mean_stab, std_stab = cluster_stability(X_mall, k=best_k)
print(f"Stability (ARI): {mean_stab:.4f} +/- {std_stab:.4f}")
```

---

## 5.5 Evaluasi Mendalam

### Regresi: Learning Curve, Residual Plot, Cross-Validation

```python
from sklearn.model_selection import learning_curve, cross_val_score
import matplotlib.pyplot as plt

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X, y, cv=5, scoring='r2',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, test_scores.mean(axis=1), label='CV')
plt.title('Learning Curve')
plt.legend()

# Residual Plot
plt.subplot(1, 3, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Cross-validation scores
cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='r2')
plt.subplot(1, 3, 3)
plt.boxplot(cv_scores)
plt.title(f'CV Scores: mean={cv_scores.mean():.3f}')
plt.tight_layout()
plt.show()
```

### Klasifikasi: Confusion Matrix, ROC-AUC, Precision-Recall Curve

```python
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import seaborn as sns

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')

# ROC Curve
plt.subplot(1, 3, 2)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.3f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

# Precision-Recall Curve
plt.subplot(1, 3, 3)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.tight_layout()
plt.show()
```

### Clustering: Silhouette Plot, Davies-Bouldin Index

```python
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

# Silhouette Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Silhouette plot
silhouette_vals = silhouette_samples(X_mall, labels_kmeans)
y_lower = 10
for i in range(best_k):
    ith_cluster_silhouette = silhouette_vals[labels_kmeans == i]
    ith_cluster_silhouette.sort()
    size = len(ith_cluster_silhouette)
    y_upper = y_lower + size
    color = cm.nipy_spectral(float(i) / best_k)
    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette,
                      facecolor=color, edgecolor=color, alpha=0.7)
    y_lower = y_upper + 10

ax1.axvline(x=silhouette_score(X_mall, labels_kmeans), color='red', linestyle='--')
ax1.set_title('Silhouette Plot')

# Davies-Bouldin Score
db_score = davies_bouldin_score(X_mall, labels_kmeans)
ax2.bar(['K-Means'], [db_score])
ax2.set_title(f'Davies-Bouldin Index: {db_score:.3f}')
ax2.set_ylabel('Score (lower is better)')
plt.tight_layout()
plt.show()
```

### Identifikasi Overfitting/Underfitting

| Indikator | Overfitting | Underfitting | Good Fit |
|-----------|-------------|--------------|----------|
| Train error | Sangat rendah | Tinggi | Rendah |
| Test/CV error | Tinggi | Tinggi | Rendah (dekat train) |
| Gap train-test | Besar (>0.1) | Kecil | Kecil (<0.05) |
| Learning curve | Train rendah, test tinggi | Keduanya tinggi | Keduanya rendah & konvergen |

```python
def diagnose_fit(train_scores, test_scores):
    train_mean = train_scores.mean(axis=1)[-1]
    test_mean = test_scores.mean(axis=1)[-1]
    gap = train_mean - test_mean
    
    if gap > 0.1 and train_mean > 0.9:
        return "OVERFITTING (gap besar)"
    elif train_mean < 0.6 and test_mean < 0.6:
        return "UNDERFITTING (skor rendah)"
    else:
        return "GOOD FIT"
```

---

## Ringkasan

| Tugas | Optimasi | Evaluasi Utama |
|-------|----------|----------------|
| Regresi | GridSearchCV, XGBoost | Learning curve, residual plot, R2 |
| Klasifikasi | ROC-AUC, SMOTE, Optuna | Confusion matrix, ROC-AUC, PR curve |
| Clustering | Silhouette score, DBSCAN | Silhouette plot, Davies-Bouldin |

**Prinsip Golden**: Selalu bandingkan performa sebelum vs setelah tuning, dan gunakan cross-validation yang tepat untuk menghindari overfitting.
