# ML Pipeline — Depression Detection Project

This document defines the machine‑learning pipeline used by the project and explains each step in detail, including data shapes, assumptions, edge cases, and runnable sklearn-style snippets for clarity.

## Pipeline overview (end-to-end)
1. Data ingestion (training): read Excel files from `data/` and match patient IDs to `clinical_data.py` labels.
2. Preprocessing (EEG): read numeric columns from each Excel file and compute the first three channel means (pad with zeros if fewer than 3).
3. Clinical data extraction: read clinical measures (age, gender, education, PHQ‑9, CTQ, LES, SSRS, GAD‑7, PSQI) from `clinical_data.py` for each patient.
4. Feature concatenation: combine EEG means (3 floats) + clinical vector (9 numbers) → 12‑element vector per sample.
5. Optional feature selection: Quantum‑inspired selector to rank and keep top K features.
6. Scaling: StandardScaler (fit on training data) applied inside model pipelines.
7. Model training: fit classical model pipelines (SVM, RandomForest, KNN, GaussianNB) with GridSearchCV or CV, optionally with quantum-inspired hyperparameter search.
8. Ensemble: VotingClassifier (soft voting) over best base estimators.
9. Evaluation: cross‑validation metrics (accuracy, F1, AUC), ROC curves, confusion matrices, calibration.
10. Persist model and artifacts: save best model to `model.pkl` and store training metadata (CV results, best params, random seed, training date) in `artifacts/`.
11. Inference: load `model.pkl`, compute EEG means + clinical input, apply same preprocessing and predict/predict_proba.

---

## Data shapes and contracts
- EEG means: numpy array shape (3,) — floats [ch1_mean, ch2_mean, ch3_mean]
- Clinical vector: numpy array shape (9,) — [age, gender, education, phq9, ctq, les, ssrs, gad7, psqi]
- Single feature vector (input to model): numpy array shape (12,) — floats
- Training matrix X: shape (n_samples, 12)
- Label vector y: shape (n_samples,) with values {0: HC, 1: MDD}

All inputs should be numeric. The front-end enforces HTML min/max for some clinical scores, but server validation casts values to float and should handle missing values explicitly (e.g., reject, impute, or raise an error depending on policy).

---

## Step-by-step breakdown

1) Data ingestion (train)
- Files: all `*.xls`/`*.xlsx` in `data/`.
- For each file, `infer_label(filename)` checks the filename for a patient id found in the mapping (see `train.py` `mapping`). If unknown, skip or log for manual labeling.
- Outcome: list of rows, each with path, patient_id, label.

2) EEG preprocessing / feature extraction
- Function: `get_means(filepath)`
  - Reads file with `pandas.read_excel(..., engine='openpyxl')` if available.
  - Selects numeric columns: `num_cols = df.select_dtypes(include=[np.number]).columns.tolist()`
  - Takes first three numeric columns `ch = num_cols[:3]` and computes column means `means = df[ch].mean(axis=0).values.astype(float)`.
  - If fewer than 3 numeric columns, pad with zeros to length 3 and add synthetic channel names.
- Edge cases: file with zero numeric columns → raise ValueError and skip with warning.

3) Clinical features extraction
- Function: `get_clinical_features(patient_id)` returns the 9-field tuple from `clinical_data.py`.
- If patient_id missing, training ingestion skips file or requires manual labeling.

4) Feature concatenation
- Build arrays: X_eeg (n,3), X_clinical (n,9) → X = np.hstack([X_eeg, X_clinical]) → shape (n,12)

5) Optional: Quantum‑inspired Feature Selection
- Class: `QuantumInspiredFeatureSelector` in `train.py`.
  - Computes an `interference` score per column (based on FFT phase / summed complex exponentials) and selects top-K features.
  - Use-case: reduce noise and dimensionality (select half by default: n_features // 2).
- Note: This is a heuristic method — compare against standard selection (e.g., SelectKBest) in ablation tests.

6) Preprocessing and scaling
- Use `StandardScaler()` to center and scale features before distance-based models (SVM, KNN) and tree-based models if desired.
- In training code, scaling is added inside each model `Pipeline` to avoid leakage and ensure transform consistency.

7) Model pipelines (training)
- Example sketched pipelines (used in `train.py`):
  - SVM pipeline: Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True))])
  - KNN pipeline: Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
  - RF pipeline: Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier())])
  - NB pipeline: Pipeline([('scaler', StandardScaler()), ('nb', GaussianNB())])

- Hyperparameter selection:
  - SVM: grid over C, gamma, kernel OR quantum-inspired optimization using differential_evolution on log10(C), log10(gamma).
  - RF: grid over n_estimators, max_depth, min_samples_split.
  - KNN: grid over n_neighbors and weights.
  - NB: typically no hyperparameters.

- Use `GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)` for each model when a grid exists.
- Cross-validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`.

8) Ensemble construction
- After training base models, create an ensemble: `VotingClassifier(estimators=[('svm', svm_best), ('rf', rf_best), ...], voting='soft')`.
- Evaluate ensemble with cross_val_score; fit ensemble on full training data before saving.

9) Evaluation
- Primary metrics: Accuracy, F1-score (macro), AUC-ROC. Also report precision and recall per class.
- Compute and save:
  - Confusion matrix
  - ROC curve and AUC (with bootstrapped CIs if desired)
  - Calibration plot (reliability diagram)
  - Feature importance (from RF) and SHAP or permutation importance for model explainability

10) Persisting artifacts
- Save:
  - `joblib.dump(best_model, 'model.pkl')`
  - `artifacts/train_log.json` containing CV results, best params, date, random_state, selected features
  - Optional: `artifacts/feature_selection_indices.npy` (if applying selection)

11) Inference (runtime)
- Steps for single file inference (same as `app.py`):
  1. Read uploaded file and compute EEG means via `get_means()`.
  2. Get clinical fields from form or `clinical_data.py`.
  3. Construct X (1,12) and cast to float.
  4. Load `model = joblib.load('model.pkl')` (done at app startup for performance).
  5. `pred = model.predict(X)` and `prob = model.predict_proba(X)[0,1]` if available.
  6. Save a CSV result record with timestamp, features, pred and prob.

---

## Example training skeleton (sketch)

```py
# pseudo-code (keep in train.py)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV

svm_pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True))])
svm_grid = {'svc__C': [0.1, 1, 10], 'svc__gamma': ['scale','auto',0.1,1]}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
svm_search = GridSearchCV(svm_pipe, svm_grid, cv=cv, scoring='accuracy', n_jobs=-1)
svm_search.fit(X_train, y_train)
svm_best = svm_search.best_estimator_

# Repeat for RF, KNN, NB
ensemble = VotingClassifier(estimators=[('svm',svm_best), ('rf', rf_best), ('knn',knn_best), ('nb', nb_best)], voting='soft')
ensemble_scores = cross_val_score(ensemble, X_train, y_train, cv=cv)
ensemble.fit(X_train, y_train)
joblib.dump(ensemble, 'model.pkl')
```

---

## Edge cases and recommended validations
- Small dataset (n≈40): prefer nested CV or repeated CV for robust performance estimates; report confidence intervals instead of single point estimates.
- Class balance: dataset here is balanced (20 MDD, 20 HC), but always check distribution and consider stratification and class weights if imbalance appears.
- Outliers (LES values range widely): consider robust scaling for features with heavy tails or winsorize extreme values after domain review.
- Missing clinical fields: decide a policy early (reject, impute with median, or ask user to fill). The current app expects clinical fields to be provided on the form.

---

## Reproducibility checklist (for each training run)
- Code commit hash
- Python version and `requirements.txt`/environment
- Random seed(s) for CV and model initialization
- Full dataset listing (`artifacts/data_file_list.txt`)
- Best hyperparameters and CV scores
- Model file `model_v{version}_{YYYYMMDD}.pkl`

---

## Recommendations for next improvements
- Add unit tests for `get_means()` and `get_clinical_features()`.
- Add a small integration test that runs `train.py` on a subset to verify pipeline completeness.
- Add permutation tests and SHAP to improve interpretability and validate feature contributions.
- If quantum-inspired methods are kept, add an ablation section comparing standard selection/optimization vs quantum-inspired variants.

---

End of ML pipeline documentation.
