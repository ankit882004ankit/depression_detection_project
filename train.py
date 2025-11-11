# train.py - trains multiple models with CV, grid search, saves best model
import os, glob, joblib, numpy as np, pandas as pd, argparse, logging
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from scipy.optimize import differential_evolution

# Quantum-inspired feature selection using interference
class QuantumInspiredFeatureSelector:
    def __init__(self, n_features=None):
        self.selected_features = None
        self.n_features = n_features
        
    def quantum_interference(self, X):
        # Simulate quantum interference pattern
        n_samples, n_features = X.shape
        interference = np.zeros(n_features)
        
        for i in range(n_features):
            # Calculate phase differences
            phase = np.angle(np.fft.fft(X[:, i]))
            # Simulate interference effect
            interference[i] = np.abs(np.sum(np.exp(1j * phase)))
            
        return interference
    
    def fit(self, X, y):
        interference_scores = self.quantum_interference(X)
        if self.n_features is None:
            self.n_features = X.shape[1] // 2
            
        # Select features with highest interference scores
        self.selected_features = np.argsort(interference_scores)[-self.n_features:]
        return self
    
    def transform(self, X):
        return X[:, self.selected_features]
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

# Quantum-inspired optimization for hyperparameters
class QuantumInspiredOptimizer:
    def __init__(self, param_bounds, objective_func, population_size=20, max_iter=100):
        self.param_bounds = param_bounds
        self.objective_func = objective_func
        self.population_size = population_size
        self.max_iter = max_iter
    
    def optimize(self):
        result = differential_evolution(
            self.objective_func,
            self.param_bounds,
            popsize=self.population_size,
            maxiter=self.max_iter,
            strategy='best1bin',  # Quantum-inspired mutation strategy
            mutation=(0.5, 1.0),  # Adaptive mutation rates
            recombination=0.7
        )
        return result.x, result.fun
from clinical_data import clinical_data

# basic logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "data"
MODEL_PATH = "model.pkl"

mapping_MDD = [
    # Original MDD patients
    "02010002","02010005","02010006","02010007","02010008",
    "02010009","02010010","02010011","02010012","02010013",
    # New MDD patients
    "02010015","02010016","02010018","02010019","02010020",
    "02010021","02010022","02010024","02010025","02010026"
]
mapping_HC = [
    # Original HC patients
    "02020007","02020008","02020010","02020011","02020014",
    "02020015","02020016","02020018","02020019","02020021",
    # New HC patients
    "02030003","02030004","02030005","02030006","02030007",
    "02030008","02030009","02030016","02030020","02030021"
]
mapping = {k:1 for k in mapping_MDD}  # 1 = MDD
mapping.update({k:0 for k in mapping_HC})  # 0 = HC

def infer_label(fname):
    for k,v in mapping.items():
        if k in fname:
            return v
    return None

def get_means(path):
    # read excel and compute first three numeric column means
    df = pd.read_excel(path, engine="openpyxl")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 1:
        raise ValueError("No numeric columns")
    ch = num_cols[:3]
    means = df[ch].mean(axis=0).values.astype(float)
    if len(means) < 3:
        means = np.pad(means, (0, 3-len(means)), constant_values=0.0)
    return means


def get_clinical_features(patient_id):
    if patient_id not in clinical_data:
        raise ValueError(f"Clinical data not found for patient {patient_id}")
    features = clinical_data[patient_id]
    return np.array([
        features['age'],
        features['gender'],
        features['education'],
        features['phq9'],
        features['ctq'],
        features['les'],
        features['ssrs'],
        features['gad7'],
        features['psqi']
    ])

def train_and_select_best(model_path=MODEL_PATH, data_dir=DATA_DIR, test_size=0.2, random_state=42):
    files = sorted(glob.glob(os.path.join(data_dir, "*.xls*")))
    X_eeg, X_clinical, y = [], [], []
    
    logger.info("Starting quantum-inspired training process...")
    
    for f in files:
        patient_id = None
        for pid in mapping.keys():
            if pid in f:
                patient_id = pid
                break
                
        if patient_id is None:
            logger.info("Skipping (no patient ID match): %s", f)
            continue
            
        lab = mapping[patient_id]
        
        try:
            eeg_means = get_means(f)
            clinical_features = get_clinical_features(patient_id)
        except Exception as e:
            logger.warning("Skip due to error: %s %s", f, e)
            continue
            
        X_eeg.append(eeg_means)
        X_clinical.append(clinical_features)
        y.append(lab)

    if len(y) == 0:
        raise SystemExit("No labeled files found in data/ — check filenames or mapping.")

    # Combine EEG and clinical features
    X_eeg = np.vstack(X_eeg)
    X_clinical = np.vstack(X_clinical)
    X = np.hstack([X_eeg, X_clinical])
    y = np.array(y)
    
    logger.info("Loaded samples: %d (MDD=%d, HC=%d)", len(y), sum(y==1), sum(y==0))
    logger.info("Features: %d (EEG=%d, Clinical=%d)", X.shape[1], X_eeg.shape[1], X_clinical.shape[1])
    
    # Apply quantum-inspired feature selection
    logger.info("Applying quantum-inspired feature selection...")
    qfs = QuantumInspiredFeatureSelector(n_features=X.shape[1] // 2)  # Select half of features
    X_quantum = qfs.fit_transform(X, y)
    logger.info("Selected %d features using quantum interference", len(qfs.selected_features))

    # Define parameter grids for grid search
    svm_params = {
        'svc__C': [0.1, 1.0, 10.0],
        'svc__gamma': ['scale', 'auto', 0.1, 1.0],
        'svc__kernel': ['rbf', 'linear']
    }
    knn_params = {
        'knn__n_neighbors': [3, 5, 7],
        'knn__weights': ['uniform', 'distance']
    }
    rf_params = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [3, 5, None],
        'rf__min_samples_split': [2, 5]
    }

    # Create base models with preprocessing
    svm_pipe = Pipeline([("scaler", StandardScaler()), 
                        ("svc", SVC(probability=True, random_state=random_state))])
    knn_pipe = Pipeline([("scaler", StandardScaler()), 
                        ("knn", KNeighborsClassifier())])
    rf_pipe = Pipeline([("scaler", StandardScaler()),
                       ("rf", RandomForestClassifier(random_state=random_state))])
    nb_pipe = Pipeline([("scaler", StandardScaler()),
                       ("nb", GaussianNB())])

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # Train and evaluate each model with quantum-inspired optimization
    results = {}
    models = {
        'svm': (svm_pipe, svm_params),
        'knn': (knn_pipe, knn_params),
        'rf': (rf_pipe, rf_params),
        'nb': (nb_pipe, {})  # No hyperparameters for NB
    }
    
    # Use quantum-inspired optimization for SVM
    logger.info("Applying quantum-inspired optimization...")
    def objective_func(params):
        C, gamma = params
        svm = SVC(C=10**C, gamma=10**gamma, probability=True, random_state=random_state)
        pipe = Pipeline([("scaler", StandardScaler()), ("svc", svm)])
        scores = cross_val_score(pipe, X_quantum, y, cv=cv, scoring='accuracy')
        return -scores.mean()  # Negative because we want to maximize
    
    # Define parameter bounds for quantum optimization
    param_bounds = [(-3, 3), (-3, 3)]  # log10(C), log10(gamma)
    qopt = QuantumInspiredOptimizer(param_bounds, objective_func)
    best_params, best_score = qopt.optimize()
    
    logger.info("Quantum-optimized SVM parameters: C=%.3f, gamma=%.3f", 10**best_params[0], 10**best_params[1])

    for name, (pipe, params) in models.items():
        if params:  # Use GridSearchCV if we have parameters to tune
            grid = GridSearchCV(pipe, params, cv=cv, scoring='accuracy', n_jobs=-1)
            grid.fit(X, y)
            best_model = grid.best_estimator_
            best_score = grid.best_score_
            logger.info("%s: Best params: %s", name.upper(), grid.best_params_)
        else:  # Use regular cross-validation for NB
            scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
            best_score = scores.mean()
            pipe.fit(X, y)  # Fit on all data
            best_model = pipe

        results[name] = (best_score, best_model)
        if params:  # GridSearchCV results
            logger.info("%s: CV accuracy: %.4f", name.upper(), best_score)
        else:  # Direct CV results
            logger.info("%s: CV accuracy: %.4f (+/- %.4f)", name.upper(),
                       best_score, scores.std() * 2)

    # Train ensemble using best base models
    best_models = [(name, model) for name, (_, model) in results.items()]
    ensemble = VotingClassifier(estimators=best_models, voting='soft')
    ensemble_scores = cross_val_score(ensemble, X, y, cv=cv, scoring='accuracy')
    ensemble.fit(X, y)
    results['ensemble'] = (ensemble_scores.mean(), ensemble)
    logger.info("ENSEMBLE: CV accuracy: %.4f (+/- %.4f)", 
               ensemble_scores.mean(), ensemble_scores.std() * 2)

    # Select best model
    best_name = max(results, key=lambda k: results[k][0])
    best_score = results[best_name][0]
    best_model = results[best_name][1]
    
    logger.info("\nBest model: %s (CV accuracy=%.4f)", best_name, best_score)
    
    # Final evaluation on full dataset
    y_pred = best_model.predict(X)
    logger.info("\nClassification Report:\n%s", 
                classification_report(y, y_pred, target_names=['HC', 'MDD']))
    
    # Save best model
    joblib.dump(best_model, model_path)
    logger.info("Saved best model (%s) to %s", best_name, model_path)

    return best_name, best_score


def predict_file(model_path, filepath):
    if not os.path.exists(model_path):
        raise SystemExit(f"Model file not found: {model_path} — run training first")
    model = joblib.load(model_path)
    
    # Extract patient ID from filename
    patient_id = None
    for pid in clinical_data.keys():
        if pid in filepath:
            patient_id = pid
            break
    
    if patient_id is None:
        raise ValueError("Could not determine patient ID from filename")
    
    # Get features
    eeg_means = get_means(filepath)
    clinical_features = get_clinical_features(patient_id)
    
    # Combine features
    X = np.hstack([eeg_means, clinical_features])
    vec = np.array([X])
    
    pred = int(model.predict(vec)[0])
    label = "MDD" if pred == 1 else "HC"
    prob = None
    if hasattr(model, 'predict_proba'):
        try:
            prob = float(model.predict_proba(vec)[0,1])
        except Exception:
            prob = None
            
    # print simple output for doctor
    if prob is None:
        print(label)
    else:
        print(f"{label} (prob_MDD={prob:.3f})")
    return label, prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models and save best; or predict a single file using saved model")
    parser.add_argument('--predict', '-p', help='Path to an .xls/.xlsx file to predict (loads saved model)')
    args = parser.parse_args()

    if args.predict:
        predict_file(MODEL_PATH, args.predict)
    else:
        train_and_select_best()
