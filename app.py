# app.py - verbose debug-ready Flask app
from flask import Flask, request, render_template, send_file, session
import os, joblib, pandas as pd, numpy as np, traceback, logging
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
from clinical_data import clinical_data

# config
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
MODEL_PATH = "model.pkl"
ALLOWED_EXTENSIONS = {'xls','xlsx'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# logging to console and file
logger = logging.getLogger("depression_app")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
ch.setFormatter(fmt)
logger.addHandler(ch)
fh = logging.FileHandler("server.log", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(fmt)
logger.addHandler(fh)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.secret_key = os.urandom(24)  # Required for session

def allowed(fname):
    return '.' in fname and fname.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    if not os.path.exists(MODEL_PATH):
        logger.warning("model.pkl not found at %s", MODEL_PATH)
        return None
    try:
        m = joblib.load(MODEL_PATH)
        logger.info("Loaded model from %s", MODEL_PATH)
        return m
    except Exception as e:
        logger.exception("Failed loading model: %s", e)
        return None

model = load_model()
if model is not None:
    try:
        logger.info("Model loaded. classes_: %s", getattr(model, 'classes_', None))
        # If ensemble, log estimators
        if hasattr(model, 'estimators'):
            names = [type(e).__name__ for e in model.estimators]
            logger.info("Ensemble estimators: %s", names)
    except Exception:
        logger.exception("Error while inspecting model object")

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

def get_means(path):
    logger.debug("Reading Excel: %s", path)
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception:
        df = pd.read_excel(path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    logger.debug("Numeric columns found: %s", numeric_cols)
    if len(numeric_cols) == 0:
        raise ValueError("Uploaded file must contain at least one numeric column.")
    # Convert column names to strings so concatenation later won't fail
    ch = [str(c) for c in numeric_cols[:3]]
    means = df[numeric_cols[:3]].mean(axis=0).values.astype(float)
    if len(means) < 3:
        pad_len = 3 - len(means)
        means = np.pad(means, (0, pad_len), constant_values=0.0)
        for i in range(pad_len):
            ch.append(f"dummy_ch_{len(ch)+1}")
    logger.debug("Computed means: %s for channels %s", means, ch)
    return means, ch

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload_single", methods=["POST"])
def upload_single():
    try:
        logger.info("Received upload_single request")
        if 'file' not in request.files:
            return render_template("error.html", error_message="No file part in request"), 400
        f = request.files['file']
        if f.filename == '':
            return render_template("error.html", error_message="No selected file"), 400
        if not allowed(f.filename):
            return render_template("error.html", error_message="Invalid file type (.xls/.xlsx allowed)"), 400

        fname = secure_filename(f.filename)
        uid = uuid.uuid4().hex[:8]
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        saved_name = f"{ts}_{uid}_{fname}"
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
        os.makedirs(os.path.dirname(saved_path), exist_ok=True)
        logger.info("Saving uploaded file to: %s", saved_path)
        f.save(saved_path)

        # Get EEG features and store in session
        means, ch = get_means(saved_path)
        try:
            means_list = [float(x) for x in means]
        except Exception:
            means_list = list(means)

        session['eeg_means'] = means_list
        session['eeg_channels'] = ch
        session['temp_file'] = saved_path

        # Try to detect patient id to pre-fill clinical form, but always show the form
        prefill = {}
        patient_id = None
        for pid in clinical_data.keys():
            if pid in fname:
                patient_id = pid
                break

        if patient_id is not None:
            try:
                cf = get_clinical_features(patient_id)
                # map to simple names for the form
                prefill = {
                    'age': int(cf[0]),
                    'gender': int(cf[1]),
                    'education': int(cf[2]),
                    'phq9': int(cf[3]),
                    'ctq': int(cf[4]),
                    'les': int(cf[5]),
                    'ssrs': int(cf[6]),
                    'gad7': int(cf[7]),
                    'psqi': int(cf[8])
                }
            except Exception:
                prefill = {}

        # Render the clinical form so user can review/edit clinical inputs
        return render_template("clinical_form.html", temp_file=saved_path, prefill=prefill)
    except Exception as e:
        logger.exception("Exception in upload_single: %s", e)
        return render_template("error.html", error_message=str(e)), 500

@app.route("/process_data", methods=["POST"])
def process_data():
    try:
        # Expect temp_file to be provided from the form
        saved_path = request.form.get('temp_file') or session.get('temp_file')
        if not saved_path or not os.path.exists(saved_path):
            return render_template("error.html", error_message="Uploaded file not found. Please upload again."), 400

        # Get EEG features (recompute to be safe)
        means, ch = get_means(saved_path)
        try:
            means_list = [float(x) for x in means]
        except Exception:
            means_list = list(means)

        # Read clinical inputs from form
        clinical_features = [
            float(request.form.get('age', 0)),
            float(request.form.get('gender', 0)),
            float(request.form.get('education', 0)),
            float(request.form.get('phq9', 0)),
            float(request.form.get('ctq', 0)),
            float(request.form.get('les', 0)),
            float(request.form.get('ssrs', 0)),
            float(request.form.get('gad7', 0)),
            float(request.form.get('psqi', 0))
        ]

        # Combine and predict
        X = np.array([means_list + clinical_features], dtype=float)
        if model is None:
            return render_template("error.html", error_message="Model not loaded (run train.py)"), 500

        pred = int(model.predict(X)[0])
        label = "MDD" if pred == 1 else "HC"
        prob = None
        if hasattr(model, 'predict_proba'):
            try:
                prob = float(model.predict_proba(X)[0,1])
            except Exception:
                prob = None

        # Save CSV result
        clinical_labels = ['age', 'gender', 'education', 'phq9', 'ctq', 'les', 'ssrs', 'gad7', 'psqi']
        result = {
            'filename': os.path.basename(saved_path),
            ch[0]+"_mean": float(means[0]),
            ch[1]+"_mean": float(means[1]),
            ch[2]+"_mean": float(means[2]),
            'prediction': label,
            'prob_MDD': prob if prob is not None else ''
        }
        for i, name in enumerate(clinical_labels):
            result[name] = clinical_features[i]

        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        uid = uuid.uuid4().hex[:8]
        out_name = f"result_{ts}_{uid}.csv"
        out_path = os.path.join(app.config['RESULTS_FOLDER'], out_name)
        pd.DataFrame([result]).to_csv(out_path, index=False)
        logger.info("Saved result to: %s", out_path)

        # Prepare result for template
        result_data = {k: result.get(k) for k in result.keys()}
        return render_template("single_result.html", pred=label, result=result_data)
    except Exception as e:
        logger.exception("Exception in process_data: %s", e)
        return render_template("error.html", error_message=str(e)), 500

@app.route("/api/upload_single", methods=["POST"])
def api_upload_single():
    try:
        logger.info("api_upload_single called")
        if 'file' not in request.files:
            return {"status":"error","message":"no file in request"}, 400
        f = request.files['file']
        if f.filename == "":
            return {"status":"error","message":"empty filename"}, 400
        if not allowed(f.filename):
            return {"status":"error","message":"invalid extension"}, 400
        fname = secure_filename(f.filename)
        tmp = os.path.join(app.config['UPLOAD_FOLDER'], f"tmp_{uuid.uuid4().hex[:8]}_{fname}")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        f.save(tmp)

        # Get EEG features
        means, ch = get_means(tmp)
        try:
            means_list = [float(x) for x in means]
        except Exception:
            means_list = list(means)

        # Get patient ID from filename (optional)
        patient_id = None
        for pid in clinical_data.keys():
            if pid in fname:
                patient_id = pid
                break

        if patient_id is None:
            return {"status": "error", "message": "Could not determine patient ID from filename"}, 400

        # Get clinical features
        try:
            clinical_features = get_clinical_features(patient_id)
            clinical_features_list = [float(x) for x in clinical_features]
        except Exception as e:
            return {"status": "error", "message": f"Error getting clinical features: {str(e)}"}, 400

        # Combine features and predict
        all_features = means_list + clinical_features_list
        X = np.array([all_features], dtype=float)
        if model is None:
            return {"status":"error","message":"model not loaded"}, 500
        logger.debug("API feature vector: %s", means_list)
        pred = int(model.predict(X)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = float(model.predict_proba(X)[0,1])
            except:
                prob = None

        # support a simple-only response when client requests minimal output
        simple = (request.form.get('simple') == '1') or (request.args.get('simple') == '1')
        if simple:
            return {"prediction": "MDD" if pred==1 else "HC"}

        clinical_labels = ['age', 'gender', 'education', 'phq9', 'ctq', 'les', 'ssrs', 'gad7', 'psqi']
        clinical_values = [float(x) for x in clinical_features]
        return {
            "status": "ok",
            "filename": fname,
            "channel_names": ch,
            "means": [float(x) for x in means],
            "clinical_features": dict(zip(clinical_labels, clinical_values)),
            "prediction": "MDD" if pred==1 else "HC",
            "prob_MDD": prob
        }
    except Exception as e:
        logger.exception("Exception in api_upload_single: %s", e)
        return {"status":"error","message":str(e)}, 500

@app.route("/download/<fn>")
def download(fn):
    path = os.path.join(RESULTS_FOLDER, fn)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    else:
        return "File not found", 404

if __name__ == "__main__":
    logger.info("Starting Flask app on http://127.0.0.1:5000")
    app.run(debug=True)
