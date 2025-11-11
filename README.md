# Depression Detection Project

A Flask-based application that combines EEG-derived features and clinical assessment data to assist in detecting Major Depressive Disorder (MDD). The project includes data processing, a machine learning training pipeline, and a web interface for uploading EEG files, reviewing/entering clinical data, and viewing results.

## Features
- EEG feature extraction from Excel (.xls/.xlsx) files
- Integration of clinical measures (PHQ-9, CTQ, LES, SSRS, GAD-7, PSQI, demographics)
- Machine learning pipeline with multiple classical models (SVM, RandomForest, KNN, Naive Bayes)
- Quantum-inspired feature selection and hyperparameter optimization (optional enhancements in `train.py`)
- Flask web UI with two-step workflow: upload EEG → clinical form → results
- CSV result storage and download support

## Quick start
1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. (Optional) Train the model using the dataset in `data/`:

```powershell
python train.py
```

This will save the best model to `model.pkl`.

4. Run the web app:

```powershell
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## File structure
- `app.py` - Flask application and routes
- `train.py` - Training and model selection script
- `clinical_data.py` - Static clinical features used to prefill forms
- `data/` - Training Excel files
- `results/` - Saved CSV result files
- `uploads/` - Uploaded EEG files
- `templates/` - Jinja2 templates (index, clinical_form, single_result, error)
- `static/styles.css` - Main stylesheet
- `model.pkl` - Saved trained model (created by `train.py`)

## Usage / Endpoints
- `GET /` - Upload page
- `POST /upload_single` - Upload an EEG file; shows clinical form
- `POST /process_data` - Process clinical form and produce prediction
- `POST /api/upload_single` - API endpoint to upload and predict (expects filename containing patient ID for prefilled clinical data)
- `GET /download/<filename>` - Download saved CSV result

## Notes on quantum-inspired code
`train.py` contains optional quantum-inspired feature selection and a differential-evolution based optimizer that simulates quantum-like search strategies. These components are "quantum-inspired" (classical simulations) and do not require quantum hardware.

## Troubleshooting
- If pages don't reflect CSS changes, hard-refresh the browser (Ctrl+F5) or restart the Flask server.
- If the model is not loaded, run `python train.py` to generate `model.pkl`.

## License
Specify license here (e.g., MIT) if you plan to open-source the project.
