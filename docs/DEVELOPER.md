# Developer Notes

This file explains how to work on and extend the project.

## Project overview
- The app extracts EEG channel means (first three numeric columns) from uploaded Excel files, combines them with clinical features, and runs a saved scikit-learn model to predict MDD vs HC.

## Running locally
1. Create virtual environment and install dependencies (see README.md).
2. Train model (optional): `python train.py`.
3. Run Flask app: `python app.py`.

## Training pipeline (`train.py`)
- Collect labeled Excel files into `/data`.
- Filenames should include patient IDs matching keys in `clinical_data.py` for automatic labeling and clinical features.
- `train_and_select_best()` builds combined feature matrices (EEG + clinical), runs cross-validation and grid search, and saves the best model to `model.pkl`.
- Quantum-inspired components are implemented as helper classes in `train.py` (feature selector and optimizer). They are optional and can be removed or disabled if undesired.

## Adding a new model
1. Add the model pipeline to the `models` dict in `train_and_select_best()` with its parameter grid (if any).
2. Ensure it follows scikit-learn estimator API.

## App internals (`app.py`)
- `upload_single` saves file to `uploads/`, computes EEG means, stores them in `session` and renders `clinical_form.html`.
- `process_data` recomputes EEG means, reads clinical inputs, constructs the feature vector, and predicts using the loaded model.

## Clinical data
- `clinical_data.py` contains a mapping from patient ID strings to clinical feature dicts.
- The clinical form supports prefill when a patient ID is detected in the uploaded filename.

## Tests
- Add unit tests around `get_means()` and `get_clinical_features()` (these are pure functions).
- Integration tests can simulate Flask requests to `/upload_single` and `/process_data` using Flask's test client.

## Contributing
- Follow PEP8 for Python code.
- Keep templates accessible and prefer semantic HTML.
- Add new dependencies to `requirements.txt` and update this document.

