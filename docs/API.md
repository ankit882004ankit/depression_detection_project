# API Reference

This document describes the HTTP endpoints exposed by the Flask app.

## POST /upload_single
- Description: Upload an Excel file (.xls or .xlsx). The server extracts EEG channel means and shows the clinical data form.
- Form fields:
  - `file` (file) - Excel file to upload
- Response: Renders `clinical_form.html` with `temp_file` hidden input and optional `prefill` data.

## POST /process_data
- Description: Accepts clinical data from the form, recomputes EEG features, combines features and runs the model to predict MDD vs HC.
- Form fields:
  - `temp_file` (string) - server path to the uploaded file (hidden)
  - `age`, `gender`, `education`, `phq9`, `ctq`, `les`, `ssrs`, `gad7`, `psqi` (numbers)
- Response: Renders `single_result.html` with the prediction and clinical features displayed.

## POST /api/upload_single
- Description: API variant used for programmatic predictions. Expects uploaded file name to contain a patient ID that maps into `clinical_data.py` for clinical features.
- Form fields / multipart:
  - `file` (file)
- Query params:
  - `simple=1` (optional) - returns a minimal JSON object with `prediction` only
- Response JSON (on success):
```
{
  "status": "ok",
  "filename": "...",
  "channel_names": ["ch1","ch2","ch3"],
  "means": [12345.0, 23456.0, 34567.0],
  "clinical_features": {"age":.., "gender":.., ...},
  "prediction": "MDD|HC",
  "prob_MDD": 0.87
}
```

## GET /download/<filename>
- Description: Download a previously saved CSV result.
- Response: File download or 404 if not found.
