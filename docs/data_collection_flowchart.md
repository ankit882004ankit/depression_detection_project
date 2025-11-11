# Data Collection Flowchart

This document describes the data collection flow used by the Depression Detection project. It includes a concise flowchart (Mermaid), step-by-step actions, required fields and file formats, naming conventions, validation rules, consent and de‑identification guidance, QA checks, and storage/backup notes.

---

## Mermaid flowchart

```mermaid
flowchart TD
  A[Recruit participant / obtain consent] --> B[Schedule EEG session]
  B --> C[Record EEG (raw signals)]
  C --> D[Export EEG to Excel (.xlsx) with channels as columns]
  D --> E[Collect clinical data (PHQ-9, GAD-7, PSQI, CTQ, LES, SSRS, demographics)]
  E --> F[Create patient record CSV / JSON]
  D --> G[Local QA on EEG file (format, numeric columns)]
  E --> H[Local QA on clinical form (ranges, completeness)]
  G --> I[Apply de-identification & assign study ID]
  H --> I
  I --> J[Store files: uploads/data_research/ (raw + clinical)]
  J --> K[Upload to project `data/` (for training) or `uploads/` (for inference)]
  K --> L[Run preprocessing (get_means) + validation script]
  L --> M[If OK → mark ready; if FAIL → return to QA]
  M --> N[Archive original & metadata into `artifacts/` and backup]

  style A fill:#fef3c7,stroke:#92400e
  style N fill:#ecfccb,stroke:#164e12
```

---

## Step-by-step data collection procedure

1. Participant recruitment & consent
   - Obtain written informed consent following your institution's IRB procedures.
   - Record consent metadata: consent_date, consenting_staff, version_of_consent_form, consent_id.
   - If the dataset will be shared, ensure consent covers data sharing and de‑identified release.

2. EEG recording session
   - Record EEG using your lab's standard protocol (note sampling rate, electrode montage, reference, filter settings).
   - Recommended metadata to record alongside raw EEG: participant local ID, session datetime (UTC), device model, sampling_rate (Hz), montage, duration, task (resting eyes-closed/open), and technician.
   - Immediately save a raw copy in the device's native format and export a sanitized Excel (`.xlsx`) with channels as columns (column headers = channel names) and rows = samples.

3. Clinical measures collection
   - Administer validated instruments: PHQ‑9, GAD‑7, PSQI, CTQ (or subscales), LES, SSRS, plus demographics (age, gender, education in years).
   - Prefer electronic forms (CSV/JSON export) or paper forms transcribed to CSV with double-entry verification.
   - Clinical record fields (recommended):
     - study_id (to be assigned at de‑identification)
     - local_patient_id (kept in a secure mapping file, not shared)
     - phq9 (0–27), gad7 (0–21), psqi (0–21), ctq_total (25–125), les_score (instrument dependent), ssrs (0–60)
     - age (years), gender (coded), education_years
     - collection_date, collector_id

4. Local QA and validation (immediately after collection)
   - EEG QA checks:
     - Ensure exported Excel contains numeric columns for channels.
     - Check expected number of samples and sampling rate metadata.
     - Verify no obvious corruption (NaNs, constant columns) and that values are within device ranges.
   - Clinical QA checks:
     - Numeric ranges enforced: PHQ‑9 0–27, GAD‑7 0–21, PSQI 0–21, CTQ 25–125 etc.
     - No missing mandatory fields (age, gender, phq9).
     - If values are out of range: re-check source form and correct transcription errors.

5. De‑identification and study ID assignment
   - Replace local identifiers with a generated Study ID (format: YYYYMMDD_SITE_## or a random stable hash).
   - Store a secure mapping (local_id → study_id) in a protected location (not in the shared dataset).
   - Remove direct identifiers from shared records (name, address, contact info). Keep minimal metadata needed for analysis (age range if required).

6. File naming conventions
   - EEG export filename: {timestamp}_{session_uid}_{study_id}_raw.xlsx
   - Clinical record filename: {timestamp}_{study_id}_clinical.csv
   - Example: 20251027_153212_ab12cd34_02010005_still.xlsx and 20251027_153212_02010005_clinical.csv
   - For training set files already in `data/`, use the existing mapping convention where patient id appears in filename (e.g., `02010005_still.xlsx`).

7. Storage & transfer to project
   - Local secure storage: encrypted drive within lab network.
   - Transfer to central project repo (project `data/` for training or `uploads/` for inference) via secure copy (SCP) or SFTP over SSH, or by using a secure cloud bucket (S3 with restricted access).
   - Keep an immutable raw archive copy in `artifacts/raw/` (read-only) for reproducibility.

8. Automated ingestion & checks (project side)
   - Drop files into `data/` (training) or `uploads/` (inference). The project contains scripts to scan new files, validate format, and run `get_means()` to compute channel means.
   - Script outputs a validation report and moves files to `data/ready/` if they pass, or `data/quarantine/` if they fail QA.

9. Annotation & labeling
   - Use filename mapping or a separate metadata CSV that links `{study_id, filename, label}` for supervised learning.
   - Maintain `clinical_data.py` or `metadata.csv` with clinical features and labels — keep this file under version control but with restricted access if it contains sensitive mappings.

10. Backups and retention
    - Backup raw and processed datasets to an offsite secure backup weekly.
    - Maintain retention policy per institutional guidelines (e.g., raw data retained for X years).

---

## Validation rules (quick checklist)
- EEG file: contains at least one numeric column; preferred ≥3 numeric channel columns.
- PHQ‑9: integer 0–27
- GAD‑7: integer 0–21
- PSQI: integer 0–21
- CTQ (total): integer 25–125
- SSRS: within instrument range
- Age: 16–100 (or dataset-specific range)
- Gender: use coded values and document mapping in README.
- If any mandatory check fails, flag record for manual review before ingestion.

---

## Privacy and ethics notes (short)
- Ensure consent explicitly covers use of EEG and clinical data for research, model training, and data sharing if applicable.
- De‑identify data before sharing outside the core study team.
- Log access to mapping files (local → study ID) and restrict access to authorised personnel.

---

## QA / audit outputs
- Validation report (per-file JSON): {filename, study_id, checks_passed:true/false, errors:[...], timestamp}
- Master ingestion ledger CSV: columns [study_id, filename, collector, collection_date, data_path, validated_at, validator]
- Keep CSVs in `artifacts/ingestion_logs/` and versioned.

---

## Example commands for secure transfer (PowerShell / Windows)

```powershell
# Upload file to secure server via SCP (example)
scp C:\local\path\20251027_153212_02010005_still.xlsx user@server:/secure/data/incoming/

# Or use AWS CLI to upload to S3 bucket (if using cloud storage)
aws s3 cp C:\local\path\20251027_153212_02010005_still.xlsx s3://my-secure-bucket/raw/ --acl private
```

---

## Deliverables to include with dataset
- Raw EEG exports (.xlsx) and original raw device files (if possible)
- Clinical CSVs or single `metadata.csv` linking study_id → clinical features
- Consent forms and IRB approval summary (kept outside public repo)
- Ingestion validation reports and ledger

---

End of data collection flowchart and procedure.
