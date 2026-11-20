# SentinelOps AI — Final Project Starter

A Flask + SQLAlchemy + MySQL-compatible security operations application built around the supplied `SentinelOps_AI_Starter_Dataset(2).zip`.

## Included modules
- Sign-up and sign-in with email/username + password + emailed OTP, three roles: Administrator, Analyst, Incident Manager
- Dashboard, event search and pagination
- CSV event ingestion and manual event entry
- ML: compare three binary classifiers, train five independent one-vs-rest attack-pattern models (Brute Force, Port Scanning, Unusual Transfer, Suspicious Privilege, Malware-like), and use a five-class family model for incoming event classification
- Alerts and incident management
- Interactive database-grounded investigation chat assistant with pattern-specific triage guidance (local rules; no external AI API)
- Asset inventory, user directory and audit log
- ML evaluation report with accuracy, precision, recall, F1, ROC-AUC, model-comparison graphs, five-pattern metric graphs, ROC curves and readable confusion matrices

## Dataset facts
The supplied starter data contains:
- 1,000 security events (800 Benign, 200 Malicious)
- 50 directory users
- 30 assets
- 150 seeded alerts
- 25 seeded incidents

All supplied records are synthetic. The 20% malicious proportion is deliberately constructed for learning and must not be represented as real-world prevalence or production model performance. The event CSV has these exact 14 columns:
`event_id,event_time,user_id,asset_id,src_ip,dst_ip,event_type,protocol,dst_port,failed_login_count,bytes_sent,severity_hint,label,attack_type`

## Windows setup
1. Install Python 3.12 (recommended for the pinned scikit-learn / NumPy versions) and MySQL Server (optional for first run).
2. Extract the ZIP file.
3. Open a terminal in the extracted `SentinelOps_AI_Final_Project` folder.
4. Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
```

5. Install packages:

```powershell
pip install -r requirements.txt
```

6. Copy `.env.example` to `.env`. Change `SECRET_KEY` to a long random value.
7. Run:

```powershell
python app.py
```

8. Open `http://127.0.0.1:5000`.

If you currently use Python 3.14, create this project environment with Python 3.12 instead; the pinned ML packages in this ZIP are tested for the Python 3.12 environment. The default database is SQLite, so the app can be launched immediately without MySQL. For MySQL, create a database named `sentinelops_ai`, then set in `.env`:

```text
DATABASE_URL=mysql+pymysql://root:YOUR_PASSWORD@localhost/sentinelops_ai
```

Restart the application. Missing tables and missing starter CSV records are created/imported automatically on launch using SQLAlchemy ORM. The importer checks primary keys, so restarting does not duplicate records.

## Sign up and sign in
- **Sign up** (`/signup`): username + email + password, then the 6-digit code emailed to that address. The account is created (active) only after the code is verified. New accounts get the role in `SIGNUP_DEFAULT_ROLE` (default `Analyst`); an Administrator can change it.
- **Sign in** (`/login`): email *or* username + password, then the 6-digit code emailed to the account's email. "Forgot password?" works the same way.
- **User Management** (Administrator): add accounts with any of the 3 roles, **Edit** username / email / role / active / new password, or remove them. Everything is stored in the `app_users` table; every change is written to the Audit Trail.
- `admin` gets `ADMIN_EMAIL` from `.env`; set `ANALYST_EMAIL` / `MANAGER_EMAIL` for the other built-in accounts (accounts without an email sign in with the password only).
- Until SMTP is configured, codes are printed in the server console (development only).

Change the secret key before any deployment. This is a capstone demonstration app, not a production-hardened SOC platform.

## Train models
Sign in as `admin`, open **ML Reports**, and click **Train & compare models**. The application compares:
- Random Forest
- Extra Trees
- Logistic Regression

It selects the model with highest held-out F1 (tie-break by recall), saves it, and separately fits a Random Forest attack-family classifier on malicious-labelled rows. The split is stratified and fixed (`random_state=42`). Re-run training to reproduce the same split.

## AI Investigation Copilot
Open **AI Investigation** and ask about an exact event ID, alert ID, source IP, or one of the five attack patterns. The chat endpoint searches the live SQLAlchemy database, returns linked event/asset/alert evidence, and selects a safe investigation checklist based on the matched attack family. It is intentionally a local evidence assistant, not a hosted generative LLM; no OpenAI/API key is required. It does not execute containment actions.

## New event prediction
Use **Ingest event**. The form stores the new event with `label` and `attack_type` empty, then writes `predicted_label`, `predicted_attack_type`, and `model_score` separately. A malicious prediction creates a new alert. Do not calculate accuracy/precision/recall/F1 for an unlabelled incoming event; metrics require known ground truth.

## Security and project limitations
- Passwords are hashed.
- Routes enforce role permissions server-side.
- AI Copilot is a local evidence-based rules assistant, not an LLM. It does not execute containment or response actions.
- CSRF protection and production-grade account lifecycle, rate limiting, MFA, secrets management, HTTPS, backup and monitoring should be added before deployment.
- Use CIC-IDS2017 (or another properly documented independent dataset) for formal external ML evaluation as required by your project brief. Do not claim the synthetic starter dataset validates real-world detection.


## MySQL database setup and tables

See `database/DATABASE_SETUP.txt` for the complete setup. On startup, `app.py` calls
`db.create_all()` and creates missing tables. The ORM models define these tables:
`app_users`, `users`, `assets`, `security_events`, `alerts`, `incidents`,
`audit_logs`, `incident_timeline`, `investigation_notes`, and `ml_model_registry`.

For a MySQL password containing `@`, encode it as `%40` inside `DATABASE_URL`.
For example, `sailu@9014` becomes `sailu%409014`. Keep your real credentials in
`.env`; never commit `.env` to Git.


## ORM data population
The startup importer loads the full supplied dataset: 50 users, 30 assets, 1,000 security events, 150 alerts, and 25 incidents. It also creates one clearly marked system timeline entry and one starter evidence note per incident. The `ml_model_registry` is populated after an Administrator trains the models; it stores the actual evaluation metrics returned by training. See `database/DATABASE_SETUP.txt` for expected counts and details.

## Dashboard redesign
The redesigned dark SOC dashboard uses live database aggregates for event totals, malicious/benign labels, alert severity and workflow status, attack-family distribution, top source IPs, and recent events. The dashboard does not use the sample values shown in the visual reference images.

## Role-based access (updated)
- **Administrator:** full dashboard, event ingestion, alerts, incidents, AI investigation, ML training, assets, user management and audit trail.
- **Analyst:** distinct threat-monitoring dashboard; event search/detail/import/create, alert triage, incident creation/view, AI investigation and asset view. Cannot update incidents, access user management/audit or train models.
- **Incident Manager:** distinct incident-oversight dashboard; alert status and escalation, incident updates, AI investigation, ML report viewing, assets and audit trail. Cannot ingest/view raw event explorer, manage users or train models.
- **Administrator:** can access every module and all Analyst/Incident Manager workflows. Dashboard KPI cards and quick links open their corresponding modules. Server-side route checks enforce role restrictions; hiding a navigation link is not the only protection.
Navigation is role-aware and protected routes enforce permissions server-side. Dashboard cards use database query results.


## ML update — five attack-pattern models
Training uses the complete supplied `security_events.csv` (1,000 rows; 800 benign and 200 malicious). The pipeline compares Random Forest, Extra Trees and Logistic Regression for benign/malicious detection. It then trains five separate Extra Trees one-vs-rest models:
`brute_force`, `port_scan_like`, `unusual_transfer`, `suspicious_privilege`, `malware_like`.
A five-class family classifier is also saved for event classification. The reports page plots model metrics and the five ROC curves, and displays confusion matrices as tables. Incoming events are scored by the binary detector; if malicious, all five pattern models are run and the highest pattern probability is recorded as the predicted family.

Metrics are from a stratified held-out split of this synthetic teaching dataset. Some results may be very high because the dataset is generated with distinguishable patterns; this is not proof of generalization to live traffic. Validate on an independent dataset and avoid using ground-truth columns as prediction features.

## Pattern-based ML Model Lab (updated)
- Administrator and Security Analyst can open the ML Model Lab; Incident Manager cannot.
- Administrator can retrain the models. Analyst can select and view trained models only.
- Each of the five attack patterns is trained with Random Forest, Extra Trees, and Logistic Regression candidates. The Lab lets the user choose the pattern and candidate model.
- Pattern graphs show precision, recall, F1, ROC-AUC, ROC curve, and a confusion matrix for the selected pattern/model.
- New manual events and CSV-imported events are scored by the binary detector and all five pattern models. The live prediction counts and attack-family chart refresh every 10 seconds.
- Unlabelled new events update prediction counts/distributions, not held-out accuracy or confusion matrices. Evaluation metrics change only after retraining/evaluation with ground-truth labels.

## Attack pattern graphs
Run `python sentinelops_pattern_graphs.py` (needs pandas, numpy, matplotlib).
PNG graphs are written to `graphs/`: brute force, port scan, unusual transfer,
suspicious privilege, malware-like, and parameter curves.
connection_count_5min, total_bytes_received, flow_duration_ms and bytes_per_second
are simulated when missing from the CSV; real columns are used if present.


## Admin management and automatic ML updates (latest)
- **Administrator** can add/remove application accounts, directory users and assets (User Management, Assets). Removing keeps the related events; see `CHANGES_ADMIN_AND_AUTO_ML.txt`.
- Every new event or imported CSV is appended to `data/security_events.csv`, scored, and the models retrain in the background; ML Model Lab refreshes itself. New events are provisional training rows (predicted label, training only); an analyst can confirm the real label on the event page. Set `TRAIN_ON_PREDICTED=0` to train on confirmed labels only.
