SentinelOps AI — Synthetic Starter Dataset
============================================

Purpose
-------
This dataset is a SAFE, SYNTHETIC starter dataset for building and demonstrating
the SentinelOps AI capstone application. It is not captured from a real company
and is not the official CIC-IDS2017 dataset.

Files
-----
1. security_events.csv  : 1,000 synthetic security events; labelled Benign/Malicious.
2. users.csv            : 50 synthetic user records.
3. assets.csv           : 30 synthetic asset records.
4. seed_alerts.csv      : 150 example alerts mapped to event IDs; includes 30 benign
                          review candidates so you can discuss false positives.
5. seed_incidents.csv   : 25 example incident records mapped to alert IDs.

Important
---------
- All names, identifiers, timestamps, IPs, and activities are synthetic.
- The 20% malicious label share is deliberately created for a teaching/demo dataset;
  it is NOT representative of real-world incident prevalence.
- IP addresses in 198.51.100.0/24 are reserved for documentation/examples.
- Labels are suitable for learning the end-to-end workflow, but patterns are simplified.
  Do not use this dataset to claim real-world model performance.
- For the formal ML evaluation in your capstone, use the recommended CIC-IDS2017 dataset,
  compare at least two models, use a held-out test set, and report precision, recall,
  F1-score, confusion matrix, and malicious-class recall.
- Treat labels as ground truth for this synthetic exercise only. In a real SOC, analysts
  must validate detections and investigate false positives.

Suggested MySQL tables
----------------------
users.csv -> users
assets.csv -> assets
security_events.csv -> security_events
seed_alerts.csv -> alerts
seed_incidents.csv -> incidents

Suggested first steps
---------------------
1. Create the database and tables.
2. Import users.csv and assets.csv first.
3. Import security_events.csv.
4. Verify user_id and asset_id references.
5. Display events in Flask.
6. Train/evaluate ML separately from alert seed data.
7. Import seed_alerts.csv only if you want example records for UI testing.
