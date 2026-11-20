# Basic smoke-test checklist. Run the app, then test manually:
# 1. GET /login -> 200
# 2. Login as admin -> dashboard
# 3. /events shows 1,000 starter events
# 4. /assets shows 30 assets
# 5. /alerts shows 150 seeded alerts
# 6. /incidents shows 25 seeded incidents
# 7. Train models from ML Reports
# 8. Add an event and verify prediction fields are separate from labels

# Live-plot checklist (added)
# 9.  Log in as analyst, open ML Model Lab, note the footer of a graph ("1000 events · 0 new").
# 10. Event > Add: login_failure, failed_login_count 30, event_time 13-11-2026 12:00 -> Predict and save.
# 11. Open ML Model Lab: within ~10 s the status line shows "1 new ML-scored" and the Brute force graph shows a gold star.
# 12. Dashboard: "Events over time", "Activity by hour", "ML score distribution" and the number cards include the event.
# 13. Incidents: open an alert from "Alerts waiting for escalation"; it appears in New. As manager use Advance.

# Admin + auto-retrain checklist (added)
# 14. Admin > User Management: add account "tester" (Analyst, 8+ char password) -> appears; sign in as tester; remove it as admin.
# 15. Admin cannot remove own account or the last active Administrator (error message shown).
# 16. Admin > Assets: add asset (IP 10.20.0.31) -> appears and is appended to data/assets.csv; remove it -> gone from the file too.
# 17. Restart the app: removed users/assets do not return; removed demo account does not return.
# 18. Event > Add (login_failure, 30 failed logins): flash says it was added to data/security_events.csv; the CSV has a new last row.
# 19. ML Model Lab: banner shows "Retraining models..." then the page reloads; "Last trained" time and "+ 1 new events" update.
# 20. Open the new event > Confirm the real outcome > Malicious / brute force: CSV row label updates; Lab retrains again.

# Email OTP login checklist (added)
# 21. Log in as admin (admin / Admin@123): you are sent to /login/verify and a 6-digit code is emailed to the admin's address
#     (with SMTP not configured the code is printed in the server console instead).
# 22. Wrong code -> "Incorrect code. N attempt(s) left"; 5 wrong tries burn the code; Resend works after ~30 s.
# 23. Correct code within 5 minutes -> dashboard. The same code cannot be reused.
# 24. Table email_otps holds only a hash of each code (code_hash), never the code itself.
# 25. Admin > User Management: add an account with an email -> that user must verify by OTP too.

# Forgot-password checklist (added)
# 26. Login page: sign in with EMAIL (or username) + password -> OTP page -> correct code -> dashboard.
# 27. Click "Forgot password?" -> enter account email -> same "if that email belongs to an account" message for known and unknown emails.
# 28. Enter the emailed reset code (separate from login codes) -> "Set a new password" page -> save -> back at login with a success message.
# 29. Old password no longer works; new password + a fresh login OTP works. email_otps rows have purpose = 'login' or 'reset'.
# 30. Opening /forgot/reset directly without verifying redirects to /forgot; reset must be finished within 10 minutes of verifying.

# Sign-up / sign-in / admin edit
# 31. /signup: username + email + password + confirm -> code emailed -> correct code -> "account is ready" -> sign in works.
# 32. Before the code is entered the account is Unverified/inactive and cannot sign in; a wrong code never activates it.
# 33. Sign in with EMAIL or USERNAME + password -> code page -> dashboard. Sign-up accounts get role SIGNUP_DEFAULT_ROLE (Analyst).
# 34. Admin > User Management: add an account (any of the 3 roles) -> row appears in app_users; restart the app -> still there.
# 35. Admin > Edit: change username/email/role/active/new password -> saved to the database, audit row "Edited application account".
# 36. Admin cannot change own role/active, nor demote or disable the last active Administrator.
# 37. Dashboard fits one screen at 1366x768, 1440x900 and 1920x1080 with no page scroll bar.
