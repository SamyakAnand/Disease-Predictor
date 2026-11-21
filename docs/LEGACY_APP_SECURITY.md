# Legacy disease application security

## Scope

This document covers the five standalone Flask demos under `healthapp/`:

- Cancer
- Diabetes
- Heart
- Kidney
- Liver

They are separate from the repository-root Render entrypoint (`app.py`). The root entrypoint is addressed independently by [PR #1](https://github.com/SamyakAnand/Disease-Predictor/pull/1).

## Request handling

Each legacy app now:

- Requires every configured feature field.
- Rejects blank, non-numeric, NaN, and infinite values before preprocessing.
- Preserves the feature order expected by its existing preprocessor.
- Returns fixed, generic client-facing validation and inference errors.
- Logs inference exceptions on the server without returning exception details to the browser.
- Loads model artifacts using context managers.
- Sends `Cache-Control: private, no-store`, `Pragma: no-cache`, and `Expires: 0` headers.
- Does not enable CORS.
- Binds its local development server to `127.0.0.1` with debug mode disabled.

## Screening history and retention

The legacy apps no longer write submitted features or predictions to shared result files. Their unauthenticated `/results` routes return HTTP 410.

Existing tracked files under `Results/` and disease-specific result directories are intentionally untouched. Their provenance, consent, and retention requirements must be reviewed before any deletion or history rewrite. Disabling future writes does not remove historical records from the repository or its Git history.

The repository-root app has its own history implementation and must be reviewed through its separate security change.

## Local use

Run one legacy app at a time from the repository root:

- `python healthapp/Cancer/app.py` — port 5003
- `python healthapp/Diabetes/app.py` — port 5000
- `python healthapp/Heart/app.py` — port 5001
- `python healthapp/Liver/app.py` — port 5002
- `python healthapp/Kidney/app.py` — port 5004

The apps are educational prototypes, not medical devices or diagnostic services. Do not submit real patient data or use their predictions for healthcare decisions.

## Verification

Install the focused test dependencies and run:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q tests/test_legacy_web_security.py
```

The tests mock model artifacts and verify route behavior, required-field validation, finite numeric input handling, generic errors, no-store headers, disabled history, and absence of shared result-file writes. They do not validate model accuracy, clinical validity, or live deployment behavior.
