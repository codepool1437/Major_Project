# Runtime Validation Report

**Generated:** 2026-10-24  
**Target:** `procurement-anomaly-detection/app.py`

## Summary

| Step | Status | Exit code | Details |
|---|---|---:|---|
| Dependency inspection | PASS | 0 | Streamlit 1.65.0, pandas 3.0.6, NumPy 2.5.4, Altair 6.3.0, matplotlib 3.11.2 available |
| Clean-environment install | PASS | 0 | Fresh temporary Python 3.14 virtual environment installed all pinned requirements successfully |
| Clean-environment checks | PASS | 0 | Imports, `pip check`, and `compileall` passed in the fresh environment |
| Clean-environment startup | PASS | 0 | Fresh-environment Streamlit process on port 8512; `/_stcore/health` returned HTTP 200 and body `ok` |
| Live dashboard startup | PASS | 0 | `python -m streamlit run app.py`; `/_stcore/health` returned HTTP 200 and body `ok` |
| Overview | PASS | 0 | Three tabs rendered without exceptions; final alert and percentage text present |
| Alerts & Investigation | PASS | 0 | Alert table, sorting controls, download button, and filtered reruns rendered |
| Event Inspector | PASS | 0 | Customer and item controls rendered; customer filtering synchronized item options |
| Date filtering | PASS | 0 | Single-date tuple, complete range, clearing, and no-alert date window completed without exceptions |
| Documentation/reproducibility | PASS | 0 | Pinned requirements and README added; retained outputs remain listed in the artifact manifest |

**Overall:** PASS

## Evidence

The dashboard reads the regenerated
`results/improved_ranked_early_reorders.csv` and
`data/processed/temporal_event_features.csv`. AppTest observed:

- Three tabs
- Final alert text: 13,861
- Final flagged percentage text: 21.36%
- Customer, item, severity, and date controls
- Customer selector synchronization in the Event Inspector
- A visible-column CSV download button
- Zero Streamlit exceptions across the successful interaction sequence

The live process was terminated after the health check. It returned a
termination exit code from the controlled shutdown; startup and health
readiness passed.

## Environment and gaps

- Docker: unavailable; not required by this file-backed Streamlit application.
- Node.js/Playwright: unavailable; browser-rendering and developer-console
  inspection were not run in this environment.
- AppTest uses Streamlit's application harness rather than a real browser, so
  visual layout, browser download clicks, and browser console warnings remain
  unverified here.
- The dashboard's required Python packages are captured in `requirements.txt`.
- A clean-environment reproduction was completed on 2026-10-24 using a temporary
  Python 3.14 virtual environment. Installation from `requirements.txt` completed
  successfully, `pip check` reported no broken requirements, source compilation
  passed, and the dashboard health endpoint returned HTTP 200 with body `ok`.
  The temporary environment and logs were removed after validation.

## Reproduction

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
