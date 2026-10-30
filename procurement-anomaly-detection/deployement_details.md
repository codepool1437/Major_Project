# Deployment Details

| Item | Value |
|---|---|
| Platform | Streamlit Community Cloud (free) |
| Application | Streamlit dashboard (`procurement-anomaly-detection/app.py`) |
| GitHub repository | https://github.com/codepool1437/Major_Project |
| Branch deployed | `dataset-inspection` (switch to `main` once merged) |
| Main file path | `procurement-anomaly-detection/app.py` |
| Python version | 3.12 (the platform default; the app is tested on 3.14 locally) |
| Live URL | https://b2b-procurement-anomaly.streamlit.app |
| Deployed by | Jaydip Chavan |
| Cost | Rs. 0 (no credit card, no paid tier) |

---

## 1. What we deployed

The deployed component is the **decision-support dashboard**, which is the last
block of the system architecture ("Early Warning / Visualization"). It lets a
procurement reviewer see which customer-part reorders look unusually early.

It has three tabs:

1. **Overview**: summary of the alerts.
2. **Alerts & Investigation**: filterable and sortable alert table, with a CSV
   download of the filtered alerts.
3. **Customer-Part Event Inspector**: full purchase history and the
   inter-purchase interval (IPI) chart for one customer-part pair.

The dashboard does **not** retrain the model while running. The heavy work
(feature building, scoring, evaluation) is done offline by the scripts in
`src/`. The deployed app only reads two pre-computed files:

| File | Size | Purpose |
|---|---|---|
| `results/improved_ranked_early_reorders.csv` | about 4 MB | Ranked alerts |
| `data/processed/temporal_event_features.csv` | about 14 MB | Event history for the inspector |

This is why a free, lightweight host is enough. There is no database, no
server-side training and no secrets.

---

## 2. Why we chose Streamlit Community Cloud

Say this in the presentation:

1. **It is free.** There is no credit card and no trial that expires. This is a
   college prototype, so there is no budget for cloud hosting.
2. **It matches our code.** The dashboard is already written in Streamlit, so
   no code changes, wrapper or rewrite were needed.
3. **It deploys straight from GitHub.** The repository is already on GitHub.
   Every `git push` redeploys the app automatically, so the live app always
   matches the latest committed code.
4. **Dependencies are handled for us.** The platform reads our pinned
   `requirements.txt`, which is the same environment we validated locally.
5. **A public link is easy to share.** Examiners can open
   `https://<name>.streamlit.app` from any browser without installing Python.
6. **Low risk and low effort.** Deployment takes a few minutes and there is no
   server to maintain, patch or pay for.
7. **The app is small enough for the free tier.** The data it loads is about
   18 MB in total, and all computation was done beforehand.

### Alternatives we considered

| Option | Why we did not pick it |
|---|---|
| Hugging Face Spaces | Also free and a good fallback. It needs a separate repository or push, so it is more setup for no extra benefit. |
| AWS / Azure / GCP (Docker) | Powerful but overkill, and cost and setup effort are higher. This is suitable for a future production version. |
| Local only | No shareable link. We keep it as a backup for the demo. |

---

## 3. Step-by-step deployment

### Step 1: Create the Streamlit Community Cloud account
1. Open https://share.streamlit.io.
2. Click **Continue with GitHub** and authorize Streamlit to read the
   repositories.

### Step 2: Create the app
1. Click **Create app**, then choose **Yup, I have an app**.
2. Fill in:
   - **Repository:** `codepool1437/Major_Project`
   - **Branch:** `dataset-inspection`
   - **Main file path:** `procurement-anomaly-detection/app.py`
   - **App URL (optional):** custom subdomain `b2b-procurement-anomaly`
3. Open **Advanced settings**, select Python **3.12**, and click **Save**.
   No secrets are needed.
4. Click **Deploy**.

### Step 3: Wait for the build
The platform installs the packages from `requirements.txt` and starts the app.
This takes a few minutes. Build logs appear on the right side of the screen.

### Step 4: Verify the live app
1. Open the live URL and check that all three tabs load.
2. In the Overview tab, check the headline numbers: **13,861 alerts from
   64,892 eligible events (21.36%)**.
3. In Alerts & Investigation, apply a filter, then test the CSV download.
4. In the Event Inspector, pick a customer and item and check that the history
   and IPI chart appear.
5. Record the URL at the top of this file (done).

**Verification result:** the live app at
https://b2b-procurement-anomaly.streamlit.app loaded successfully. The
Overview metrics matched our experimental results exactly: 64,892 eligible
purchase events, 13,861 potential early reorders, 21.36% flagged, and 136,372
reorder events with a prior purchase. The sidebar filters (customer, item/part,
severity, request date range) and the three tabs (Overview, Alerts &
Investigation, Event Inspector) rendered correctly.

### Step 5: Updating the app later
Push to the deployed branch. The app redeploys automatically:
```bash
git add .
git commit -m "Update dashboard"
git push origin dataset-inspection
```

---

---

## 4. FAQS

**Q: Why did you deploy on Streamlit Community Cloud?**
It is free, built for Streamlit apps, deploys directly from GitHub with
automatic redeploys, and needs no server management. That is a good fit for a
data-science prototype.

**Q: Is the model running live on the cloud?**
No. Training and scoring run offline using the scripts in `src/`. The cloud app
loads the pre-computed ranked alerts and event features, so it is fast and fits
in the free tier.

**Q: How would this scale in a real company?**
Package the app in Docker and host it on AWS, Azure or GCP. Connect it to the
company's procurement database instead of CSVs, and schedule the scoring
pipeline to run daily or weekly. Add login and role-based access.

**Q: What about data security?**
We use the public anonymized dataset, so there is no sensitive data. The repo
holds no secrets, and `.env` and `secrets.toml` are in `.gitignore`. A real
deployment would need private hosting and authentication.

**Q: What if the internet fails during the demo?**
We have a local fallback. Run `python -m streamlit run app.py` and open
`http://localhost:8501`. This was validated in `runtime-validation-report.md`.

**Q: How do you update the live app?**
Commit and push to GitHub, and the platform redeploys automatically.

**Q: What does the dashboard show, in one line?**
It ranks customer-part reorders whose interval is unusually short compared with
their own history (alert rule: `IPI_DAYS <= 0.30 x historical median IPI`), and
lets a reviewer inspect each case.

---
