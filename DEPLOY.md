# Deploy: GitHub (private) + Streamlit Community Cloud

1. Create a **private** GitHub repository (plant drawings are confidential) and push this folder:
   ```bash
   git init && git add . && git commit -m "PID Foundry"
   git branch -M main && git remote add origin https://github.com/<you>/pid-foundry.git && git push -u origin main
   ```
   The repo is ~35 MB (the DXF source is 24 MB, the database a few MB). `data/cache/` is rebuilt automatically and is not committed.
2. On share.streamlit.io → **Create app** → pick the repo, branch `main`, file `app.py`.
   `requirements.txt` installs the Python packages; `packages.txt` installs **tesseract-ocr** and **poppler-utils** (needed for OCR and PDF pages).
3. Private repo apps are private by default; invite viewers from the app's Share menu.

4. Optional, for the Claude copilot: app → Settings → **Secrets** → `ANTHROPIC_API_KEY = "sk-ant-..."`. Without it the copilot runs offline.
   Keep the key out of git (`.streamlit/secrets.toml` is ignored; see `secrets.toml.example`).

## Notes
* The database ships with the results already computed, so the app opens instantly. Re-running step 1 on all sheets takes ~4 minutes
  on Community Cloud; run one sheet at a time from the Ingest tab, or rerun in batch locally and push `data/foundry.db`.
* Reviews written on Community Cloud live in the container's copy of `foundry.db` and are lost on restart. For a team, run the app on a
  company VM, or download the Excel / review log regularly, or point the store at a managed database (the `Store` class is the only place to change).
* REST service: `pip install fastapi uvicorn && uvicorn scripts.api:app` exposes the same steps as the Azure design's endpoints.
