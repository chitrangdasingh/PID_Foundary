# PID Foundry — single tube reactor P&ID ingestion and digitisation

A small, self-hosted "foundry" for P&IDs: it ingests a static drawing, extracts symbols, tags, lines and connections,
keeps every fact traceable to the CAD entity it came from, lets engineers approve or correct the results, learns from
those corrections, and publishes a governed asset hierarchy and knowledge graph.

Built for the **Skid Mounted Single Tube Reactor Unit** P&ID (SPL/2024-25/P-47/PID/R5, 9 process sheets + legend),
and usable for other DXF drawings.

## What it does

| Stage | How | Output |
|---|---|---|
| Ingest | Dependency-free DXF reader (no ezdxf needed); blocks exploded; sheets split by border frames; title block read | `sources`, `sheets` |
| 1 Symbol detection | Small geometry grouped into candidates; lettering (exploded into strokes by the PDF import) removed by OCR; instrument bubbles by circle geometry + voted OCR of the letters; arrows by fills; equipment by tag + outline; everything else by the **active model** | `symbols` + `output_*_symbol_detection.png` |
| 2 Text detection | DXF text read exactly, stroke lettering read by Tesseract, labelled (equipment / instrument / valve / connector / size …), valve tags built from prefix + number | `texts` |
| 3 Graph construction (job) | Exact vector segments (no Hough needed for CAD; Hough is used only for scans), tees split, dashed signal lines chained, objects attached; **flow direction** seeded from drawn arrows and FROM/TO connectors, carried along pipes through in-line valves and up to 3 junctions, signal direction by ISA-5.1 loop order; conflicting seeds flagged | `lines`, candidate edges, `job_status.json` |
| 4 Graph persistence | SQL graph tables, ISA-95 hierarchy (Enterprise › Site › Area › Sheet › Equipment › Loop › Instrument/Valve), BOM reconciliation, full-text search | `graph_nodes`, `graph_edges`, `assets` |
| Review → learn | Low-confidence queue; approve / correct / reject; approved crops become training data; retrain; model registry | `reviews`, `training_samples`, `models` |
| Copilot (chatbot) | Asks the extracted graph, hierarchy, BOM match and checks in plain language. Offline intent model by default; Claude with tool calls when `ANTHROPIC_API_KEY` is set in Secrets. Each answer shows the data it used | Copilot tab |
| Publish | GraphML, Neo4j Cypher, DEXPI-shaped JSON, Excel asset register | Exports tab |

Training data funnels: **seed** (the drawing's own legend sheet, 51 symbols), **upload** (labelled crops, e.g. from PID2Graph / Digitize-PID),
**approved** (reviewed detections). Everything is stored in one SQLite file, `data/foundry.db`, so the repo is the platform.

## Run locally
```bash
pip install -r requirements.txt          # plus tesseract-ocr and poppler-utils from your OS
streamlit run app.py
python scripts/run_pipeline.py data/sources/Thermax_Single_Tube_Reactor_System_PID.dxf --pid STR-PID-R5   # batch rerun (~4 min)
python scripts/train.py --algo rf        # retrain after reviews
python -m pytest -q tests                # smoke tests
```
DWG input: convert to DXF first (ODA File Converter, free), then ingest the DXF.

## Results on the supplied drawing
See the Overview and BOM reconciliation tabs. Every item is a candidate until reviewed; the BOM comparison is the quickest
way to see what the OCR missed and what the BOM may have wrong.

See `DEPLOY.md` for GitHub + Streamlit Community Cloud, and `docs/REFERENCES.md` for how SymphonyAI IRIS Foundry,
the Azure and AWS samples, PID2Graph and DEXPI shaped the design.
