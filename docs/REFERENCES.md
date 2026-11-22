### How others ingest P&IDs, and what this platform takes from each

| Approach | How it ingests | What PID Foundry reuses |
|---|---|---|
| **SymphonyAI IRIS Foundry** (P&ID Ingestion) | Vision AI trained on P&IDs finds assets, tags, valves, instruments and connectors; results are verified, then mapped onto the existing asset model (CMMS / EAM / knowledge graph), or a hierarchy is built from the drawings when none exists; ISA-95 structure, asset and process templates, unified namespace | Verify-before-publish (candidate → approved), ISA-95 hierarchy built from the drawing, mapping to an external list (BOM / index), knowledge-graph export |
| **Azure-Samples / digitization-of-piping-and-instrument-diagrams** | Four endpoints in sequence: symbol detection (object detector on Digitize-PID), text detection (OCR), graph construction (Hough line detection + candidate matching, background job), graph persistence (SQL graph); corrected outputs between steps; three funnels of training data; active learning from approved inference | The same step API, folder layout, corrected-output rule, job status and training funnels; versioning added (the design lists it as missing) |
| **AWS guidance (Amazon SageMaker)** | Faster R-CNN symbol detection on page images, OCR, rules for connectivity | Raster path (scripts/run_raster.py) for scans and PDF pages |
| **PID2Graph** (Zenodo record 14803338) | Real and synthetic P&IDs with graph-level ground truth | Pre-training data and a graph-level metric (scripts/fetch_datasets.py) |
| **Digitize-PID** (Paliwal et al., 500 synthetic P&IDs, 32 symbols) and **SynthPID** | Synthetic sheets with symbol / line / text labels | Upload funnel for extra training crops |
| **DEXPI / pyDEXPI** (Proteus XML 1.3 → NetworkX) | Vendor-neutral P&ID exchange | DEXPI-shaped JSON export; the class names map to DEXPI 1.3 |
| **SFILES 2.0** (Vogel et al., Optimization and Engineering 24, 2023) | Text notation of flowsheet topology | Planned export from the approved graph |
| Rule-based P&ID autocorrection (arXiv 2502.18493), Talking like P&IDs (arXiv 2502.18928, Neo4j graph RAG) | Engineering rules and graph retrieval over digitised P&IDs | Review rules and the Search & ask tab; Cypher export for a Neo4j copilot |

### What is different here
The supplied drawing is a **DXF made from a PDF**: lettering is exploded into strokes and most symbols are loose polylines, not blocks.
So the platform reads the CAD geometry directly (exact coordinates and entity handles for every fact), uses OCR only where letters became strokes,
and uses the drawing's own **legend sheet** as the seed training set instead of a generic 32-symbol set.
