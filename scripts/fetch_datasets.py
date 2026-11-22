"""Download public P&ID datasets for pre-training (run where internet is available, e.g. Colab).

  PID2Graph (Zenodo 14803338)      - real + synthetic P&IDs with graph ground truth (symbols, lines, connections)
  SynthPID  (github LatentSpaceIITB/SynthPID) - synthetic P&IDs with symbol boxes
  Digitize-PID (Paliwal et al.)    - 500 synthetic P&IDs, 32 symbol classes (used by the Azure sample)

Crops exported from these datasets enter training through scripts/upload_training_data.py (funnel 'upload').
Map their class names onto this project's <Category>/<Sub>/<Name> labels in data/seed/class_map.csv first.
"""
import os
import subprocess
import sys
import urllib.request

OUT = sys.argv[1] if len(sys.argv) > 1 else "external_datasets"
os.makedirs(OUT, exist_ok=True)
print("PID2Graph record:", "https://zenodo.org/records/14803338")
try:
    import json
    meta = json.load(urllib.request.urlopen("https://zenodo.org/api/records/14803338"))
    for f in meta.get("files", []):
        url, name = f["links"]["self"], f["key"]
        print("downloading", name); urllib.request.urlretrieve(url, os.path.join(OUT, name))
except Exception as ex:
    print("PID2Graph download failed:", ex)
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/LatentSpaceIITB/SynthPID", os.path.join(OUT, "SynthPID")])
print("Digitize-PID: request access via the paper (Paliwal et al., 'Digitize-PID', PAKDD 2021) and unpack into", OUT)
