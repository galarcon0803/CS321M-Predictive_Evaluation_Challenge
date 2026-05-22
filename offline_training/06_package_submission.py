"""
Package the trained artifacts into my_submission/ and create submission.zip.

Copies from data/ -> my_submission/:
  subject_theta_full.json  -> subject_theta.json

Then creates my_submission.zip from inside my_submission/.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEST = ROOT / "my_submission"
DEST.mkdir(exist_ok=True)

ARTIFACTS = [
    (DATA / "subject_theta_full.json",              DEST / "subject_theta.json"),
    (DATA / "subject_benchmark_fingerprint.json",   DEST / "subject_benchmark_fingerprint.json"),
    (DATA / "irt_bridge.pt",                        DEST / "irt_bridge.pt"),
    (DATA / "irt_bridge_meta.json",                 DEST / "irt_bridge_meta.json"),
    (DATA / "item_difficulty_bridge.pt",            DEST / "item_difficulty_bridge.pt"),
    (DATA / "item_difficulty_meta.json",            DEST / "item_difficulty_meta.json"),
]

REQUIRED = {"subject_theta_full.json"}

print("Copying artifacts to my_submission/ ...", flush=True)
missing_required = []
for src, dst in ARTIFACTS:
    if not src.exists():
        marker = "(REQUIRED)" if src.name in REQUIRED else "(optional)"
        print(f"  MISSING {marker}: {src.name}")
        if src.name in REQUIRED:
            missing_required.append(str(src))
        continue
    shutil.copy2(src, dst)
    size_kb = dst.stat().st_size / 1024
    print(f"  {src.name:35s} -> {dst.name}  ({size_kb:.0f} KB)")

if missing_required:
    print(f"\n{len(missing_required)} REQUIRED artifact(s) missing.")
    raise SystemExit(1)

# Create ZIP (only include files needed at runtime)
INCLUDE = {
    "model.py", "labeling.py", "subject_theta.json", "requirements.txt",
    "models.txt",
    "subject_benchmark_fingerprint.json",
    "irt_bridge.pt", "irt_bridge_meta.json",
    "item_difficulty_bridge.pt", "item_difficulty_meta.json",
}
zip_path = ROOT / "my_submission.zip"
print(f"\nCreating {zip_path.name} ...", flush=True)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(DEST.iterdir()):
        if f.is_file() and f.name in INCLUDE:
            zf.write(f, arcname=f.name)
            print(f"  + {f.name}")

total_kb = zip_path.stat().st_size / 1024
print(f"\nDone. {zip_path.name} is {total_kb:.0f} KB")
print("\nNext: validate with")
print(f"  python starting_kit/tools/check_submission_zip.py my_submission.zip")
print(f"  python starting_kit/tools/run_smoke_test.py my_submission/")
