import os
import zipfile
from osfclient import OSF

"""
Downloading the pushT data from the osf project for DINO-WM which the data was first
collected for but LeWM also used it so running this script accesses the project and
downloads and unzips the dataset
"""

PROJECT_ID = "bmw48"
TARGET_ZIP_NAME = "pusht_noise.zip"

osf = OSF()
project = osf.project(PROJECT_ID)
storage = project.storage('osfstorage')

download_dir = "data"
os.makedirs(download_dir, exist_ok=True)

found = False
for f in storage.files:
    if f.name != TARGET_ZIP_NAME:
        continue
    found = True

    out_path = os.path.join(download_dir, TARGET_ZIP_NAME)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "wb") as fp:
        f.write_to(fp)

    print(f"Downloaded: {storage.provider}:{f.path} -> {out_path}")

    extract_dir = os.path.join(download_dir, os.path.splitext(TARGET_ZIP_NAME)[0])
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(out_path, "r") as zf:
        zf.extractall(extract_dir)

    os.remove(out_path)
    print(f"Unzipped to: {extract_dir}")
    print(f"Deleted zip: {out_path}")

    break

if not found:
    raise SystemExit(f"Could not find {TARGET_ZIP_NAME!r} in OSF project {PROJECT_ID!r}.")

