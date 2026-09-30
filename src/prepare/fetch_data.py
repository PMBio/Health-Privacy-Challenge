#!/usr/bin/env python3
"""
Fetch preprocessed data and split indices from Zenodo, verify integrity,
and extract into the layout the pipeline expects.

Pinned to a specific Zenodo version record. 

Post-fetch layout:
    data/processed/        *.tsv   (from the data archive)
    data/meta/             *.csv   (subtype / metadata)
    data_splits/split_indices/ *.yaml  (from the splits archive)


Usage:
    python fetch_data.py            # fetch + verify + extract (skips work already done)
    python fetch_data.py --force    # re-download and re-extract everything
"""

from __future__ import annotations
import click
import hashlib
import sys
import zipfile
from pathlib import Path

import requests  

# Zenodo VERSION record id (the numeric id in the version-specific URL / DOI)
ZENODO_RECORD_ID = "22996320" #22996320

def find_repo_root(marker=".git"):
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"repo root ({marker}) not found above {__file__}")

REPO_ROOT = find_repo_root()

# One entry per archive to pull. `filename` must match the file name on the
# Zenodo record exactly. `extract_to` is where its contents are unzipped.
ARCHIVES = [
    {"filename": "TCGA-BRCA.zip",     
     "mode": "route"},
    {"filename": "TCGA-COMBINED.zip", 
     "mode": "route"},
    {"filename": "split_indices.zip", 
     "mode": "extract_to",
     "dest": "data_splits/split_indices"},
]
# Where downloaded archives are cached (gitignored).
DOWNLOAD_DIR = REPO_ROOT / "downloads"

ZENODO_API = "https://zenodo.org/api/records"

# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _fetch_record_files(record_id: str) -> dict[str, dict]:
    """Return {filename: {url, md5}} for every file on the Zenodo record."""
    resp = requests.get(f"{ZENODO_API}/{record_id}", timeout=60)
    resp.raise_for_status()
    files = {}
    for f in resp.json().get("files", []):
        # Zenodo reports checksum as "md5:<hex>"
        checksum = f.get("checksum", "")
        md5 = checksum.split(":", 1)[1] if ":" in checksum else checksum
        url = f.get("links", {}).get("self") or f.get("links", {}).get("download")
        files[f["key"]] = {"url": url, "md5": md5}
    return files


def _download(url: str, dest: Path, expected_md5: str, force: bool) -> None:
    if dest.exists() and not force:
        if expected_md5 and _md5(dest) == expected_md5:
            print(f"  cached & verified: {dest.name}")
            return
        print(f"  re-downloading (checksum mismatch or unknown): {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading: {dest.name}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    if expected_md5:
        actual = _md5(dest)
        if actual != expected_md5:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {dest.name}: "
                f"expected {expected_md5}, got {actual}"
            )
        print(f"  verified: {dest.name}")


def _route_dest(member_name: str) -> Path | None:
    lower = member_name.lower()
    if lower.endswith("_subtypes.csv"):
        return REPO_ROOT / "data" / "meta"
    if lower.endswith(".tsv"):
        return REPO_ROOT / "data" / "processed"
    return None


def _extract(archive: Path, spec: dict, force: bool) -> None:
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        for member in members:
            if spec["mode"] == "route":
                dest_dir = _route_dest(member)
                if dest_dir is None:
                    print(f"  skip (unrouted): {member}")
                    continue
            else:  # extract_to
                dest_dir = REPO_ROOT / spec["dest"]

            flat = Path(member.split("/", 1)[-1] if "/" in member else member).name
            target = dest_dir / flat
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not force:
                print(f"  skip (exists): {target.relative_to(REPO_ROOT)}")
                continue
            with zf.open(member) as src, open(target, "wb") as out:
                out.write(src.read())
            print(f"  {member} -> {target.relative_to(REPO_ROOT)}")



@click.command()
@click.option("--force", is_flag=True, default=False,
              help="Re-download and re-extract even if files are present.")
def main(force):
    """Fetch data + splits from Zenodo, verify, and extract."""
    if "{{" in ZENODO_RECORD_ID:
        raise click.ClickException("Set ZENODO_RECORD_ID (and archive filenames) first.")

    print(f"Zenodo record {ZENODO_RECORD_ID}")
    remote = _fetch_record_files(ZENODO_RECORD_ID)
    for spec in ARCHIVES:
        name = spec["filename"]
        if name not in remote:
            raise click.ClickException(f"'{name}' not on record. Available: {sorted(remote)}")
        info = remote[name]
        local = DOWNLOAD_DIR / name
        print(f"\n{name}")
        _download(info["url"], local, info["md5"], force)
        _extract(local, spec, force)
    print("\nDone.")


if __name__ == "__main__":
    main()

# python src/prepare/fetch_data.py 