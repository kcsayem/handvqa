#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, Iterable, Optional

import mlcroissant as mlc
import requests


def normalize_url(url: str) -> str:
    match = re.search(r"file\.xhtml\?fileId=(\d+)", url)
    if match:
        return f"https://dataverse.harvard.edu/api/access/datafile/{match.group(1)}"
    return url


def build_session(dataverse_token: Optional[str], hf_token: Optional[str]) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "handvqa-downloader/1.0"})

    if dataverse_token:
        session.headers.update({"X-Dataverse-key": dataverse_token})

    if hf_token:
        session.headers.update({"Authorization": f"Bearer {hf_token}"})

    return session


def stream_download(session: requests.Session, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def extract_member_from_zip(zip_path: Path, member_name: str, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        candidates = [member_name, member_name.replace("\\", "/"), Path(member_name).name]
        archive_names = {name.replace("\\", "/"): name for name in zf.namelist()}

        selected_name = None
        for candidate in candidates:
            candidate = candidate.replace("\\", "/")
            if candidate in archive_names:
                selected_name = archive_names[candidate]
                break

        if selected_name is None:
            basename = Path(member_name).name
            for archive_name in zf.namelist():
                if Path(archive_name).name == basename:
                    selected_name = archive_name
                    break

        if selected_name is None:
            raise FileNotFoundError(f"Could not find '{member_name}' inside '{zip_path.name}'.")

        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(selected_name) as src, open(dest, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)


def is_archive_member(file_object: dict) -> bool:
    content_url = normalize_url(file_object["contentUrl"])
    name = file_object["name"]
    return content_url.lower().endswith(".zip") and not Path(name).name.lower() == Path(content_url).name.lower()


def group_by_url(distribution: Iterable[dict]) -> Dict[str, list]:
    grouped: Dict[str, list] = {}
    for file_object in distribution:
        url = normalize_url(file_object["contentUrl"])
        grouped.setdefault(url, []).append(file_object)
    return grouped


def filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    return name or "downloaded_file"


def download_from_croissant(manifest_path: str, out_dir: Path, dataverse_token: Optional[str], hf_token: Optional[str]) -> None:
    ds = mlc.Dataset(jsonld=manifest_path)
    meta = ds.metadata.to_json()
    session = build_session(dataverse_token=dataverse_token, hf_token=hf_token)

    grouped_files = group_by_url(meta["distribution"])

    for url, file_objects in grouped_files.items():
        if len(file_objects) == 1 and not is_archive_member(file_objects[0]):
            file_object = file_objects[0]
            dest = out_dir / file_object["name"]
            print(f"Downloading {file_object['name']} ...")
            stream_download(session, url, dest)
            continue

        archive_name = filename_from_url(url)
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / archive_name
            print(f"Downloading shared archive {archive_name} ...")
            stream_download(session, url, archive_path)

            for file_object in file_objects:
                dest = out_dir / file_object["name"]

                if is_archive_member(file_object):
                    print(f"Extracting {file_object['name']} from {archive_name} ...")
                    extract_member_from_zip(archive_path, file_object["name"], dest)
                else:
                    print(f"Saving {file_object['name']} ...")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(archive_path, dest)

    print(f"\nAll files downloaded under {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download all files listed in a Croissant manifest from Dataverse or Hugging Face"
    )
    parser.add_argument("manifest", help="Path to Croissant.json")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="downloaded_data",
        help="Folder to write the downloaded files",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("DATAVERSE_API_TOKEN"),
        help="Optional Dataverse API token. Falls back to DATAVERSE_API_TOKEN if set.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        help="Optional Hugging Face token for gated/private repos.",
    )
    args = parser.parse_args()

    download_from_croissant(
        manifest_path=str(args.manifest),
        out_dir=Path(args.out_dir),
        dataverse_token=args.token,
        hf_token=args.hf_token,
    )
