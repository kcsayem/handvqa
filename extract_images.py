#!/usr/bin/env python3
import argparse
import zipfile
import shutil
import tempfile
from pathlib import Path


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract all contents of zip_path into extract_to."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(path=extract_to)


def merge_directories(src: Path, dest: Path):
    """Merge src/* into dest/*, creating dest if needed."""
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            # Python 3.8+ supports dirs_exist_ok
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def main():
    p = argparse.ArgumentParser(
        description="Unzip datasets and merge FPHA parts into `fpha/`"
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("downloaded_data/data"),
        help="Directory containing your ZIP files"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("unpacked_data"),
        help="Where to place the extracted folders"
    )
    args = p.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Simply unzip these two into their own folders:
    for name in ["FreiHAND-002.zip", "InterHand2.6M_5fps_batch1.zip"]:
        zip_path = data_dir / name
        dest = out_dir / name.replace(".zip", "")
        print(f"Extracting {zip_path} → {dest}")
        dest.mkdir(exist_ok=True, parents=True)
        extract_zip(zip_path, dest)

    # 2) Merge the two FPHA parts into a single fpha/ directory
    fpha_out = out_dir / "fpha/Video_files"
    fpha_out.mkdir(exist_ok=True, parents=True)

    for part in ["fpha_part_1.zip", "fpha_part_2.zip"]:
        zip_path = data_dir / part
        print(f"Processing {zip_path}")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            extract_zip(zip_path, tmp)

            # If zip unpacks into one top-level folder, drill in
            children = [c for c in tmp.iterdir() if not c.name.startswith("__MACOSX")]
            if len(children) == 1 and children[0].is_dir():
                src_root = children[0]
            else:
                src_root = tmp

            print(f" Merging {src_root} → {fpha_out}")
            merge_directories(src_root, fpha_out)

    print("✅ Done extracting and merging.")


if __name__ == "__main__":
    main()
