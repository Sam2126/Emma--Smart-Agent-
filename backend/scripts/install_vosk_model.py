"""
Download the small offline English Vosk model for the wake-word listener.

Usage (from the backend folder):
    python scripts/install_vosk_model.py

Then set WAKE_WORD_ENGINE=vosk in .env. The model is about 40 MB and is
extracted to backend/data/vosk-model-small-en-us-0.15. The download resumes
if the connection drops.
"""

from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

import httpx

MODEL = "vosk-model-small-en-us-0.15"
URL = f"https://alphacephei.com/vosk/models/{MODEL}.zip"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def download(dest: Path, attempts: int = 8) -> None:
    for attempt in range(1, attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with httpx.stream("GET", URL, headers=headers, timeout=60.0, follow_redirects=True) as r:
                if r.status_code == 416:  # already complete
                    return
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                total = int(r.headers.get("content-length", 0)) + (have if mode == "ab" else 0)
                with dest.open(mode) as f:
                    for chunk in r.iter_bytes(1 << 16):
                        f.write(chunk)
                        if total:
                            print(f"\r  {dest.stat().st_size / 1e6:5.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            print()
            return
        except (httpx.HTTPError, OSError) as e:
            print(f"\n  attempt {attempt} failed: {e}; retrying...")
            time.sleep(min(30, 3 * attempt))
    raise SystemExit("Download failed after several attempts.")


def main() -> None:
    target = DATA_DIR / MODEL
    if target.exists():
        print(f"Vosk model already installed: {target}")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / f"{MODEL}.zip"
    print(f"Downloading {URL}")
    download(archive)
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise SystemExit(f"Corrupt archive member: {bad}. Delete {archive} and run again.")
        z.extractall(DATA_DIR)
    archive.unlink(missing_ok=True)
    print(f"Installed: {target}")
    print("Set WAKE_WORD_ENGINE=vosk in .env to use it.")


if __name__ == "__main__":
    sys.exit(main())
