"""Download the IBM Telco Customer Churn CSV reproducibly.

Usage:  python -m src.data.download [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import urllib.request
from pathlib import Path

from src.config import RAW_DATA_PATH, RAW_DATA_URL

logger = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str = RAW_DATA_URL, dest: Path = RAW_DATA_PATH, force: bool = False) -> Path:
    if dest.exists() and not force:
        logger.info("Raw data already present at %s (use --force to re-download)", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https URL
        tmp.write_bytes(resp.read())
    tmp.replace(dest)
    return dest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if file exists")
    args = parser.parse_args()
    path = download(force=args.force)
    n_rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
    logger.info("Saved %s | rows=%d | sha256=%s", path, n_rows, sha256(path))


if __name__ == "__main__":
    main()
