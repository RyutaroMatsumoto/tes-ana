"""
lecroy_trc_converter.py
----------------------
Utility script to convert Teledyne‑LeCroy *.trc binary waveform files into

* one *.npy file containing the raw waveform samples for each trace file
* a single JSON file holding the metadata for the whole run

Directory layout (fixed):
    Raw traces  : ../tes01/generated_data/raw_trc/pXX/rYYY/C1--Trace--00000.trc
    Waveform npy: ../tes01/generated_data/raw/pXX/rYYY/C1--Trace--00000.npy
    Metadata    : ../tes01/teststand_metadata/scope/pXX/rYYY/lecroy_metadata_pXX_rYYY.json

The script exposes two layers:
    1. **Static processing functions** (`parse_trc_file`, `save_waveform`, ...)
       – Pure, deterministic helpers with no I/O side effects except for the
         explicit path passed in.
    2. **Dynamic orchestration wrapper** (`process_run`) – Glues the helpers
       together for a given (p, r) pair, taking care of directory discovery,
       iteration, error handling, logging, and optional CLI invocation.

Author: Ryutaro Matsumoto – 2025‑04‑07
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Any, List

import numpy as np

# 3rd‑party: parser for LeCroy *.trc files.
try:
    from lecroyparser import ScopeData  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'lecroyparser'. Install via 'pip install lecroyparser'"
    ) from exc

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger().setLevel(logging.DEBUG)
###############################################################################
# ──────────────────────────── 1. PROCESSOR LAYER ─────────────────────────── #
###############################################################################

def parse_trc_file(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Parse a single LeCroy *.trc file.

    Parameters
    ----------
    path   : Path to the *.trc file.

    Returns
    -------
    waveform : np.ndarray  – The Y‑axis samples (voltage) in native units.
    metadata : dict        – Header information and scope settings.
    """
    logging.debug("Parsing %s", path)
    data = ScopeData(str(path))  # lecroyparser handles I/O & endianess
    print(data.y)
    waveform: np.ndarray = np.asarray(data.y, dtype=np.float32)
    metadata: Dict[str, Any] = {
        "path": str(path),
        "samples": waveform.size,
        "dt": float(data.horiz_interval),
        "t_origin": float(data.x_origin),
        "vertical_gain": float(data.vertical_gain),
        "vertical_offset": float(data.vertical_offset),
        "instrument": getattr(data, "instrument", None),
        "channel": getattr(data, "channel", None),
        "trigger_time": str(getattr(data, "trigger_time", "")),
    }
    return waveform, metadata


def save_waveform(array: np.ndarray, dest: Path, overwrite: bool = False) -> None:
    """Save *array* to *dest* (.npy). Creates parent directories if required."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        logging.debug("Skipping existing %s", dest)
        return
    np.save(dest, array)
    logging.debug("Saved waveform → %s", dest)


def append_metadata(meta_dict: Dict[str, Any], dest: Path, overwrite: bool = False) -> None:
    """Write *meta_dict* as JSON to *dest*.

    If the file exists and *overwrite* is False, merge the new data in‑place.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        with dest.open("r", encoding="utf-8") as fh:
            existing: Dict[str, Any] = json.load(fh)
        existing.update(meta_dict)
        meta_dict = existing
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(meta_dict, fh, indent=2, ensure_ascii=False)
    logging.info("Metadata written → %s (%d traces)", dest, len(meta_dict))

###############################################################################
# ──────────────────────────── 2. WRAPPER LAYER ───────────────────────────── #
###############################################################################

def process_run(p_id: str, r_id: str, *, overwrite: bool = False) -> None:
    """Convert every *.trc in the specified run and persist waveforms + metadata.

    Parameters
    ----------
    p_id, r_id : Identifiers like "01", "002" (do **not** include the leading 'p'/'r').
    overwrite  : If *True*, overwrite existing *.npy / JSON files.
    """
    # Resolve directories ----------------------------------------------------
    Base_dir = Path(__file__).resolve().parent.parent.parent/"tes01"
    raw_dir = Base_dir/"generated_data"/"raw_trc"/f"p{p_id}"/f"r{r_id}"
    out_dir = Base_dir/"generated_data"/"raw"/f"p{p_id}"/f"r{r_id}"
    meta_path = (Base_dir/"teststand_data"/"scope"/f"p{p_id}"/f"r{r_id}"/
                     f"lecroy_metadata_p{p_id}_r{r_id}.json")

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    trc_files: List[Path] = sorted(raw_dir.glob("C*--Trace--*.trc"))
    if not trc_files:
        logging.warning("No .trc files found in %s", raw_dir)
        return

    logging.info("Processing %d traces in %s", len(trc_files), raw_dir)
    meta_all: Dict[str, Any] = {}

    for f in trc_files:
        waveform, meta = parse_trc_file(f)
        save_waveform(waveform, out_dir / (f.stem + ".npy"), overwrite=overwrite)
        meta_all[f.name] = meta

    append_metadata(meta_all, meta_path, overwrite=overwrite)
    logging.info("✓ Run p%s r%s completed", p_id, r_id)

###############################################################################
# ────────────────────────────────── CLI ──────────────────────────────────── #
###############################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert LeCroy *.trc files to *.npy + metadata JSON.")
    parser.add_argument("p", help="p‑number (e.g. 01)")
    parser.add_argument("r", help="r‑number (e.g. 001)")
    parser.add_argument("--overwrite", "-f", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    try:
        process_run(args.p, args.r, overwrite=args.overwrite)
    except Exception as exc:
        logging.exception("Processing failed: %s", exc)
        sys.exit(1)
