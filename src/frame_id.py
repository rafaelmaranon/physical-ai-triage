"""Deterministic frame identifier.

The join key across BigQuery (metadata), LanceDB (embeddings), and MCAP
(playback). Hashing means two machines independently ingesting the same
segment land on the same ids, so the index can be rebuilt without re-ingesting.
"""

from __future__ import annotations

import hashlib


def frame_id(dataset: str, segment: str, ts_ns: int, camera: str) -> str:
    """Return a 16-hex-char id for one frame.

    16 hex chars = 64 bits. Birthday collision odds at 100M frames are ~3e-4,
    which is fine for a research index; bump to 24 chars if going to 1B+.
    """
    key = f"{dataset}|{segment}|{ts_ns}|{camera}".encode()
    return hashlib.blake2b(key, digest_size=8).hexdigest()


def parse_frame_id_components(dataset: str, segment: str, ts_ns: int, camera: str) -> dict:
    """Return the component dict alongside the hash — for writing BQ rows."""
    return {
        "frame_id": frame_id(dataset, segment, ts_ns, camera),
        "dataset": dataset,
        "segment": segment,
        "ts_ns": int(ts_ns),
        "camera": camera,
    }
