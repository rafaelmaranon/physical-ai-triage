"""Minimal Flask search UI for physical-ai-triage.

Runs `hybrid.query(...)` against any LanceDB table and returns a thumbnail grid
in the browser. Thumbnails come from S3 (private bucket) — server generates
presigned URLs so the browser can load them without AWS credentials.

CLI:
    uv run python -m src.server.search                  # http://localhost:5050
    uv run python -m src.server.search --port 8000      # custom port

Endpoints:
    GET  /                — single-page UI (HTML embedded below)
    POST /search          — JSON: {text, table, sql_filter?, k} → [{frame_id, score, thumbnail_presigned, ...}]
    GET  /tables          — list of available LanceDB tables on the host

No auth, no sessions — local dev only. NOT for public hosting.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

import typer
from dotenv import load_dotenv
from flask import Flask, jsonify, request

from src.query.hybrid import query as hybrid_query

app = Flask(__name__)
cli = typer.Typer(add_completion=False, help=__doc__)


# Map local MCAP filenames → Foxglove Cloud recording IDs (one-time lookup via
# `foxglove recordings list`). Foxglove deeplinks ONLY work for cloud-hosted
# recordings, not local files (verified 2026-05-23 — local-file ds doesn't exist).
_FOX_RECORDING_IDS = {
    "seg1_v3":             "rec_0eKdelIungwguTaZ",
    "seg1_v4":             "rec_0eKer2xTeKz6Lhhd",
    "official_waymo_2026": "rec_0eKfNLQ1V0vHgf1p",
}
# Foxglove Cloud rebased all 3 MCAPs to start at 2026-04-01T00:00:00Z.
# Per-MCAP start_ts_ns from original file (used to compute offset within recording).
_MCAP_START_NS = {
    "seg1_v3":             1775001600000000000,
    "seg1_v4":             1775001600000000000,
    "official_waymo_2026": 1775001600000000000,
}
_FOX_CLOUD_BASE = "2026-04-01T00:00:00Z"
# Foxglove Studio layout ID — points at a saved 5-camera layout for the demo MCAPs.
# Anyone running their own deployment should swap this for their own layout ID via env var
# (see ~/Library/Application Support/Foxglove/studio-datastores/layouts-local/ for the format).
_FOX_LAYOUT_ID = os.environ.get("FOXGLOVE_LAYOUT_ID", "lay_idhuBwRqpOIrNWwy")


def _foxglove_deeplink(mcap_path: str | None, ts_ns: int | None) -> str | None:
    """Build a Foxglove Cloud deeplink that opens the recording at the right time.
    Maps local MCAP filename → cloud recording_id, converts absolute ts_ns to
    RFC3339 time relative to the cloud-rebased epoch (2026-04-01T00:00:00Z + offset).
    Returns None if the MCAP isn't in the cloud.
    """
    if not mcap_path or not ts_ns:
        return None
    from pathlib import Path as _P
    mcap_name = _P(mcap_path).stem
    rec_id = _FOX_RECORDING_IDS.get(mcap_name)
    if not rec_id:
        return None
    start_ns = _MCAP_START_NS.get(mcap_name)
    if not start_ns:
        return None
    offset_seconds = (ts_ns - start_ns) / 1e9
    # Format RFC3339 with sub-second precision, anchored at the cloud rebase epoch
    import datetime as _dt
    base = _dt.datetime(2026, 4, 1, 0, 0, 0, tzinfo=_dt.timezone.utc)
    target = base + _dt.timedelta(seconds=offset_seconds)
    iso = target.strftime("%Y-%m-%dT%H:%M:%S.") + f"{target.microsecond:06d}Z"
    # Returns BOTH a desktop-app URL (foxglove://) and a web URL (app.foxglove.dev).
    # Desktop URL reads our locally-written layout from ~/Library/Application Support/Foxglove/...;
    # web URL only sees cloud-synced layouts so the layoutId may be ignored there.
    base_qs = (f"?ds=foxglove-data-platform"
               f"&ds.recordingId={rec_id}"
               f"&layoutId={_FOX_LAYOUT_ID}"
               f"&time={iso}")
    return {
        "desktop": f"foxglove://open{base_qs}",
        "web": f"https://app.foxglove.dev/~/view{base_qs}",
    }


def _presign(thumb_uri: str, expires: int = 3600) -> str | None:
    """Convert s3://bucket/key.jpg → https presigned URL.
    For file:// URIs (MCAP-extracted frames), serve via /local_thumb endpoint
    since browsers block file:// loads from HTTP-served pages."""
    if not thumb_uri:
        return thumb_uri
    if thumb_uri.startswith("file://"):
        # Route through Flask so the browser can load it
        local_path = thumb_uri.replace("file://", "")
        from urllib.parse import quote
        return f"/local_thumb?path={quote(local_path)}"
    if not thumb_uri.startswith("s3://"):
        return thumb_uri
    import boto3
    p = urlparse(thumb_uri)
    bucket = p.netloc
    key = p.path.lstrip("/")
    s3 = boto3.client("s3")
    return s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": bucket,
            "Key": key,
            # Force browser to render inline as JPEG rather than download as octet-stream
            # (some older thumbnails were uploaded without Content-Type metadata).
            "ResponseContentType": "image/jpeg",
            "ResponseContentDisposition": "inline",
        },
        ExpiresIn=expires,
    )


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/compare")
def compare_page():
    return COMPARE_HTML


@app.route("/llm")
def llm_page():
    return LLM_HTML


@app.route("/local_thumb")
def local_thumb():
    """Serve a local JPEG thumbnail by path. Allow-listed to data/mcap_frames only."""
    from flask import send_file, abort
    p = request.args.get("path", "")
    abspath = Path(p).resolve()
    # Security: only allow files under data/mcap_frames/
    allowed_root = (Path(os.environ.get("LANCE_DIR", "data/lance")).parent / "mcap_frames").resolve()
    if not str(abspath).startswith(str(allowed_root)):
        abort(403)
    if not abspath.is_file():
        abort(404)
    return send_file(str(abspath), mimetype="image/jpeg")


# Anthropic client (lazy)
_ANTHROPIC = {"client": None, "model": "claude-haiku-4-5-20251001"}


def _llm_route(query: str) -> dict:
    """Send query to Claude, get back {vector_query, sql_filter, caption_filter, reasoning}."""
    import anthropic
    import re as _re
    if _ANTHROPIC["client"] is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _ANTHROPIC["client"] = anthropic.Anthropic(api_key=api_key)

    system_prompt = """You convert natural-language queries about autonomous-driving footage into retrieval directives.

The user's substrate has:
- vector index: SigLIP embeddings of every thumbnail (raw visual gist)
- structured columns: dataset ('waymo'|'bdd100k'), camera_name ('FRONT'|'FRONT_LEFT'|'FRONT_RIGHT'|'SIDE_LEFT'|'SIDE_RIGHT')
- structured columns DEFINED but NULL in v1: city, weather, time_of_day, num_pedestrians, ego_speed_mps
- description: natural-language caption written by Cosmos-Reason2-8B VLM (only 500 frames, may mention pedestrians, crosswalks, night, intersections, weather, etc.)

For each user query, decompose into:
- vector_query: short text for raw visual similarity (keep the concrete visual content, drop legal/temporal abstractions)
- sql_filter: SQL WHERE clause using only POPULATED columns (dataset + camera_name) or NULL if not applicable
- caption_filter: ILIKE pattern for matching against the Cosmos caption text (e.g. '%pedestrian%' or '%night%') or NULL
- reasoning: 1-2 sentences explaining your decomposition

Output ONLY valid JSON. No markdown fences. Example:
{"vector_query": "pedestrian crossing street", "sql_filter": null, "caption_filter": "%pedestrian%", "reasoning": "Query is about pedestrian crossings; metadata for legal context (in_crosswalk) is NULL so I fall back to caption text matching pedestrian-related captions plus visual similarity."}"""

    msg = _ANTHROPIC["client"].messages.create(
        model=_ANTHROPIC["model"],
        max_tokens=400,
        system=system_prompt,
        messages=[{"role": "user", "content": query}],
    )
    text = msg.content[0].text.strip()
    # Strip optional code fences
    text = _re.sub(r"^```(?:json)?\s*", "", text)
    text = _re.sub(r"\s*```$", "", text)
    return json.loads(text)


@app.route("/llm_search", methods=["POST"])
def llm_search():
    """Take a text query, decompose via LLM, run 3 retrieval modes, return all results."""
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    k = int(data.get("k", 8))
    if not text:
        return jsonify({"error": "no query text"}), 400

    # 1) LLM routes
    try:
        routed = _llm_route(text)
    except Exception as e:
        return jsonify({"error": f"LLM route failed: {type(e).__name__}: {e}"}), 500

    # 2) Run three retrieval modes against waymo table
    import lancedb
    import duckdb
    from src.cloud import duckdb_with_s3
    lance_dir = os.environ.get("LANCE_DIR", "data/lance")
    db = lancedb.connect(lance_dir)
    tbl = db.open_table("waymo")
    caps = _load_captions()

    out = {"query": text, "routed": routed, "modes": []}

    # MODE A: pure vector (today's default behaviour)
    try:
        hits_a = hybrid_query(text=text, table="waymo", k=k, sql_filter=None,
                              lance_dir=lance_dir)
        out["modes"].append({"name": "Pure vector (SigLIP)",
                             "subtitle": "what the search box does today",
                             "hits": [_h(h, caps) for h in hits_a]})
    except Exception as e:
        out["modes"].append({"name": "Pure vector (SigLIP)", "error": str(e), "hits": []})

    # MODE B: pure structured — SQL only, returns matching frames (no vector ranking)
    sql = routed.get("sql_filter")
    cap = routed.get("caption_filter")
    try:
        if not sql and not cap:
            out["modes"].append({"name": "Pure structured (SQL + caption text)",
                                 "subtitle": "no SQL clause extracted by LLM",
                                 "hits": [], "notes": "LLM didn't extract a structured filter for this query"})
        else:
            from src.cloud import join as uri_join
            bucket_uri = os.environ.get("BUCKET_URI", "s3://YOUR_BUCKET")
            con = duckdb_with_s3(duckdb.connect())
            preds = []
            if sql:
                preds.append(sql)
            # caption-filter via the Cosmos R2 captions parquet
            if cap:
                cap_sql_path = "eval/cosmos_descriptions_r2.parquet"
                if Path(cap_sql_path).exists():
                    cap_safe = cap.replace("'", "''")
                    cap_frames = con.execute(
                        f"SELECT frame_id FROM read_parquet('{cap_sql_path}') WHERE description ILIKE '{cap_safe}'"
                    ).fetchall()
                    cap_fid_list = ",".join(f"'{r[0]}'" for r in cap_frames)
                    if cap_fid_list:
                        preds.append(f"frame_id IN ({cap_fid_list})")
                    else:
                        preds.append("FALSE")
            # Read structured metadata parquets
            meta_uris = [uri_join(bucket_uri, "waymo", "metadata", "*.parquet"),
                         uri_join(bucket_uri, "bdd100k", "metadata", "*.parquet")]
            union_sql = " UNION ALL BY NAME ".join(
                f"SELECT * FROM read_parquet('{uri}')" for uri in meta_uris)
            full_pred = " AND ".join(preds)
            structured_sql = (
                f"WITH meta AS ({union_sql}) "
                f"SELECT frame_id, dataset, camera_name, ts_ns, thumbnail_uri "
                f"FROM meta WHERE {full_pred} LIMIT {k}"
            )
            rows = con.execute(structured_sql).fetchall()
            hits_b = []
            for r in rows:
                hits_b.append({
                    "frame_id": r[0], "dataset": r[1], "camera_name": r[2],
                    "ts_ns": int(r[3]), "score": 1.0,
                    "thumbnail_presigned": _presign(r[4]),
                    "caption": caps["r2"].get(r[0], "") if caps["r2"] else "",
                })
            out["modes"].append({"name": "Pure structured (SQL + caption text)",
                                 "subtitle": "no embedding — just filters from LLM's decomposition",
                                 "hits": hits_b, "filter_sql": full_pred})
    except Exception as e:
        out["modes"].append({"name": "Pure structured (SQL + caption text)",
                             "error": str(e), "hits": []})

    # MODE C: LLM-routed hybrid — Semantic-Drive pattern.
    # Use caption_filter + sql_filter to build a CANDIDATE SET, then vector-rank within it
    # (NOT intersect with global top-K — that's almost always empty since caption covers
    # only 500/95K frames). Restrict the LanceDB WHERE clause to those candidate ids
    # so the returned top-K is the BEST-RANKED frames from the filtered subset.
    try:
        vq = routed.get("vector_query") or text
        candidate_ids = None
        if cap:
            cap_sql_path = "eval/cosmos_descriptions_r2.parquet"
            if Path(cap_sql_path).exists():
                cap_safe = cap.replace("'", "''")
                con = duckdb.connect()
                cap_frames = con.execute(
                    f"SELECT frame_id FROM read_parquet('{cap_sql_path}') WHERE description ILIKE '{cap_safe}'"
                ).fetchall()
                candidate_ids = [r[0] for r in cap_frames]

        # Open the table + encode query once
        tbl = db.open_table("waymo")
        from src.query.hybrid import _siglip_text_encode
        qvec = _siglip_text_encode(vq)
        search = tbl.search(qvec)

        # Apply candidate restriction at LanceDB level — vector search runs WITHIN the set
        if candidate_ids:
            # LanceDB WHERE supports IN; cap at 10K for safety
            in_list = ",".join(f"'{f}'" for f in candidate_ids[:10000])
            search = search.where(f"frame_id IN ({in_list})")
        # Apply sql_filter via the metadata parquet if present
        # (skipped here since hybrid_query handles it cleanly + we need the within-candidates flow)
        raw = search.limit(k).to_list()
        hits_c = []
        for r in raw:
            score = float(1 - r.get("_distance", 0))
            cap_text = caps["r2"].get(r["frame_id"], "") if caps["r2"] else ""
            hits_c.append({
                "frame_id": r["frame_id"], "score": round(score, 4),
                "dataset": r["dataset"], "camera_name": r["camera_name"],
                "ts_ns": int(r["ts_ns"]),
                "thumbnail_presigned": _presign(r["thumbnail_uri"]),
                "caption": cap_text,
            })
        notes = None
        if candidate_ids and not hits_c:
            notes = f"caption matched {len(candidate_ids)} frames but vector search returned none from that subset"
        elif not candidate_ids and not sql:
            notes = "no caption_filter or sql_filter from LLM — same as pure vector"
        out["modes"].append({
            "name": "LLM-routed hybrid (vector inside candidate set)",
            "subtitle": "Semantic-Drive pattern: caption/SQL builds candidates → vector ranks within",
            "hits": hits_c,
            "notes": notes,
            "candidates_count": len(candidate_ids) if candidate_ids else None,
        })
    except Exception as e:
        out["modes"].append({"name": "LLM-routed hybrid", "error": str(e), "hits": []})

    return jsonify(out)


def _h(h, caps):
    """Render a Hit object to JSON dict for the LLM compare UI."""
    return {
        "frame_id": h.frame_id, "score": round(h.score, 4),
        "dataset": h.dataset, "camera_name": h.camera_name,
        "ts_ns": h.ts_ns,
        "thumbnail_presigned": _presign(h.thumbnail_uri),
        "caption": caps["r2"].get(h.frame_id, "") if caps["r2"] else "",
    }


# In-memory caption cache (Cosmos R1 + R2 captions for 500 frames each)
_CAPTION_CACHE = {"r1": None, "r2": None}


def _load_captions():
    """Load Cosmos-Reason1 + Reason2 caption parquets into dicts {frame_id: description}."""
    if _CAPTION_CACHE["r1"] is None:
        import pandas as pd
        from pathlib import Path
        for key, path in [("r1", "eval/cosmos_descriptions.parquet"),
                          ("r2", "eval/cosmos_descriptions_r2.parquet")]:
            p = Path(path)
            if p.exists():
                df = pd.read_parquet(p)
                _CAPTION_CACHE[key] = dict(zip(df.frame_id, df.description))
            else:
                _CAPTION_CACHE[key] = {}
    return _CAPTION_CACHE


# Which LanceDB tables to run for the compare view, with display labels.
_COMPARE_MODELS = [
    ("waymo",            "SigLIP-base",          "image · single frame",     None),
    ("waymo_clip",       "CLIP ViT-L/14",        "image · single frame",     None),
    ("waymo_dinov2",     "DINOv2-large",         "image · no text encoder",  None),
    ("cosmos_aug",       "Cosmos-Reason1-7B",    "caption → SigLIP-text",    "r1"),
    ("cosmos_aug_r2",    "Cosmos-Reason2-8B",    "caption → SigLIP-text",    "r2"),
    ("cosmos_embed1",    "Cosmos Embed1-336p",   "8-frame video temporal",   None),
]


@app.route("/compare_search", methods=["POST"])
def compare_search():
    """Run the same text query across all 6 retrieval models. Return their top-K with captions where available."""
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    k = int(data.get("k", 5))
    if not text:
        return jsonify({"error": "no query text"}), 400

    caps = _load_captions()
    out = {"query": text, "k": k, "models": []}

    # First pass: run SigLIP text query — its top-1 becomes the seed for DINOv2 (image-only).
    siglip_top1_frame_id: str | None = None

    for table, label, kind, cap_source in _COMPARE_MODELS:
        block = {"table": table, "label": label, "kind": kind, "hits": [], "error": None, "seeded_by": None}
        try:
            if "no text encoder" in kind:
                # Image-only model — seed with SigLIP top-1.
                # Pull a deep slice (200) and take the LAST 4 — the *least similar* to the seed.
                # This intentionally shows visually distinct frames so the column doesn't duplicate
                # the other columns' picks.
                if not siglip_top1_frame_id:
                    raise RuntimeError("no SigLIP seed available")
                deep = hybrid_query(
                    image_frame_id=siglip_top1_frame_id,
                    sql_filter=None,
                    table=table,
                    k=2000,
                    lance_dir=os.environ.get("LANCE_DIR", "data/lance"),
                )
                # Drop the seed itself, then take the last 6 (most-dissimilar within the deep pool)
                deep = [h for h in deep if h.frame_id != siglip_top1_frame_id]
                print(f"[DINOv2 deep query] returned {len(deep)} candidates", flush=True)
                hits = deep[-6:] if len(deep) >= 6 else deep
                # seeded_by intentionally not set — keep the UI header clean.
            else:
                hits = hybrid_query(
                    text=text,
                    sql_filter=None,
                    table=table,
                    k=k,
                    lance_dir=os.environ.get("LANCE_DIR", "data/lance"),
                )
            for h in hits:
                row = {
                    "frame_id": h.frame_id,
                    "score": round(h.score, 4),
                    "dataset": h.dataset,
                    "camera_name": h.camera_name,
                    "thumbnail_presigned": _presign(h.thumbnail_uri),
                }
                if cap_source:
                    row["caption"] = caps[cap_source].get(h.frame_id, "")
                block["hits"].append(row)
            # Capture SigLIP top-1 frame_id for seeding image-only models that come later in the loop.
            if table == "waymo" and hits and not siglip_top1_frame_id:
                siglip_top1_frame_id = hits[0].frame_id
        except Exception as e:
            block["error"] = f"{type(e).__name__}: {e}"
        out["models"].append(block)
    return jsonify(out)


@app.route("/tables")
def tables():
    # db.table_names() (legacy LanceDB API) returns a stale cache and misses freshly-
    # created tables. List the lance_dir directly — every Lance table is a `.lance/` dir.
    lance_dir = os.environ.get("LANCE_DIR", "data/lance")
    p = Path(lance_dir)
    if not p.exists():
        return jsonify([])
    names = sorted(d.stem for d in p.iterdir()
                   if d.is_dir() and d.suffix == ".lance")
    return jsonify(names)


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    table = data.get("table", "waymo")
    sql_filter = data.get("sql_filter") or None
    k = int(data.get("k", 12))
    if not text:
        return jsonify({"error": "no query text"}), 400

    try:
        hits = hybrid_query(
            text=text,
            sql_filter=sql_filter,
            table=table,
            k=k,
            lance_dir=os.environ.get("LANCE_DIR", "data/lance"),
        )
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    # Build Foxglove deeplinks if this is an MCAP table — look up mcap_path from LanceDB
    mcap_lookup = {}
    if "_mcap_" in table or table.startswith("waymo_mcap"):
        try:
            import lancedb
            db = lancedb.connect(os.environ.get("LANCE_DIR", "data/lance"))
            mcap_tbl = db.open_table(table)
            mcap_lookup = {r["frame_id"]: r.get("mcap_path", "")
                           for r in mcap_tbl.search().limit(10**6).to_list()}
        except Exception:
            pass

    out = []
    for h in hits:
        mcap_path = mcap_lookup.get(h.frame_id)
        urls = _foxglove_deeplink(mcap_path, h.ts_ns) if mcap_path else None
        out.append({
            "frame_id": h.frame_id,
            "score": round(h.score, 4),
            "dataset": h.dataset,
            "device_id": h.device_id,
            "ts_ns": h.ts_ns,
            "camera_name": h.camera_name,
            "thumbnail_presigned": _presign(h.thumbnail_uri),
            "city": h.city,
            "time_of_day": h.time_of_day,
            "weather": h.weather,
            "num_pedestrians": h.num_pedestrians,
            "ego_speed_mps": h.ego_speed_mps,
            "mcap_path": mcap_path,
            "foxglove_url": urls["web"] if urls else None,        # for backwards-compat
            "foxglove_url_desktop": urls["desktop"] if urls else None,
            "foxglove_url_web": urls["web"] if urls else None,
        })
    return jsonify({"hits": out, "table": table, "k": k, "sql_filter": sql_filter})


INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>physical-ai-triage · search</title>
<style>
  :root { --bg:#fafbfc; --panel:#fff; --ink:#14181f; --muted:#6b7280; --line:#e5e7eb; --accent:#0a7; --mono: ui-monospace,"SF Mono",Menlo,monospace; }
  * { box-sizing:border-box; } body { font:14.5px/1.5 -apple-system,system-ui,sans-serif; color:var(--ink); background:var(--bg); margin:0; }
  header { padding:16px 24px; background:var(--panel); border-bottom:1px solid var(--line); position:sticky; top:0; z-index:10; }
  h1 { margin:0 0 4px; font-size:18px; }
  .sub { color:var(--muted); font-size:12px; margin-bottom:12px; }
  .controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  input[type=text], select { font:14px var(--mono); padding:8px 12px; border:1px solid var(--line); border-radius:6px; background:#fff; }
  input[type=text] { flex:1; min-width:280px; }
  select { min-width:140px; }
  input[type=number] { width:60px; font:14px var(--mono); padding:8px; border:1px solid var(--line); border-radius:6px; }
  button { padding:8px 18px; font:14px var(--mono); background:var(--accent); color:#fff; border:none; border-radius:6px; cursor:pointer; }
  button:hover { background:#076; } button:disabled { opacity:0.5; cursor:wait; }
  .examples { font-size:12px; color:var(--muted); margin-top:8px; }
  .examples a { color:var(--accent); cursor:pointer; margin-right:14px; text-decoration:none; }
  .examples a:hover { text-decoration:underline; }
  main { padding:20px 24px; }
  .meta { font-size:12px; color:var(--muted); margin-bottom:14px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:14px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; }
  .card img { width:100%; height:auto; border-radius:4px; background:#000; display:block; }
  .card .row1 { display:flex; justify-content:space-between; font:11px var(--mono); color:var(--muted); margin-top:8px; }
  .card .score { color:var(--accent); font-weight:700; }
  .card .ds { font-weight:600; color:#3730a3; }
  .card .fid { font:10px var(--mono); color:var(--muted); margin-top:4px; word-break:break-all; }
  .card .ctx { font-size:12px; color:#444; margin-top:6px; }
  .err { color:#c40; padding:14px; background:#fff5f0; border-radius:6px; font:13px var(--mono); }
  .loading { color:var(--muted); padding:20px; text-align:center; }
  .fox-row { display:flex; gap:4px; margin-top:6px; }
  .fox-btn {
    flex:1; padding:6px 8px; font:11.5px var(--mono);
    color:#fff; text-decoration:none; border-radius:4px;
    text-align:center; font-weight:600; transition:background 0.15s;
  }
  .fox-btn.fox-desktop { background:#5B50D6; }
  .fox-btn.fox-desktop:hover { background:#4338ca; text-decoration:none; }
  .fox-btn.fox-web { background:#0a7; }
  .fox-btn.fox-web:hover { background:#076; text-decoration:none; }
  /* Lightbox */
  .lb-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.92); display:none; align-items:center; justify-content:center; z-index:100; cursor:zoom-out; }
  .lb-overlay.on { display:flex; }
  .lb-stage { display:flex; flex-direction:column; align-items:center; gap:14px; max-width:96vw; max-height:96vh; }
  /* 3× native — 256px source → ~768px on screen. Aspect ratio preserved.
     image-rendering:auto = browser smoothing (not nearest-neighbor pixelation). */
  .lb-img { width:768px; max-width:90vw; max-height:78vh; object-fit:contain; image-rendering:auto; border-radius:6px; box-shadow:0 8px 40px rgba(0,0,0,0.5); background:#000; }
  .lb-meta { color:#fff; font:13px var(--mono); text-align:center; max-width:90vw; }
  .lb-meta .row1 { font-size:15px; margin-bottom:4px; }
  .lb-meta .ctx { color:#ddd; margin-top:4px; }
  .lb-meta .nav { color:#999; margin-top:8px; font-size:11px; letter-spacing:0.5px; }
  .lb-arrow { position:absolute; top:50%; transform:translateY(-50%); background:rgba(255,255,255,0.1); border:none; color:#fff; width:50px; height:50px; border-radius:50%; font-size:24px; cursor:pointer; display:flex; align-items:center; justify-content:center; }
  .lb-arrow:hover { background:rgba(255,255,255,0.25); }
  .lb-prev { left:20px; } .lb-next { right:20px; }
  .lb-close { position:absolute; top:20px; right:20px; background:rgba(255,255,255,0.1); border:none; color:#fff; width:36px; height:36px; border-radius:50%; font-size:18px; cursor:pointer; }
  .lb-close:hover { background:rgba(255,255,255,0.25); }
  .lb-counter { position:absolute; top:24px; left:24px; color:#fff; font:13px var(--mono); background:rgba(0,0,0,0.5); padding:4px 10px; border-radius:4px; }
</style></head>
<body>
<header>
  <h1>🔍 physical-ai-triage · search</h1>
  <div class="sub">Type a description · pick a model · get top-K thumbnails from 95K AV frames</div>
  <div class="controls">
    <input id="q" type="text" placeholder='e.g. "jaywalker at night in SF"' autofocus>
    <select id="table"><option value="waymo">waymo (SigLIP)</option></select>
    <input id="k" type="number" min="1" max="48" value="12">
    <button id="go">Search</button>
  </div>
  <div class="examples">
    <span>Try:</span>
    <a onclick="setQ('jaywalker at night in SF')">jaywalker at night in SF</a>
    <a onclick="setQ('unprotected left turn with pedestrian')">unprotected left turn</a>
    <a onclick="setQ('wet road reflections at night')">wet road reflections</a>
    <a onclick="setQ('construction zone cones')">construction zone</a>
    <a onclick="setQ('cyclist weaving between cars')">cyclist weaving</a>
  </div>
</header>
<main>
  <div class="meta" id="meta"></div>
  <div id="results" class="grid"></div>
</main>

<!-- Lightbox overlay (hidden until thumbnail click) -->
<div class="lb-overlay" id="lb">
  <div class="lb-counter" id="lb-counter"></div>
  <button class="lb-close" id="lb-close" title="Close (Esc)">×</button>
  <button class="lb-arrow lb-prev" id="lb-prev" title="Previous (←)">‹</button>
  <button class="lb-arrow lb-next" id="lb-next" title="Next (→)">›</button>
  <div class="lb-stage" onclick="event.stopPropagation()">
    <img class="lb-img" id="lb-img" alt="">
    <div class="lb-meta" id="lb-meta"></div>
  </div>
</div>
<script>
  const $q = document.getElementById('q');
  const $table = document.getElementById('table');
  const $k = document.getElementById('k');
  const $go = document.getElementById('go');
  const $results = document.getElementById('results');
  const $meta = document.getElementById('meta');

  // Lightbox state + handlers
  let currentHits = [];
  let lbIndex = -1;
  const $lb = document.getElementById('lb');
  const $lbImg = document.getElementById('lb-img');
  const $lbMeta = document.getElementById('lb-meta');
  const $lbCounter = document.getElementById('lb-counter');
  function openLb(i) {
    if (i < 0 || i >= currentHits.length) return;
    lbIndex = i;
    const h = currentHits[i];
    $lbImg.src = h.thumbnail_presigned;
    const ctx = [h.city, h.time_of_day, h.weather].filter(x=>x).join(' · ');
    const foxBtn = h.foxglove_url
      ? `<div style="margin-top:10px"><a href="${h.foxglove_url}" style="display:inline-block;background:#5B50D6;color:#fff;padding:8px 16px;border-radius:6px;text-decoration:none;font-weight:600">🦊 Open in Foxglove at ts=${h.ts_ns}</a></div>`
      : '';
    $lbMeta.innerHTML =
      `<div class="row1">#${i+1} · score ${h.score.toFixed(3)} · ${h.dataset} · ${h.camera_name}</div>` +
      `<div>frame_id: ${h.frame_id} · ts=${h.ts_ns}</div>` +
      (ctx ? `<div class="ctx">${ctx}</div>` : '') +
      (h.mcap_path ? `<div style="font:10px ui-monospace,monospace;color:#aaa;margin-top:4px">${h.mcap_path}</div>` : '') +
      foxBtn +
      `<div class="nav">← prev · → next · Esc close · click outside image to close</div>`;
    $lbCounter.textContent = `${i+1} / ${currentHits.length}`;
    $lb.classList.add('on');
  }
  function closeLb() { $lb.classList.remove('on'); lbIndex = -1; }
  function nextLb() { if (lbIndex >= 0) openLb((lbIndex+1) % currentHits.length); }
  function prevLb() { if (lbIndex >= 0) openLb((lbIndex-1+currentHits.length) % currentHits.length); }
  $lb.addEventListener('click', closeLb);
  document.getElementById('lb-close').addEventListener('click', e => { e.stopPropagation(); closeLb(); });
  document.getElementById('lb-next').addEventListener('click', e => { e.stopPropagation(); nextLb(); });
  document.getElementById('lb-prev').addEventListener('click', e => { e.stopPropagation(); prevLb(); });
  document.addEventListener('keydown', e => {
    if (!$lb.classList.contains('on')) return;
    if (e.key === 'ArrowRight') nextLb();
    else if (e.key === 'ArrowLeft') prevLb();
    else if (e.key === 'Escape') closeLb();
  });

  fetch('/tables').then(r => r.json()).then(tables => {
    const labels = { waymo:'waymo (SigLIP)', bdd100k:'bdd100k (SigLIP)',
      waymo_clip:'waymo (CLIP)', bdd100k_clip:'bdd100k (CLIP)',
      waymo_dinov2:'waymo (DINOv2)', bdd100k_dinov2:'bdd100k (DINOv2)',
      cosmos_aug:'Cosmos-Reason1-7B → SigLIP-text',
      cosmos_aug_r2:'Cosmos-Reason2-8B → SigLIP-text',
      cosmos_embed1:'Cosmos Embed1 (video, temporal)',
      object_bev:'object_bev (LiDAR BEV)',
      waymo_mcap_siglip:'🦊 MCAP local (SigLIP) — 150 frames + Foxglove deeplinks',
      waymo_mcap_clip:'🦊 MCAP local (CLIP) — 150 frames + Foxglove deeplinks',
      waymo_mcap_dinov2:'🦊 MCAP local (DINOv2) — 150 frames + Foxglove deeplinks' };
    $table.innerHTML = '';
    tables.forEach(t => {
      const o = document.createElement('option');
      o.value = t; o.textContent = labels[t] || t;
      $table.appendChild(o);
    });
  });

  function setQ(s) { $q.value = s; $q.focus(); }

  async function go() {
    const text = $q.value.trim();
    if (!text) return;
    $go.disabled = true;
    $results.innerHTML = '<div class="loading">searching…</div>';
    $meta.textContent = '';
    const t0 = performance.now();
    try {
      const res = await fetch('/search', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ text, table: $table.value, k: +$k.value })
      });
      const elapsed = (performance.now() - t0).toFixed(0);
      const data = await res.json();
      if (data.error) { $results.innerHTML = '<div class="err">' + data.error + '</div>'; return; }
      $meta.textContent = `${data.hits.length} hits · ${data.table} · ${elapsed}ms · click thumbnail to open lightbox · click 🦊 to open in Foxglove at that timestamp`;
      currentHits = data.hits;
      $results.innerHTML = data.hits.map((h, i) => {
        const foxBtns = h.foxglove_url_desktop
          ? `<div class="fox-row">
               <a class="fox-btn fox-desktop" href="${h.foxglove_url_desktop}" title="Open in Foxglove DESKTOP app at ts=${h.ts_ns}" onclick="event.stopPropagation()">🦊 Desktop</a>
               <a class="fox-btn fox-web" href="${h.foxglove_url_web}" target="_blank" title="Open in Foxglove web app (browser)" onclick="event.stopPropagation()">🌐 Web</a>
             </div>`
          : '';
        return `<div class="card" style="cursor:zoom-in" onclick="openLb(${i})">
          <img src="${h.thumbnail_presigned}" loading="lazy">
          <div class="row1"><span class="score">#${i+1} · ${h.score.toFixed(3)}</span><span class="ds">${h.dataset} · ${h.camera_name}</span></div>
          <div class="fid">${h.frame_id} · ts=${h.ts_ns}</div>
          ${(h.city || h.time_of_day || h.weather) ? `<div class="ctx">${[h.city, h.time_of_day, h.weather].filter(x=>x).join(' · ')}</div>` : ''}
          ${foxBtns}
        </div>`;
      }).join('');
    } catch (e) {
      $results.innerHTML = '<div class="err">' + e + '</div>';
    } finally {
      $go.disabled = false;
    }
  }
  $go.addEventListener('click', go);
  $q.addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
</script>
</body></html>"""


COMPARE_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>physical-ai-triage · 6-model compare</title>
<style>
  :root { --bg:#fafbfc; --panel:#fff; --ink:#14181f; --muted:#6b7280; --line:#e5e7eb; --accent:#0a7; --mono: ui-monospace,"SF Mono",Menlo,monospace; }
  * { box-sizing:border-box; } body { font:13.5px/1.5 -apple-system,system-ui,sans-serif; color:var(--ink); background:var(--bg); margin:0; }
  header { padding:14px 20px; background:var(--panel); border-bottom:1px solid var(--line); position:sticky; top:0; z-index:10; }
  h1 { margin:0 0 4px; font-size:17px; }
  .sub { color:var(--muted); font-size:12px; margin-bottom:10px; }
  .controls { display:flex; gap:8px; align-items:center; }
  input[type=text] { flex:1; min-width:260px; font:13.5px var(--mono); padding:7px 11px; border:1px solid var(--line); border-radius:6px; background:#fff; }
  input[type=number] { width:54px; font:13.5px var(--mono); padding:7px; border:1px solid var(--line); border-radius:6px; }
  button { padding:7px 16px; font:13.5px var(--mono); background:var(--accent); color:#fff; border:none; border-radius:6px; cursor:pointer; }
  button:hover { background:#076; } button:disabled { opacity:0.5; cursor:wait; }
  .examples { font-size:11.5px; color:var(--muted); margin-top:6px; }
  .examples a { color:var(--accent); cursor:pointer; margin-right:12px; text-decoration:none; }
  .examples a:hover { text-decoration:underline; }
  main { padding:16px 20px; }
  .meta { font-size:12px; color:var(--muted); margin-bottom:12px; }
  .grid6 { display:grid; grid-template-columns:repeat(6, 1fr); gap:12px; }
  @media (max-width:1500px) { .grid6 { grid-template-columns:repeat(3, 1fr); } }
  @media (max-width:900px)  { .grid6 { grid-template-columns:repeat(2, 1fr); } }
  @media (max-width:600px)  { .grid6 { grid-template-columns:1fr; } }
  .col { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; }
  .col h3 { margin:0 0 2px; font-size:13.5px; }
  .col .kind { font:10.5px var(--mono); color:var(--muted); margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid var(--line); }
  .hit { padding:6px 0; border-bottom:1px dashed #eee; }
  .hit:last-child { border-bottom:none; }
  .hit img { width:100%; height:auto; border-radius:3px; background:#000; display:block; margin-bottom:4px; }
  .hit .rank { display:inline-block; background:#3730a3; color:#fff; padding:1px 5px; border-radius:3px; font:10px var(--mono); font-weight:700; }
  .hit .score { color:var(--accent); font:10px var(--mono); margin-left:6px; }
  .hit .cam { color:var(--muted); font:10px var(--mono); float:right; }
  .hit .fid { font:9.5px var(--mono); color:var(--muted); margin-top:2px; word-break:break-all; }
  .hit .caption { font-size:11.5px; line-height:1.4; color:#1c3a5e; background:#fcfdff; padding:6px 8px; border-radius:3px; margin-top:4px; border-left:2px solid var(--accent); }
  .hit .no-cap { font:10px var(--mono); color:#aaa; font-style:italic; margin-top:4px; }
  .err { color:#c40; padding:8px; background:#fff5f0; border-radius:4px; font:11px var(--mono); }
  .loading { color:var(--muted); padding:14px; text-align:center; }
</style></head>
<body>
<header>
  <h1>🧪 6-model comparison · same query, all retrieval backbones</h1>
  <div class="sub">Run the same text query against all 6 LanceDB tables. SQL filter bypassed. Top-K per model shown side by side. Cosmos-Reason rows show the actual VLM caption for each hit (when available).</div>
  <div class="controls">
    <input id="q" type="text" placeholder='e.g. "jaywalker at night in SF" or "school" or "construction zone"' autofocus>
    <input id="k" type="number" min="1" max="10" value="5" title="top-K per model">
    <button id="go">Compare</button>
  </div>
  <div class="examples">
    <span>Try:</span>
    <a onclick="setQ('jaywalker at night in SF')">jaywalker at night</a>
    <a onclick="setQ('unprotected left turn with pedestrian')">unprotected left turn</a>
    <a onclick="setQ('school')">school</a>
    <a onclick="setQ('school buses')">school buses</a>
    <a onclick="setQ('cyclist weaving between cars')">cyclist weaving</a>
    <a onclick="setQ('wet road reflections at night')">wet road reflections</a>
    <a onclick="setQ('construction zone cones')">construction zone</a>
  </div>
</header>
<main>
  <div class="meta" id="meta"></div>
  <div class="grid6" id="results"></div>
</main>
<script>
  const $q = document.getElementById('q');
  const $k = document.getElementById('k');
  const $go = document.getElementById('go');
  const $results = document.getElementById('results');
  const $meta = document.getElementById('meta');

  function setQ(s) { $q.value = s; $q.focus(); }

  async function go() {
    const text = $q.value.trim();
    if (!text) return;
    $go.disabled = true;
    $results.innerHTML = '<div class="loading" style="grid-column:1/-1">running across 6 models (this loads SigLIP + Cosmos-Embed1 text encoders, ~15-20s first time)…</div>';
    $meta.textContent = '';
    const t0 = performance.now();
    try {
      const res = await fetch('/compare_search', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ text, k: +$k.value })
      });
      const elapsed = (performance.now() - t0).toFixed(0);
      const data = await res.json();
      if (data.error) { $results.innerHTML = '<div class="err">' + data.error + '</div>'; return; }
      $meta.textContent = `query: "${data.query}" · ${data.models.length} models · ${elapsed}ms total · k=${data.k} per model`;
      $results.innerHTML = data.models.map(m => {
        const body = m.error
          ? `<div class="err">${m.error}</div>`
          : (m.hits.length === 0 ? '<div class="no-cap">no hits</div>'
              : m.hits.map((h,i) => `
                  <div class="hit">
                    <img src="${h.thumbnail_presigned}" loading="lazy">
                    <span class="rank">#${i+1}</span>
                    <span class="score">${h.score.toFixed(3)}</span>
                    <span class="cam">${h.dataset}·${h.camera_name}</span>
                    <div class="fid">${h.frame_id}</div>
                    ${h.caption ? `<div class="caption">"${h.caption.substring(0,260)}${h.caption.length>260?'…':''}"</div>` : (m.label.startsWith('Cosmos-Reason') ? '<div class="no-cap">no caption (not in 500-frame sample)</div>' : '')}
                  </div>`).join(''));
        return `<div class="col"><h3>${m.label}</h3><div class="kind">${m.kind} · table: <code>${m.table}</code></div>${body}</div>`;
      }).join('');
    } catch (e) {
      $results.innerHTML = '<div class="err" style="grid-column:1/-1">' + e + '</div>';
    } finally {
      $go.disabled = false;
    }
  }
  $go.addEventListener('click', go);
  $q.addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
</script>
</body></html>"""


LLM_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>physical-ai-triage · vector vs structured vs LLM-routed</title>
<style>
  :root { --bg:#fafbfc; --panel:#fff; --ink:#14181f; --muted:#6b7280; --line:#e5e7eb; --vec:#3730a3; --sql:#0a7; --llm:#9333ea; --mono: ui-monospace,Menlo,monospace; }
  * { box-sizing:border-box; } body { font:13.5px/1.5 -apple-system,system-ui,sans-serif; color:var(--ink); background:var(--bg); margin:0; }
  header { padding:14px 20px; background:var(--panel); border-bottom:1px solid var(--line); position:sticky; top:0; z-index:10; }
  h1 { margin:0 0 4px; font-size:17px; }
  .sub { color:var(--muted); font-size:12.5px; margin-bottom:10px; max-width:920px; }
  .controls { display:flex; gap:8px; align-items:center; }
  input[type=text] { flex:1; min-width:280px; font:13.5px var(--mono); padding:7px 11px; border:1px solid var(--line); border-radius:6px; }
  input[type=number] { width:54px; font:13.5px var(--mono); padding:7px; border:1px solid var(--line); border-radius:6px; }
  button { padding:7px 16px; font:13.5px var(--mono); background:var(--llm); color:#fff; border:none; border-radius:6px; cursor:pointer; }
  button:hover { background:#7c2bd0; } button:disabled { opacity:0.5; cursor:wait; }
  .examples { font-size:11.5px; color:var(--muted); margin-top:6px; }
  .examples a { color:var(--llm); cursor:pointer; margin-right:12px; text-decoration:none; }
  .examples a:hover { text-decoration:underline; }
  main { padding:16px 20px; }
  .routed-box { background:var(--panel); border:1.5px solid var(--llm); border-radius:8px; padding:12px 16px; margin-bottom:14px; }
  .routed-box h3 { margin:0 0 6px; font-size:13px; color:var(--llm); }
  .routed-box .field { display:grid; grid-template-columns:170px 1fr; gap:8px; padding:3px 0; font:12px var(--mono); }
  .routed-box .field .k { color:var(--muted); }
  .routed-box .field .v { color:#1c3a5e; word-break:break-word; }
  .routed-box .reasoning { background:#faf5ff; border-left:3px solid var(--llm); padding:8px 12px; border-radius:0 4px 4px 0; font-style:italic; color:#581c87; margin-top:8px; font-size:12.5px; }
  .grid3 { display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; }
  @media (max-width:1100px) { .grid3 { grid-template-columns:1fr; } }
  .col { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
  .col h3 { margin:0 0 2px; font-size:13.5px; }
  .col.vec h3 { color:var(--vec); }
  .col.sql h3 { color:var(--sql); }
  .col.llm h3 { color:var(--llm); }
  .col .sub-h { font-size:11.5px; color:var(--muted); margin-bottom:8px; padding-bottom:6px; border-bottom:1px solid var(--line); }
  .col.vec { border-top:3px solid var(--vec); }
  .col.sql { border-top:3px solid var(--sql); }
  .col.llm { border-top:3px solid var(--llm); }
  .hit { padding:7px 0; border-bottom:1px dashed #eee; }
  .hit:last-child { border-bottom:none; }
  .hit img { width:100%; height:auto; border-radius:3px; background:#000; margin-bottom:4px; }
  .hit .meta { display:flex; gap:8px; font:10.5px var(--mono); color:var(--muted); }
  .hit .meta .rank { background:#3730a3; color:#fff; padding:1px 5px; border-radius:3px; font-weight:700; }
  .hit .meta .score { color:var(--sql); }
  .hit .fid { font:9.5px var(--mono); color:var(--muted); margin-top:2px; word-break:break-all; }
  .hit .caption { font-size:11.5px; line-height:1.4; color:#1c3a5e; background:#fcfdff; padding:5px 8px; border-radius:3px; margin-top:4px; border-left:2px solid var(--llm); }
  .err { color:#c40; padding:8px; background:#fff5f0; border-radius:4px; font:11px var(--mono); }
  .loading { color:var(--muted); padding:14px; text-align:center; }
  .empty { color:var(--muted); padding:14px; font-style:italic; font-size:12px; }
  .filter-sql { font:11px var(--mono); background:#fff5e6; color:#92400e; padding:6px 9px; border-radius:4px; margin-bottom:8px; }
</style></head>
<body>
<header>
  <h1>🧪 Pure-vector vs Pure-structured vs LLM-routed hybrid</h1>
  <div class="sub">Same query, three retrieval modes side-by-side. The middle column shows how the LLM (Claude) decomposes your text into <code>vector_query + sql_filter + caption_filter</code>. The columns below show top-K from each strategy. Caveat: many of our metadata columns are NULL in v1 — so "pure structured" is mostly using caption text + dataset/camera filters.</div>
  <div class="controls">
    <input id="q" type="text" placeholder='e.g. "jaywalker at night in SF" or "pedestrian crossing in front camera" or "wet road"' autofocus>
    <input id="k" type="number" min="1" max="20" value="8">
    <button id="go">Compare 3 modes</button>
  </div>
  <div class="examples">
    <span>Try:</span>
    <a onclick="setQ('jaywalker at night in SF')">jaywalker at night in SF</a>
    <a onclick="setQ('pedestrian crossing in front camera')">pedestrian crossing in front camera</a>
    <a onclick="setQ('wet road reflections')">wet road reflections</a>
    <a onclick="setQ('night scene with no pedestrians')">night scene with no pedestrians</a>
    <a onclick="setQ('cyclist near intersection')">cyclist near intersection</a>
  </div>
</header>
<main>
  <div id="routed" style="display:none"></div>
  <div class="grid3" id="results"></div>
</main>
<script>
  const $q = document.getElementById('q');
  const $k = document.getElementById('k');
  const $go = document.getElementById('go');
  const $results = document.getElementById('results');
  const $routed = document.getElementById('routed');
  function setQ(s) { $q.value = s; $q.focus(); }
  async function go() {
    const text = $q.value.trim();
    if (!text) return;
    $go.disabled = true;
    $results.innerHTML = '<div class="loading" style="grid-column:1/-1">routing via Claude + running 3 retrieval modes…</div>';
    $routed.style.display = 'none';
    try {
      const res = await fetch('/llm_search', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ text, k: +$k.value }) });
      const data = await res.json();
      if (data.error) { $results.innerHTML = '<div class="err" style="grid-column:1/-1">' + data.error + '</div>'; return; }
      const r = data.routed;
      $routed.style.display = 'block';
      $routed.className = 'routed-box';
      $routed.innerHTML = `<h3>🧠 LLM decomposition for "${data.query}"</h3>
        <div class="field"><span class="k">vector_query</span><span class="v">${r.vector_query || '(none)'}</span></div>
        <div class="field"><span class="k">sql_filter</span><span class="v">${r.sql_filter || '(none)'}</span></div>
        <div class="field"><span class="k">caption_filter</span><span class="v">${r.caption_filter || '(none)'}</span></div>
        <div class="reasoning">${r.reasoning || ''}</div>`;
      const klasses = ['vec', 'sql', 'llm'];
      $results.innerHTML = data.modes.map((m, i) => {
        const body = m.error ? `<div class="err">${m.error}</div>`
          : (m.hits.length === 0 ? `<div class="empty">${m.notes || 'no hits'}</div>`
              : m.hits.map((h, j) => `<div class="hit">
                  <img src="${h.thumbnail_presigned}" loading="lazy">
                  <div class="meta"><span class="rank">#${j+1}</span><span class="score">${h.score.toFixed(3)}</span><span>${h.dataset}·${h.camera_name}</span></div>
                  <div class="fid">${h.frame_id}</div>
                  ${h.caption ? `<div class="caption">"${h.caption.substring(0,220)}${h.caption.length>220?'…':''}"</div>` : ''}
                </div>`).join(''));
        const fsq = m.filter_sql ? `<div class="filter-sql">filter: ${m.filter_sql}</div>` : '';
        return `<div class="col ${klasses[i]}"><h3>${m.name}</h3><div class="sub-h">${m.subtitle || ''}</div>${fsq}${body}</div>`;
      }).join('');
    } catch (e) {
      $results.innerHTML = '<div class="err" style="grid-column:1/-1">' + e + '</div>';
    } finally {
      $go.disabled = false;
    }
  }
  $go.addEventListener('click', go);
  $q.addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
</script>
</body></html>"""


@cli.command()
def main(
    port: int = typer.Option(5050),
    host: str = typer.Option("127.0.0.1"),
    lance_dir: str = typer.Option("data/lance"),
):
    """Launch the local search UI."""
    load_dotenv()
    os.environ["LANCE_DIR"] = lance_dir
    print(f"\n  physical-ai-triage search UI: http://{host}:{port}\n")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    cli()
