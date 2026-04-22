from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


APP_DIR = Path(__file__).resolve().parent


def _list_mp4s() -> List[Path]:
    return sorted([p for p in APP_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"], key=lambda p: p.name)


def _group_videos(files: List[Path]) -> List[Dict[str, str]]:
    groups: Dict[str, Dict[str, str]] = {}
    for p in files:
        name = p.name
        kind = "other"
        base = p.stem
        if name.endswith("_clip_only.mp4"):
            base = name[: -len("_clip_only.mp4")]
            kind = "clip_only"
        elif name.endswith("_processedcontrols_overlay_compact.mp4"):
            base = name[: -len("_processedcontrols_overlay_compact.mp4")]
            kind = "processed_overlay"
        item = groups.setdefault(base, {"name": base})
        item[kind] = name
    return [groups[k] for k in sorted(groups.keys())]


def _video_html(src_name: str, title: str) -> str:
    esc_title = html.escape(title)
    esc_src = html.escape(src_name)
    return (
        f"<div class='video-block'>"
        f"<div class='video-title'>{esc_title}</div>"
        f"<video controls preload='metadata' playsinline>"
        f"<source src='/files/{esc_src}' type='video/mp4'>"
        f"</video>"
        f"<div class='file-line'><a href='/files/{esc_src}' target='_blank'>{esc_src}</a></div>"
        f"</div>"
    )


def _render_index() -> str:
    groups = _group_videos(_list_mp4s())
    cards: List[str] = []
    for item in groups:
        name = html.escape(item["name"])
        videos: List[str] = []
        if "clip_only" in item:
            videos.append(_video_html(item["clip_only"], "Clip Only"))
        if "processed_overlay" in item:
            videos.append(_video_html(item["processed_overlay"], "Processed Overlay"))
        if "other" in item:
            videos.append(_video_html(item["other"], "Other"))
        cards.append(
            "<section class='card'>"
            f"<h2>{name}</h2>"
            "<div class='video-grid'>"
            + "".join(videos)
            + "</div></section>"
        )

    count_files = len(_list_mp4s())
    count_groups = len(groups)
    body = "".join(cards) if cards else "<p>No .mp4 files found in this folder.</p>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Zelda Video Viewer</title>
  <style>
    :root {{
      --bg: #0f1418;
      --panel: #182028;
      --panel-2: #22303a;
      --text: #edf3f7;
      --muted: #9fb2c1;
      --accent: #65d7ff;
      --border: #31414d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #0b1116 0%, var(--bg) 100%);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    .sub {{
      color: var(--muted);
      margin-bottom: 22px;
    }}
    .card {{
      background: rgba(24, 32, 40, 0.92);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      margin-bottom: 18px;
      box-shadow: 0 10px 26px rgba(0, 0, 0, 0.22);
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 18px;
      color: var(--accent);
      word-break: break-word;
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 16px;
    }}
    .video-block {{
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
    }}
    .video-title {{
      margin-bottom: 10px;
      font-size: 14px;
      color: var(--muted);
      font-weight: 600;
      letter-spacing: 0.02em;
    }}
    video {{
      display: block;
      width: 100%;
      max-height: 70vh;
      background: #000;
      border-radius: 10px;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .file-line {{
      margin-top: 8px;
      font-size: 12px;
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Zelda Video Viewer</h1>
    <div class="sub">{count_groups} groups, {count_files} videos from {html.escape(str(APP_DIR))}</div>
    {body}
  </div>
</body>
</html>"""


app = FastAPI(title="Zelda Video Viewer")
app.mount("/files", StaticFiles(directory=str(APP_DIR)), name="files")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _render_index()


@app.get("/api/videos")
def api_videos():
    files = _list_mp4s()
    return JSONResponse(
        {
            "root": str(APP_DIR),
            "count": len(files),
            "groups": _group_videos(files),
        }
    )


@app.get("/healthz")
def healthz():
    return {"ok": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8018)
    args = ap.parse_args()

    try:
        import uvicorn
    except Exception as e:
        raise SystemExit(f"uvicorn import failed: {e}")

    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
