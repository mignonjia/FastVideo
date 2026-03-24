from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


WM_LAB_ROOT = Path("/mnt/weka/home/hao.zhang/alex/wm-lab")
MANIFEST_ROOT = WM_LAB_ROOT / "datas/datasets/zeldam2-v2/action_latent"
IDX_FILE = WM_LAB_ROOT / "datas/spec/zeldam2/inpaint.txt"

CLIP_CACHE: dict[int, list[dict[str, Any]]] = {}
ACTION_CACHE: dict[str, dict[str, Any] | None] = {}
VIDEO_CACHE: dict[str, dict[str, Any]] = {}


def _parse_idx_file(idx_file: Path) -> list[int]:
    values: list[int] = []
    for raw_line in idx_file.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.split(","):
            token = token.strip()
            if token:
                values.append(int(token))
    return values


def _available_idxs() -> list[int]:
    return _parse_idx_file(IDX_FILE)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return WM_LAB_ROOT / path


def _load_idx_clips(idx: int) -> list[dict[str, Any]]:
    idx = int(idx)
    cached = CLIP_CACHE.get(idx)
    if cached is not None:
        return cached

    by_chunk: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(MANIFEST_ROOT.rglob("manifest.rank*.jsonl")):
        with manifest_path.open() as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if int(record["idx"]) != idx:
                    continue
                chunk = str(record["chunk"])
                if chunk in by_chunk:
                    continue
                by_chunk[chunk] = {
                    "sample_id": chunk,
                    "idx": idx,
                    "chunk": chunk,
                    "fps": float(record.get("train_fps") or 30.0),
                    "video_source_path": record["video_source_path"],
                }

    clips = sorted(by_chunk.values(), key=lambda item: item["chunk"])
    CLIP_CACHE[idx] = clips
    return clips


def _find_clip(idx: int, sample_id: str) -> dict[str, Any]:
    for clip in _load_idx_clips(idx):
        if clip["sample_id"] == sample_id:
            return clip
    raise HTTPException(
        status_code=404,
        detail=f"clip not found for idx={idx}: {sample_id}",
    )


def _chunk_dir_for_clip(clip: dict[str, Any]) -> Path:
    return _resolve_path(clip["video_source_path"]).parent


def _original_video_path_for_clip(clip: dict[str, Any]) -> Path:
    chunk_dir = _chunk_dir_for_clip(clip)
    for name in ("clip.mp4", "clip_fps30_s960.mp4"):
        path = chunk_dir / name
        if path.is_file():
            return path
    raise HTTPException(status_code=404, detail=f"original clip not found under {chunk_dir}")


def _load_video_meta(video_path: Path) -> dict[str, Any]:
    key = str(video_path)
    cached = VIDEO_CACHE.get(key)
    if cached is not None:
        return cached

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=404, detail=f"failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    cached = {
        "fps": float(fps if fps > 0.0 else 30.0),
        "num_frames": num_frames,
    }
    VIDEO_CACHE[key] = cached
    return cached


def _load_chunk_action_arrays(
    chunk_dir: Path,
    parquet_name: str,
) -> dict[str, Any] | None:
    parquet_path = chunk_dir / parquet_name
    key = str(parquet_path)
    cached = ACTION_CACHE.get(key)
    if key in ACTION_CACHE:
        return cached

    import numpy as np
    import pandas as pd

    if not parquet_path.is_file():
        ACTION_CACHE[key] = None
        return None

    df = pd.read_parquet(parquet_path, columns=["j_left", "j_right"])
    left = np.asarray(df["j_left"].tolist(), dtype=np.float32)
    right = np.asarray(df["j_right"].tolist(), dtype=np.float32)
    if left.ndim != 2 or left.shape[1] != 2:
        raise HTTPException(status_code=400, detail=f"invalid j_left shape in {parquet_path}")
    if right.ndim != 2 or right.shape[1] != 2:
        raise HTTPException(status_code=400, detail=f"invalid j_right shape in {parquet_path}")

    cached = {
        "parquet_path": str(parquet_path),
        "left": left,
        "right": right,
        "num_frames": int(len(df)),
    }
    ACTION_CACHE[key] = cached
    return cached


def _load_action_arrays(
    clip: dict[str, Any],
) -> dict[str, Any]:
    import math

    chunk_dir = _chunk_dir_for_clip(clip)

    def _build_variant(parquet_name: str, label: str) -> dict[str, Any]:
        payload = _load_chunk_action_arrays(chunk_dir, parquet_name)
        if payload is None:
            return {
                "available": False,
                "label": label,
                "action_source": str(chunk_dir / parquet_name),
                "left_stick": [],
                "right_stick": [],
                "left_scale": 1.0,
                "right_scale": 1.0,
            }

        end = int(payload["num_frames"]) - 1
        if end < 0:
            raise HTTPException(
                status_code=400,
                detail=f"empty action parquet for {clip['sample_id']}: {payload['parquet_path']}",
            )

        left_slice = payload["left"][: end + 1]
        right_slice = payload["right"][: end + 1]
        target_extent = 0.95
        left_extent = max(
            math.hypot(float(row[0]), float(row[1])) for row in left_slice
        )
        right_extent = max(
            math.hypot(float(row[0]), float(row[1])) for row in right_slice
        )
        left_scale = (
            float(left_extent) / target_extent if float(left_extent) > 1e-6 else 1.0
        )
        right_scale = (
            float(right_extent) / target_extent if float(right_extent) > 1e-6 else 1.0
        )
        return {
            "available": True,
            "label": label,
            "action_source": str(payload["parquet_path"]),
            "left_stick": left_slice.astype(float).tolist(),
            "right_stick": right_slice.astype(float).tolist(),
            "left_scale": float(left_scale),
            "right_scale": float(right_scale),
        }

    return {
        "processed": _build_variant("actions_processed.parquet", "processed"),
        "raw": _build_variant("actions_raw.parquet", "raw"),
    }


def _spawn_ffmpeg_stream(
    *,
    src: Path,
    start_sec: float,
    duration_sec: float,
    width: int,
    view: str,
) -> subprocess.Popen[bytes]:
    if str(view) == "controller":
        vf = (
            "crop=iw/6:ih/6:0:ih*5/6,"
            f"scale=w={int(width)}:h=-2:flags=bicubic"
        )
    else:
        vf = f"scale=w={int(width)}:h=-2:flags=bicubic"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if start_sec > 0.0:
        cmd.extend(["-ss", f"{start_sec:.6f}"])
    cmd.extend(["-i", str(src)])
    if duration_sec > 0.0:
        cmd.extend(["-t", f"{duration_sec:.6f}"])
    cmd.extend(
        [
            "-an",
            "-vf",
            vf,
            "-c:v",
            "libopenh264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "frag_keyframe+empty_moov+default_base_moof",
            "-f",
            "mp4",
            "pipe:1",
        ]
    )
    return subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=1024 * 1024,
    )


def _stream_from_process(proc: subprocess.Popen[bytes]):
    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(1024 * 512)
            if not chunk:
                break
            yield chunk
    finally:
        try:
            proc.kill()
        except Exception:
            pass


def _render_index() -> str:
    idx_buttons = "".join(
        f"<button class='idx-btn' onclick='loadIdx({idx})'>idx{idx}</button>"
        for idx in _available_idxs()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Zeldam2-v2 Clip Inspector</title>
  <style>
    :root {{
      --bg: #0f1418;
      --panel: #182028;
      --panel-2: #22303a;
      --text: #edf3f7;
      --muted: #9fb2c1;
      --accent: #65d7ff;
      --border: #31414d;
      --chip: #24323c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #0b1116 0%, var(--bg) 100%);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1700px;
      margin: 0 auto;
      padding: 24px 20px 40px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    .sub {{
      color: var(--muted);
      margin-bottom: 16px;
    }}
    .card {{
      background: rgba(24, 32, 40, 0.92);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      margin-bottom: 18px;
      box-shadow: 0 10px 26px rgba(0, 0, 0, 0.22);
    }}
    .idx-picker {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .idx-btn {{
      padding: 7px 11px;
      border-radius: 999px;
      background: var(--chip);
      border: 1px solid var(--border);
      color: var(--text);
      cursor: pointer;
    }}
    .idx-btn.active {{
      background: #1f4f63;
      border-color: #2b6e8a;
    }}
    .row {{
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 18px;
      align-items: start;
    }}
    .controls {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .field {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .field input, .field select {{
      width: 100%;
      padding: 10px 12px;
      background: var(--panel);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
    }}
    .actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .btn {{
      padding: 10px 14px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      cursor: pointer;
    }}
    .preview {{
      display: grid;
      grid-template-columns: minmax(500px, 1fr) 420px;
      gap: 18px;
      align-items: start;
    }}
    .side-stack {{
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}
    video {{
      width: 100%;
      background: #000;
      border-radius: 12px;
      border: 1px solid var(--border);
    }}
    .mini-video {{
      width: 100%;
      max-width: 396px;
      display: block;
      margin: 0 auto;
    }}
    .pad-box {{
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
    }}
    canvas {{
      width: 100%;
      max-width: 396px;
      height: auto;
      display: block;
      background: #11171d;
      border-radius: 10px;
      margin: 0 auto;
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    .muted {{
      color: var(--muted);
    }}
    @media (max-width: 1200px) {{
      .row {{
        grid-template-columns: 1fr;
      }}
      .preview {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Zeldam2-v2 Clip Inspector</h1>
    <div class="sub">One idx at a time. Full 20-second chunk clips on the left, full shard <span class="mono">actions_processed.parquet</span> sticks on the right.</div>

    <section class="card">
      <div class="field">
        <label for="idx_input">idx</label>
        <input id="idx_input" placeholder="108">
      </div>
      <div class="actions" style="margin-top:12px;">
        <button class="btn" onclick="loadIdxFromInput()">Load idx</button>
      </div>
      <div class="idx-picker" id="idx_picker">{idx_buttons}</div>
    </section>

    <div class="row">
      <section class="card controls">
        <div class="field">
          <label for="clip_filter">filter clips</label>
          <input id="clip_filter" placeholder="chunk_0842" oninput="filterClips()">
        </div>
        <div class="field">
          <label for="clip_select">clips</label>
          <select id="clip_select" size="24" onchange="loadSelectedClip()"></select>
        </div>
        <div class="muted mono" id="meta">idx: - | clips: -</div>
      </section>

      <section class="card preview">
        <div>
          <div class="muted" style="margin-bottom:8px;">video</div>
          <video id="vid" controls playsinline></video>
          <div class="muted mono" id="video_meta" style="margin-top:8px;">clip: -</div>
        </div>
        <div class="side-stack">
          <div class="pad-box">
            <div class="muted" style="margin-bottom:8px;">controller crop</div>
            <video id="controller_vid" class="mini-video" playsinline muted></video>
          </div>
          <div class="pad-box">
            <div class="muted" style="margin-bottom:8px;">processed action (left Y flipped in circle; right not flipped)</div>
            <canvas id="processed_canvas" width="396" height="220"></canvas>
            <div class="mono" id="processed_meta" style="margin-top:10px;">frame: -</div>
          </div>
          <div class="pad-box">
            <div class="muted" style="margin-bottom:8px;">raw action (left Y flipped in circle; right not flipped)</div>
            <canvas id="raw_canvas" width="396" height="220"></canvas>
            <div class="mono" id="raw_meta" style="margin-top:10px;">frame: -</div>
          </div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const state = {{
      idx: null,
      clips: [],
      filtered: [],
      clip: null,
      fps: 30.0,
      processed: null,
      raw: null,
      raf: null,
    }};

    const idxInput = document.getElementById('idx_input');
    const clipFilter = document.getElementById('clip_filter');
    const clipSelect = document.getElementById('clip_select');
    const meta = document.getElementById('meta');
    const vid = document.getElementById('vid');
    const controllerVid = document.getElementById('controller_vid');
    const videoMeta = document.getElementById('video_meta');
    const processedMeta = document.getElementById('processed_meta');
    const rawMeta = document.getElementById('raw_meta');
    const processedCanvas = document.getElementById('processed_canvas');
    const rawCanvas = document.getElementById('raw_canvas');
    const processedCtx = processedCanvas.getContext('2d');
    const rawCtx = rawCanvas.getContext('2d');

    function setActiveIdxButton(idx) {{
      document.querySelectorAll('.idx-btn').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.idx-btn').forEach(btn => {{
        if (btn.textContent === `idx${{idx}}`) btn.classList.add('active');
      }});
    }}

    async function loadIdx(idx) {{
      idxInput.value = String(idx);
      setActiveIdxButton(idx);
      const res = await fetch(`/api/load?idx=${{idx}}`);
      const data = await res.json();
      state.idx = idx;
      state.clips = data.clips || [];
      meta.textContent = `idx: ${{idx}} | clips: ${{state.clips.length}}`;
      clipFilter.value = '';
      filterClips();
      if (state.filtered.length > 0) {{
        clipSelect.selectedIndex = 0;
        await loadSelectedClip();
      }}
    }}

    async function loadIdxFromInput() {{
      const value = Number(idxInput.value.trim());
      if (!Number.isFinite(value)) return;
      await loadIdx(value);
    }}

    function clipLabel(item) {{
      return `${{item.chunk}} | 20s clip`;
    }}

    function filterClips() {{
      const q = clipFilter.value.trim().toLowerCase();
      state.filtered = state.clips.filter(item => {{
        if (!q) return true;
        return clipLabel(item).toLowerCase().includes(q) || item.sample_id.toLowerCase().includes(q);
      }});
      clipSelect.innerHTML = '';
      state.filtered.forEach((item, i) => {{
        const opt = document.createElement('option');
        opt.value = item.sample_id;
        opt.textContent = clipLabel(item);
        clipSelect.appendChild(opt);
      }});
    }}

    async function loadSelectedClip() {{
      const sampleId = clipSelect.value;
      if (!sampleId || !state.idx) return;
      const res = await fetch(`/api/clip?idx=${{state.idx}}&sample_id=${{encodeURIComponent(sampleId)}}`);
      const data = await res.json();
      state.clip = data.clip;
      state.processed = data.processed;
      state.raw = data.raw;
      state.fps = data.clip.fps || 30.0;
      const processedLabel = data.processed?.available ? data.processed.action_source.split('/').slice(-2).join('/') : 'missing';
      const rawLabel = data.raw?.available ? data.raw.action_source.split('/').slice(-2).join('/') : 'missing';
      videoMeta.textContent = `clip: ${{data.clip.sample_id}} | fps: ${{state.fps}} | frames: ${{data.clip.num_frames}} | processed: ${{processedLabel}} | raw: ${{rawLabel}}`;
      vid.src = `/video?idx=${{state.idx}}&sample_id=${{encodeURIComponent(sampleId)}}&width=960`;
      controllerVid.src = `/video?idx=${{state.idx}}&sample_id=${{encodeURIComponent(sampleId)}}&width=396&view=controller`;
      vid.load();
      controllerVid.load();
      drawCurrentAction();
    }}

    function clamp(v, lo, hi) {{
      return Math.max(lo, Math.min(hi, v));
    }}

    function currentFrameIndex() {{
      if (!state.clip) return 0;
      const idx = Math.round((vid.currentTime || 0) * state.fps);
      return clamp(idx, 0, (state.clip.num_frames || 1) - 1);
    }}

    function drawCirclePad(ctx, cx, cy, radius, x, y, valueText) {{
      ctx.strokeStyle = '#7f8c96';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();

      ctx.strokeStyle = '#47545f';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx - radius, cy);
      ctx.lineTo(cx + radius, cy);
      ctx.moveTo(cx, cy - radius);
      ctx.lineTo(cx, cy + radius);
      ctx.stroke();

      const dotX = cx + x * radius * 0.7;
      const dotY = cy - y * radius * 0.7;
      ctx.fillStyle = '#65d7ff';
      ctx.beginPath();
      ctx.arc(dotX, dotY, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = '#9fb2c1';
      ctx.font = '13px monospace';
      ctx.fillText(valueText, cx - radius, cy + radius + 24);
    }}

    function drawActionPanel(ctx, canvas, metaEl, actionState) {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#11171d';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      if (!actionState?.available) {{
        ctx.fillStyle = '#9fb2c1';
        ctx.font = '16px sans-serif';
        ctx.fillText('missing', 20, 30);
        metaEl.textContent = 'frame: -';
        return;
      }}

      const frameIdx = currentFrameIndex();
      const jl = actionState.left_stick[frameIdx] || [0, 0];
      const jr = actionState.right_stick[frameIdx] || [0, 0];

      const leftX = clamp(Number(jl[0] || 0) / actionState.left_scale, -1, 1);
      const leftY = clamp(-Number(jl[1] || 0) / actionState.left_scale, -1, 1);
      const rightX = clamp(Number(jr[0] || 0) / actionState.right_scale, -1, 1);
      const rightY = clamp(Number(jr[1] || 0) / actionState.right_scale, -1, 1);

      drawCirclePad(
        ctx,
        105,
        92,
        58,
        leftX,
        leftY,
        `x=${{Number(jl[0]||0).toFixed(3)}} y=${{Number(jl[1]||0).toFixed(3)}}`
      );
      drawCirclePad(
        ctx,
        291,
        92,
        58,
        rightX,
        rightY,
        `x=${{Number(jr[0]||0).toFixed(3)}} y=${{Number(jr[1]||0).toFixed(3)}}`
      );

      metaEl.textContent = `frame: ${{frameIdx}} / ${{actionState.left_stick.length - 1}}`;
    }}

    function drawCurrentAction() {{
      drawActionPanel(processedCtx, processedCanvas, processedMeta, state.processed);
      drawActionPanel(rawCtx, rawCanvas, rawMeta, state.raw);
    }}

    function tick() {{
      drawCurrentAction();
      state.raf = requestAnimationFrame(tick);
    }}

    function syncControllerTime(force = false) {{
      if (!controllerVid.src) return;
      const delta = Math.abs((controllerVid.currentTime || 0) - (vid.currentTime || 0));
      if (force || delta > 0.05) {{
        try {{
          controllerVid.currentTime = vid.currentTime || 0;
        }} catch (err) {{}}
      }}
    }}

    vid.addEventListener('play', () => {{
      controllerVid.playbackRate = vid.playbackRate || 1.0;
      syncControllerTime(true);
      controllerVid.play().catch(() => null);
    }});
    vid.addEventListener('pause', () => controllerVid.pause());
    vid.addEventListener('seeking', () => syncControllerTime(true));
    vid.addEventListener('seeked', () => syncControllerTime(true));
    vid.addEventListener('timeupdate', () => syncControllerTime(false));
    vid.addEventListener('ratechange', () => {{
      controllerVid.playbackRate = vid.playbackRate || 1.0;
    }});
    controllerVid.addEventListener('loadedmetadata', () => {{
      controllerVid.playbackRate = vid.playbackRate || 1.0;
      syncControllerTime(true);
    }});

    state.raf = requestAnimationFrame(tick);

    const params = new URLSearchParams(window.location.search);
    const initialIdx = Number(params.get('idx') || '');
    if (Number.isFinite(initialIdx)) {{
      loadIdx(initialIdx);
    }}
  </script>
</body>
</html>"""


app = FastAPI(title="Zeldam2-v2 Clip Inspector")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _render_index()


@app.get("/api/idxs")
def api_idxs() -> JSONResponse:
    return JSONResponse({"idxs": _available_idxs()})


@app.get("/api/load")
def api_load(idx: int = Query(...)) -> JSONResponse:
    clips = _load_idx_clips(int(idx))
    return JSONResponse(
        {
            "idx": int(idx),
            "count": len(clips),
            "clips": clips,
        }
    )


@app.get("/api/clip")
def api_clip(
    idx: int = Query(...),
    sample_id: str = Query(...),
) -> JSONResponse:
    clip = _find_clip(int(idx), sample_id)
    video_path = _original_video_path_for_clip(clip)
    video_meta = _load_video_meta(video_path)
    actions = _load_action_arrays(clip)
    clip_payload = dict(clip)
    clip_payload["fps"] = float(video_meta["fps"])
    clip_payload["num_frames"] = int(video_meta["num_frames"])
    return JSONResponse(
        {
            "clip": clip_payload,
            "processed": actions["processed"],
            "raw": actions["raw"],
        }
    )


@app.get("/video")
def video(
    idx: int = Query(...),
    sample_id: str = Query(...),
    width: int = Query(default=960, ge=128, le=1920),
    view: str = Query(default="full"),
) -> StreamingResponse:
    clip = _find_clip(int(idx), sample_id)
    src = _original_video_path_for_clip(clip)
    if not src.is_file():
        raise HTTPException(status_code=404, detail=f"video not found: {src}")
    if view not in {"full", "controller"}:
        raise HTTPException(status_code=400, detail="view must be full or controller")
    proc = _spawn_ffmpeg_stream(
        src=src,
        start_sec=0.0,
        duration_sec=0.0,
        width=int(width),
        view=view,
    )
    return StreamingResponse(
        _stream_from_process(proc),
        media_type="video/mp4",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8021)
    args = parser.parse_args()

    try:
        import uvicorn
    except Exception as exc:
        raise SystemExit(f"uvicorn import failed: {exc}")

    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
