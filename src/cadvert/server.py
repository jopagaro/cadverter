"""CADVERT web server.

Endpoints:
  GET  /                    → UI (index.html)
  POST /convert             → upload CAD file, run full pipeline, return HSD + images
  POST /chat/{session_id}   → stream GPT-4o response with HSD as system context
  DELETE /session/{session_id} → clean up temp files
"""

from __future__ import annotations
import asyncio
import json
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Paths ─────────────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "cadvert_sessions"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="CADVERT")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve rendered images directly as static files
app.mount("/sessions", StaticFiles(directory=str(UPLOAD_DIR)), name="sessions")

_executor = ThreadPoolExecutor(max_workers=2)

# In-memory session store: session_id → result dict
# For production, replace with Redis or a database.
_sessions: dict[str, dict] = {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="UI not found — ensure static/index.html exists")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    """Upload a CAD file and run the full CADVERT pipeline.

    Returns JSON:
      session_id, hsd (text), images (list of {name, data: base64 PNG}),
      format, is_mesh, filename, units, summary
    """
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True)

    suffix = Path(file.filename or "upload.step").suffix.lower() or ".step"
    input_path = session_dir / f"input{suffix}"
    input_path.write_bytes(await file.read())

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, _run_pipeline, input_path, session_dir),
            timeout=600,
        )
    except asyncio.TimeoutError:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail="Processing timed out after 10 minutes.")
    except Exception as exc:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(exc))

    _sessions[session_id] = result

    # Return image URLs (served as static files) instead of base64 blobs
    images = []
    for img_path_str in result.get("image_paths", []):
        p = Path(img_path_str)
        if p.exists():
            # URL: /sessions/<session_id>/views/<filename>
            rel = p.relative_to(UPLOAD_DIR)
            images.append({"name": p.stem, "url": f"/sessions/{rel.as_posix()}"})

    return JSONResponse({
        "session_id": session_id,
        "hsd": result["hsd"],
        "images": images,
        "format": result["format"],
        "is_mesh": result["is_mesh"],
        "filename": file.filename,
        "units": result["units"],
        "summary": result["summary"],
    })


@app.post("/chat/{session_id}")
async def chat(
    session_id: str,
    request: Request,
    x_openai_key: str = Header(..., alias="X-OpenAI-Key"),
    x_model: str = Header(default="gpt-4o", alias="X-Model"),
):
    """Stream a GPT chat response with the HSD injected as system context.

    Request body JSON: { "messages": [{role, content}, ...] }
    The API key is used for this request only and never stored.
    """
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    body = await request.json()
    messages: list[dict] = body.get("messages", [])

    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="openai package not installed — run: pip install openai",
        )

    client = AsyncOpenAI(api_key=x_openai_key)
    hsd_text = session["hsd"]

    # GPT-4o has a 128k token limit (~400k chars with overhead).
    # Truncate large HSD documents to keep the most useful sections
    # (header, global props, features, spatial) which appear first.
    MAX_HSD_CHARS = 280_000
    if len(hsd_text) > MAX_HSD_CHARS:
        hsd_text = (
            hsd_text[:MAX_HSD_CHARS]
            + f"\n\n[HSD TRUNCATED — document too large for context window. "
            f"{len(session['hsd']) - MAX_HSD_CHARS:,} characters omitted. "
            f"Global properties, features, and spatial relationships above are complete.]"
        )

    system_msg = (
        "You are an expert mechanical engineer and manufacturing consultant. "
        "You have been given a Hierarchical Spatial Document (HSD) — a machine-readable "
        "description of a CAD part with exact geometry extracted from the B-REP model.\n\n"
        "Rules:\n"
        "- Answer questions using the exact numerical values in the HSD.\n"
        "- Reference faces as F1, F3, etc. and edges as E5, E12, etc.\n"
        "- If a value is not in the HSD, say so — do not guess.\n"
        "- For mesh-format files (STL/OBJ), note that exact geometry is unavailable.\n"
        "- Be concise but precise. Units are specified in the document header.\n\n"
        f"<HSD>\n{hsd_text}\n</HSD>"
    )

    openai_messages = [{"role": "system", "content": system_msg}] + messages

    async def stream_response() -> None:
        try:
            response = await client.chat.completions.create(
                model=x_model,
                messages=openai_messages,
                stream=True,
                max_tokens=2048,
            )
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield f"data: {json.dumps({'content': delta.content})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    _sessions.pop(session_id, None)
    session_dir = UPLOAD_DIR / session_id
    shutil.rmtree(session_dir, ignore_errors=True)
    return {"ok": True}


# ── Pipeline runner (synchronous, runs in thread pool) ────────────────────────

def _run_pipeline(input_path: Path, session_dir: Path) -> dict:
    from .ingest import load_step, IngestError

    try:
        shape, body_count, metadata = load_step(input_path)
    except IngestError as exc:
        raise ValueError(str(exc))

    graph = features = spatial = validation_text = None

    if not metadata.is_mesh:
        from .topology import build_topology
        graph = build_topology(shape, body_count)

        # Skip validation for large assemblies — O(n) per face is too slow above ~200 faces
        if len(graph.faces) <= 200:
            try:
                from .validate import validate_extraction, format_validation_report
                val = validate_extraction(shape, graph)
                validation_text = format_validation_report(val)
            except Exception:
                pass

        try:
            from .features import detect_features
            features = detect_features(graph)
        except Exception:
            pass

        if features:
            try:
                from .spatial import compute_spatial_relationships
                spatial = compute_spatial_relationships(graph, features, shape=shape)
            except Exception:
                pass

    # Render views
    image_paths: list[str] = []
    try:
        render_dir = session_dir / "views"
        image_paths = _render_subprocess(shape, render_dir, (900, 675), input_path.stem)
    except Exception:
        pass

    # Mesh global properties
    mesh_info = None
    if metadata.is_mesh:
        mesh_info = _compute_mesh_info(metadata, shape)

    # Assemble HSD document
    from .document import render_document
    hsd = render_document(
        graph, input_path,
        features=features,
        spatial=spatial,
        rendered_views=[Path(p) for p in image_paths] if image_paths else None,
        validation_report=validation_text,
        units=metadata.units,
        gdt_annotations=metadata.gdt_annotations if metadata.gdt_annotations else None,
        mesh_info=mesh_info,
    )

    # Build a short summary for the UI info panel
    summary = _build_summary(graph, features, metadata, body_count)

    return {
        "hsd": hsd,
        "image_paths": image_paths,
        "format": metadata.source_format,
        "is_mesh": metadata.is_mesh,
        "units": metadata.units,
        "summary": summary,
    }


def _render_subprocess(shape, output_dir: Path, image_size: tuple, stem: str) -> list[str]:
    """Render VTK views in a subprocess to avoid macOS NSWindow threading crash.

    VTK's Cocoa backend requires NSWindow creation on the main thread.
    Running in a subprocess gives us a fresh main thread.
    """
    import subprocess
    import sys
    import tempfile
    import os

    # Serialize shape to a temp BREP file so the subprocess can load it
    from OCP.BRepTools import BRepTools

    with tempfile.NamedTemporaryFile(suffix=".brep", delete=False) as f:
        brep_path = f.name
    try:
        BRepTools.Write_s(shape, brep_path)

        # Find the src/ directory so the subprocess can import cadvert
        src_dir = str(Path(__file__).parent.parent)

        script = f"""
import os, sys
os.environ["VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN"] = "1"
sys.path.insert(0, {repr(src_dir)})
from OCP.BRep import BRep_Builder
from OCP.BRepTools import BRepTools
from OCP.TopoDS import TopoDS_Shape
from cadvert.renderer import render_shape

builder = BRep_Builder()
shape = TopoDS_Shape()
BRepTools.Read_s(shape, {repr(brep_path)}, builder)
output_dir = {repr(str(output_dir))}
paths = render_shape(shape, output_dir, image_size={image_size!r}, stem={repr(stem)})
for p in paths:
    print(p)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return [p for p in result.stdout.strip().splitlines() if p]
        # Log stderr for debugging but don't crash
        return []
    finally:
        try:
            os.unlink(brep_path)
        except OSError:
            pass


def _compute_mesh_info(metadata, shape) -> dict:
    info = {
        "format": metadata.source_format,
        "triangle_count": metadata.triangle_count,
        "units": metadata.units,
    }
    try:
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib

        vol_props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, vol_props)
        info["volume"] = vol_props.Mass()

        surf_props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(shape, surf_props)
        info["surface_area"] = surf_props.Mass()

        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        if not box.IsVoid():
            xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
            info["bbox"] = {"X": (xmin, xmax), "Y": (ymin, ymax), "Z": (zmin, zmax)}
    except Exception:
        pass
    return info


def _build_summary(graph, features, metadata, body_count: int) -> dict:
    s: dict = {
        "format": metadata.source_format,
        "schema": metadata.schema,
        "units": metadata.units,
        "is_mesh": metadata.is_mesh,
    }
    if metadata.is_mesh:
        s["triangles"] = metadata.triangle_count
    else:
        if graph:
            s["faces"] = len(graph.faces)
            s["edges"] = len(graph.edges)
            s["bodies"] = body_count
        if features:
            from collections import Counter
            counts = Counter(f.feature_type for f in features)
            s["features"] = dict(counts)
        if metadata.gdt_annotations:
            s["gdt_count"] = len(metadata.gdt_annotations)
    return s


# ── Dev server entry point ────────────────────────────────────────────────────

def run():
    import uvicorn
    uvicorn.run("cadvert.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
