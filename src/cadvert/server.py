"""CADVERT web server.

Endpoints:
  GET  /                        → UI (index.html)
  POST /convert                 → upload CAD file, run full pipeline, return Tier 0 + images
  POST /chat/{session_id}       → stream an LLM response (OpenAI or Claude) with Tier 0 context + tool calling
  POST /tool/{session_id}       → run one geometry tool directly (for clients running their own LLM)
  GET  /tools                   → tool definitions
  DELETE /session/{session_id}  → clean up temp files
"""

from __future__ import annotations
import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid

# Load .env file from project root if it exists (stdlib only, no dotenv needed)
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Paths ─────────────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"
# Where uploads and rendered views live. Everything here is regenerable from the user's
# own CAD file, so it belongs in a cache: the desktop app points CADVERT_DATA_DIR at its
# sandbox Caches directory, and the CLI falls back to the system temp folder.
UPLOAD_DIR = Path(os.environ.get("CADVERT_DATA_DIR") or (Path(tempfile.gettempdir()) / "cadvert_sessions"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Limits ────────────────────────────────────────────────────────────────────
# The engine runs on the machine that owns the file — there is no hosted service and
# no accounts. These are sanity bounds, not a business model.

# Max upload size
MAX_FILE_MB = int(os.environ.get("MAX_FILE_MB", "500"))
# Max messages in chat history accepted from client
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "30"))
# Max length of a single user message
MAX_MESSAGE_CHARS = int(os.environ.get("MAX_MESSAGE_CHARS", "4000"))
# Allowed OpenAI models (keeps a typo from reaching the API as a bill)
ALLOWED_MODELS = {
    "gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-5.4", "o4-mini",
}
# Keys come from the environment the app starts the engine with, or from a request
# header. They are the user's own; nothing is billed centrally.
SERVER_OPENAI_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY")

# ── App ───────────────────────────────────────────────────────────────────────

_ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]

app = FastAPI(title="CADVERT")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/sessions", StaticFiles(directory=str(UPLOAD_DIR)), name="sessions")

_executor = ThreadPoolExecutor(max_workers=2)

# Session store with timestamps for TTL cleanup
_sessions: dict[str, dict] = {}
_session_timestamps: dict[str, float] = {}
SESSION_TTL_HOURS = int(os.environ.get("SESSION_TTL_HOURS", "24"))


@app.on_event("startup")
async def _start_cleanup_task():
    # Sweep once at startup: sessions left behind by a previous run are not in memory,
    # so a memory-only sweep never sees them and they live forever.
    _sweep_expired(startup=True)
    asyncio.create_task(_cleanup_loop())


def _sweep_expired(startup: bool = False) -> int:
    """Delete expired sessions from memory and disk. Returns how many went.

    Walks the directory rather than the in-memory table, so orphans from earlier runs
    are collected too — that is the difference between a cache with a limit and one that
    only grows.
    """
    cutoff = time.time() - SESSION_TTL_HOURS * 3600
    removed = 0

    for sid, ts in list(_session_timestamps.items()):
        if ts < cutoff:
            _sessions.pop(sid, None)
            _session_timestamps.pop(sid, None)
            shutil.rmtree(UPLOAD_DIR / sid, ignore_errors=True)
            removed += 1

    try:
        for entry in UPLOAD_DIR.iterdir():
            if not entry.is_dir() or entry.name in _sessions:
                continue
            try:
                if entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
    except OSError:
        pass

    if removed:
        print(f"[cleanup] removed {removed} expired session(s){' at startup' if startup else ''}")
    return removed


async def _cleanup_loop():
    """Sweep expired sessions every hour."""
    while True:
        await asyncio.sleep(3600)
        _sweep_expired()


def _cache_usage() -> dict:
    """Size and session count on disk, for the app's cache control."""
    total = files = sessions = 0
    try:
        for entry in UPLOAD_DIR.iterdir():
            if not entry.is_dir():
                continue
            sessions += 1
            for root, _dirs, names in os.walk(entry):
                for n in names:
                    try:
                        total += os.path.getsize(os.path.join(root, n))
                        files += 1
                    except OSError:
                        continue
    except OSError:
        pass
    return {
        "path": str(UPLOAD_DIR),
        "sessions": sessions,
        "files": files,
        "bytes": total,
        "megabytes": round(total / (1024 * 1024), 1),
        "active_sessions": len(_sessions),
        "ttl_hours": SESSION_TTL_HOURS,
    }


# _sessions and _session_timestamps declared above near app startup


# ── OpenAI tool definitions ───────────────────────────────────────────────────

CADVERT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_feature",
            "description": (
                "Get full geometric detail for a specific feature (hole, boss, fillet, "
                "countersink, pattern, etc.) including all constituent faces with exact "
                "surface parameters, boundary edges, and measurements."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "feature_id": {
                        "type": "string",
                        "description": "Feature ID from the part summary, e.g. 'hole_1', 'fillet_3', 'pattern_1'",
                    }
                },
                "required": ["feature_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_face",
            "description": (
                "Get exact geometry for a specific B-REP face including surface type, "
                "parameters (normal, radius, axis, etc.), area, and all boundary edges "
                "with their dihedral angles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "face_id": {
                        "type": "string",
                        "description": "Face ID, e.g. 'F12' or '12'",
                    }
                },
                "required": ["face_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_edge",
            "description": (
                "Get exact geometry for a specific edge: curve type, length, "
                "the two faces it connects, and the dihedral angle between them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "edge_id": {
                        "type": "string",
                        "description": "Edge ID, e.g. 'E5' or '5'",
                    }
                },
                "required": ["edge_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "measure_distance",
            "description": (
                "Compute the exact minimum distance between two geometric entities "
                "using OCC's BRepExtrema. Returns distance and the closest points on each entity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_a": {
                        "type": "string",
                        "description": "First entity: face ID like 'F12', feature like 'hole_1', or point 'x,y,z'",
                    },
                    "entity_b": {
                        "type": "string",
                        "description": "Second entity: same format as entity_a",
                    },
                },
                "required": ["entity_a", "entity_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_neighbors",
            "description": (
                "Get all faces adjacent to a given face within N edge hops, "
                "with their geometry and the connecting edge info."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "face_id": {
                        "type": "string",
                        "description": "Starting face ID, e.g. 'F12'",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Number of edge hops to traverse (default 1 = immediate neighbors)",
                        "default": 1,
                    },
                },
                "required": ["face_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_component",
            "description": (
                "Get one named assembly component: how many are fitted, its volume and "
                "bounding box, its face IDs and the features detected in it. Use the part "
                "name exactly as listed in the ASSEMBLY table of the summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Component name, e.g. 'Belt S5M-300' or 'ISO 4762 - M8 x 20'. Partial names match.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_mass",
            "description": (
                "Compute mass from the modelled geometry for a given material density. "
                "Returns per-part and total mass in g, kg, oz and lb. Optionally restrict "
                "to parts whose name contains a filter, e.g. 'ISO 4762' for screws. "
                "Always use this rather than multiplying volumes yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "density_g_cm3": {
                        "type": "number",
                        "description": "Material density in g/cm³ (steel 7.85, aluminium 2.70, brass 8.50, ABS 1.04). Default 7.85.",
                    },
                    "name_filter": {
                        "type": "string",
                        "description": "Only include parts whose name contains this text. Omit for the whole assembly.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_faces",
            "description": "Find faces matching geometric criteria (type, radius range, area range).",
            "parameters": {
                "type": "object",
                "properties": {
                    "surface_type": {
                        "type": "string",
                        "enum": ["plane", "cylinder", "cone", "sphere", "torus", "nurbs"],
                        "description": "Filter by surface type",
                    },
                    "radius_min": {"type": "number", "description": "Minimum radius in part units"},
                    "radius_max": {"type": "number", "description": "Maximum radius in part units"},
                    "area_min":   {"type": "number", "description": "Minimum face area in part units²"},
                    "area_max":   {"type": "number", "description": "Maximum face area in part units²"},
                },
            },
        },
    },
]


# ── LLM providers ─────────────────────────────────────────────────────────────
# Two hosted providers share the same tool set. Clients pick one with `X-Provider`
# (openai | anthropic); if absent it is inferred from the model name.
SERVER_ANTHROPIC_KEY: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
ALLOWED_ANTHROPIC_MODELS = {
    "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5", "claude-opus-4-8", "claude-sonnet-4-6",
}
DEFAULT_OPENAI_MODEL    = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
MAX_TOOL_ROUNDS = 8
TOOL_NAMES = {t["function"]["name"] for t in CADVERT_TOOLS}


def _anthropic_tools() -> list[dict]:
    """CADVERT_TOOLS (OpenAI function format) → Anthropic tool format."""
    return [
        {
            "name":         t["function"]["name"],
            "description":  t["function"]["description"],
            "input_schema": t["function"]["parameters"],
        }
        for t in CADVERT_TOOLS
    ]


def _resolve_provider(x_provider: Optional[str], x_model: Optional[str]) -> tuple[str, str]:
    """Return (provider, model) with unknown models clamped to each provider's default."""
    p = (x_provider or "").strip().lower()
    if p not in ("openai", "anthropic"):
        p = "anthropic" if (x_model or "").startswith("claude-") else "openai"
    if p == "anthropic":
        model = x_model if x_model in ALLOWED_ANTHROPIC_MODELS else DEFAULT_ANTHROPIC_MODEL
    else:
        model = x_model if x_model in ALLOWED_MODELS else DEFAULT_OPENAI_MODEL
    return p, model


def _provider_available(provider: str) -> bool:
    try:
        if provider == "anthropic":
            import anthropic  # noqa: F401
        else:
            import openai  # noqa: F401
        return True
    except ImportError:
        return False


def _system_prompt(tier0: str) -> str:
    return (
        "You are an expert mechanical engineer and manufacturing consultant. "
        "You have direct access to the exact B-REP geometry of a CAD part through tool calls. "
        "You have been given a compact Tier-0 summary — use it to orient yourself, "
        "then call tools freely to get any geometry you need.\n\n"
        "CRITICAL RULES:\n"
        "- ALWAYS use tools when you need geometry details not in the summary. "
        "Never say 'I recommend using external tools' or 'I cannot access this data' — "
        "you HAVE the tools, call them.\n"
        "- When asked to 'go deeper', call get_feature() or get_face() immediately "
        "for the relevant features and report the exact numbers.\n"
        "- When asked about a specific hole, fillet, boss etc., call get_feature() "
        "with its ID from the summary.\n"
        "- When asked about dimensions between two faces, call measure_distance().\n"
        "- When asked about adjacent faces or local topology, call get_neighbors().\n"
        "- Chain multiple tool calls in one response when needed — e.g. get all "
        "hole features then measure distances between them.\n"
        "- Reference faces as F12, edges as E5, features by ID (hole_1, fillet_3).\n"
        "- Units are specified in the document header. Be precise with numbers.\n"
        "- For mesh files (STL/OBJ): exact geometry is unavailable — say so clearly.\n\n"
        f"<PART_SUMMARY>\n{tier0}\n</PART_SUMMARY>"
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/config")
async def config():
    """What this engine supports. No accounts, no plans — it runs locally."""
    return JSONResponse({
        "local_only": True,
        "providers": {
            "openai": {
                "available":     _provider_available("openai"),
                "key_present":   bool(SERVER_OPENAI_KEY),
                "models":        sorted(ALLOWED_MODELS),
                "default_model": DEFAULT_OPENAI_MODEL,
            },
            "anthropic": {
                "available":     _provider_available("anthropic"),
                "key_present":   bool(SERVER_ANTHROPIC_KEY),
                "models":        sorted(ALLOWED_ANTHROPIC_MODELS),
                "default_model": DEFAULT_ANTHROPIC_MODEL,
            },
        },
        "tools": sorted(TOOL_NAMES),
        "max_file_mb": MAX_FILE_MB,
    })


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="UI not found — ensure static/index.html exists")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/developers", response_class=HTMLResponse)
async def developers():
    html_path = STATIC_DIR / "developers.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="developers.html not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/about", response_class=HTMLResponse)
async def about():
    html_path = STATIC_DIR / "about.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="about.html not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/convert")
async def convert(
    request: Request,
    file: UploadFile = File(...),
):
    """Upload a CAD file and run the full CADVERT pipeline."""
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True)

    suffix = Path(file.filename or "upload.step").suffix.lower() or ".step"
    input_path = session_dir / f"input{suffix}"
    content = await file.read()
    if len(content) > MAX_FILE_MB * 1024 * 1024:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=413, detail=f"File too large — maximum size is {MAX_FILE_MB}MB.")
    input_path.write_bytes(content)

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

    result["message_count"] = 0
    _sessions[session_id] = result
    _session_timestamps[session_id] = time.time()

    images = []
    for img_path_str in result.get("image_paths", []):
        p = Path(img_path_str)
        if p.exists():
            rel = p.relative_to(UPLOAD_DIR)
            images.append({"name": p.stem, "url": f"/sessions/{rel.as_posix()}"})

    return JSONResponse({
        "session_id": session_id,
        "hsd":        result["hsd"],
        "tier0":      result["tier0"],
        "images":     images,
        "format":     result["format"],
        "is_mesh":    result["is_mesh"],
        "filename":   file.filename,
        "units":      result["units"],
        "summary":    result["summary"],
    })


@app.post("/chat/{session_id}")
async def chat(
    session_id: str,
    request: Request,
    x_openai_key: Optional[str] = Header(default=None, alias="X-OpenAI-Key"),
    x_anthropic_key: Optional[str] = Header(default=None, alias="X-Anthropic-Key"),
    x_provider: Optional[str] = Header(default=None, alias="X-Provider"),
    x_model: str = Header(default="gpt-4o-mini", alias="X-Model"),
):
    """Stream an LLM chat response using Tier 0 context + tool calling.

    Body JSON: ``{"messages": [{role, content}, ...]}``

    The key is the caller's own: sent per-request in ``X-OpenAI-Key`` /
    ``X-Anthropic-Key``, or taken from the environment the engine was started with.
    There are no accounts, plans or quotas — the engine runs on the machine that owns
    the file, and any API spend is the user's own.
    """
    provider, model = _resolve_provider(x_provider, x_model)
    env_key   = SERVER_ANTHROPIC_KEY if provider == "anthropic" else SERVER_OPENAI_KEY
    header_key = x_anthropic_key if provider == "anthropic" else x_openai_key
    vendor    = "Anthropic" if provider == "anthropic" else "OpenAI"

    if not _provider_available(provider):
        raise HTTPException(
            status_code=500,
            detail=f"The {vendor} SDK is not installed in this engine — pip install cadvert[llm]",
        )

    api_key = (header_key or "").strip() or env_key
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "api_key_required",
                "provider": provider,
                "message": f"Add your {vendor} API key to ask questions about this part.",
            },
        )

    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    body = await request.json()
    user_messages: list[dict] = body.get("messages", [])
    user_messages = user_messages[-MAX_HISTORY_MESSAGES:]
    for msg in user_messages:
        if isinstance(msg.get("content"), str) and len(msg["content"]) > MAX_MESSAGE_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"Message too long — maximum {MAX_MESSAGE_CHARS} characters.",
            )

    session["message_count"] = session.get("message_count", 0) + 1
    tier0      = session.get("tier0") or session.get("hsd", "")
    system_msg = _system_prompt(tier0)
    use_tools  = not session.get("is_mesh", False)

    if provider == "anthropic":
        gen = _stream_anthropic(api_key, model, system_msg, user_messages, session, use_tools)
    else:
        gen = _stream_openai(api_key, model, system_msg, user_messages, session, use_tools)
    return StreamingResponse(gen, media_type="text/event-stream")


async def _stream_openai(api_key: str, model: str, system_msg: str, user_messages: list[dict],
                         session: dict, use_tools: bool):
    """SSE generator: OpenAI chat completions with the CADVERT tool loop."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    openai_messages = [{"role": "system", "content": system_msg}] + user_messages
    tools = CADVERT_TOOLS if use_tools else []

    try:
        # Stream the initial response — detect tool calls mid-stream
        stream = await client.chat.completions.create(
            model=model,
            messages=openai_messages,
            tools=tools or None,
            tool_choice="auto" if tools else None,
            max_completion_tokens=2048,
            stream=True,
        )

        accumulated_content = ""
        accumulated_tool_calls: dict[int, dict] = {}
        finish_reason = None

        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta

            if delta.content:
                accumulated_content += delta.content
                yield f"data: {json.dumps({'content': delta.content})}\n\n"

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "id":        tc_delta.id or "",
                            "name":      (tc_delta.function.name or "") if tc_delta.function else "",
                            "arguments": "",
                        }
                    if tc_delta.function:
                        if tc_delta.function.name:
                            accumulated_tool_calls[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            accumulated_tool_calls[idx]["arguments"] += tc_delta.function.arguments

        # Tool-call loop
        rounds = 0
        while finish_reason == "tool_calls" and accumulated_tool_calls and rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            tool_calls_list = [
                accumulated_tool_calls[i]
                for i in sorted(accumulated_tool_calls.keys())
            ]

            # Notify frontend — show "Analyzing geometry…" for each tool call
            for tc in tool_calls_list:
                yield f"data: {json.dumps({'tool_call': tc['name']})}\n\n"

            # Append assistant turn with tool_calls
            openai_messages.append({
                "role":    "assistant",
                "content": accumulated_content or None,
                "tool_calls": [
                    {
                        "id":   tc["id"],
                        "type": "function",
                        "function": {
                            "name":      tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in tool_calls_list
                ],
            })

            # Execute each tool call and append results
            for tc in tool_calls_list:
                try:
                    args = json.loads(tc["arguments"] or "{}")
                except Exception:
                    args = {}
                result_data = _execute_tool(session, tc["name"], args)
                openai_messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      json.dumps(result_data),
                })

            # Next round — stream again
            accumulated_content = ""
            accumulated_tool_calls = {}
            finish_reason = None

            stream = await client.chat.completions.create(
                model=model,
                messages=openai_messages,
                tools=tools,
                tool_choice="auto",
                max_completion_tokens=2048,
                stream=True,
            )

            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta

                if delta.content:
                    accumulated_content += delta.content
                    yield f"data: {json.dumps({'content': delta.content})}\n\n"

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id":        tc_delta.id or "",
                                "name":      (tc_delta.function.name or "") if tc_delta.function else "",
                                "arguments": "",
                            }
                        if tc_delta.function:
                            if tc_delta.function.name:
                                accumulated_tool_calls[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                accumulated_tool_calls[idx]["arguments"] += tc_delta.function.arguments

        yield "data: [DONE]\n\n"

    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"


async def _stream_anthropic(api_key: str, model: str, system_msg: str, user_messages: list[dict],
                            session: dict, use_tools: bool):
    """SSE generator: Claude (Messages API, streaming) with the CADVERT tool loop.

    Same event vocabulary as the OpenAI path — `content`, `tool_call`, `error`, `[DONE]` —
    so every client works unchanged. Claude Opus 5 runs adaptive thinking by default;
    thinking blocks are echoed back untouched when the tool loop continues the turn.
    """
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=api_key)
    tools = _anthropic_tools() if use_tools else []
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in user_messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    if not messages or messages[0]["role"] != "user":
        yield f"data: {json.dumps({'error': 'Conversation must start with a user message'})}\n\n"
        return

    try:
        for _round in range(MAX_TOOL_ROUNDS + 1):
            kwargs: dict = dict(model=model, max_tokens=8192, system=system_msg, messages=messages)
            if tools:
                kwargs["tools"] = tools

            async with client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    etype = getattr(event, "type", "")
                    if etype == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if getattr(block, "type", "") == "tool_use":
                            yield f"data: {json.dumps({'tool_call': block.name})}\n\n"
                    elif etype == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if getattr(delta, "type", "") == "text_delta" and delta.text:
                            yield f"data: {json.dumps({'content': delta.text})}\n\n"
                final = await stream.get_final_message()

            if final.stop_reason == "refusal":
                yield f"data: {json.dumps({'error': 'Claude declined to answer this request.'})}\n\n"
                return

            tool_uses = [b for b in final.content if b.type == "tool_use"]
            if final.stop_reason != "tool_use" or not tool_uses:
                break

            # Continue the turn: assistant blocks (incl. thinking) + one user message of tool results
            messages.append({"role": "assistant", "content": final.content})
            results = []
            for tu in tool_uses:
                args = tu.input if isinstance(tu.input, dict) else {}
                result_data = _execute_tool(session, tu.name, args)
                results.append({
                    "type":        "tool_result",
                    "tool_use_id": tu.id,
                    "content":     json.dumps(result_data),
                    "is_error":    "error" in result_data,
                })
            messages.append({"role": "user", "content": results})

        yield "data: [DONE]\n\n"

    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"


@app.post("/tool/{session_id}")
async def call_tool(
    session_id: str,
    request: Request,
):
    """Run one geometry tool directly and return its JSON result.

    For clients that drive the LLM themselves — e.g. the Mac/iPad app using Apple's
    on-device model — so the model's tool calls hit the same exact-geometry code as
    the hosted providers. Body: {"name": "get_feature", "arguments": {"feature_id": "hole_1"}}
    """
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    body = await request.json()
    name = str(body.get("name", ""))
    args = body.get("arguments") or {}
    if name not in TOOL_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown tool: {name}")
    if not isinstance(args, dict):
        raise HTTPException(status_code=400, detail="arguments must be an object")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _execute_tool, session, name, args)
    return JSONResponse(result)


@app.get("/cache")
async def cache_usage():
    """How much disk the cached parts are using, for the app's cache control."""
    return JSONResponse(_cache_usage())


@app.delete("/cache")
async def clear_cache():
    """Delete every cached part, including the one currently open.

    Everything here can be rebuilt by reopening the CAD file, so this is always safe;
    the caller just has to reopen whatever it was showing.
    """
    before = _cache_usage()
    _sessions.clear()
    _session_timestamps.clear()
    removed = 0
    try:
        for entry in UPLOAD_DIR.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
    except OSError:
        pass
    return JSONResponse({
        "cleared_sessions": removed,
        "freed_megabytes": before["megabytes"],
        "now": _cache_usage(),
    })


@app.get("/tools")
async def list_tools():
    """The geometry tools every provider gets, as name / description / JSON-schema parameters."""
    return JSONResponse([t["function"] for t in CADVERT_TOOLS])


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    _sessions.pop(session_id, None)
    session_dir = UPLOAD_DIR / session_id
    shutil.rmtree(session_dir, ignore_errors=True)
    return {"ok": True}


# ── Tool execution ────────────────────────────────────────────────────────────

def _execute_tool(session: dict, tool_name: str, args: dict) -> dict:
    """Dispatch a tool call to the appropriate handler."""
    graph        = session.get("graph")
    features     = session.get("features") or []
    feature_ids  = session.get("feature_ids") or []
    shape        = session.get("shape")
    face_shape_map = session.get("face_shape_map") or {}
    units        = session.get("units", "mm")
    assembly     = session.get("assembly")

    try:
        if tool_name == "get_feature":
            return _tool_get_feature(args.get("feature_id", ""), features, feature_ids, graph, units, assembly)
        elif tool_name == "get_face":
            return _tool_get_face(args.get("face_id", ""), graph, units, assembly)
        elif tool_name == "get_edge":
            return _tool_get_edge(args.get("edge_id", ""), graph, units)
        elif tool_name == "measure_distance":
            return _tool_measure(
                args.get("entity_a", ""), args.get("entity_b", ""),
                graph, shape, face_shape_map, features, feature_ids,
            )
        elif tool_name == "get_neighbors":
            return _tool_neighbors(args.get("face_id", ""), int(args.get("depth", 1)), graph, units)
        elif tool_name == "search_faces":
            return _tool_search(args, graph, units)
        elif tool_name == "get_component":
            return _tool_get_component(args.get("name", ""), session)
        elif tool_name == "compute_mass":
            return _tool_compute_mass(args, session)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    except Exception as exc:
        return {"error": f"Tool error: {exc}"}


def _edge_length(edge) -> float | None:
    """Edge length from its curve geometry (lines: length, circles: arc_length)."""
    g = edge.geometry or {}
    v = g.get("length", g.get("arc_length"))
    try:
        return round(float(v), 4) if v is not None else None
    except (TypeError, ValueError):
        return None


def _face_edge_convexity(face, edge_by_id) -> list[str]:
    """Faces have no convexity of their own — summarise their boundary edges."""
    return sorted({edge_by_id[e].convexity for e in face.edge_ids if e in edge_by_id})


def _component_of(assembly, face_ids) -> dict:
    """``{"component": name}`` when the file has a product tree, else nothing.

    Answering "which part is this hole in" is most of what makes an assembly legible.
    """
    if not assembly:
        return {}
    owner = assembly.owner_of_faces(face_ids or ())
    if owner is None:
        return {}
    out = {"component": owner.name}
    if owner.path:
        out["component_path"] = owner.location
    return out


def _tool_get_component(name: str, session: dict) -> dict:
    assembly = session.get("assembly")
    if not assembly:
        return {"error": "This file has no assembly structure (single part, or a format without one)."}
    q = (name or "").strip().lower()
    if not q:
        return {"error": "Provide a component name."}
    rows = [r for r in assembly.bill_of_materials() if q in r["name"].lower()]
    if not rows:
        return {
            "error": f"No component matching '{name}'",
            "available": [r["name"] for r in assembly.bill_of_materials()[:40]],
        }

    features = session.get("features") or []
    feature_ids = session.get("feature_ids") or []
    out = []
    for row in rows:
        insts = [i for i in assembly.instances if i.name == row["name"]]
        first = insts[0]
        feats = []
        for fid, feat in zip(feature_ids, features):
            owner = assembly.owner_of_faces(getattr(feat, "face_ids", ()) or ())
            if owner is not None and owner.name == row["name"]:
                feats.append(f"{fid}:{feat.feature_type}")
        out.append({
            "name": row["name"],
            "quantity": row["quantity"],
            "faces_each": row["faces_each"],
            "volume_each_mm3": round(row["volume_each"], 3),
            "bbox_each": first.bbox,
            "fitted_in": row["locations"][:6],
            "face_ids_first_instance": [f"F{f}" for f in first.face_ids[:60]],
            "features_in_this_part": feats[:40],
            "feature_count": len(feats),
        })
    return {"matches": len(out), "components": out, "units": session.get("units", "mm")}


def _tool_compute_mass(args: dict, session: dict) -> dict:
    assembly = session.get("assembly")
    if not assembly:
        return {"error": "This file has no assembly structure; per-part mass is unavailable."}
    try:
        density = float(args.get("density_g_cm3") or 7.85)
    except (TypeError, ValueError):
        density = 7.85
    if density <= 0:
        return {"error": "density_g_cm3 must be positive"}
    name_filter = args.get("name_filter") or None

    density_mm3 = density / 1000.0
    rows, total, qty = [], 0.0, 0
    for row in assembly.bill_of_materials():
        if name_filter and name_filter.lower() not in row["name"].lower():
            continue
        each = row["volume_each"] * density_mm3
        tot = each * row["quantity"]
        total += tot
        qty += row["quantity"]
        rows.append({
            "name": row["name"], "quantity": row["quantity"],
            "volume_each_mm3": round(row["volume_each"], 3),
            "mass_each_g": round(each, 4), "mass_total_g": round(tot, 3),
        })
    if not rows:
        return {"error": f"No parts matching '{name_filter}'"}
    return {
        "density_g_cm3": density, "filter": name_filter,
        "part_types": len(rows), "total_parts": qty,
        "total_mass_g": round(total, 3),
        "total_mass_kg": round(total / 1000.0, 5),
        "total_mass_oz": round(total / 28.349523125, 4),
        "total_mass_lb": round(total / 453.59237, 5),
        "parts": rows,
        "assumptions": [
            f"density {density} g/cm³ applied to every matched part",
            "volumes are of the modelled solid; unmodelled threads make fastener mass slightly high",
        ],
    }


def _parse_fid(s: str) -> int:
    return int(str(s).lstrip("Ff"))

def _parse_eid(s: str) -> int:
    return int(str(s).lstrip("Ee"))


def _tool_get_feature(feature_id, features, feature_ids, graph, units, assembly=None) -> dict:
    if not features:
        return {"error": "No features available for this part"}

    feat = None
    for fid, f in zip(feature_ids, features):
        if fid == feature_id:
            feat = f
            break

    if feat is None:
        return {
            "error": f"Feature '{feature_id}' not found",
            "available_ids": feature_ids[:30],
        }

    result: dict = {
        "id":         feature_id,
        "type":       feat.feature_type,
        "parameters": feat.parameters,
        "confidence": feat.confidence,
        "face_ids":   [f"F{x}" for x in feat.face_ids],
        "edge_ids":   [f"E{x}" for x in feat.edge_ids],
    }
    if feat.standard_match:
        result["standard_match"] = feat.standard_match
    if feat.notes:
        result["notes"] = feat.notes

    if graph:
        face_by_id = {f.id: f for f in graph.faces}
        edge_by_id = {e.id: e for e in graph.edges}

        result["faces"] = []
        for fid in feat.face_ids:
            face = face_by_id.get(fid)
            if face:
                result["faces"].append({
                    "id":             f"F{fid}",
                    "geometry":       face.geometry,
                    "area":           round(face.area, 4),
                    "edge_convexity": _face_edge_convexity(face, edge_by_id),
                })

        result["edges"] = []
        for eid in feat.edge_ids[:12]:
            edge = edge_by_id.get(eid)
            if edge:
                result["edges"].append({
                    "id":            f"E{eid}",
                    "geometry":      edge.geometry,
                    "length":        _edge_length(edge),
                    "connects":      [f"F{x}" for x in edge.face_ids],
                    "dihedral_angle": round(edge.dihedral_angle, 3) if edge.dihedral_angle else None,
                })

    result["units"] = units

    result.update(_component_of(assembly, feat.face_ids))

    return result


def _tool_get_face(face_id, graph, units, assembly=None) -> dict:
    if not graph:
        return {"error": "No B-REP graph available"}
    try:
        fid = _parse_fid(face_id)
    except (ValueError, TypeError):
        return {"error": f"Invalid face ID: {face_id}"}

    face_by_id = {f.id: f for f in graph.faces}
    face = face_by_id.get(fid)
    if not face:
        return {"error": f"Face F{fid} not found (total: {len(graph.faces)})"}

    edge_by_id = {e.id: e for e in graph.edges}
    edges = []
    for eid in face.edge_ids:
        edge = edge_by_id.get(eid)
        if edge:
            other = [f"F{x}" for x in edge.face_ids if x != fid]
            edges.append({
                "id":            f"E{eid}",
                "geometry":      edge.geometry,
                "length":        _edge_length(edge),
                "connects_to":   other,
                "dihedral_angle": round(edge.dihedral_angle, 3) if edge.dihedral_angle else None,
                "convexity":     edge.convexity,
            })

    return {
        "id":        f"F{fid}",
        "geometry":  face.geometry,
        "area":      round(face.area, 4),
        "edge_convexity": _face_edge_convexity(face, edge_by_id),
        "edges":     edges,
        "units":     units,
        **_component_of(assembly, [fid]),
    }


def _tool_get_edge(edge_id, graph, units) -> dict:
    if not graph:
        return {"error": "No B-REP graph available"}
    try:
        eid = _parse_eid(edge_id)
    except (ValueError, TypeError):
        return {"error": f"Invalid edge ID: {edge_id}"}

    edge_by_id = {e.id: e for e in graph.edges}
    edge = edge_by_id.get(eid)
    if not edge:
        return {"error": f"Edge E{eid} not found"}

    return {
        "id":            f"E{eid}",
        "geometry":      edge.geometry,
        "length":        _edge_length(edge),
        "connects":      [f"F{x}" for x in edge.face_ids],
        "dihedral_angle": round(edge.dihedral_angle, 3) if edge.dihedral_angle else None,
        "convexity":     edge.convexity,
        "units":         units,
    }


def _tool_measure(entity_a, entity_b, graph, shape, face_shape_map, features, feature_ids) -> dict:
    if shape is None:
        return {"error": "OCC shape not available for distance measurement"}

    def resolve(entity_str: str):
        entity_str = str(entity_str).strip()

        # Point "x,y,z"
        if "," in entity_str and not entity_str.upper().startswith("F"):
            try:
                coords = [float(x.strip()) for x in entity_str.split(",")]
                if len(coords) == 3:
                    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
                    from OCP.gp import gp_Pnt
                    return BRepBuilderAPI_MakeVertex(gp_Pnt(*coords)).Vertex()
            except ValueError:
                pass

        # Feature ID → first face of that feature
        if features and feature_ids:
            for fid, feat in zip(feature_ids, features):
                if fid == entity_str and feat.face_ids:
                    occ_face = face_shape_map.get(feat.face_ids[0])
                    if occ_face is not None:
                        return occ_face

        # Face ID "F12" or "12"
        try:
            fid = _parse_fid(entity_str)
            occ_face = face_shape_map.get(fid)
            if occ_face is not None:
                return occ_face
        except (ValueError, TypeError):
            pass

        return None

    shape_a = resolve(entity_a)
    shape_b = resolve(entity_b)

    if shape_a is None:
        return {"error": f"Cannot resolve entity: {entity_a}"}
    if shape_b is None:
        return {"error": f"Cannot resolve entity: {entity_b}"}

    try:
        from OCP.BRepExtrema import BRepExtrema_DistShapeShape
        calc = BRepExtrema_DistShapeShape(shape_a, shape_b)
        calc.Perform()
        if calc.IsDone():
            p1 = calc.PointOnShape1(1)
            p2 = calc.PointOnShape2(1)
            return {
                "distance":    round(calc.Value(), 4),
                "point_on_a":  (round(p1.X(), 3), round(p1.Y(), 3), round(p1.Z(), 3)),
                "point_on_b":  (round(p2.X(), 3), round(p2.Y(), 3), round(p2.Z(), 3)),
                "entity_a":    entity_a,
                "entity_b":    entity_b,
            }
        return {"error": "BRepExtrema did not converge"}
    except Exception as exc:
        return {"error": f"Distance computation failed: {exc}"}


def _tool_neighbors(face_id, depth, graph, units) -> dict:
    if not graph:
        return {"error": "No B-REP graph available"}
    try:
        start = _parse_fid(face_id)
    except (ValueError, TypeError):
        return {"error": f"Invalid face ID: {face_id}"}

    face_by_id = {f.id: f for f in graph.faces}
    edge_by_id = {e.id: e for e in graph.edges}

    if start not in face_by_id:
        return {"error": f"Face F{start} not found"}

    visited  = {start}
    frontier = {start}
    results  = []

    for _ in range(max(1, min(depth, 3))):  # cap depth at 3
        new_frontier: set[int] = set()
        for fid in frontier:
            face = face_by_id.get(fid)
            if not face:
                continue
            for eid in face.edge_ids:
                edge = edge_by_id.get(eid)
                if not edge:
                    continue
                for nfid in edge.face_ids:
                    if nfid not in visited:
                        visited.add(nfid)
                        new_frontier.add(nfid)
                        nbr = face_by_id.get(nfid)
                        if nbr and len(results) < 30:
                            results.append({
                                "id":             f"F{nfid}",
                                "geometry":       nbr.geometry,
                                "area":           round(nbr.area, 4),
                                "convexity":      edge.convexity,
                                "via_edge":       f"E{eid}",
                                "dihedral_angle": round(edge.dihedral_angle, 3) if edge.dihedral_angle else None,
                            })
        frontier = new_frontier
        if not frontier:
            break

    return {
        "center":    f"F{start}",
        "depth":     depth,
        "count":     len(results),
        "neighbors": results,
        "units":     units,
    }


def _tool_search(args: dict, graph, units) -> dict:
    if not graph:
        return {"error": "No B-REP graph available"}

    stype    = (args.get("surface_type") or "").lower()
    rmin     = args.get("radius_min")
    rmax     = args.get("radius_max")
    amin     = args.get("area_min")
    amax     = args.get("area_max")

    matches = []
    for face in graph.faces:
        g = face.geometry
        t = g.get("type", "").lower()

        if stype and t != stype:
            continue
        if amin is not None and face.area < amin:
            continue
        if amax is not None and face.area > amax:
            continue

        r = g.get("radius") or g.get("minor_radius")
        if rmin is not None and (r is None or r < rmin):
            continue
        if rmax is not None and (r is None or r > rmax):
            continue

        matches.append({
            "id":   f"F{face.id}",
            "type": t,
            "area": round(face.area, 4),
            "key":  _compact_geom(g),
        })

    return {
        "query":  args,
        "count":  len(matches),
        "faces":  matches[:50],
        "units":  units,
    }


def _compact_geom(g: dict) -> str:
    t = g.get("type", "")
    if t == "CYLINDER":
        return f"r={g.get('radius', 0):.3f}"
    if t == "PLANE":
        n = g.get("normal", [0, 0, 0])
        return f"n=({n[0]:.2f},{n[1]:.2f},{n[2]:.2f})"
    if t == "TORUS":
        return f"R={g.get('major_radius', 0):.3f} r={g.get('minor_radius', 0):.3f}"
    if t == "CONE":
        return f"angle={g.get('half_angle', 0):.1f}°"
    return ""


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _run_pipeline(input_path: Path, session_dir: Path) -> dict:
    from .ingest import load_step, IngestError

    try:
        shape, body_count, metadata = load_step(input_path)
    except IngestError as exc:
        raise ValueError(str(exc))

    graph = features = spatial = validation_text = None
    feature_ids: list[str] = []

    if not metadata.is_mesh:
        from .topology import build_topology
        graph = build_topology(shape, body_count)

        # Skip validation for large assemblies — O(faces²) is too slow above ~200
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
            from .document import assign_feature_ids
            feature_ids = assign_feature_ids(features)
            try:
                from .spatial import compute_spatial_relationships
                spatial = compute_spatial_relationships(graph, features, shape=shape)
            except Exception:
                pass

    # Render views in subprocess (avoids macOS NSWindow crash)
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

    # Build face → OCC shape map for tool calls (1-based, matches topology.py)
    face_shape_map: dict[int, object] = {}
    if not metadata.is_mesh and graph is not None:
        face_shape_map = _build_face_shape_map(shape)

    # Tier 0 — compact executive summary (goes in LLM system prompt)
    from .document import render_tier0
    tier0 = render_tier0(
        graph, input_path,
        feature_ids=feature_ids,
        features=features,
        spatial=spatial,
        units=metadata.units,
        gdt_annotations=metadata.gdt_annotations or None,
        mesh_info=mesh_info,
        validation_report=validation_text,
        metadata=metadata,
    )

    # Tier 2 — full HSD document (stored server-side, available via hsd field)
    from .document import render_document
    hsd = render_document(
        graph, input_path,
        features=features,
        spatial=spatial,
        rendered_views=[Path(p) for p in image_paths] if image_paths else None,
        validation_report=validation_text,
        units=metadata.units,
        gdt_annotations=metadata.gdt_annotations or None,
        mesh_info=mesh_info,
        metadata=metadata,
    )

    summary = _build_summary(graph, features, metadata, body_count)

    return {
        "hsd":            hsd,
        "tier0":          tier0,
        "graph":          graph,
        "features":       features,
        "feature_ids":    feature_ids,
        "spatial":        spatial,
        "shape":          shape,
        "face_shape_map": face_shape_map,
        "assembly":       metadata.assembly,
        "image_paths":    image_paths,
        "format":         metadata.source_format,
        "is_mesh":        metadata.is_mesh,
        "units":          metadata.units,
        "summary":        summary,
    }


def _build_face_shape_map(shape) -> dict:
    """Map face ID (1-based, matching topology.py) → TopoDS_Face."""
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopAbs  import TopAbs_FACE
    from OCP.TopExp  import TopExp
    from OCP.TopoDS  import TopoDS

    face_imap = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_imap)
    result = {}
    for fid in range(1, face_imap.Extent() + 1):
        result[fid] = TopoDS.Face_s(face_imap.FindKey(fid))
    return result


def _render_subprocess(shape, output_dir: Path, image_size: tuple, stem: str) -> list[str]:
    """Render VTK views in a subprocess to avoid macOS NSWindow crash."""
    import subprocess, sys, os, tempfile
    from OCP.BRepTools import BRepTools

    with tempfile.NamedTemporaryFile(suffix=".brep", delete=False) as f:
        brep_path = f.name
    try:
        BRepTools.Write_s(shape, brep_path)
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
paths = render_shape(shape, {repr(str(output_dir))}, image_size={image_size!r}, stem={repr(stem)})
for p in paths:
    print(p)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            return [p for p in result.stdout.strip().splitlines() if p]
        return []
    finally:
        try:
            os.unlink(brep_path)
        except OSError:
            pass


def _compute_mesh_info(metadata, shape) -> dict:
    info = {
        "format":         metadata.source_format,
        "triangle_count": metadata.triangle_count,
        "units":          metadata.units,
    }
    try:
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib

        vp = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, vp)
        info["volume"] = vp.Mass()

        sp = GProp_GProps()
        BRepGProp.SurfaceProperties_s(shape, sp)
        info["surface_area"] = sp.Mass()

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
        "format":  metadata.source_format,
        "schema":  metadata.schema,
        "units":   metadata.units,
        "is_mesh": metadata.is_mesh,
    }
    if metadata.is_mesh:
        s["triangles"] = metadata.triangle_count
    else:
        if graph:
            s["faces"]  = len(graph.faces)
            s["edges"]  = len(graph.edges)
            s["bodies"] = body_count
        if features:
            from collections import Counter
            s["features"] = dict(Counter(f.feature_type for f in features))
        if metadata.gdt_annotations:
            s["gdt_count"] = len(metadata.gdt_annotations)
    return s


# ── Dev server entry point ────────────────────────────────────────────────────

def run():
    import uvicorn
    uvicorn.run("cadvert.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
