"""CADVERT web server.

Endpoints:
  GET  /                        → UI (index.html)
  POST /convert                 → upload CAD file, run full pipeline, return Tier 0 + images
  POST /chat/{session_id}       → stream GPT response with Tier 0 context + tool calling
  DELETE /session/{session_id}  → clean up temp files
"""

from __future__ import annotations
import asyncio
import json
import os
import shutil
import sqlite3
import tempfile
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
UPLOAD_DIR = Path(tempfile.gettempdir()) / "cadvert_sessions"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Auth ───────────────────────────────────────────────────────────────────────
DISABLE_AUTH = os.environ.get("DISABLE_AUTH", "0") == "1"
GOOGLE_CLIENT_ID = os.environ.get(
    "GOOGLE_CLIENT_ID",
    "257835361477-e8d4ebui9tm1pa6dssb5gguh7v1mjt0g.apps.googleusercontent.com",
)

# ── Stripe ────────────────────────────────────────────────────────────────────
STRIPE_SECRET_KEY     = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID       = os.environ.get("STRIPE_PRICE_ID", "")       # $9/month Pro
STRIPE_BYOK_PRICE_ID  = os.environ.get("STRIPE_BYOK_PRICE_ID", "")  # $3/month BYOK
TRIAL_DAYS            = 3
TRIAL_AMOUNT_CENTS    = 100  # $1

if STRIPE_SECRET_KEY:
    try:
        import stripe as _stripe
        _stripe.api_key = STRIPE_SECRET_KEY
    except ImportError:
        pass

# ── Tier limits ───────────────────────────────────────────────────────────────

# Server pays for this many messages per session (free tier), then BYOK wall
FREE_MESSAGES_PER_SESSION = int(os.environ.get("FREE_MESSAGES_PER_SESSION", "3"))
# Max files processed per user per day (resets at midnight)
FREE_FILES_PER_DAY = int(os.environ.get("FREE_FILES_PER_DAY", "1"))
# BYOK users cap (0 = unlimited)
BYOK_MESSAGES_PER_SESSION = int(os.environ.get("BYOK_MESSAGES_PER_SESSION", "20"))
# Server's own OpenAI key (pays for free + pro messages)
SERVER_OPENAI_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY")

# ── SQLite users DB ────────────────────────────────────────────────────────────
DB_PATH = Path(tempfile.gettempdir()) / "cadvert_users.db"


def _init_db() -> None:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            google_id              TEXT PRIMARY KEY,
            email                  TEXT NOT NULL,
            name                   TEXT,
            picture                TEXT,
            tier                   TEXT DEFAULT 'free',
            stripe_customer_id     TEXT,
            stripe_subscription_id TEXT,
            files_today            INTEGER DEFAULT 0,
            last_file_date         TEXT,
            created_at             TEXT DEFAULT (date('now'))
        )
    """)
    # Migrate existing DBs that lack the new columns
    for col, definition in [
        ("tier",                   "TEXT DEFAULT 'free'"),
        ("stripe_customer_id",     "TEXT"),
        ("stripe_subscription_id", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col} {definition}")
        except Exception:
            pass
    conn.commit()
    conn.close()


_init_db()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="CADVERT")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/sessions", StaticFiles(directory=str(UPLOAD_DIR)), name="sessions")

_executor = ThreadPoolExecutor(max_workers=2)

# In-memory session store: session_id → result dict
# Keys: hsd, tier0, graph, features, feature_ids, spatial, shape,
#       face_shape_map, image_paths, format, is_mesh, units, summary,
#       message_count (int)
_sessions: dict[str, dict] = {}


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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/config")
async def config():
    return JSONResponse({
        "disable_auth":        DISABLE_AUTH,
        "stripe_enabled":      bool(STRIPE_SECRET_KEY and STRIPE_PRICE_ID),
        "stripe_byok_enabled": bool(STRIPE_SECRET_KEY and STRIPE_BYOK_PRICE_ID),
    })


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="UI not found — ensure static/index.html exists")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


def _verify_google_token(credential: str) -> dict:
    """Verify a Google ID token (from GIS) and return the decoded payload."""
    try:
        from google.oauth2 import id_token as gid_token
        from google.auth.transport import requests as grequests
        payload = gid_token.verify_oauth2_token(
            credential,
            grequests.Request(),
            GOOGLE_CLIENT_ID,
        )
        return payload
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {exc}")


def _get_current_user(authorization: Optional[str]) -> dict:
    """Extract & verify Bearer token; return user dict with google_id, email, name, picture."""
    if DISABLE_AUTH:
        return {"sub": "dev", "email": "dev@local", "name": "Dev", "picture": ""}
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Login required")
    token = authorization[len("Bearer "):]
    return _verify_google_token(token)


def _upsert_user(payload: dict) -> None:
    """Create or update user row in SQLite from a verified token payload."""
    google_id = payload["sub"]
    email     = payload.get("email", "")
    name      = payload.get("name", "")
    picture   = payload.get("picture", "")
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """INSERT INTO users (google_id, email, name, picture)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(google_id) DO UPDATE SET
               email   = excluded.email,
               name    = excluded.name,
               picture = excluded.picture""",
        (google_id, email, name, picture),
    )
    conn.commit()
    conn.close()


def _check_file_limit_user(google_id: str) -> None:
    """Raise 429 if this user has already used their daily file quota, else increment."""
    today = date.today().isoformat()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        row = conn.execute(
            "SELECT files_today, last_file_date FROM users WHERE google_id = ?",
            (google_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=401, detail="User not found — please sign in again")
        files_today, last_file_date = row
        if last_file_date != today:
            files_today = 0
        if files_today >= FREE_FILES_PER_DAY:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "file_limit",
                    "message": f"You've used your {FREE_FILES_PER_DAY} free file(s) today.",
                    "reset": "Resets at midnight. Get the desktop app for unlimited.",
                },
            )
        conn.execute(
            "UPDATE users SET files_today = ?, last_file_date = ? WHERE google_id = ?",
            (files_today + 1, today, google_id),
        )
        conn.commit()
    finally:
        conn.close()


def _get_user_tier(google_id: str) -> str:
    """Return 'pro' or 'free' for a user."""
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute("SELECT tier FROM users WHERE google_id = ?", (google_id,)).fetchone()
    conn.close()
    return (row[0] or "free") if row else "free"


def _set_user_tier_by_customer(customer_id: str, tier: str, subscription_id: Optional[str]) -> None:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "UPDATE users SET tier = ?, stripe_subscription_id = ? WHERE stripe_customer_id = ?",
        (tier, subscription_id, customer_id),
    )
    conn.commit()
    conn.close()


@app.post("/auth/verify")
async def auth_verify(request: Request):
    """Verify a Google credential and upsert the user. Returns user info."""
    body = await request.json()
    credential = body.get("credential", "")
    if not credential:
        raise HTTPException(status_code=400, detail="credential required")
    payload = _verify_google_token(credential)
    _upsert_user(payload)
    tier = _get_user_tier(payload["sub"])
    return JSONResponse({
        "google_id": payload["sub"],
        "email":     payload.get("email", ""),
        "name":      payload.get("name", ""),
        "picture":   payload.get("picture", ""),
        "tier":      tier,
    })


@app.post("/create-checkout")
async def create_checkout(
    request: Request,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    """Create a Stripe Checkout session. plan: 'pro' ($9/mo) or 'byok' ($3/mo)."""
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Payments not configured")
    import stripe as _stripe

    user = _get_current_user(authorization)
    google_id = user["sub"]

    body = await request.json()
    origin = body.get("origin", "http://localhost:8080")
    plan   = body.get("plan", "pro")  # 'pro' or 'byok'

    price_id = STRIPE_BYOK_PRICE_ID if plan == "byok" else STRIPE_PRICE_ID
    if not price_id:
        raise HTTPException(status_code=503, detail=f"Stripe price not configured for plan: {plan}")

    plan_name = "CADVERT BYOK" if plan == "byok" else "CADVERT Pro"

    # Get or create Stripe customer
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute(
        "SELECT stripe_customer_id FROM users WHERE google_id = ?", (google_id,)
    ).fetchone()
    conn.close()
    customer_id = row[0] if row and row[0] else None

    if not customer_id:
        customer = _stripe.Customer.create(
            email=user.get("email", ""),
            metadata={"google_id": google_id},
        )
        customer_id = customer.id
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "UPDATE users SET stripe_customer_id = ? WHERE google_id = ?",
            (customer_id, google_id),
        )
        conn.commit()
        conn.close()

    session = _stripe.checkout.Session.create(
        customer=customer_id,
        mode="subscription",
        payment_method_types=["card"],
        line_items=[{"price": price_id, "quantity": 1}],
        subscription_data={
            "trial_period_days": TRIAL_DAYS,
            "metadata": {"plan": plan},
            "add_invoice_items": [{
                "price_data": {
                    "currency": "usd",
                    "product_data": {"name": f"{plan_name} — 3-day trial"},
                    "unit_amount": TRIAL_AMOUNT_CENTS,
                },
            }],
        },
        success_url=f"{origin}?upgraded={plan}",
        cancel_url=f"{origin}?upgraded=0",
    )
    return JSONResponse({"checkout_url": session.url})


@app.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events to update user tiers."""
    import stripe as _stripe
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = _stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    etype = event["type"]
    data  = event["data"]["object"]

    if etype == "checkout.session.completed":
        sub_id = data.get("subscription")
        # Retrieve subscription to read the plan metadata
        plan = "pro"
        if sub_id:
            try:
                sub = _stripe.Subscription.retrieve(sub_id)
                plan = sub.get("metadata", {}).get("plan", "pro")
            except Exception:
                pass
        tier = "byok" if plan == "byok" else "pro"
        _set_user_tier_by_customer(data["customer"], tier, sub_id)

    elif etype in ("customer.subscription.deleted", "customer.subscription.paused"):
        _set_user_tier_by_customer(data["customer"], "free", None)

    elif etype == "customer.subscription.updated":
        status = data.get("status", "")
        if status in ("active", "trialing"):
            plan = data.get("metadata", {}).get("plan", "pro")
            tier = "byok" if plan == "byok" else "pro"
        else:
            tier = "free"
        _set_user_tier_by_customer(data["customer"], tier, data.get("id"))

    elif etype == "invoice.payment_failed":
        # Grace — don't downgrade immediately, Stripe will retry
        pass

    return JSONResponse({"ok": True})


@app.post("/convert")
async def convert(
    request: Request,
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    """Upload a CAD file and run the full CADVERT pipeline."""
    user = _get_current_user(authorization)
    if not DISABLE_AUTH:
        _check_file_limit_user(user["sub"])

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

    result["message_count"] = 0
    _sessions[session_id] = result

    images = []
    for img_path_str in result.get("image_paths", []):
        p = Path(img_path_str)
        if p.exists():
            rel = p.relative_to(UPLOAD_DIR)
            images.append({"name": p.stem, "url": f"/sessions/{rel.as_posix()}"})

    return JSONResponse({
        "session_id": session_id,
        "hsd":        result["hsd"],
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
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    x_openai_key: Optional[str] = Header(default=None, alias="X-OpenAI-Key"),
    x_model: str = Header(default="gpt-4o", alias="X-Model"),
):
    """Stream a GPT chat response using Tier 0 context + tool calling.

    Body JSON: { "messages": [{role, content}, ...] }

    Tiering:
      - First FREE_MESSAGES_PER_SESSION messages: server key, no X-OpenAI-Key needed
      - After that: X-OpenAI-Key required (BYOK), up to BYOK_MESSAGES_PER_SESSION
    """
    user = None if DISABLE_AUTH else _get_current_user(authorization)
    user_tier = "free"
    if user and not DISABLE_AUTH:
        user_tier = _get_user_tier(user["sub"])

    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    msg_count = session.get("message_count", 0)

    # Decide which key to use based on tier
    if user_tier == "pro":
        # Pro: server pays, no cap
        if not SERVER_OPENAI_KEY:
            raise HTTPException(status_code=503, detail="Server API key not configured.")
        api_key = SERVER_OPENAI_KEY
    elif user_tier == "byok":
        # BYOK paid tier: their key required, but no message cap
        if not x_openai_key:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "byok_key_required",
                    "message": "Enter your OpenAI API key to continue (included in your BYOK plan).",
                },
            )
        api_key = x_openai_key
    elif msg_count < FREE_MESSAGES_PER_SESSION:
        # Free tier: server pays first N messages
        if not SERVER_OPENAI_KEY:
            raise HTTPException(
                status_code=503,
                detail="Server API key not configured. Please provide your own OpenAI key.",
            )
        api_key = SERVER_OPENAI_KEY
    else:
        # Free tier exhausted — show upgrade wall
        if not x_openai_key:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "byok_required",
                    "message": "You've used your 3 free messages. Upgrade to continue.",
                    "messages_used": msg_count,
                },
            )
        api_key = x_openai_key
        # Cap BYOK sessions too
        if BYOK_MESSAGES_PER_SESSION > 0 and msg_count >= FREE_MESSAGES_PER_SESSION + BYOK_MESSAGES_PER_SESSION:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "session_limit",
                    "message": "Session message limit reached. Start a new session by uploading a file.",
                    "messages_used": msg_count,
                },
            )

    body = await request.json()
    user_messages: list[dict] = body.get("messages", [])

    # Increment before the call so concurrent requests don't double-dip free quota
    session["message_count"] = msg_count + 1

    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise HTTPException(status_code=500, detail="openai package not installed")

    client = AsyncOpenAI(api_key=api_key)
    tier0  = session.get("tier0") or session.get("hsd", "")

    system_msg = (
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

    openai_messages = [{"role": "system", "content": system_msg}] + user_messages
    is_mesh = session.get("is_mesh", False)
    tools = CADVERT_TOOLS if not is_mesh else []

    async def stream_response():
        try:
            # Stream the initial response — detect tool calls mid-stream
            stream = await client.chat.completions.create(
                model=x_model,
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
            while finish_reason == "tool_calls" and accumulated_tool_calls:
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
                    model=x_model,
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

    return StreamingResponse(stream_response(), media_type="text/event-stream")


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

    try:
        if tool_name == "get_feature":
            return _tool_get_feature(args.get("feature_id", ""), features, feature_ids, graph, units)
        elif tool_name == "get_face":
            return _tool_get_face(args.get("face_id", ""), graph, units)
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
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    except Exception as exc:
        return {"error": f"Tool error: {exc}"}


def _parse_fid(s: str) -> int:
    return int(str(s).lstrip("Ff"))

def _parse_eid(s: str) -> int:
    return int(str(s).lstrip("Ee"))


def _tool_get_feature(feature_id, features, feature_ids, graph, units) -> dict:
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
                    "id":        f"F{fid}",
                    "geometry":  face.geometry,
                    "area":      round(face.area, 4),
                    "convexity": face.convexity,
                })

        result["edges"] = []
        for eid in feat.edge_ids[:12]:
            edge = edge_by_id.get(eid)
            if edge:
                result["edges"].append({
                    "id":            f"E{eid}",
                    "geometry":      edge.geometry,
                    "length":        round(edge.length, 4) if edge.length else None,
                    "connects":      [f"F{x}" for x in edge.face_ids],
                    "dihedral_angle": round(edge.dihedral_angle, 3) if edge.dihedral_angle else None,
                })

    result["units"] = units
    return result


def _tool_get_face(face_id, graph, units) -> dict:
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
                "length":        round(edge.length, 4) if edge.length else None,
                "connects_to":   other,
                "dihedral_angle": round(edge.dihedral_angle, 3) if edge.dihedral_angle else None,
                "convexity":     edge.convexity,
            })

    return {
        "id":        f"F{fid}",
        "geometry":  face.geometry,
        "area":      round(face.area, 4),
        "convexity": face.convexity,
        "edges":     edges,
        "units":     units,
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
        "length":        round(edge.length, 4) if edge.length else None,
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
                                "convexity":      nbr.convexity,
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
