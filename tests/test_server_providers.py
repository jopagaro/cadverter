"""Provider plumbing in cadvert.server: OpenAI + Anthropic share one tool set and one SSE
vocabulary; /tool exposes the geometry tools to clients that run the LLM themselves."""
import json
import os

import pytest

os.environ.setdefault("DISABLE_AUTH", "1")

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cadvert import server  # noqa: E402


@pytest.fixture
def client():
    return TestClient(server.app)


def test_anthropic_tool_translation():
    tools = server._anthropic_tools()
    names = {t["name"] for t in tools}
    assert names == server.TOOL_NAMES
    for t in tools:
        assert set(t) == {"name", "description", "input_schema"}
        assert t["input_schema"]["type"] == "object"
    by_name = {t["name"]: t for t in tools}
    assert by_name["get_feature"]["input_schema"]["required"] == ["feature_id"]


@pytest.mark.parametrize("hdr,model,expected", [
    (None, "gpt-4o", ("openai", "gpt-4o")),
    (None, "claude-opus-5", ("anthropic", "claude-opus-5")),
    ("anthropic", "gpt-4o", ("anthropic", server.DEFAULT_ANTHROPIC_MODEL)),   # wrong model → provider default
    ("openai", "claude-opus-5", ("openai", server.DEFAULT_OPENAI_MODEL)),
    ("ANTHROPIC", "claude-sonnet-5", ("anthropic", "claude-sonnet-5")),
    ("bogus", "claude-haiku-4-5", ("anthropic", "claude-haiku-4-5")),
    (None, "not-a-model", ("openai", server.DEFAULT_OPENAI_MODEL)),
])
def test_resolve_provider(hdr, model, expected):
    assert server._resolve_provider(hdr, model) == expected


def test_config_lists_providers_and_tools(client):
    cfg = client.get("/config").json()
    assert set(cfg["providers"]) == {"openai", "anthropic"}
    assert cfg["providers"]["anthropic"]["default_model"] == "claude-opus-5"
    assert "get_feature" in cfg["tools"]
    assert isinstance(cfg["providers"]["anthropic"]["available"], bool)
    assert cfg["local_only"] is True


def test_tools_endpoint(client):
    tools = client.get("/tools").json()
    assert {t["name"] for t in tools} == server.TOOL_NAMES
    assert all("parameters" in t for t in tools)


def test_tool_endpoint_routes_to_executor(client, monkeypatch):
    server._sessions["t1"] = {"graph": None, "features": [], "units": "mm", "is_mesh": False}
    calls = []

    def fake_exec(session, name, args):
        calls.append((name, args))
        return {"ok": True, "echo": args}

    monkeypatch.setattr(server, "_execute_tool", fake_exec)
    r = client.post("/tool/t1", json={"name": "get_feature", "arguments": {"feature_id": "hole_1"}})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "echo": {"feature_id": "hole_1"}}
    assert calls == [("get_feature", {"feature_id": "hole_1"})]

    assert client.post("/tool/t1", json={"name": "nope", "arguments": {}}).status_code == 400
    assert client.post("/tool/t1", json={"name": "get_face", "arguments": "F1"}).status_code == 400
    assert client.post("/tool/missing", json={"name": "get_face", "arguments": {}}).status_code == 404
    server._sessions.pop("t1", None)


def test_chat_without_a_key_asks_for_one(client):
    """No accounts, no quotas — the only reason chat stops is a missing key."""
    server._sessions["c1"] = {"message_count": 0, "tier0": "PART: x", "is_mesh": False}
    saved_openai, saved_anthropic = server.SERVER_OPENAI_KEY, server.SERVER_ANTHROPIC_KEY
    server.SERVER_OPENAI_KEY = server.SERVER_ANTHROPIC_KEY = None
    try:
        r = client.post("/chat/c1", json={"messages": [{"role": "user", "content": "hi"}]},
                        headers={"X-Provider": "anthropic", "X-Model": "claude-opus-5"})
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert detail["error"] == "api_key_required"
        assert detail["provider"] == "anthropic"
        assert "Anthropic" in detail["message"]

        # A key in the header is enough; no sign-in, no plan.
        r = client.post("/chat/c1", json={"messages": [{"role": "user", "content": "hi"}]},
                        headers={"X-Provider": "anthropic", "X-Anthropic-Key": "sk-ant-x"})
        assert r.status_code == 200      # streams, then errors inside the stream on a bad key
    finally:
        server.SERVER_OPENAI_KEY, server.SERVER_ANTHROPIC_KEY = saved_openai, saved_anthropic
        server._sessions.pop("c1", None)


def test_config_reports_local_only(client):
    cfg = client.get("/config").json()
    assert cfg["local_only"] is True
    assert set(cfg["providers"]) == {"openai", "anthropic"}
    assert "max_file_mb" in cfg
    # Nothing about accounts, plans or payment survives.
    assert not any(k in cfg for k in ("disable_auth", "stripe_enabled", "stripe_byok_enabled"))


def test_no_account_or_payment_routes_remain(client):
    for path in ("/auth/verify", "/create-checkout", "/stripe-webhook"):
        assert client.post(path, json={}).status_code == 404, f"{path} should be gone"


# ── Anthropic stream translation, with a fake SDK (no network, no key) ─────────

class _Block:
    def __init__(self, **kw): self.__dict__.update(kw)

class _Event:
    def __init__(self, **kw): self.__dict__.update(kw)

class _FakeStream:
    """Mimics `async with client.messages.stream(...) as stream` for one round."""
    def __init__(self, events, final):
        self._events, self._final = events, final
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    def __aiter__(self):
        async def gen():
            for e in self._events:
                yield e
        return gen()
    async def get_final_message(self): return self._final

class _FakeMessages:
    def __init__(self, rounds): self.rounds = list(rounds); self.calls = []
    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return self.rounds.pop(0)

class _FakeAnthropic:
    instances = []
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.messages = _FakeAnthropic.next_messages
        _FakeAnthropic.instances.append(self)


@pytest.mark.asyncio
async def test_stream_anthropic_tool_loop(monkeypatch):
    pytest.importorskip("anthropic")
    import anthropic as real

    # Round 1: Claude asks for a tool. Round 2: Claude answers with text.
    tool_use = _Block(type="tool_use", id="toolu_1", name="get_feature", input={"feature_id": "hole_1"})
    round1 = _FakeStream(
        events=[_Event(type="content_block_start", content_block=tool_use)],
        final=_Block(stop_reason="tool_use", content=[tool_use]),
    )
    round2 = _FakeStream(
        events=[
            _Event(type="content_block_delta", delta=_Block(type="text_delta", text="d=8.000 ")),
            _Event(type="content_block_delta", delta=_Block(type="text_delta", text="mm")),
        ],
        final=_Block(stop_reason="end_turn", content=[_Block(type="text", text="d=8.000 mm")]),
    )
    _FakeAnthropic.next_messages = _FakeMessages([round1, round2])
    monkeypatch.setattr(real, "AsyncAnthropic", _FakeAnthropic)

    executed = []
    monkeypatch.setattr(server, "_execute_tool", lambda s, n, a: executed.append((n, a)) or {"diameter": 8.0})

    out = []
    async for chunk in server._stream_anthropic("sk-ant-test", "claude-opus-5", "SYS",
                                                [{"role": "user", "content": "hole sizes?"}], {}, True):
        out.append(chunk)

    payloads = [json.loads(c[6:]) for c in out if c.startswith("data: ") and c.strip() != "data: [DONE]"]
    assert payloads[0] == {"tool_call": "get_feature"}
    assert "".join(p.get("content", "") for p in payloads) == "d=8.000 mm"
    assert out[-1] == "data: [DONE]\n\n"
    assert executed == [("get_feature", {"feature_id": "hole_1"})]

    # Second request carried the assistant tool_use turn + a tool_result user turn, with tools attached.
    calls = _FakeAnthropic.next_messages.calls
    assert calls[0]["model"] == "claude-opus-5" and calls[0]["system"] == "SYS" and "tools" in calls[0]
    msgs = calls[1]["messages"]
    assert msgs[1]["role"] == "assistant" and msgs[2]["role"] == "user"
    tr = msgs[2]["content"][0]
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "toolu_1" and tr["is_error"] is False
    assert json.loads(tr["content"]) == {"diameter": 8.0}


@pytest.mark.asyncio
async def test_stream_anthropic_refusal(monkeypatch):
    pytest.importorskip("anthropic")
    import anthropic as real
    _FakeAnthropic.next_messages = _FakeMessages([
        _FakeStream(events=[], final=_Block(stop_reason="refusal", content=[])),
    ])
    monkeypatch.setattr(real, "AsyncAnthropic", _FakeAnthropic)
    out = [c async for c in server._stream_anthropic("k", "claude-opus-5", "SYS",
                                                     [{"role": "user", "content": "x"}], {}, False)]
    assert '"error"' in out[-1]


# ── Cache: the sweep must see orphans, not just live sessions ────────────────

def test_sweep_removes_orphans_from_previous_runs(tmp_path, monkeypatch):
    """The original bug: cleanup walked only the in-memory table, so a session left
    behind when the app quit was invisible forever. Five-day-old files survived a
    one-day limit."""
    import time
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(server, "SESSION_TTL_HOURS", 24)

    orphan = tmp_path / "left-by-a-previous-run"
    orphan.mkdir()
    (orphan / "input.step").write_text("x" * 1024)
    import os
    old = time.time() - 200 * 3600
    os.utime(orphan, (old, old))

    fresh = tmp_path / "opened-just-now"
    fresh.mkdir()
    (fresh / "input.step").write_text("y" * 512)

    assert server._sweep_expired(startup=True) == 1
    assert not orphan.exists(), "the orphan should be gone"
    assert fresh.exists(), "a recent part must survive"


def test_cache_usage_reports_what_is_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path)
    (tmp_path / "s1").mkdir()
    (tmp_path / "s1" / "input.step").write_bytes(b"0" * 2048)
    (tmp_path / "s2").mkdir()
    (tmp_path / "s2" / "view.png").write_bytes(b"0" * 1024)

    usage = server._cache_usage()
    assert usage["sessions"] == 2
    assert usage["files"] == 2
    assert usage["bytes"] == 3072
    assert usage["ttl_hours"] == server.SESSION_TTL_HOURS


def test_clear_cache_endpoint_empties_it(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f.png").write_bytes(b"0" * 4096)
    server._sessions["a"] = {"units": "mm"}

    r = client.delete("/cache")
    assert r.status_code == 200
    body = r.json()
    assert body["cleared_sessions"] == 1
    assert body["now"]["sessions"] == 0
    assert not server._sessions
    assert client.get("/cache").json()["sessions"] == 0


def test_data_dir_is_configurable():
    """The desktop app points this at its sandbox Caches directory."""
    import importlib, os
    saved = os.environ.get("CADVERT_DATA_DIR")
    try:
        os.environ["CADVERT_DATA_DIR"] = "/tmp/cadvert-dir-test"
        mod = importlib.reload(server)
        assert str(mod.UPLOAD_DIR) == "/tmp/cadvert-dir-test"
    finally:
        if saved is None:
            os.environ.pop("CADVERT_DATA_DIR", None)
        else:
            os.environ["CADVERT_DATA_DIR"] = saved
        importlib.reload(server)
