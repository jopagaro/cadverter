"""End-to-end tests for the CADVERT pipeline and high-level API.

These lock in the analysis of the bundled sample part
(``samples/test_block_with_holes.step``): a 60×40×20 mm block with two
∅8 mm through holes. The geometric facts asserted here (dimensions, hole
diameter, face/edge counts) come straight from the B-REP and must never
drift — that exactness is CADVERT's entire value proposition.
"""

import json
from pathlib import Path

import pytest

import cadvert
from cadvert import analyze, IngestError

SAMPLE = Path(__file__).parent.parent / "samples" / "test_block_with_holes.step"


@pytest.fixture(scope="module")
def result():
    return analyze(SAMPLE)


# ── Package surface ───────────────────────────────────────────────────────────

def test_version():
    assert cadvert.__version__ == "0.2.0"


def test_top_level_exports():
    assert callable(cadvert.analyze)
    assert hasattr(cadvert, "CadvertResult")
    assert hasattr(cadvert, "SUPPORTED_EXTENSIONS")


# ── Ingest / metadata ─────────────────────────────────────────────────────────

def test_metadata(result):
    assert result.format == "STEP"
    assert result.is_mesh is False
    assert result.units == "mm"
    assert result.body_count == 1


def test_missing_file_raises():
    with pytest.raises(IngestError):
        analyze("does_not_exist.step")


def test_unsupported_format_raises(tmp_path):
    bogus = tmp_path / "part.xyz"
    bogus.write_text("nope")
    with pytest.raises(IngestError):
        analyze(bogus)


# ── Global geometry (exact, from B-REP) ───────────────────────────────────────

def test_global_properties(result):
    g = result.graph
    assert len(g.faces) == 8
    assert len(g.edges) == 18
    bb = g.bounding_box
    assert bb["X"][1] - bb["X"][0] == pytest.approx(60.0, abs=1e-3)
    assert bb["Y"][1] - bb["Y"][0] == pytest.approx(40.0, abs=1e-3)
    assert bb["Z"][1] - bb["Z"][0] == pytest.approx(20.0, abs=1e-3)
    # center of mass sits at the block centroid
    cx, cy, cz = g.center_of_mass
    assert (cx, cy, cz) == pytest.approx((30.0, 20.0, 10.0), abs=1e-3)


# ── Feature detection ─────────────────────────────────────────────────────────

def test_two_through_holes_and_a_pattern(result):
    types = sorted(f.feature_type for f in result.features)
    assert types == ["PATTERN", "THROUGH_HOLE", "THROUGH_HOLE"]


def test_hole_diameter_is_exact(result):
    holes = [f for f in result.features if f.feature_type == "THROUGH_HOLE"]
    assert len(holes) == 2
    for h in holes:
        # diameter stored as 8.000 mm — the authored value, not a measurement
        dia = h.parameters.get("diameter") or 2 * h.parameters.get("radius", 0)
        assert dia == pytest.approx(8.0, abs=1e-3)


# ── Spatial relationships ─────────────────────────────────────────────────────

def test_spatial_relationships_present(result):
    assert len(result.spatial) > 0
    # the two holes are 30 mm apart centre-to-centre
    centres = [r for r in result.spatial if "center" in r.description.lower()]
    assert any(r.value == pytest.approx(30.0, abs=1e-3) for r in centres)


# ── Text representation ───────────────────────────────────────────────────────

def test_to_text(result):
    txt = result.to_text()
    assert isinstance(txt, str)
    assert "PART:" in txt
    assert "test_block_with_holes" in txt
    assert "8.000" in txt          # hole diameter surfaced
    assert len(txt) < 20_000       # tier-0 stays compact


# ── Dict / JSON representation ─────────────────────────────────────────────────

def test_to_dict_shape(result):
    d = result.to_dict()
    assert d["part"] == "test_block_with_holes"
    assert d["format"] == "STEP"
    assert d["global"]["face_count"] == 8
    assert len(d["faces"]) == 8
    assert len(d["edges"]) == 18
    assert len(d["features"]) == 3


def test_to_json_roundtrips(result):
    parsed = json.loads(result.to_json())
    assert parsed["global"]["face_count"] == 8
    # every value must be JSON-native (no tuples / OCC objects leaked)
    assert isinstance(parsed["global"]["center_of_mass"], list)


# ── Graph representation ──────────────────────────────────────────────────────

def test_to_graph():
    nx = pytest.importorskip("networkx")
    g = analyze(SAMPLE).to_graph()
    assert isinstance(g, nx.Graph)
    assert g.number_of_nodes() == 8      # faces
    assert g.number_of_edges() == 18     # shared B-REP edges
    # nodes carry surface type + area
    for _, attrs in g.nodes(data=True):
        assert "type" in attrs and "area" in attrs
    # edges carry convexity + dihedral angle
    for _, _, attrs in g.edges(data=True):
        assert "convexity" in attrs
