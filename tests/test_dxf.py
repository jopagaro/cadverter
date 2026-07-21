"""Tests for the DXF (2D drawing) pipeline.

A sample drawing is generated with ezdxf, then read back through
``cadvert.analyze`` — verifying the 2D dispatch, entity extraction, and both
output representations against known-exact values.
"""

import json
from pathlib import Path

import pytest

import cadvert

ezdxf = pytest.importorskip("ezdxf")


@pytest.fixture()
def sample_dxf(tmp_path) -> Path:
    """A 100×60 mm plate on named layers with two ⌀8 holes and a note."""
    doc = ezdxf.new(setup=True)
    doc.header["$INSUNITS"] = 4  # mm
    for name in ("OUTLINE", "HOLES", "NOTES"):
        doc.layers.add(name)
    msp = doc.modelspace()
    msp.add_lwpolyline(
        [(0, 0), (100, 0), (100, 60), (0, 60)], close=True,
        dxfattribs={"layer": "OUTLINE"},
    )
    msp.add_circle((25, 30), radius=4, dxfattribs={"layer": "HOLES"})
    msp.add_circle((75, 30), radius=4, dxfattribs={"layer": "HOLES"})
    msp.add_text("BRACKET REV A", dxfattribs={"layer": "NOTES"}).set_placement((10, 70))
    path = tmp_path / "plate.dxf"
    doc.saveas(path)
    return path


def test_analyze_dispatches_to_dxf(sample_dxf):
    r = cadvert.analyze(sample_dxf)
    assert type(r).__name__ == "DxfResult"
    assert r.format == "DXF"
    assert r.units == "mm"
    assert r.is_mesh is False


def test_named_layers_present(sample_dxf):
    r = cadvert.analyze(sample_dxf)
    assert {"OUTLINE", "HOLES", "NOTES"}.issubset(set(r.layers))


def test_entity_counts(sample_dxf):
    counts = cadvert.analyze(sample_dxf).counts
    assert counts.get("CIRCLE") == 2
    assert counts.get("LWPOLYLINE") == 1


def test_holes_have_exact_diameter(sample_dxf):
    r = cadvert.analyze(sample_dxf)
    circles = [e for e in r.entities if e.kind == "CIRCLE"]
    assert len(circles) == 2
    for c in circles:
        assert c.params["diameter"] == pytest.approx(8.0, abs=1e-6)


def test_extents(sample_dxf):
    # Extents enclose every entity — the 100×60 plate plus the note above it,
    # so the Y span legitimately exceeds 60.
    r = cadvert.analyze(sample_dxf)
    assert r.bounds is not None
    assert r.bounds["X"][0] <= 0.0 and r.bounds["X"][1] >= 100.0 - 1e-3
    assert r.bounds["Y"][0] <= 0.0 and r.bounds["Y"][1] >= 60.0 - 1e-3


def test_to_text_mentions_holes_and_note(sample_dxf):
    txt = cadvert.analyze(sample_dxf).to_text()
    assert "DRAWING:" in txt
    assert "8.000" in txt
    assert "BRACKET REV A" in txt


def test_to_json_roundtrips(sample_dxf):
    r = cadvert.analyze(sample_dxf)
    parsed = json.loads(r.to_json())
    assert parsed["format"] == "DXF"
    assert parsed["entity_counts"]["CIRCLE"] == 2
