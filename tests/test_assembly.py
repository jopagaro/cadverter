"""Assembly structure: face → named component, quantities, and mass.

These cover the invariants that matter. Face IDs must line up with the topology graph
(they come from the same loaded shape), every face of an assembly should be attributed,
and files with no product structure must degrade quietly rather than inventing one.
"""
from pathlib import Path

import pytest

import cadvert
from cadvert.assembly import AssemblyInfo, ComponentInstance, extract_assembly

SAMPLES = Path(__file__).parent.parent / "samples"
SAMPLE = SAMPLES / "test_block_with_holes.step"


class TestDataModel:
    def test_empty_assembly_is_falsey(self):
        a = AssemblyInfo()
        assert not a
        assert a.instance_count == 0 and a.part_count == 0
        assert a.owner_of_face(1) is None
        assert a.owner_of_faces([1, 2]) is None
        assert a.bill_of_materials() == []

    def test_extract_with_no_shape_is_safe(self):
        assert not extract_assembly(None, None)

    def test_instance_location(self):
        assert ComponentInstance(0, "x").location == "(top level)"
        assert ComponentInstance(0, "x", path=("A", "B")).location == "A / B"

    def test_bom_groups_instances_and_totals_volume(self):
        a = AssemblyInfo(instances=[
            ComponentInstance(0, "Screw", face_ids=[1], volume=10.0),
            ComponentInstance(1, "Screw", face_ids=[2], volume=10.0),
            ComponentInstance(2, "Plate", face_ids=[3], volume=100.0),
        ])
        bom = a.bill_of_materials()
        assert bom[0]["name"] == "Screw" and bom[0]["quantity"] == 2       # qty first
        assert bom[0]["volume_total"] == pytest.approx(20.0)
        assert bom[1]["name"] == "Plate" and bom[1]["quantity"] == 1

    def test_owner_of_faces_uses_majority(self):
        a = AssemblyInfo(
            instances=[ComponentInstance(0, "A"), ComponentInstance(1, "B")],
            face_owner={1: 0, 2: 0, 3: 1},
        )
        assert a.owner_of_faces([1, 2, 3]).name == "A"
        assert a.owner_of_faces([3]).name == "B"
        assert a.owner_of_faces([99]) is None


class TestRealFile:
    @pytest.fixture(scope="class")
    def result(self):
        if not SAMPLE.exists():
            pytest.skip("sample missing")
        return cadvert.analyze(SAMPLE)

    def test_single_part_still_analyses(self, result):
        assert len(result.graph.faces) == 8
        assert len(result.features) == 3

    def test_face_ids_align_with_the_topology_graph(self, result):
        """The whole design rests on this: IDs must come from the same loaded shape."""
        a = result.assembly
        if not a:
            pytest.skip("no structure in this sample")
        graph_ids = {f.id for f in result.graph.faces}
        assert set(a.face_owner) <= graph_ids, "attributed a face the graph does not have"

    def test_mass_properties_needs_structure_or_says_so(self, result):
        m = result.mass_properties()
        assert "error" in m or m["total_mass_g"] > 0

    def test_mass_scales_with_density(self, result):
        if not result.assembly:
            pytest.skip("no structure")
        a = result.mass_properties(density_g_cm3=1.0)["total_mass_g"]
        b = result.mass_properties(density_g_cm3=2.0)["total_mass_g"]
        # Outputs are rounded to 3 dp, so allow a rounding step either way.
        assert b == pytest.approx(a * 2, abs=0.002)

    def test_mass_reports_its_assumptions(self, result):
        m = result.mass_properties()
        if "error" not in m:
            assert m["assumptions"], "a mass figure must carry its assumptions"


class TestGracefulWithoutStructure:
    """Meshes and IGES have no product tree; nothing may pretend otherwise."""

    @pytest.mark.parametrize("stem", ["block.stl", "block.iges"])
    def test_no_invented_structure(self, tmp_path, stem):
        src = SAMPLES / stem
        if not src.exists():
            pytest.skip(f"{stem} not generated")
        r = cadvert.analyze(src)
        assert r.assembly is None or not r.assembly
        assert "ASSEMBLY —" not in r.to_text()
        assert "error" in r.mass_properties()


def test_tool_definitions_include_the_new_tools():
    from cadvert.server import TOOL_NAMES
    assert {"get_component", "compute_mass"} <= TOOL_NAMES


def test_component_tools_handle_a_file_without_structure():
    from cadvert.server import _execute_tool
    session = {"graph": None, "assembly": None, "units": "mm"}
    assert "error" in _execute_tool(session, "get_component", {"name": "x"})
    assert "error" in _execute_tool(session, "compute_mass", {})
