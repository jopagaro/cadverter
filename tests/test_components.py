"""Assembly component names, project name and authoring system from a STEP header.

These are the strongest plain-language clues about what an assembly *is* — catalog parts
arrive as real order codes — so they go into the LLM summary alongside the geometry.
Fixtures are synthetic: no dependency on any local CAD file.
"""
import pytest

from cadvert.ingest import (
    _decode_step_string,
    _parse_components,
    _parse_originating_system,
    _parse_project,
)

# A header in the shape ST-Developer writes: a wrapped, escaped non-ASCII path.
HEADER = r"""ISO-10303-21;
HEADER;
FILE_NAME(
/* name */
'D:\\Projects\\2026\\Demo-01-Geh\X\E4use M\X2\00FC\X0\hle
 Pr\X2\00FC\X0\fung\\Demo-01-A001.stp',
/* time_stamp */ '2026-01-01T00:00:00+00:00',
/* author */ ('ACME'),
/* organization */ (''),
/* preprocessor_version */ 'ST-DEVELOPER v18.1',
/* originating_system */ 'Autodesk Inventor 2021',
/* authorisation */ '');
FILE_SCHEMA (('AUTOMOTIVE_DESIGN { 1 0 10303 214 3 1 1 }'));
ENDSEC;
DATA;
#1=PRODUCT('Belt S5M-300','Belt S5M-300','',(#2));
#3=PRODUCT('ISO 4762 - M8 x 20','ISO 4762 - M8 x 20','',(#4));
#5=PRODUCT('DIN 625 T1 - 6205 - 25 x 52 x 15','','',(#6));
#7=PRODUCT('belt s5m-300','dup in different case','',(#8));
#9=PRODUCT('Part1','generic, should be skipped','',(#10));
#11=PRODUCT('','empty, should be skipped','',(#12));
#13=PRODUCT('Demo-01-T002','custom part','',(#14));
ENDSEC;
END-ISO-10303-21;
"""


class TestDecoding:
    def test_utf16_run(self):
        assert _decode_step_string(r"V\X2\1EA1\X0\t") == "Vạt"

    def test_single_byte(self):
        assert _decode_step_string(r"M\X\E1y") == "Máy"

    def test_doubled_quote(self):
        assert _decode_step_string("Bob''s part") == "Bob's part"

    def test_plain_text_untouched(self):
        assert _decode_step_string("Belt S5M-300") == "Belt S5M-300"

    def test_malformed_escape_left_alone(self):
        # Odd-length hex run must not raise.
        bad = "\\X2\\ZZZZ\\X0\\"          # a raw string cannot end with a backslash
        assert _decode_step_string(bad) == bad


class TestComponents:
    def test_extracts_names(self):
        got = _parse_components(HEADER)
        assert "Belt S5M-300" in got
        assert "ISO 4762 - M8 x 20" in got
        assert "DIN 625 T1 - 6205 - 25 x 52 x 15" in got
        assert "Demo-01-T002" in got

    def test_deduplicates_case_insensitively(self):
        got = _parse_components(HEADER)
        assert sum(1 for c in got if c.lower() == "belt s5m-300") == 1

    def test_skips_generic_and_empty(self):
        got = _parse_components(HEADER)
        assert "Part1" not in got
        assert "" not in got

    def test_respects_limit(self):
        assert len(_parse_components(HEADER, limit=2)) == 2

    def test_no_products_is_empty(self):
        assert _parse_components("DATA;\nENDSEC;") == []


class TestHeaderFields:
    def test_originating_system_survives_a_wrapped_file_name(self):
        # The old single-line regex failed here, leaving the field empty.
        assert _parse_originating_system(HEADER) == "Autodesk Inventor 2021"

    def test_project_name_decoded_from_the_path(self):
        assert _parse_project(HEADER) == "Demo-01-Gehäuse Mühle Prüfung"

    def test_project_absent_when_path_has_no_folder(self):
        assert _parse_project("FILE_NAME(\n'part.stp',\n'');") == ""

    def test_missing_header_is_empty_not_an_error(self):
        assert _parse_originating_system("") == ""
        assert _parse_project("") == ""


class TestSummaryIntegration:
    """The components block must reach the text the model actually reads."""

    def test_block_rendered_into_tier0(self):
        from cadvert.document import _render_components
        from cadvert.ingest import PartMetadata

        meta = PartMetadata(
            components=["Belt S5M-300", "ISO 4762 - M8 x 20", "WIDGET-001"],
            project="Chamfering Machine",
            originating_system="Autodesk Inventor 2021",
        )
        text = "\n".join(_render_components(meta, limit=40))
        assert "DESIGN: Chamfering Machine" in text
        assert "AUTHORED IN: Autodesk Inventor 2021" in text
        assert "COMPONENTS — 3 named in the file" in text
        # Catalog parts rank above internal drawing numbers.
        assert text.index("Belt S5M-300") < text.index("WIDGET-001")

    def test_cap_keeps_catalog_parts_and_reports_the_remainder(self):
        from cadvert.document import _render_components
        from cadvert.ingest import PartMetadata

        meta = PartMetadata(components=[f"INTERNAL-{i:03d}" for i in range(50)] + ["Belt S5M-300"])
        text = "\n".join(_render_components(meta, limit=10))
        assert "Belt S5M-300" in text, "a catalog part must survive the cap"
        assert "and 41 more" in text

    def test_no_metadata_renders_nothing(self):
        from cadvert.document import _render_components
        assert _render_components(None, limit=40) == []


class TestPublicAPI:
    """`pip install cadvert` users reach this through analyze() → to_dict()."""

    def test_to_dict_carries_components(self):
        from pathlib import Path
        import cadvert
        from cadvert.ingest import PartMetadata

        sample = Path(__file__).parent.parent / "samples" / "test_block_with_holes.step"
        if not sample.exists():
            pytest.skip("sample STEP missing")
        d = cadvert.analyze(sample).to_dict()
        assert "components" in d and isinstance(d["components"], list)
        assert "project" in d

    def test_metadata_defaults_are_safe(self):
        from cadvert.ingest import PartMetadata
        m = PartMetadata()
        assert m.components == [] and m.project == ""
        # Separate instances must not share the same list.
        m.components.append("x")
        assert PartMetadata().components == []
