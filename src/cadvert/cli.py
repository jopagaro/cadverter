"""Command-line entry point: cadvert <file> [-o output.txt]

Supported input formats:
  B-REP (full pipeline): .step .stp .iges .igs .brep
  Mesh  (partial pipeline — global props + views only): .stl .obj
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a CAD file to a Hierarchical Spatial Document (HSD)."
    )
    parser.add_argument("cad_file", help="Input CAD file (.step, .stp, .iges, .igs, .brep, .stl, .obj)")
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: <stem>.hsd.txt)",
        default=None,
    )
    parser.add_argument(
        "--part-name",
        help="Part name override (default: filename stem)",
        default=None,
    )
    parser.add_argument(
        "--no-features", action="store_true",
        help="Skip feature detection",
    )
    parser.add_argument(
        "--no-spatial", action="store_true",
        help="Skip spatial relationships",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Skip image rendering",
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip extraction validation",
    )
    parser.add_argument(
        "--pull-direction", default="0,0,1",
        help="Pull direction for draft/undercut analysis X,Y,Z (default: 0,0,1)",
    )
    parser.add_argument(
        "--nurbs-detail", action="store_true",
        help="Append full NURBS control-point dumps (B-REP only)",
    )
    parser.add_argument(
        "--image-size", default="1200x900",
        help="Render image size WxH (default: 1200x900)",
    )
    args = parser.parse_args()

    cad_path = Path(args.cad_file)
    if not cad_path.exists():
        print(f"ERROR: file not found: {cad_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else cad_path.with_suffix(".hsd.txt")
    img_w, img_h = (int(x) for x in args.image_size.split("x"))
    pull_direction = tuple(float(x) for x in args.pull_direction.split(","))

    # ── 1: Ingest ─────────────────────────────────────────────────────────────
    print(f"Loading {cad_path} ...", file=sys.stderr)
    from .ingest import load_step, IngestError, SUPPORTED_EXTENSIONS
    try:
        shape, body_count, metadata = load_step(cad_path)
    except IngestError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    fmt_line = f"  Format: {metadata.source_format}"
    if metadata.is_mesh:
        fmt_line += f"  (mesh — {metadata.triangle_count:,} triangles, B-REP pipeline skipped)"
    print(fmt_line, file=sys.stderr)
    if metadata.schema:
        print(f"  Schema: {metadata.schema}", file=sys.stderr)
    print(f"  Units:  {metadata.units}", file=sys.stderr)
    if body_count:
        print(f"  Bodies: {body_count}", file=sys.stderr)
    if metadata.gdt_annotations:
        print(f"  GD&T:   {len(metadata.gdt_annotations)} annotations", file=sys.stderr)

    # ── 2: Topology (B-REP only) ──────────────────────────────────────────────
    graph = None
    validation_report_text = None
    features = None
    spatial = None

    if not metadata.is_mesh:
        print("Building topology graph ...", file=sys.stderr)
        from .topology import build_topology
        graph = build_topology(shape, body_count)
        print(
            f"  Faces: {len(graph.faces)}  Edges: {len(graph.edges)}"
            f"  Vertices: {len(graph.vertices)}",
            file=sys.stderr,
        )

        # ── 2b: Validation ────────────────────────────────────────────────────
        if not args.no_validate:
            print("Validating extraction ...", file=sys.stderr)
            try:
                from .validate import validate_extraction, format_validation_report
                val_report = validate_extraction(shape, graph)
                validation_report_text = format_validation_report(val_report)
                status = "PASSED" if val_report.overall_passed else "FAILED"
                print(
                    f"  Validation {status}: "
                    f"{val_report.faces_passed}/{len(val_report.face_results)} faces OK, "
                    f"max deviation {val_report.max_surface_deviation:.2e} mm",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"  Validation skipped: {e}", file=sys.stderr)

        # ── 3: Feature detection ──────────────────────────────────────────────
        if not args.no_features:
            print("Detecting features ...", file=sys.stderr)
            from .features import detect_features
            features = detect_features(graph)
            print(
                f"  Found: {len(features)} features"
                f" ({sum(1 for f in features if f.feature_type != 'PATTERN')} primary,"
                f" {sum(1 for f in features if f.feature_type == 'PATTERN')} patterns)",
                file=sys.stderr,
            )
            for feat in features:
                print(f"    {feat.feature_type}: {feat.notes}", file=sys.stderr)

        # ── 4: Spatial relationships ──────────────────────────────────────────
        if not args.no_spatial and features is not None:
            print("Computing spatial relationships ...", file=sys.stderr)
            from .spatial import compute_spatial_relationships
            spatial = compute_spatial_relationships(
                graph, features,
                shape=shape,
                pull_direction=pull_direction,
            )
            print(f"  Found: {len(spatial)} relationships", file=sys.stderr)

    else:
        print(
            "  Mesh format: skipping topology, feature detection, and spatial analysis.",
            file=sys.stderr,
        )
        print(
            "  To get full analysis, export to STEP from your CAD tool.",
            file=sys.stderr,
        )

    # ── 5: Rendering ──────────────────────────────────────────────────────────
    rendered_views = None
    if not args.no_render:
        print("Rendering views ...", file=sys.stderr)
        try:
            from .renderer import render_shape
            render_dir = output_path.parent / (output_path.stem + "_views")
            rendered_views = render_shape(
                shape, render_dir,
                image_size=(img_w, img_h),
                stem=cad_path.stem,
            )
            print(f"  Written {len(rendered_views)} PNG files to {render_dir}/",
                  file=sys.stderr)
        except Exception as e:
            print(f"  Rendering skipped: {e}", file=sys.stderr)

    # ── Assemble document ─────────────────────────────────────────────────────
    print("Rendering document ...", file=sys.stderr)
    from .document import render_document, render_nurbs_detail
    doc = render_document(
        graph, cad_path,
        part_name=args.part_name,
        features=features,
        spatial=spatial,
        rendered_views=rendered_views,
        validation_report=validation_report_text,
        units=metadata.units,
        gdt_annotations=metadata.gdt_annotations if metadata.gdt_annotations else None,
        mesh_info=_mesh_info(metadata, shape) if metadata.is_mesh else None,
    )

    if args.nurbs_detail and graph is not None:
        nurbs_blocks = []
        for face in graph.faces:
            if face.geometry.get("type") == "NURBS_SURFACE" and "error" not in face.geometry:
                nurbs_blocks.append(render_nurbs_detail(face.id, face.geometry))
        if nurbs_blocks:
            from .document import _section_header
            doc += "\n\n" + "\n".join(_section_header("NURBS DETAIL"))
            doc += "\n\n".join(nurbs_blocks)

    output_path.write_text(doc, encoding="utf-8")
    print(f"Written: {output_path}", file=sys.stderr)
    print(doc)


def _mesh_info(metadata, shape) -> dict:
    info = {
        "format": metadata.source_format,
        "triangle_count": metadata.triangle_count,
        "units": metadata.units,
    }
    # Compute basic properties from the mesh shape via OCC
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
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        info["bbox"] = {"X": (xmin, xmax), "Y": (ymin, ymax), "Z": (zmin, zmax)}
    except Exception:
        pass
    return info


if __name__ == "__main__":
    main()
