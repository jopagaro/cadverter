"""Headless rendering of OCC shapes to PNG using VTK.

Pipeline:
  1. Tessellate OCC shape with BRepMesh_IncrementalMesh
  2. Extract triangles per face, preserving face colors
  3. Build a single vtkPolyData with per-face coloring
  4. Render 8 standard views (front/back/top/bottom/left/right/iso + optional section)
  5. Write PNG files to output directory

View conventions (engineering drawing standard):
  Front  → camera looking in +Y direction (from Y-)
  Back   → camera looking in -Y direction (from Y+)
  Left   → camera looking in +X direction (from X-)
  Right  → camera looking in -X direction (from X+)
  Top    → camera looking in -Z direction (from Z+)
  Bottom → camera looking in +Z direction (from Z-)
  Iso    → camera at (1, -1, 1) normalized (standard isometric)
"""

from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_views(
    step_path: str | Path,
    output_dir: str | Path,
    image_size: tuple[int, int] = (1200, 900),
) -> list[Path]:
    """Tessellate the STEP file and render 8 standard views to PNG.

    Returns list of output PNG paths.
    """
    from .ingest import load_step
    shape, _ = load_step(step_path)
    return render_shape(shape, output_dir, image_size=image_size,
                        stem=Path(step_path).stem)


def render_shape(
    shape,
    output_dir: str | Path,
    image_size: tuple[int, int] = (1200, 900),
    stem: str = "part",
) -> list[Path]:
    """Tessellate *shape* and render standard views."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    verts, tris, colors = _tessellate(shape)
    if len(verts) == 0:
        raise RuntimeError("Tessellation produced no geometry")

    poly_data = _build_vtk_polydata(verts, tris, colors)
    bounds = poly_data.GetBounds()  # (xmin,xmax, ymin,ymax, zmin,zmax)
    center = (
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2,
    )
    diag = math.sqrt(
        (bounds[1] - bounds[0]) ** 2 +
        (bounds[3] - bounds[2]) ** 2 +
        (bounds[5] - bounds[4]) ** 2
    )

    views = [
        ("front",    ( 0, -1,  0), (0, 0, 1)),
        ("back",     ( 0,  1,  0), (0, 0, 1)),
        ("left",     (-1,  0,  0), (0, 0, 1)),
        ("right",    ( 1,  0,  0), (0, 0, 1)),
        ("top",      ( 0,  0,  1), (0, 1, 0)),
        ("bottom",   ( 0,  0, -1), (0, 1, 0)),
        ("iso",      ( 1, -1,  1), (0, 0, 1)),
    ]

    out_paths = []
    for name, cam_dir, cam_up in views:
        path = output_dir / f"{stem}_{name}.png"
        _render_view(poly_data, center, diag, cam_dir, cam_up,
                     path, image_size)
        out_paths.append(path)

    return out_paths


# ---------------------------------------------------------------------------
# Location helper
# ---------------------------------------------------------------------------

def _compose_loc_trsf(loc) -> "gp_Trsf":
    """Walk a (possibly compound) TopLoc_Location and return a single gp_Trsf.

    OCP does not expose loc.IsTopLevel() in all versions, so we walk the
    datum chain manually: result = datum_n^power_n * ... * datum_1^power_1
    """
    from OCP.gp import gp_Trsf
    result = gp_Trsf()  # identity
    if loc.IsIdentity():
        return result
    try:
        current = loc
        while not current.IsIdentity():
            t = current.FirstDatum().Transformation()
            power = current.FirstPower()
            if power < 0:
                t.Invert()
                power = -power
            for _ in range(power):
                result.Multiply(t)
            current = current.NextLocation()
    except Exception:
        pass  # fallback: return identity (nodes stay in local space)
    return result


# ---------------------------------------------------------------------------
# Tessellation
# ---------------------------------------------------------------------------

def _tessellate(
    shape,
    linear_deflection: float = 0.05,
    angular_deflection: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (vertices Nx3, triangles Mx3, colors Mx3 uint8)."""
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Trsf

    BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)

    all_verts: list[np.ndarray] = []
    all_tris: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []

    vert_offset = 0
    face_index = 0

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = face.Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is None:
            exp.Next()
            continue

        n = tri.NbNodes()
        m = tri.NbTriangles()

        # Extract node positions in local (triangulation) space
        pts = np.empty((n, 3), dtype=np.float64)
        for i in range(1, n + 1):
            node = tri.Node(i)
            pts[i - 1] = (node.X(), node.Y(), node.Z())

        # Apply location transform to get global coordinates.
        # loc.IsTopLevel() is not bound in this version of OCP —
        # walk the datum chain manually to compose the gp_Trsf.
        if not loc.IsIdentity():
            from OCP.gp import gp_Pnt, gp_Trsf
            trsf = _compose_loc_trsf(loc)
            for i in range(n):
                p = gp_Pnt(pts[i, 0], pts[i, 1], pts[i, 2])
                p.Transform(trsf)
                pts[i] = (p.X(), p.Y(), p.Z())

        # Extract triangle connectivity
        face_tris = np.empty((m, 3), dtype=np.int64)
        for i in range(1, m + 1):
            n1, n2, n3 = tri.Triangle(i).Get()
            # Respect face orientation
            if face.Orientation().name == "TopAbs_REVERSED":
                face_tris[i - 1] = [n1 - 1 + vert_offset,
                                     n3 - 1 + vert_offset,
                                     n2 - 1 + vert_offset]
            else:
                face_tris[i - 1] = [n1 - 1 + vert_offset,
                                     n2 - 1 + vert_offset,
                                     n3 - 1 + vert_offset]

        # Assign a color per face (cycle through a palette)
        color = _FACE_COLORS[face_index % len(_FACE_COLORS)]
        face_colors = np.tile(color, (m, 1)).astype(np.uint8)

        all_verts.append(pts)
        all_tris.append(face_tris)
        all_colors.append(face_colors)

        vert_offset += n
        face_index += 1
        exp.Next()

    if not all_verts:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64), np.empty((0, 3), dtype=np.uint8)

    verts = np.concatenate(all_verts, axis=0)
    tris = np.concatenate(all_tris, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    return verts, tris, colors


# Muted steel-blue palette for multi-face coloring
_FACE_COLORS = [
    [180, 200, 220],  # light steel blue
    [160, 185, 210],
    [170, 195, 215],
    [155, 180, 205],
    [175, 198, 218],
    [165, 190, 212],
]


# ---------------------------------------------------------------------------
# VTK polydata construction
# ---------------------------------------------------------------------------

def _build_vtk_polydata(
    verts: np.ndarray,
    tris: np.ndarray,
    colors: np.ndarray,
):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(verts.astype(np.float64), deep=True))

    cells = vtk.vtkCellArray()
    # VTK cell array format: [3, v0, v1, v2, 3, v0, v1, v2, ...]
    flat = np.empty(len(tris) * 4, dtype=np.int64)
    flat[0::4] = 3
    flat[1::4] = tris[:, 0]
    flat[2::4] = tris[:, 1]
    flat[3::4] = tris[:, 2]
    cell_arr = vtk.vtkIdTypeArray()
    cell_arr.SetNumberOfValues(len(flat))
    for i, v in enumerate(flat):
        cell_arr.SetValue(i, int(v))
    cells.SetCells(len(tris), cell_arr)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cells)

    # Per-cell colors
    color_arr = numpy_to_vtk(colors.astype(np.uint8), deep=True)
    color_arr.SetName("Colors")
    poly.GetCellData().SetScalars(color_arr)

    # Compute normals for shading
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.SplittingOff()
    normals.Update()

    return normals.GetOutput()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_view(
    poly_data,
    center: tuple,
    diag: float,
    cam_direction: tuple,
    cam_up: tuple,
    out_path: Path,
    image_size: tuple[int, int],
) -> None:
    import vtk

    # Mapper — explicitly select the Colors array so normals don't override it
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    mapper.ScalarVisibilityOn()
    mapper.SetColorModeToDirectScalars()
    mapper.SelectColorArray("Colors")
    mapper.SetScalarModeToUseCellFieldData()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetAmbient(0.3)
    prop.SetDiffuse(0.7)
    prop.SetSpecular(0.2)
    prop.SetSpecularPower(30)

    # Renderer with dark background for contrast
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.12, 0.13, 0.17)
    renderer.AddActor(actor)

    # Camera — set direction then let VTK auto-fit via ResetCamera
    cam_dist = diag * 3.0
    cx, cy, cz = center
    dx, dy, dz = cam_direction
    mag = math.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / mag, dy / mag, dz / mag

    cam = renderer.GetActiveCamera()
    cam.SetFocalPoint(cx, cy, cz)
    cam.SetPosition(cx + dx * cam_dist, cy + dy * cam_dist, cz + dz * cam_dist)
    cam.SetViewUp(*cam_up)
    cam.ParallelProjectionOn()
    renderer.ResetCamera()  # auto-fit geometry within the view

    # Lighting
    light1 = vtk.vtkLight()
    light1.SetPosition(cx + cam_dist, cy - cam_dist, cz + cam_dist)
    light1.SetFocalPoint(cx, cy, cz)
    light1.SetIntensity(0.8)
    renderer.AddLight(light1)

    light2 = vtk.vtkLight()
    light2.SetPosition(cx - cam_dist * 0.5, cy + cam_dist * 0.3, cz + cam_dist * 0.5)
    light2.SetFocalPoint(cx, cy, cz)
    light2.SetIntensity(0.4)
    renderer.AddLight(light2)

    # Render window (offscreen)
    renwin = vtk.vtkRenderWindow()
    renwin.SetOffScreenRendering(1)
    renwin.SetSize(*image_size)
    renwin.AddRenderer(renderer)
    renwin.Render()

    # Write PNG
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(renwin)
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()

    renwin.Finalize()
