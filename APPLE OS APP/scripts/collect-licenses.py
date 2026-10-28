#!/usr/bin/env python3
"""Build CADVERT/Resources/Acknowledgements.json: every third-party component the app ships,
with its license text, for the in-app Acknowledgements screen.

Two sources:
  * Licenses/components.json — OpenCASCADE, the native libraries bundled beside it, and the
    pieces compiled into the Python runtime. None of these carry license metadata of their
    own, so they are listed by hand with texts under Licenses/.
  * The bundled engines' site-packages — each Python package's METADATA and license files.

Run after rebuilding or refreshing an engine (`make licenses`). It fails if a library listed
in the manifest is no longer in the engine, so the screen can't silently drift from what ships.
Output is deterministic: an unchanged engine produces a byte-identical file.
"""
from __future__ import annotations

import json
import re
import sys
from email.parser import HeaderParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LICENSES = ROOT / "Licenses"
OUT = ROOT / "CADVERT" / "Resources" / "Acknowledgements.json"
ENGINES = sorted((ROOT / "Engine").glob("engine-*"))

# Shipped engines leave these out (scripts/copy-engine.sh), or they are CADVERT itself.
SKIP = {"pip", "cadvert"}
# Packages whose license file lives in the package rather than the dist-info.
EXTRA_LICENSE = {"cadquery-ocp": ["cadquery_ocp/LICENSE"]}
# Packages whose METADATA omits the license; the label comes from their license file.
LICENSE_LABEL = {"cadquery-ocp": "Apache-2.0"}
# Packages that ship code but whose authors declare no license. Listed honestly, never guessed.
NO_LICENSE = {
    "cadquery-ocp-proxy": "A version-tracking stub (two short files) published alongside cadquery-ocp. "
                          "Its author has not declared a license for it.",
}
LICENSE_NAME = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|COPYRIGHT)", re.I)


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").rstrip() + "\n"


def site_packages(engine: Path) -> Path:
    found = list(engine.glob("lib/python3.*/site-packages"))
    if not found:
        sys.exit(f"collect-licenses: no site-packages in {engine}")
    return found[0]


def python_version(engine: Path) -> str:
    header = next(engine.glob("include/python3.*/patchlevel.h"), None)
    if header:
        m = re.search(r'#define PY_VERSION\s+"([^"]+)"', header.read_text())
        if m:
            return m.group(1)
    return "3"


def license_label(meta) -> str:
    expr = meta.get("License-Expression")
    if expr:
        return expr.strip()
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or []
                   if c.startswith("License ::") and "OSI Approved ::" in c]
    if classifiers:
        return " / ".join(dict.fromkeys(classifiers))
    lic = (meta.get("License") or "").strip()
    if lic and "\n" not in lic and len(lic) <= 60:
        return lic
    return "See license text"


def homepage(meta) -> str:
    urls = {}
    for entry in meta.get_all("Project-URL") or []:
        label, _, url = entry.partition(",")
        urls[label.strip().lower()] = url.strip()
    for key in ("homepage", "home", "source", "source code", "repository", "github", "documentation"):
        if key in urls:
            return urls[key]
    return (meta.get("Home-page") or "").strip()


def ships_code(dist: Path, sp: Path) -> bool:
    """False for metadata-only packages (e.g. version proxies) — nothing of theirs ships."""
    record = dist / "RECORD"
    if not record.exists():
        return True
    for line in record.read_text().splitlines():
        path = line.split(",")[0]
        if path and not path.startswith(dist.name + "/"):
            return True
    return False


def python_packages() -> dict:
    packages: dict[str, dict] = {}
    for engine in ENGINES:
        sp = site_packages(engine)
        for dist in sorted(sp.glob("*.dist-info")):
            meta = HeaderParser().parsestr((dist / "METADATA").read_text(encoding="utf-8", errors="replace"))
            name = meta.get("Name", dist.name.split("-")[0]).strip()
            if norm(name) in SKIP or not ships_code(dist, sp):
                continue
            files = [p for p in sorted(dist.rglob("*")) if p.is_file() and
                     (LICENSE_NAME.match(p.name) or "licenses" in p.relative_to(dist).parts[:-1])]
            files += [sp / extra for extra in EXTRA_LICENSE.get(norm(name), []) if (sp / extra).exists()]
            entry = packages.setdefault(norm(name), {
                "name": name, "version": meta.get("Version", "").strip(),
                "license": license_label(meta), "url": homepage(meta),
                "group": "Python packages", "texts": [],
            })
            for f in files:
                body = read(f)
                if all(t["body"] != body for t in entry["texts"]):
                    title = str(f.relative_to(dist)) if dist in f.parents else f.name
                    entry["texts"].append({"title": title.removeprefix("licenses/"), "body": body})
    for name, entry in packages.items():
        if name in LICENSE_LABEL:
            entry["license"] = LICENSE_LABEL[name]
        if name in NO_LICENSE:
            entry["license"] = "Not declared"
            entry["notice"] = NO_LICENSE[name]
        elif not entry["texts"]:
            sys.exit(f"collect-licenses: {name} ships code but no license text — add it to EXTRA_LICENSE")
    return packages


def manifest_components() -> tuple[list[str], list[dict]]:
    manifest = json.loads((LICENSES / "components.json").read_text())
    reference = ENGINES[0]
    sp = site_packages(reference)
    out = []
    for c in manifest["components"]:
        for f in c.get("files", []):
            for engine in ENGINES:
                if not (site_packages(engine) / f).exists():
                    sys.exit(f"collect-licenses: {c['name']} lists {f}, which is not in {engine.name}. "
                             "Update Licenses/components.json to match the engine.")
        texts = []
        for t in c.get("texts", []):
            path = reference / t.removeprefix("engine:") if t.startswith("engine:") else LICENSES / t
            if not path.exists():
                sys.exit(f"collect-licenses: missing license text {path}")
            texts.append({"title": path.name, "body": read(path)})
        if "inline" in c:
            texts.append({"title": "Notice", "body": c["inline"].rstrip() + "\n"})
        version = python_version(reference) if c.get("version") == "@python" else c.get("version", "")
        out.append({"name": c["name"], "version": version, "license": c["license"], "url": c.get("url", ""),
                    "group": c["group"], "notice": c.get("notice", ""), "texts": texts})
    return manifest["groups"], out


def main() -> None:
    if not ENGINES:
        sys.exit("collect-licenses: no engine under Engine/ — build one with scripts/bundle-engine.sh")
    groups, components = manifest_components()
    packages = sorted(python_packages().values(), key=lambda p: p["name"].lower())
    components += [{"notice": "", **p} for p in packages]
    order = {g: i for i, g in enumerate(groups)}
    components.sort(key=lambda c: order.get(c["group"], len(order)))
    for i, c in enumerate(components):
        c["id"] = f"{i:03d}-{norm(c['name'])}"
    data = {"groups": groups, "components": components}
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    size = OUT.stat().st_size
    print(f"collect-licenses: {len(components)} components ({len(packages)} Python packages) "
          f"from {', '.join(e.name for e in ENGINES)} → {OUT.relative_to(ROOT)} ({size // 1024} KB)")


if __name__ == "__main__":
    main()
