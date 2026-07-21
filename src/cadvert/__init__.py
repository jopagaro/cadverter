"""CADVERT — CAD to LLM-readable Hierarchical Spatial Document converter.

Quick start::

    import cadvert
    result = cadvert.analyze("bracket.step")
    print(result.to_text())     # structured text for an LLM prompt
    result.to_dict()            # JSON-safe analysis
    result.to_graph()           # networkx face-adjacency graph
"""

__version__ = "0.3.0"

from .api import analyze, CadvertResult
from .ingest import IngestError, PartMetadata, SUPPORTED_EXTENSIONS

__all__ = [
    "analyze",
    "CadvertResult",
    "IngestError",
    "PartMetadata",
    "SUPPORTED_EXTENSIONS",
    "__version__",
]
