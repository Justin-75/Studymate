# src/knowledge_map.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.db.models import Chunk

# NEW: deterministic concept graph with typed edges + query focus
from src.knowledge_graph import build_document_graph


def build_knowledge_map(
    doc_id: str,
    chunks: List[Chunk],
    max_nodes: int = 60,
    max_edges: int = 200,
    query: Optional[str] = None,
    focus_depth: int = 2,
) -> Dict[str, Any]:
    """
    Build a deterministic knowledge graph.

    Improvements vs your previous output:
      - No "the/of/to/use/way/key" nodes (strict hygiene)
      - Concepts come from definitional patterns + doc-wide scoring
      - Edges are typed: prerequisite-of, part-of, causes, enables, used-for, contrasts-with
      - Optional query focus returns only the relevant neighborhood + learning paths

    NOTE:
      - The MCP tool currently calls this without `query`. That's fine: it returns a clean
        full-document map.
      - If you later add `query` to the MCP tool, you will get focused subgraphs and paths automatically.
    """
    return build_document_graph(
        doc_id=doc_id,
        chunks=chunks,
        query=query or "",
        max_nodes=max_nodes,
        max_edges=max_edges,
        focus_depth=focus_depth,
    )


def save_knowledge_map(doc_id: str, km: Dict[str, Any], out_dir: str | Path) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fp = out / f"knowledge_map_{doc_id}.json"
    fp.write_text(json.dumps(km, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(fp)


def load_knowledge_map(doc_id: str, out_dir: str | Path) -> Dict[str, Any] | None:
    fp = Path(out_dir) / f"knowledge_map_{doc_id}.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text(encoding="utf-8"))
