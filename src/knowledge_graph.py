# src/knowledge_graph.py
"""Deterministic, query-driven knowledge graph builder.

Design goals:
  - Nodes are *concepts* (not surface tokens like "use"/"method")
  - Edges are typed: prerequisite-of, part-of, causes, enables, used-for, contrasts-with
  - Build a global concept graph from the full document, then filter to the query neighborhood (k-hop)
  - The query anchor node MUST exist and must be central in the focused view
  - Output includes human-readable paths like A → B → C and branching like B → D

This module avoids LLM calls. It uses:
  - conservative definition-pattern concept extraction
  - strict term hygiene (reusing rules from src.learning_items)
  - relation pattern matching on sentences

Everything is deterministic and backed by evidence.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.db.models import Chunk

# Reuse strict hygiene rules from learning_items
from src.learning_items import _is_bad_term, _norm_space, _term_in_text, _query_tokens, _truncate

# Query anchoring (hyphen/space tolerant phrase matching)
from src.query_anchor import build_query_anchor, phrase_in_text, relevance_score


_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+|\n+")

# Definition patterns: capture "TERM is/means/refers to ..."
# Keep conservative to avoid extracting "We" or "This".
_DEF_PATTERNS = [
    re.compile(r"\b([A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,3})\s+(is|are|means)\b", re.IGNORECASE),
    re.compile(r"\b([A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,3})\s+refers\s+to\b", re.IGNORECASE),
    re.compile(r"\bwe\s+(call|define)\s+([A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,3})\b", re.IGNORECASE),
]

# Relation cue patterns (typed edges)
_REL_CUES = {
    "prerequisite-of": ["requires", "depend", "depends", "needed", "must", "before", "in order to"],
    "part-of": ["part of", "consists of", "includes", "composed of"],
    "causes": ["causes", "leads to", "results in", "therefore", "thus"],
    "enables": ["enables", "allows", "lets", "so that"],
    "used-for": ["used to", "used for", "helps"],
    "contrasts-with": ["unlike", "whereas", "in contrast", "as opposed to", "different from"],
}

_ALLOWED_EDGE_TYPES = set(_REL_CUES.keys())


def _split_sentences(text: str) -> List[str]:
    t = _norm_space(text)
    if not t:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(t) if s.strip()]


def _slug(term: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (term or "").lower()).strip("-")
    return s or (term or "").lower().strip() or "concept"


@dataclass
class KGNode:
    id: str
    label: str
    pages: List[int]
    evidence: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "pages": sorted(set(int(p) for p in self.pages if p)),
            "evidence": self.evidence,
        }


@dataclass
class KGEdge:
    source: str
    target: str
    type: str
    weight: int = 1
    pages: List[int] = dataclasses.field(default_factory=list)
    evidence: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "weight": int(self.weight),
            "pages": sorted(set(int(p) for p in self.pages if p)),
            "evidence": self.evidence[:3],
        }


def _page_texts_from_chunks(chunks: Sequence[Chunk]) -> Dict[int, str]:
    """Concatenate chunk texts per page in stable order."""
    pages: Dict[int, List[Tuple[int, str]]] = {}
    for c in chunks:
        pages.setdefault(int(c.page_no), []).append((int(c.chunk_index), c.text or ""))
    out: Dict[int, str] = {}
    for p, items in sorted(pages.items(), key=lambda kv: kv[0]):
        items_sorted = sorted(items, key=lambda t: t[0])
        joined = " ".join(_norm_space(t) for _, t in items_sorted if _norm_space(t))
        out[p] = joined
    return out


def extract_concepts_from_document(
    chunks: Sequence[Chunk],
    query: str = "",
    max_nodes: int = 80,
) -> Dict[str, KGNode]:
    """Extract key concepts using definitional patterns + hygiene filters."""
    page_texts = _page_texts_from_chunks(chunks)
    qtok = _query_tokens(query)

    # term -> pages, best evidence sentence
    stats: Dict[str, Dict[str, Any]] = {}

    for page_no, text in page_texts.items():
        for sent in _split_sentences(text):
            sl = sent.lower()

            for pat in _DEF_PATTERNS:
                m = pat.search(sent)
                if not m:
                    continue

                # pattern variants: either term in group1 or group2
                term = ""
                if m.lastindex and m.lastindex >= 2 and "we " in pat.pattern:
                    term = m.group(2)
                else:
                    term = m.group(1)

                term = _norm_space(term)
                if _is_bad_term(term, query=query):
                    continue

                tl = term.lower()
                st = stats.setdefault(tl, {"term": term, "pages": set(), "evidence": None, "def_count": 0})
                st["pages"].add(int(page_no))
                st["def_count"] += 1

                # prefer evidence sentences that mention query tokens
                if st["evidence"] is None:
                    st["evidence"] = sent
                else:
                    cur = str(st["evidence"])
                    cur_score = sum(1 for w in qtok if w in cur.lower())
                    new_score = sum(1 for w in qtok if w in sl)
                    if new_score > cur_score:
                        st["evidence"] = sent

    # If no definition patterns found, fall back to query token concepts (limited)
    if not stats and query:
        for w in sorted(qtok):
            if _is_bad_term(w, query=query):
                continue
            stats[w] = {"term": w, "pages": set(), "evidence": None, "def_count": 0}

    # Score and select
    scored: List[Tuple[float, str, Dict[str, Any]]] = []
    for tl, st in stats.items():
        pages = set(st.get("pages", set()) or set())
        def_count = int(st.get("def_count", 0) or 0)
        term = st.get("term", tl)

        score = 0.0
        score += 3.0 * min(3, def_count)
        score += 1.0 * min(5, len(pages))
        if any(qw in tl for qw in qtok):
            score += 4.0

        scored.append((score, tl, st))

    scored.sort(key=lambda x: (-x[0], x[1]))

    nodes: Dict[str, KGNode] = {}
    for score, tl, st in scored[:max_nodes]:
        term = str(st.get("term", tl))
        node_id = _slug(term)
        ev = st.get("evidence", None)
        if ev:
            ev = _truncate(str(ev), 200)
        nodes[node_id] = KGNode(
            id=node_id,
            label=term,
            pages=sorted(set(int(p) for p in st.get("pages", set()) or set())),
            evidence=ev,
        )

    return nodes


def _infer_relation_type(sentence: str) -> Optional[str]:
    sl = sentence.lower()
    for typ, cues in _REL_CUES.items():
        for cue in cues:
            if cue in sl:
                return typ
    return None


def _ensure_anchor_node(
    nodes: Dict[str, KGNode],
    page_texts: Dict[int, str],
    query: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Ensure the query node exists.

    Returns (anchor_id, anchor_label). Both may be None if query is empty.
    """
    q = (query or "").strip()
    if not q:
        return None, None

    anchor = build_query_anchor(q, text_pool="")
    label = (anchor.anchor_phrase or q).strip()
    if not label:
        return None, None

    anchor_id = _slug(label)

    if anchor_id in nodes:
        return anchor_id, nodes[anchor_id].label

    # Find pages and evidence for the anchor phrase.
    pages: List[int] = []
    best_evidence = None
    best_score = -1

    for pno, text in page_texts.items():
        if not text:
            continue

        # Treat a page as relevant if it contains the anchor phrase or enough query tokens.
        rel = relevance_score(text, anchor)
        if phrase_in_text(label, text) or rel >= 2:
            pages.append(int(pno))

            # find best supporting sentence
            for sent in _split_sentences(text):
                if not sent:
                    continue
                srel = relevance_score(sent, anchor)
                if phrase_in_text(label, sent):
                    srel += 10
                if srel > best_score:
                    best_score = srel
                    best_evidence = sent

    nodes[anchor_id] = KGNode(
        id=anchor_id,
        label=label,
        pages=sorted(set(pages)),
        evidence=_truncate(str(best_evidence), 200) if best_evidence else None,
    )

    return anchor_id, label


def build_document_graph(
    doc_id: str,
    chunks: Sequence[Chunk],
    query: str = "",
    max_nodes: int = 60,
    max_edges: int = 200,
    focus_depth: int = 2,
) -> Dict[str, Any]:
    """Build a document-level knowledge graph.

    Output keys:
      - doc_id
      - query (if provided)
      - nodes: [{id,label,pages,evidence}]
      - edges: [{source,target,type,weight,pages,evidence}]
      - paths: [{for, type, path:[node_ids], path_str:"A → B → C"}]
      - focus: {anchor_nodes:[...], depth:int}
    """

    # Global extraction
    nodes = extract_concepts_from_document(chunks, query=query, max_nodes=max(80, max_nodes * 2))
    if not nodes:
        return {"doc_id": doc_id, "nodes": [], "edges": [], "paths": []}

    # Build page texts once
    page_texts = _page_texts_from_chunks(chunks)

    # Ensure query node exists (contract)
    anchor_id, anchor_label = _ensure_anchor_node(nodes, page_texts, query=query)

    # Build quick lookup by label and id
    id_to_label = {nid: n.label for nid, n in nodes.items()}
    labels = list(id_to_label.items())  # (id,label)

    # Build edges by scanning sentences and matching concepts
    edges_map: Dict[Tuple[str, str, str], KGEdge] = {}

    for page_no, text in page_texts.items():
        for sent in _split_sentences(text):
            rel_type = _infer_relation_type(sent)
            if rel_type is None:
                continue

            present: List[str] = []
            for nid, label in labels:
                if _term_in_text(label, sent):
                    present.append(nid)

            if len(present) < 2:
                continue

            # Create edges between pairs in sentence
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    a = present[i]
                    b = present[j]

                    src, tgt = a, b
                    if rel_type == "contrasts-with":
                        src, tgt = sorted([a, b])
                    elif rel_type == "prerequisite-of":
                        # If "requires/depends on/needed" appears, assume earlier depends on later => later prerequisite-of earlier
                        if any(cue in sent.lower() for cue in ["requires", "depends", "depend", "needed", "must"]):
                            src, tgt = b, a
                    elif rel_type == "part-of":
                        # "A includes B" => B part-of A
                        if any(cue in sent.lower() for cue in ["includes", "consists of", "composed of"]):
                            src, tgt = b, a
                    else:
                        # causes/enables/used-for: assume left-to-right
                        src, tgt = a, b

                    if rel_type not in _ALLOWED_EDGE_TYPES:
                        continue

                    key = (src, tgt, rel_type)
                    e = edges_map.get(key)
                    if e is None:
                        e = KGEdge(source=src, target=tgt, type=rel_type, weight=0, pages=[], evidence=[])
                        edges_map[key] = e
                    e.weight += 1
                    e.pages.append(int(page_no))
                    if len(e.evidence) < 3:
                        e.evidence.append(_truncate(sent, 200))

    edges = list(edges_map.values())
    edges.sort(key=lambda e: (-e.weight, e.type, e.source, e.target))
    edges = edges[:max_edges]

    # Reduce nodes to those participating in edges
    used_nodes: Set[str] = set()
    for e in edges:
        used_nodes.add(e.source)
        used_nodes.add(e.target)

    # Always keep anchor node if query provided
    if anchor_id:
        used_nodes.add(anchor_id)

    # If graph became empty, fall back to node-only result (but keep anchor)
    if not edges:
        used_nodes = set(list(nodes.keys())[:max_nodes])
        if anchor_id:
            used_nodes.add(anchor_id)

    node_list = [nodes[nid] for nid in used_nodes if nid in nodes]
    node_list.sort(key=lambda n: (-len(n.pages), n.label.lower()))

    # Ensure anchor survives max_nodes truncation
    if len(node_list) > max_nodes and anchor_id:
        keep: List[KGNode] = []
        anchor_node = nodes.get(anchor_id)
        if anchor_node:
            keep.append(anchor_node)
        for n in node_list:
            if anchor_node and n.id == anchor_node.id:
                continue
            keep.append(n)
            if len(keep) >= max_nodes:
                break
        node_list = keep
    else:
        node_list = node_list[:max_nodes]

    kept_ids = {n.id for n in node_list}
    edges = [e for e in edges if e.source in kept_ids and e.target in kept_ids]

    # Focus to query neighborhood if query provided
    focus = None
    paths: List[Dict[str, Any]] = []
    if query:
        anchors = _anchor_nodes(kept_ids, id_to_label, query=query, force_anchor_id=anchor_id)
        focus = {"anchor_nodes": anchors, "depth": int(focus_depth)}
        sub_ids = _bfs_neighborhood(kept_ids, edges, anchors, depth=focus_depth)
        node_list = [n for n in node_list if n.id in sub_ids]
        edges = [e for e in edges if e.source in sub_ids and e.target in sub_ids]
        paths = _build_paths(edges, anchors, id_to_label)

    return {
        "doc_id": doc_id,
        "query": query or None,
        "nodes": [n.to_dict() for n in node_list],
        "edges": [e.to_dict() for e in edges],
        "paths": paths,
        "focus": focus,
    }


def _anchor_nodes(node_ids: Set[str], id_to_label: Dict[str, str], query: str, force_anchor_id: Optional[str] = None) -> List[str]:
    qtok = _query_tokens(query)
    anchors: List[str] = []

    if force_anchor_id and force_anchor_id in node_ids:
        anchors.append(force_anchor_id)

    ql = query.lower()
    for nid in sorted(node_ids):
        if force_anchor_id and nid == force_anchor_id:
            continue
        label = (id_to_label.get(nid, "") or "").lower()
        if not label:
            continue
        if label in ql:
            anchors.append(nid)
            continue
        if any(qw in label for qw in qtok):
            anchors.append(nid)

    # Dedup, keep at most 5
    seen: Set[str] = set()
    out: List[str] = []
    for a in anchors:
        if a in seen:
            continue
        seen.add(a)
        out.append(a)
        if len(out) >= 5:
            break
    return out


def _bfs_neighborhood(node_ids: Set[str], edges: List[KGEdge], anchors: List[str], depth: int = 2) -> Set[str]:
    if not anchors:
        return set(node_ids)

    adj: Dict[str, Set[str]] = {nid: set() for nid in node_ids}
    for e in edges:
        adj.setdefault(e.source, set()).add(e.target)
        adj.setdefault(e.target, set()).add(e.source)

    visited: Set[str] = set()
    frontier: Set[str] = set(anchors)
    visited |= frontier

    for _ in range(max(1, depth)):
        nxt: Set[str] = set()
        for u in frontier:
            for v in adj.get(u, set()):
                if v not in visited:
                    visited.add(v)
                    nxt.add(v)
        frontier = nxt
        if not frontier:
            break

    return visited


def _build_paths(edges: List[KGEdge], anchors: List[str], id_to_label: Dict[str, str]) -> List[Dict[str, Any]]:
    """Build a small set of example paths for UI/debug.

    We include:
      - prerequisite chains (if present)
      - a few shortest paths from anchors to other nodes

    Each path also includes a human-readable `path_str` using node labels.
    """

    paths: List[Dict[str, Any]] = []

    # 1) Prerequisite chains
    prereq_adj: Dict[str, Set[str]] = {}
    for e in edges:
        if e.type != "prerequisite-of":
            continue
        prereq_adj.setdefault(e.target, set()).add(e.source)  # source prerequisite for target

    for a in anchors:
        chain = [a]
        cur = a
        for _ in range(3):
            preds = sorted(list(prereq_adj.get(cur, set())))
            if not preds:
                break
            cur = preds[0]
            chain.append(cur)
        if len(chain) >= 2:
            chain2 = list(reversed(chain))
            paths.append(
                {
                    "for": a,
                    "type": "prerequisites",
                    "path": chain2,
                    "path_str": " → ".join(id_to_label.get(x, x) for x in chain2),
                }
            )

    # 2) General shortest paths from anchors (up to depth 3)
    # Build undirected adjacency for exploration
    adj: Dict[str, Set[str]] = {}
    for e in edges:
        adj.setdefault(e.source, set()).add(e.target)
        adj.setdefault(e.target, set()).add(e.source)

    def bfs_paths(start: str, max_depth: int = 3) -> List[List[str]]:
        out_paths: List[List[str]] = []
        queue: List[Tuple[str, List[str]]] = [(start, [start])]
        seen: Set[str] = {start}

        while queue:
            node, path = queue.pop(0)
            if len(path) - 1 >= max_depth:
                continue
            for nb in sorted(adj.get(node, set())):
                if nb in path:
                    continue
                npath = path + [nb]
                # capture 3-hop-ish paths
                if len(npath) >= 3:
                    out_paths.append(npath)
                queue.append((nb, npath))

        return out_paths

    for a in anchors[:2]:
        for p in bfs_paths(a, max_depth=3)[:4]:
            paths.append(
                {
                    "for": a,
                    "type": "example",
                    "path": p,
                    "path_str": " → ".join(id_to_label.get(x, x) for x in p),
                }
            )

        # also include a 1-hop branch example: a → neighbor
        for nb in sorted(adj.get(a, set()))[:4]:
            paths.append(
                {
                    "for": a,
                    "type": "neighbor",
                    "path": [a, nb],
                    "path_str": f"{id_to_label.get(a,a)} → {id_to_label.get(nb,nb)}",
                }
            )

    # Deduplicate paths by node-id sequence
    seen_seq: Set[str] = set()
    uniq: List[Dict[str, Any]] = []
    for p in paths:
        seq = ",".join(p.get("path", []) or [])
        if not seq or seq in seen_seq:
            continue
        seen_seq.add(seq)
        uniq.append(p)

    return uniq[:12]
