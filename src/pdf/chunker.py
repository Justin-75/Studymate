# src/pdf/chunker.py
"""Sentence-aware chunking for StudyMate.

1. Language, once per PDF: sample 5 text pages. A page is Chinese if 汉字 are at least
   10% of its letters (汉字 + A-Z); digits, punctuation and spaces don't count. Majority vote.
2. Sentences, per page: en -> Moses punctuation normalizer + Moses sentence splitter
                        zh -> split after 。！？； (closing quotes stay attached)
3. Words, counted like a human: en = text.split(), zh = jieba (我们去图书馆 = 3).
   Punctuation is not a word.
4. Pack whole sentences up to 512 words, carry 1 sentence over as overlap.
   No hard cut: a sentence longer than 512 words becomes its own chunk.
5. Parent-child: each chunk is a parent, cut into the fewest children of up to CHILD_WORDS words (near-equal
   parts at sentence boundaries). With an overlap (e.g. 150 words, 40 overlap = child set "150o40") each
   child also starts ~40 words before the previous one ends, so a sentence at a cut is in both.
   Only the children are indexed and searched; the LLM reads their parents.

Chunks never cross pages, so every chunk cites exactly one page.
Moses runs as Perl subprocesses; Perl comes from conda-forge (environment.yml).
"""
from __future__ import annotations

import math
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import jieba
from mosestokenizer import MosesPunctuationNormalizer, MosesSentenceSplitter

from src.db.models import ChildChunk, Chunk, child_unit

jieba.setLogLevel(60)   # hide jieba's "Building prefix dict..." messages

MAX_WORDS = 512
CHILD_WORDS = 150      # parent-child retrieval: words per child chunk (the parents are MAX_WORDS)
CHILD_OVERLAP = 0      # words a child shares with the previous child of its parent (0 = none). Benchmarked with the
                       # reranker: 150 no overlap 0.923 MRR > 150o15 0.918, 160o20 0.917, 150o20 0.913, 160o40 0.915
OVERLAP = 1            # sentences carried into the next chunk
ZH_CHAR_RATIO = 0.10   # 汉字 / (汉字 + A-Z letters) at or above this -> Chinese page
SAMPLE_PAGES = 5
MIN_PAGE_CHARS = 50    # pages shorter than this (page numbers, blank pages) are not sampled


class NoTextLayerError(ValueError):
    """No page has extractable text, so the language can't be detected. Run OCR first."""


# ---------- 1. language detection ----------

_HAN = re.compile(r"[\u3400-\u9fff]")   # 汉字
_LATIN = re.compile(r"[A-Za-z]")


def page_is_zh(text: str) -> bool:
    # English pages score 0; chi_sim OCR of an English page about 0.01 (stray 汉字 on math symbols);
    # Chinese pages, table-of-contents pages included, 0.1 and up
    han, latin = len(_HAN.findall(text)), len(_LATIN.findall(text))
    return han > 0 and han / (han + latin) >= ZH_CHAR_RATIO


def detect_lang(page_texts: List[str], seed: int = 0) -> str:
    """'zh' or 'en' for the whole document. Seeded, so re-chunking gives the same answer."""
    pages = [t for t in page_texts if len(t.strip()) >= MIN_PAGE_CHARS] \
        or [t for t in page_texts if t.strip()]
    if not pages:
        raise NoTextLayerError("no extractable text on any page; OCR it, or pass language=")
    sample = random.Random(seed).sample(pages, min(SAMPLE_PAGES, len(pages)))
    zh_votes = sum(page_is_zh(t) for t in sample)
    return "zh" if zh_votes > len(sample) / 2 else "en"   # a tie goes to 'en'


# ---------- Moses (Perl subprocesses) ----------

def _find_perl() -> str:
    """
    The env's own Perl first: conda-forge puts it in <env>/bin, which is on PATH only after
    `conda activate`. On Windows that copy sits next to its perl5xx.dll (the Library/bin one does not),
    and Git for Windows' Perl, often on PATH, can't find Moses' data files.
    """
    env_bin = os.pathsep.join(str(Path(sys.prefix) / d) for d in ("bin", "Library/bin"))
    return shutil.which("perl", path=env_bin) or shutil.which("perl") or "perl"


class _MosesTool:
    """Runs a Moses wrapper with _find_perl(), and raises instead of hanging when Perl exits."""

    def start(self):
        self.argv = [_find_perl(), *self.argv[1:]]
        self.stdbuf = False     # the scripts get -b (no output buffering) already
        super().start()

    def readline(self):
        line = self.stdout.readline()
        if not line:            # EOF: toolwrapper would keep reading "" forever
            raise RuntimeError(f"Moses stopped (is Perl installed?): {' '.join(self.argv)}")
        return line.rstrip("\n")


class _Normalizer(_MosesTool, MosesPunctuationNormalizer):
    pass


class _Splitter(_MosesTool, MosesSentenceSplitter):
    pass


# ---------- 2-4. sentences, counting, packing ----------

_ZH_SENT = re.compile(r".+?(?:[。！？；!?;]+[”’」』）)]*|$)", re.S)
_HAS_WORD_CHAR = re.compile(r"\w")             # \w also matches 汉字
_CJK = re.compile(r"[\u3000-\u303f\u3400-\u9fff\uff00-\uffef]")   # 汉字, CJK and full-width punctuation


def _join_zh_lines(lines: List[str]) -> str:
    """Glue Chinese lines with no space, but keep one between two non-Chinese ends ("data\\npoint")."""
    parts = [lines[0]]
    for prev, line in zip(lines, lines[1:]):
        parts.append(line if _CJK.match(prev[-1]) or _CJK.match(line[0]) else " " + line)
    return "".join(parts)


class Chunker:
    """Create once, reuse for every PDF, close at the end (Moses runs as Perl subprocesses)."""

    def __init__(self, max_words: int = MAX_WORDS, overlap: int = OVERLAP):
        self.max_words = max_words
        self.overlap = overlap
        self._norm = None   # Moses starts lazily, only when an English document shows up
        self._split = None

    def close(self):
        for tool in (self._norm, self._split):
            if tool is not None:
                tool.close()
        self._norm = self._split = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _moses(self):
        if self._norm is None:
            self._norm = _Normalizer("en")
            self._split = _Splitter("en")
        return self._norm, self._split

    def sentences(self, text: str, lang: str) -> List[str]:
        out = []
        # text is cleaned already: cleaner.py rejoined "exam-\nple" (letters only, so "3-\ngram" stays)
        for para in re.split(r"\n\s*\n", text):
            lines = [l.strip() for l in para.splitlines() if l.strip()]
            if not lines:
                continue
            if lang == "zh":
                # PDF lines break mid-sentence; Chinese lines are glued with no space
                out += [s.strip() for s in _ZH_SENT.findall(_join_zh_lines(lines)) if s.strip()]
            else:
                norm, split = self._moses()
                lines = [norm(l) for l in lines]   # one line at a time: the Moses wrapper is line-based
                out += [s for s in split([l for l in lines if l.strip()]) if s.strip()]
        return out

    @staticmethod
    def count_words(sentence: str, lang: str) -> int:
        if lang == "zh":
            return sum(1 for w in jieba.lcut(sentence) if _HAS_WORD_CHAR.search(w))
        return len(sentence.split())

    def pack(self, sentences: List[str], lang: str) -> List[str]:
        sep = "" if lang == "zh" else " "
        chunks, cur, n = [], [], 0          # cur holds (sentence, word_count)
        for s in sentences:
            c = self.count_words(s, lang)
            if cur and n + c > self.max_words:
                chunks.append(sep.join(x for x, _ in cur))
                cur = cur[-self.overlap:] if self.overlap else []
                n = sum(k for _, k in cur)
                if cur and n + c > self.max_words:   # the overlap alone would overflow
                    cur, n = [], 0
            cur.append((s, c))
            n += c
        if cur:
            chunks.append(sep.join(x for x, _ in cur))
        return chunks

    def chunk_pages(self, doc_id: str, pages: List[Tuple[int, str]], language: Optional[str] = None) -> List[Chunk]:
        """pages: [(page_no, clean_text)], page_no 1-based. Pass language= to skip detection (e.g. after OCR)."""
        lang = language or detect_lang([t for _, t in pages])
        out = []
        for page_no, text in sorted(pages, key=lambda p: p[0]):
            for i, chunk_text in enumerate(self.pack(self.sentences(text, lang), lang)):
                out.append(Chunk(f"{doc_id}:p{page_no}:c{i}", doc_id, page_no, i, chunk_text))
        return out

    def _parent_sentences(self, text: str, lang: str) -> List[str]:
        """The sentences of a stored chunk: split again but NOT normalized again, so joining them gives the text back."""
        if lang == "zh":
            return [x for x in _ZH_SENT.findall(text) if x.strip()]
        _, split = self._moses()
        return [x for x in split([text.strip()]) if x.strip()] if text.strip() else []

    def children(self, parents: List[Chunk], lang: str, child_words: int = CHILD_WORDS,
                 child_overlap: int = CHILD_OVERLAP) -> List[ChildChunk]:
        """
        Cut every parent chunk into children of at most child_words words: the fewest children that fit,
        of near-equal size, at sentence boundaries (a parent of <= child_words words -> 1 child). Only a single
        sentence longer than child_words makes a longer child (no cut inside a sentence).
        child_overlap=0: no overlap (a 512-word parent -> 4 children of ~128); the children joined in order
            give back the parent text.
        child_overlap>0: each child after the first starts at the sentence boundary closest to child_overlap
            words before the previous child ends, so neighbours share ~child_overlap words (whole sentences;
            none when the sentence at the cut is too long to fit, e.g. a code block).
            n = ceil((total - overlap) / (child_words - overlap)): with 150 words, 40 overlap a 512-word parent
            -> 5 children of ~134; with 160/40 -> 4 of ~158.
        """
        sep = "" if lang == "zh" else " "
        unit = child_unit(child_words, child_overlap)
        out = []
        for p in parents:
            sents = self._parent_sentences(p.text, lang) or [p.text]
            sizes = [self.count_words(x, lang) for x in sents]
            total = sum(sizes)
            if child_overlap and total > child_words:
                n = math.ceil((total - child_overlap) / (child_words - child_overlap))
            else:
                n = math.ceil(total / child_words)
            n = max(1, min(len(sents), n))
            ranges = _cut(sizes, n, child_overlap)
            # child_words is a hard maximum: one more child while a child of 2+ sentences is too long
            # (a single sentence longer than child_words stays whole and doesn't count)
            while n < len(sents) and any(b - a > 1 and sum(sizes[a:b]) > child_words for a, b in ranges):
                n += 1
                ranges = _cut(sizes, n, child_overlap)
            for i, (a, b) in enumerate(ranges):
                group = sents[a:b]
                text = sep.join(x.strip() for x in group) if sep else "".join(group).strip()
                out.append(ChildChunk(f"{p.chunk_id}:w{unit}:k{i}", p.chunk_id, p.doc_id, p.page_no, i,
                                      text, child_words, child_overlap))
        return out


def _cut(sizes: List[int], n: int, overlap: int = 0) -> List[Tuple[int, int]]:
    return _overlap_cut(sizes, n, overlap) if overlap else _balanced_cut(sizes, n)


def _overlap_cut(sizes: List[int], n: int, overlap: int) -> List[Tuple[int, int]]:
    """
    Like _balanced_cut, but range k+1 starts about `overlap` words before range k ends. Every range is
    length = (total + (n - 1) * overlap) / n words long and starts stride = length - overlap after the previous
    one; both cut points snap to the closest sentence boundary. Ranges are non-empty and always move forward.
    """
    if n <= 1:
        return [(0, len(sizes))]
    cum = [0]
    for size in sizes:
        cum.append(cum[-1] + size)          # cum[c] = size of items[:c]
    length = (cum[-1] + (n - 1) * overlap) / n
    stride = length - overlap
    out: List[Tuple[int, int]] = []
    a = 0
    for k in range(n - 1):
        hi = len(sizes) - (n - 1 - k)       # leave at least one item for each later range
        b = min(range(a + 1, hi + 1), key=lambda c: abs(cum[c] - (k * stride + length)))
        out.append((a, b))
        a = min(range(a + 1, b + 1), key=lambda c: abs(cum[c] - (k + 1) * stride))   # next start: the overlap
    out.append((a, len(sizes)))
    return out


def _balanced_cut(sizes: List[int], n: int) -> List[Tuple[int, int]]:
    """
    Cut a run of items with these sizes into n consecutive, non-empty (start, end) ranges whose totals are as
    close to total / n as the boundaries allow.
    """
    if n <= 1:
        return [(0, len(sizes))]
    cum = [0]
    for size in sizes:
        cum.append(cum[-1] + size)          # cum[c] = size of items[:c]
    cuts: List[int] = []
    for k in range(1, n):
        lo, hi = (cuts[-1] if cuts else 0) + 1, len(sizes) - (n - k)    # leave at least one item per later range
        cuts.append(min(range(lo, hi + 1), key=lambda c: abs(cum[c] - cum[-1] * k / n)))
    bounds = [0, *cuts, len(sizes)]
    return list(zip(bounds, bounds[1:]))
