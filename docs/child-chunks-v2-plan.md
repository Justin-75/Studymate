# Plan: child chunks v2 (200 words, 40-word overlap)

Status (9 Oct 2026, night): **the app uses 150-word children with no overlap** (`CHILD_WORDS = 150`, `CHILD_OVERLAP = 0`). Overlap support is in the code: a child set is `(child_words, child_overlap)`, written `150` or `160o40`, so the benchmark can still compare overlaps.

- Hybrid RRF + reranker MRR (eval_out/20261009-212147): 150 **0.9235** (en 0.913, zh 0.938), 150o15 0.918, 160o20 0.917, 150o20 0.913; 160o40 0.915 (earlier run); old 800-char chunks 0.920.
- Hybrid RRF alone (eval_out/20261009-195448): 160o40 0.833, 160 0.825, 150 0.823, 150o40 0.821.
- Not done: the long-sentence split, the Chinese join fix and Phase 2 below.

Original plan (the numbers below come from a prototype run on a copy of `app.db`):

## Why

- **The old 800-char chunks were unfair between languages.** Median 114–130 words per chunk in the English books, ~200 jieba words in the Chinese ones. A Chinese chunk held about 1.6× more content.
- **Counting words is fair.** BGE-M3 needs about 1.4–1.65 tokens per word in both languages (LLMbook 1.43, d2l 1.65, slp3 1.61, clrs 1.58, macro 1.41, investor 1.46). So 200 words is about 280–330 tokens in either language, below the dense `MAX_LENGTH` of 512: nothing is cut off.
- **The current 100-word children are too small for the reranker.** It reads one child of about 82 words, with no overlap, so the question's key words and the answer often end up in different children. Reranker MRR is 0.904, against 0.920 for the old chunks. Giving it the neighbouring children brought it back to 0.9175.
- **Parents stay as they are** (up to 512 words, one page, 1-sentence overlap). Children are for retrieval, parents are what the LLM reads.

## Prototype result (271 answerable questions, page-level)

| setup | hybrid RRF MRR (all / en / zh) | + reranker R@1 / MRR |
|---|---|---|
| old 800-char chunks | 0.822 / 0.835 / 0.803 | 0.871 / 0.920 |
| children 100 words (current) | 0.812 / 0.816 / 0.807 | 0.856 / 0.904 |
| children 200 words, no overlap | 0.805 / 0.791 / 0.826 | 0.867 / 0.918 |
| **children 200 words, 40 overlap** | **0.823 / 0.818 / 0.831** | 0.860 / 0.912 |

- With 200/40, hybrid RRF ties the old chunks overall. The Chinese books gain (0.803 → 0.831) and the English books lose a little (0.835 → 0.818). Both languages now get chunks of the same size, so this is the fairness change showing up in the numbers.
- The 40-word overlap helps the first stage (+0.018 MRR vs no overlap) but not the reranker (−0.007).
- Every difference in this table is within noise. The 95% confidence interval of each row against the old chunks is about ±0.02 to ±0.03 MRR, because there are only 271 questions. Also, the benchmark questions were written from old 800-char chunks, which slightly favours the old boundaries.
- Children produced by the prototype: median 159–183 words, median overlap 37–41 words.

## Design

### Children: sliding windows inside each parent

- Target 200 words, 40-word overlap, cuts only at sentence boundaries.
- Number of children for a parent of W words: 1 if W ≤ 250, otherwise ceil((W − 40) / 160). A full 512-word parent gives 3 children of about 197 words. A 300-word parent gives 2 children of about 170 words.
- Child i ends at the sentence boundary closest to its ideal end. Child i+1 starts at the sentence boundary closest to that end minus 40 words, which creates the overlap. The overlap is made of whole sentences, so it is 0 when the last sentence alone is much longer than 40 words.
- **A sentence longer than 200 words is split at word boundaries.** Code blocks, tables and reference lists in the prototype produced children of up to 651 words (slp3) and 460 words (d2l).
- **Chinese joining:** join two pieces with `""` only when either side is a Chinese character, otherwise with one space. This is the rule `_join_zh_lines` already uses, and it fixes text like `print(data)NumRooms`.

### Why not a fixed 4 children per parent

Four children of 200 words fit a 512-word parent only with about 96 words of overlap. At real parent sizes (median 289–442 words), parents over 400 words would need 104–127 words of overlap, so over half of every child would repeat. Letting the size set the count (1–3 children) keeps every child at 145–200 words.

### Storage key: size and overlap

A child set is identified by `(child_words, child_overlap)`, so the benchmark can compare 200/0 against 200/40.

- child_id: `<parent_id>:w200o40:k<i>`
- dense cache: `Data/Cache/bge_m3_children/w200o40/<doc_id>/`
- benchmark row names: `hybrid_rrf+reranker@200o40`; `@200` means no overlap, as before

## Implementation steps

### Phase 1: children only (no PDF re-read, no LLM, chunk cards kept)

1. `src/pdf/chunker.py`
   - `CHILD_WORDS = 200`, new `CHILD_OVERLAP = 40`
   - new `_windows(sizes, target, overlap)`, which replaces `_balanced_cut` for children
   - new `_split_long(sentence, lang, target)` for sentences over the target
   - new `_join(parts, lang)` with the Chinese joining rule
   - `children(parents, lang, child_words, child_overlap)`; update the docstring: children without their overlap give back the parent text
2. `src/db/models.py`: `ChildChunk.child_overlap: int`
3. `src/db/schema.py`: `ALTER TABLE child_chunks ADD COLUMN child_overlap INTEGER NOT NULL DEFAULT 0` (existing children really have 0 overlap); index on `(doc_id, child_words, child_overlap)`
4. `src/db/repository.py`: `replace_children`, `get_children`, `has_children` and `delete_children` take `(child_words, child_overlap)`
5. `src/ingest.py`: `child_overlap` in the ingest state; `build_children(doc_id, db, size, overlap)`; CLI `python -m src.ingest --children all 200 40` and `--drop-children 100 0`
6. `src/retrieval/bm25_index.py`, `src/retrieval/dense_index.py`: the unit is `"chunks"` or a `(words, overlap)` pair; in-memory cache keys and the cache folder `w200o40` follow it
7. `src/retrieval/hybrid.py`: `search_unit()` returns `(CHILD_WORDS, CHILD_OVERLAP)`; the parent-child logic stays the same (rank children, return each parent once)
8. `scripts/eval_retrieval.py`: `parse_row` accepts `@200o40`; `sweep` accepts sizes like `160o40 200o40 240o40 200`
9. Run `python -m src.ingest --children all 200 40`, then:
   - `python scripts/eval_retrieval.py sweep hybrid_rrf 160o40 200o40 240o40 200 100`
   - `python scripts/eval_retrieval.py sweep hybrid_rrf+reranker 160o40 200o40 240o40 200 100`
10. Decision rule: take the setting with the best reranker MRR. If the top settings are within about 0.02 of each other, take the cheaper one (fewer children).
11. Clean up the sweep sizes (64–140): `--drop-children <size> 0` for each, and delete `Data/Cache/bge_m3_children/w64` … `w140`.

### Phase 2: parent fixes (optional; needs a re-ingest, changed parents get new chunk cards from the LLM)

- `pack()` uses the same Chinese joining rule. Today 10% of sentence joins in LLMbook and 25% in d2l glue two ASCII words together.
- Balanced parents per page: a page of W words becomes ceil(W / 512) parents of near-equal size, instead of filling 512 words and leaving a short tail. Today 7.2% of parents are page tails under 150 words, for example a 600-word page gives 512 + 88.
- Cost: every parent whose text changes loses its chunk card, which the LLM regenerates. For the Chinese books that is most parents.

### Not planned

- Parents that cross pages so that every parent is exactly 512 words. This would need page ranges for citations and benchmark scoring, and would change every chunk card.

## Changelog entry (fill in after the run)

```
## Child chunks v2 (<date>)

- Children are now ~200-word sliding windows with a 40-word overlap, cut at sentence boundaries
  inside each 512-word parent (1–3 children per parent). Before: up to 100 words, no overlap.
- Same size in both languages: ~1.4–1.65 BGE-M3 tokens per word in English and Chinese, so 200 words
  is ~300 tokens and nothing is truncated.
- Sentences over 200 words are split at word boundaries. Chinese pieces are joined with a space when
  neither side is a Chinese character.
- child_chunks has a new child_overlap column; a child set is (child_words, child_overlap).
- Results (271 questions): hybrid RRF MRR <x> (was <y>), + reranker MRR <x> (was <y>).
```
