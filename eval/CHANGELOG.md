# Benchmark changelog

## v1.1 (2 Oct 2026)

The original file is kept as `questions_v1.0.jsonl`. Every changed row has a `notes` field (why it changed) and a `changed_v1_1` field (its old level and gold pages).

### New format: `gold_groups`

Each group is one part of the answer, and any page inside a group satisfies that part.

- `[[719]]`: one part; page 719 answers it.
- `[[177], [180, 181]]`: two parts. You need 177, plus either 180 or 181.
- `[[108, 122], [122, 123]]`: page 122 alone covers both parts, and 108 + 123 together also do.

`gold_pages` is kept as the flat list of every gold page, so older scripts still work.

### Rules used, applied to every reviewed question

1. A page counts as gold only if it answers the question (or its part) by itself. A code-only or summary page that doesn't state the answer is not gold.
2. A question is L4 only if no single page answers it. If one page already states the comparison, the question is re-leveled.
3. If several pages each answer it, they all go in the same group.

### Reviewed

72 questions were reviewed by reading the gold pages and the retrieved pages:

- all 20 L4 questions
- all 20 L5 questions
- the 20 questions where 4 or more retrievers agreed on a non-gold page
- every question the reranker didn't rank at 1

### Summary

- **Levels:** L1 75 · L2 61 · L3 31 · L4 14 · L5 19 (was L1 70 · L2 60 · L3 30 · L4 20 · L5 20).
- **Re-leveled from L4** (one page answers it): d2l-zh-042, LLMbookzh-043, p300-043, p300-044, p300-045 → L1; clrs-041 → L2.
- **Wrong gold page:** clrs-013 (912 → 913), p300-043 (306 was the chapter summary → 302).
- **L5 that was answerable:** LLMbookzh-049. Page 110 says Mixtral has 8 experts per layer and uses 2 per token. Changed to L3 with gold 110.
- **Gold pages added** (another page answers it too): d2l-zh-013, -032, -036, -037; p300-007, -036; clrs-027, -034.
- **Split into groups** (real multi-part questions): the remaining 14 L4 questions.

### Judgment calls to double-check

- LLMbookzh-042: page 207 also compares DPO with RLHF. It stays L4.
- p300-045: re-leveled to L1 because page 471 compares the two parsers.
- clrs-037 was kept. Its label is correct, and the miss is the RAG's fault. To drop it from scoring, add `"excluded": true`.
- clrs-025 is tagged as a distractor question.
