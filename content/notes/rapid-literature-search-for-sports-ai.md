---
title: Rapid Literature Search for Sports AI
tags:
- tools
date: '2026-05-28'
lastmod: '2026-05-28'
worked_example: literature-reviews/skill-ranking-algorithms.md
---

# Rapid Literature Search for Sports AI

## Definition

> [!definition] Plain-language
> **Rapid Sports** is a local arXiv search warehouse — every arXiv paper since 2007 (~3 million) sitting in a single DuckDB file on my laptop, with a hybrid retrieval stack layered on top (lexical full-text search, dense embeddings, cross-encoder rerank, LLM-driven query rewriting). The point of it is to take a sports question — "what's the state of the art in injury-risk prediction from wearables?" — and return a ranked, cited set of frontier papers in seconds, with retrieval quality comparable to a hosted production system, but running entirely on a single device with no rate limits and no data leaving the machine.

It is the upstream of every note on this site. When I want to know what AI has to say about a sport problem, this is what I ask first.

## Why this matters for sport

Elite sport is small, distributed, and structurally under-resourced for keeping up with frontier AI. The literature that matters — in vision, sequence modelling, reinforcement learning, evaluation, time-series — lives in arXiv preprints that move faster than journals, faster than newsletters, faster than the staff time most institutes can spare. By the time a relevant paper surfaces through conferences or social media, it has often been superseded.

The bottleneck is not idea generation. Coaches and practitioners have abundant questions. The bottleneck is **the speed and accuracy of literature triage**. If finding the right five frontier papers on a specific sport question takes a day, the question doesn't get asked. If it takes ninety seconds, the question gets asked routinely — and the answer feeds the next question.

That asymmetry is what Rapid Sports exists to compress.

## The pipeline

Rapid Sports is organised as a five-phase bronze → silver → gold architecture, with each phase independently shippable and gated by retrieval-quality metrics (nDCG@10, Recall@100) computed against TREC-format qrels via `pytrec_eval`.

1. **Bronze — raw metadata.** Full arXiv metadata harvested via OAI-PMH, stored as immutable rows keyed on `id`. ~3M papers, daily incremental refresh. This is the source of truth; everything downstream rebuilds from it.
2. **Silver — normalisation + lexical FTS.** A BM25 full-text index over normalised titles and abstracts. Cheap, fast, and surprisingly hard to beat for keyword-rich queries ("Polar Verity Sense heart-rate variability").
3. **Gold — dense embeddings + HNSW.** Multilingual Matryoshka encoders (e.g. [m3BERT](https://arxiv.org/abs/2605.19568)) indexed via DuckDB's `vss` extension. Catches the queries BM25 misses — semantically adjacent phrasing, paraphrase, conceptual overlap ("how do we measure recovery without HRV?").
4. **Cross-encoder rerank.** The top ~100 hybrid candidates are reranked by a small cross-encoder, trading a few hundred milliseconds for precision at the top of the list. This is where "good enough" becomes "good".
5. **Query understanding.** LLM-driven rewriting, HyDE, pseudo-relevance feedback. A sports practitioner's question rarely matches the vocabulary of an ML paper; this layer translates between them.

End-to-end latency budget: **p95 ≤ 3.5 s cold, ≤ 500 ms cached**, on a single MacBook Pro. No cloud round-trip, no rate limits, no data leaving the device. Every retrieval decision is reproducible from the warehouse file alone.

## The sports workflow

A practical loop looks like this:

1. **Frame the question in sport vocabulary** — "can we predict hamstring strain from GPS load and sleep variance?"
2. **Query Rapid Sports.** The query-understanding layer rewrites the question into ML-paper vocabulary (time-series anomaly detection, multimodal injury prediction, survival modelling).
3. **Triage the top 10–20 hits.** Filter for recency, citation density, and methodological fit.
4. **Read deeply.** The ranked list is short enough that close reading is possible in an evening.
5. **Synthesise into a note** on this site, with the sports lens applied — what does this method assume, what data does it need, where does it break in our setting?

Step 5 is what closes the loop. Rapid Sports is the search engine; this blog is the synthesis layer.

## What this is not

Not a replacement for domain expertise. Not a substitute for talking to authors. Not a guarantee that the top hit is the right paper — retrieval quality is high but not perfect, and the cross-encoder is tuned for general relevance, not sport-specific correctness. The human-in-the-loop is the validator, always.

It is also not novel research infrastructure. The architecture is standard hybrid retrieval; every design choice in the search-enrichment plan cites the paper that motivates it. The contribution is putting all of it in one local file, scoped to a single user's sport-AI synthesis loop, with the speed and privacy properties that come from running on-device.

## A worked example — skill rating for sport

To make this concrete: on 2026-05-28 I ran the Discipline loop end-to-end on a sports-adjacent question — *what are the algorithms for rating skill (Elo, Glicko, TrueSkill, …) and which is state of the art?* Eight BM25 sweeps over the 3.05 M-paper silver layer returned ~80 unique skill-rating papers across theory, sports applications, esports matchmaking and LLM arena evaluation. The full synthesis is in [Skill Rating Algorithms](/notes/skill-rating-algorithms); the headline findings:

- **Foundational layer is Bradley–Terry / Plackett–Luce.** Elo, Glicko-2 and TrueSkill are all special cases of a Gaussian state-space filter over a BT/PL likelihood ([2104.14012](https://arxiv.org/abs/2104.14012), [2312.13619](https://arxiv.org/abs/2312.13619)).
- **For sport specifically,** the modern SOTA is *score-driven* / *margin-of-victory* rating — [2604.09143](https://arxiv.org/abs/2604.09143) and [2506.00348](https://arxiv.org/abs/2506.00348) — which beat vanilla Elo + Glicko + TrueSkill on calibrated log-loss across soccer, NBA, tennis and Test cricket backtests. Both recover Elo as a degenerate special case, so adopting them is strictly an upgrade.
- **For multiplayer / team settings,** OpenSkill ([2401.05451](https://arxiv.org/abs/2401.05451)) is the production-ready closed-form Plackett–Luce update, ~3× faster than TrueSkill at equal accuracy.
- **Vanilla Elo is bad in two distinct ways:** wrong step-size for most real distributions ([2502.10985](https://arxiv.org/abs/2502.10985)), and structurally unable to represent non-transitive (rock-paper-scissors) interactions ([2206.12301](https://arxiv.org/abs/2206.12301)).

### Speed and accuracy of the search itself

The point of showing this in the worked example is the retrieval meta-evaluation — *how fast and how good* is the loop?

| Metric | Value |
| --- | ---: |
| 8 BM25 queries over 3.05 M papers, warm-state median end-to-end | **~380 ms** |
| First query (cold `LOAD fts` + ART postings warmup) | ~2 s |
| `title_to_abstract` Recall@10 (corpus gate) | **0.969** |
| `citation_qrels` Recall@100 (corpus gate) | **0.7649** |
| `citation_qrels` nDCG@10 (current; carried to Phase 2 hybrid) | 0.2879 |
| `citation_qrels` p95 latency @ k=100 | 419 ms |

Total wall-clock for the entire skill-rating survey — eight retrievals, ~80 candidate papers triaged, structured synthesis written — was under an hour. The same exercise via Google Scholar + arXiv search + manual deduplication is a day's work and usually less complete. The retrieval-quality numbers say the candidate set is reliable (0.76 Recall@100) but the in-list ordering will tighten further once the Phase 2 dense leg and Phase 3 cross-encoder ship; I'll re-run this exact survey then and report the lift.

That is the loop this site is built around: a sport-relevant question, eight queries, an evening of close reading, one note. The skill-rating example transposes directly — every elite sport runs internal player-quality rankings of some kind, and the algorithms above (especially [2604.09143](https://arxiv.org/abs/2604.09143) and [2506.00348](https://arxiv.org/abs/2506.00348)) are immediately applicable to that work.
