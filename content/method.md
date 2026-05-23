---
title: Method
---

The corpus on this site is produced by an agent pipeline I'm building on top of Claude Code. Every note is drafted by the agent, validated by me before publication, and structured to the same template. This page describes how the pipeline works and what it does and doesn't do well.

## The pipeline

Each topic moves through five stages:

1. **Scoping** — I pick a topic from an ontology I maintain (~177 topics across alignment, training, evaluation, interpretability, agents, foundations). The ontology has explicit dependencies so foundational topics are written first.
2. **Research** — the agent runs a multi-pass research process against an arxiv-anchored bibliography, building a `papers:` list and a `resources:` list in YAML front matter. Every claim in the note is sourced.
3. **Drafting** — the agent writes the note against a fixed 15-section template (Definition, Mechanism, Historical lineage, State of the art, Key papers, Key people, Practical applications, Limitations, Open questions, Sources, etc.).
4. **Validation** — I read the draft against the source papers, fix factual errors, write the Personal Synthesis section in my own voice, and mark `coverage_status: validated` once the note is publishable. Notes that fail validation stay in the private vault until they pass.
5. **Publishing** — `publish.py` strips pipeline-internal YAML, converts inline `(paper_id: X)` references into IEEE numbered citations, drops local-only Zotero links, and copies the cleaned note into this site's content directory.

## What works

<!-- TODO: Dan to fill in his voice. Suggested points:
- The structure-first approach catches gaps before drafting (sections that would be left empty are flagged)
- Per-paper grounding (paper_id + arxiv URL) makes claims auditable
- Math goes into a separate sub-vault with its own explanation notes — keeps the main note readable
- Open questions section forces the agent to surface uncertainty rather than papering over it
-->

## What fails, specifically

<!-- TODO: Dan to fill in. The plan calls out that this is the section signalling seriousness. Suggested failure modes to name:
- Hallucinated citations the validator catches (X% of drafts) — concrete examples
- Sycophancy in the agent's framing of contested claims
- Mode collapse when the topic isn't well-covered in training data
- Specific cases where the agent missed important recent papers
-->

## Why publish this

<!-- TODO: Dan's voice. Plan framing: portfolio / collaboration calling card, not a blog. The site exists to be findable by AI safety researchers, MATS/SPAR programmes, ANZ AI hiring (Canva, APS, Anthropic-adjacent). Transparency about the agent pipeline is the differentiator. -->

## Tools

- **Source vault:** Obsidian
- **Research agent:** Claude Code with custom skills
- **Bibliography manager:** Zotero (Zotero links are stripped during publishing)
- **Static site generator:** [Quartz 4](https://quartz.jzhao.xyz/)
- **Hosting:** Cloudflare Pages
- **Citation style:** IEEE numbered, in-document anchors
