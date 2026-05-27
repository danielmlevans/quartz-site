---
title: Skill Rating Algorithms
tags:
- foundations
- sport
date: '2026-05-28'
lastmod: '2026-05-28'
---

# Skill Rating Algorithms

## Definition

> [!definition] Plain-language
> A **skill-rating algorithm** turns a stream of pairwise (or multi-player) match outcomes into a numeric estimate of each competitor's underlying ability, updated online as more matches arrive. Elo, Glicko, Glicko-2, TrueSkill, OpenSkill and the modern arena/score-driven systems are all instances of the same idea: assume a probabilistic model that links latent skills to observed results (logistic for Elo / Bradley–Terry, Gaussian for Thurstone / TrueSkill), then update each player's skill after every match using either a fixed-step gradient (Elo's K-factor) or a Bayesian posterior update (Glicko, TrueSkill). The output is a calibrated win-probability for any future matchup, plus — in the Bayesian variants — an uncertainty interval that shrinks as more games are played.

**Formal.** Let $r_i \in \mathbb{R}$ denote the latent skill of player $i$. The **Bradley–Terry** model gives the win probability of $i$ over $j$ as

$$
\Pr[i \succ j] = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i - r_j),
$$

with $\sigma$ the logistic function. **Elo** is stochastic gradient ascent on the BT log-likelihood with constant step size $K$:

$$
r_i \leftarrow r_i + K \cdot (s_{ij} - \sigma(r_i - r_j)),
$$

where $s_{ij} \in \{0, \tfrac{1}{2}, 1\}$ is the observed outcome. **Glicko / Glicko-2** replace the point estimate with a Gaussian posterior $(r_i, \sigma_i^2)$ and inflate $\sigma_i^2$ over time to handle inactivity. **TrueSkill** generalises to teams and multi-player rankings via factor-graph message passing on a Thurstone (Gaussian) observation model. **Plackett–Luce** generalises BT to full multi-player rankings; **OpenSkill** is a closed-form PL update that drops in for TrueSkill at ~3× the speed.

**Intuition.** Every system in the family is a state-space filter over a latent-skill vector. The choices are: what's the link function (logistic vs Gaussian), how is the posterior represented (point estimate vs Gaussian vs particles), how does it evolve between matches (random walk vs static), and how is the inference done (SGD vs analytic Bayes vs message passing). Everything below — Elo, Glicko-2, TrueSkill, OpenSkill, score-driven rating, margin-of-victory rating — is a different point in that 4-dimensional design space ([2104.14012](https://arxiv.org/abs/2104.14012)).

## SOTA by use case (2026)

There is no single best algorithm. The field has fragmented by application, and the right answer depends on what you're rating and why.

| Setting | Use today | Why |
| --- | --- | --- |
| **LLM / preference arenas** | **Bradley–Terry MLE with bootstrap CIs**, optionally annotator-mixture (am-ELO) | Chatbot Arena moved off rolling Elo for stability; [2412.18407](https://arxiv.org/abs/2412.18407) + [2505.03475](https://arxiv.org/abs/2505.03475) are the current best practice |
| **Multiplayer video games (≥3 players, asymmetric teams)** | **OpenSkill** ([2401.05451](https://arxiv.org/abs/2401.05451)) or **TrueSkill 2** | Closed-form PL update; OpenSkill is faster at comparable accuracy. TrueSkill 2 still strong with draw / partial-credit data |
| **Sports with informative score margins** (soccer, NBA, tennis, cricket) | **Margin-of-victory relative to expectation** ([2506.00348](https://arxiv.org/abs/2506.00348)) or **score-driven / GAS** ([2604.09143](https://arxiv.org/abs/2604.09143)) | Beats Elo + Glicko + TrueSkill on calibrated log-loss; both recover Elo as a degenerate special case so adoption is strictly an upgrade |
| **1v1 with sparse / irregular play** (chess-like) | **Glicko-2** or **Whole-History Rating** | Glicko-2 has explicit RD inflation under inactivity; WHR re-fits all past games for better offline historical rankings |
| **Intransitive games** (counters, rock-paper-scissors structure) | **Nash-averaging** / disc-decomposition + a base rating ([2206.12301](https://arxiv.org/abs/2206.12301), [2502.03998](https://arxiv.org/abs/2502.03998), [2502.20170](https://arxiv.org/abs/2502.20170)) | Elo is provably wrong here; scalar skill cannot represent cyclic dominance — needs a non-scalar representation |
| **Online cold-start (novices)** | **QuickSkill-style DL warm-start** ([2208.07704](https://arxiv.org/abs/2208.07704)) feeding into Glicko-2 / OpenSkill | Single best-documented production fix to the novice problem |
| **Unifying / research view** | **Gaussian state-space (Kalman) formulation** ([2104.14012](https://arxiv.org/abs/2104.14012)) | One framework that recovers Elo / Glicko / Glicko-2 / TrueSkill as special cases — useful when designing a custom system |

## Recommendations

> [!tip] If you only read three papers
> 1. [2104.14012](https://arxiv.org/abs/2104.14012) — the Kalman-filter unification. Read first; everything else is a special case.
> 2. [2506.00348](https://arxiv.org/abs/2506.00348) — margin-of-victory-relative-to-expectation. The most consequential recent advance for **sport**.
> 3. [2401.05451](https://arxiv.org/abs/2401.05451) — OpenSkill. The most consequential recent advance for **multiplayer / esports**.

### General principles

1. **Do not deploy vanilla Elo in 2026.** It is broken in two distinct ways: the step size $K$ is misspecified for most real distributions ([2502.10985](https://arxiv.org/abs/2502.10985), [2604.09143](https://arxiv.org/abs/2604.09143)), and a scalar skill cannot represent non-transitive interactions ([2206.12301](https://arxiv.org/abs/2206.12301)). Both have well-developed fixes. There is no excuse.

2. **If your sport has score margins that carry signal, use them.** A 4–0 result is not a 1–0 result, and any rating system that treats them identically is throwing away information. Both [2506.00348](https://arxiv.org/abs/2506.00348) (margin-of-victory-relative-to-expectation) and [2604.09143](https://arxiv.org/abs/2604.09143) (score-driven GAS) handle this principally and outperform Elo / Glicko / TrueSkill on calibrated predictive log-loss in backtested soccer, NBA, tennis and cricket data.

3. **If you have teams of variable size and composition, use a multi-player model.** Sum-of-individual-skills is empirically the wrong aggregator across Halo / DOTA / CS:GO ([2106.11397](https://arxiv.org/abs/2106.11397)); OpenSkill's Plackett–Luce update handles it correctly out of the box.

4. **If your game has informative chance or hidden information** (Skat, poker, hearthstone-like card games), use a rating system that models the noise floor explicitly: [2104.05422](https://arxiv.org/abs/2104.05422), [2512.18858](https://arxiv.org/abs/2512.18858), [2410.14363](https://arxiv.org/abs/2410.14363).

5. **For LLM-arena-style preference ranking, do not use Elo updates** — fit a Bradley–Terry MLE with bootstrap CIs, with rater random effects if your annotator pool is heterogeneous ([2412.18407](https://arxiv.org/abs/2412.18407)). Chatbot Arena ([2403.04132](https://arxiv.org/abs/2403.04132)) made this switch partway through and the literature has consolidated around it. The "Elo Uncovered" robustness paper ([2311.17295](https://arxiv.org/abs/2311.17295)) documents the path-dependence and identifiability artifacts you avoid by doing this.

6. **Audit your leaderboard for adversarial pressure.** If your rating is publicly visible and consequential, expect vote rigging ([2501.17858](https://arxiv.org/abs/2501.17858)) and adversarial manipulation ([2501.07493](https://arxiv.org/abs/2501.07493)). Rankings can also be brittle to small subsets of votes ([2508.11847](https://arxiv.org/abs/2508.11847)) — run influence-function audits before publication.

### Hardware

All of these run trivially on a MacBook Pro M5 Max. BT-MLE over 10⁶ pairwise comparisons is sub-second CPU; the algorithmic cost is in *update logic*, not compute. No GPU or MLX needed. The only places hardware matters are the neural warm-start systems (QuickSkill, PandaSkill ([2501.10049](https://arxiv.org/abs/2501.10049)), graph-embedding-augmented rating ([2304.08257](https://arxiv.org/abs/2304.08257))) — those are <1 GB models and fit comfortably in 128 GB unified memory.

## A short tour of the families

The full ~80-paper review with retrieval meta-evaluation lives in the Discipline repo at `literature-reviews/skill-ranking-algorithms.md`. The compressed tour:

### Theoretical foundations

Every system reduces to a pairwise comparison model. Bradley–Terry (logistic) and Thurstone–Mosteller (Gaussian) are the two dominant choices; Plackett–Luce and Thurstone-multi are their multi-player generalisations.

- [2312.13619](https://arxiv.org/abs/2312.13619) — *The many routes to the ubiquitous Bradley–Terry model.* Derives BT from 8 independent axioms. Best single entry point.
- [2110.03874](https://arxiv.org/abs/2110.03874) — Entrywise CLT for BT-Luce ability estimates; the basis for confidence bounds on any BT-derived leaderboard.
- [2003.00083](https://arxiv.org/abs/2003.00083) — *Dynamic Bradley–Terry*: kernel-smoothed nonparametric estimation for time-varying skill.
- [2402.07811](https://arxiv.org/abs/2402.07811) — PageRank is BT with a specific damping prior; unifies the link- and game-ranking literatures.

### Elo and its analyses

- [2406.05869](https://arxiv.org/abs/2406.05869), [2410.09180](https://arxiv.org/abs/2410.09180) — Elo as an irreducible Markov chain; stationary distribution and mixing time.
- [2502.10985](https://arxiv.org/abs/2502.10985) — *Is Elo reliable under model misspecification?* Quantifies bias when the win model isn't BT-logistic.
- [2206.12301](https://arxiv.org/abs/2206.12301) — *On the limitations of Elo: real-world games are transitive, not additive.* The non-transitivity argument.

### Glicko, Glicko-2, Kalman unification

- [2104.14012](https://arxiv.org/abs/2104.14012) — Elo / Glicko / Glicko-2 / TrueSkill as instances of a single Gaussian state-space filter.
- [2310.11459](https://arxiv.org/abs/2310.11459), [2603.02574](https://arxiv.org/abs/2603.02574) — Glicko-2 adaptations to football and Test cricket.
- [1902.07378](https://arxiv.org/abs/1902.07378) — Gaussian-process priors for dynamic paired comparison; the principled "infinite-K Glicko" generalisation.

### TrueSkill, OpenSkill, team rating

- [2401.05451](https://arxiv.org/abs/2401.05451) — **OpenSkill.** Closed-form Plackett–Luce update; drop-in TrueSkill replacement, ~3× faster at equal accuracy.
- [2106.11397](https://arxiv.org/abs/2106.11397) — Team-aggregation rules across Halo / DOTA / CS:GO.
- [2604.09143](https://arxiv.org/abs/2604.09143) — Score-driven (GAS) rating for sports.
- [2506.00348](https://arxiv.org/abs/2506.00348) — Margin-of-victory-relative-to-expectation.

### Esports / matchmaking systems

- [2208.07704](https://arxiv.org/abs/2208.07704) — QuickSkill (NetEase). Production cold-start solution.
- [2501.10049](https://arxiv.org/abs/2501.10049) — PandaSkill (League of Legends). Behaviour-aware individual rating from team outcomes.
- [2410.02831](https://arxiv.org/abs/2410.02831) — CS:GO skill-rating bake-off across families.
- [1702.06820](https://arxiv.org/abs/1702.06820) — EOMM: engagement-optimised matchmaking; the counter-argument to fair-match objectives.
- [2502.03998](https://arxiv.org/abs/2502.03998) — Online counter-category learning for PvP games; handles non-transitivity head-on.

### LLM arena evaluation

- [2403.04132](https://arxiv.org/abs/2403.04132) — Chatbot Arena. Foundational.
- [2412.18407](https://arxiv.org/abs/2412.18407) — Cleanest current statistical framework: BT with ties, BT with rater random effects, identifiability conditions.
- [2505.03475](https://arxiv.org/abs/2505.03475) — am-ELO. Annotator-mixture extension; lower rank volatility than vanilla Arena Elo.
- [2311.17295](https://arxiv.org/abs/2311.17295) — Elo Uncovered: path-dependence, non-uniqueness, ordering artifacts.
- [2502.20170](https://arxiv.org/abs/2502.20170) — Re-evaluating open-ended LLM evaluation; Nash-averaging approach to cyclic dominance.

## What this is not

Not a survey of *every* skill-rating paper — the BM25 retrieval used to build this note has corpus-level Recall@100 ≈ 0.76 against single-relevant-doc qrels, so roughly a quarter of relevant papers may have been missed. Notable absences likely outside the arXiv-indexed cs corpus: Herbrich-Minka-Graepel 2007 (TrueSkill original), Glickman 1999 / 2012 (Glicko / Glicko-2 originals), Coulom 2008 (Whole-History Rating). The arXiv-indexed surrounding literature is covered above.

The provenance and retrieval meta-evaluation for this review is in [Rapid Literature Search for Sports AI](rapid-literature-search-for-sports-ai); this note is the synthesis layer on top of that loop.
