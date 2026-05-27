---
title: Skill Rating for Olympic and Paralympic Sport
tags:
- sport
- foundations
date: '2026-05-28'
lastmod: '2026-05-28'
parent: skill-rating-algorithms
---

# Skill Rating for Olympic and Paralympic Sport

## Definition

> [!definition] Plain-language
> **Olympic-sport skill rating** is a different statistical problem from general skill rating. It is defined by three structural constraints that off-the-shelf systems (Elo, Glicko, TrueSkill, OpenSkill) do not address: (1) the *benchmark calendar is sparse* — one World Championships and one Olympic Games per cycle, with World Cups attended unevenly across nations for geographic or qualification reasons; (2) the *comparison graph is disconnected* — top athletes from different regions may never directly compete, so their ratings are local maxima inside isolated pockets, not points on a shared global scale; and (3) *within-event volatility is high and consequential* — a single bad day at a peak event, weighted as if it were ordinary evidence, can crater the rating of the actually-best athlete. The correct response is not a better Elo K-factor but a different kind of model entirely: a hierarchical Bayesian state-space model with heavy-tailed observation noise, geometric event-importance weighting, partial-identification handling for disconnected graph components, and explicit between-/within-season trajectory decomposition. The rating is no longer a single number per athlete — it is a posterior distribution, often a set-valued one when the data does not support a point estimate.

**Formal sketch.** Let $r_{i,t} \in \mathbb{R}$ be the latent skill of athlete $i$ at time $t$. A multi-competitor scored event $e$ at time $t_e$ with importance weight $w_e$ generates an observed score $y_{i,e}$ for each athlete $i \in \mathcal{A}_e$. The model is

$$
\begin{aligned}
r_{i,t} &= r_{i,t-1} + \eta_i + \epsilon^{\text{state}}_{i,t}, &\epsilon^{\text{state}}_{i,t} &\sim \mathcal{N}(0, \tau_i^2) \\
y_{i,e} &= g(\phi_{i,e} - r_{i,t_e}) + \epsilon^{\text{obs}}_{i,e}, &\epsilon^{\text{obs}}_{i,e} &\sim t_\nu(0, \sigma_i^2) \\
\Pr[\pi_e \mid \{r_{i,t_e}\}, w_e] &\propto \prod_{k} \frac{\exp(w_e \cdot r_{\pi_e(k), t_e})}{\sum_{j \in \mathcal{A}_e^{(k)}} \exp(w_e \cdot r_{j, t_e})},
\end{aligned}
$$

where $\eta_i$ is athlete-specific drift (career arc), $\phi_{i,e}$ is a within-season form term, $g$ is a monotone score transform ([2205.10746](https://arxiv.org/abs/2205.10746)), $t_\nu$ is Student-t noise replacing Gaussian to give heavy-tail robustness ([2502.18206](https://arxiv.org/abs/2502.18206)), and the last line is Plackett–Luce over the observed finishing order $\pi_e$ with geometric event weight $w_e$ ([1907.05082](https://arxiv.org/abs/1907.05082)). When the comparison graph induced by $\{\mathcal{A}_e\}$ is disconnected, the posterior over $r_{i,t}$ is reported as a *set* rather than a point ([2410.18272](https://arxiv.org/abs/2410.18272)).

**Intuition.** The thing the Olympic problem is asking — *how good is this athlete, given that they race the world's best only once a year, never meet half the field, and might have food poisoning on the day* — is fundamentally a problem of *honest uncertainty quantification*. Off-the-shelf rating systems answer it by producing a single number with overconfident precision. The hierarchical Bayesian formulation above answers it by producing a posterior whose width *reflects what the data actually supports*. That includes producing intervals rather than point estimates when the comparison graph is too sparse to identify a global ranking. The honest answer to "is the Belgian or the Australian better?" is sometimes "we don't have the data to say"; Elo and Glicko cannot give that answer, this model can.

## The three failure modes

The off-the-shelf rating literature ([Skill Rating Algorithms](skill-rating-algorithms)) is dominated by problems with dense, well-connected comparison graphs — chess servers, MOBA matchmaking queues, LLM arenas. Olympic sport has none of those properties. The failure modes:

| Failure mode | Why off-the-shelf systems break | Fix |
| --- | --- | --- |
| **Sparse benchmark calendar** | Elo/Glicko/TrueSkill assume regular comparison density; their *bias* under attendance selection is unaddressed | Hierarchical Bayesian state-space with event-importance weights |
| **Disconnected regional pockets** | A point estimate across disconnected components is mathematically meaningless | Partial identification + stochastic block model |
| **High volatility / bad-day weighting** | Gaussian noise / fixed-K SGD treats a meltdown as skill decline | Student-t observation noise + season-trajectory decomposition |

## The recommended stack

### 1. Plackett–Luce + monotone score transform

[2205.10746](https://arxiv.org/abs/2205.10746) — *Athlete rating in multi-competitor games with scored outcomes via monotone transformations.* The closest paper in the literature to the Olympic-discipline problem. Multi-competitor scored events (athletics, swimming, biathlon, cross-country) get a monotone transform applied to observed scores before the Plackett–Luce fit. The transform is what gives robustness to a bad day: a 0.5 s slow 100m is not 10× worse than a 0.05 s slow one, but raw-time Elo would treat it that way.

For **judged disciplines** (gymnastics, diving, figure skating, freestyle skiing): add a judge-bias term per [1807.10055](https://arxiv.org/abs/1807.10055), which separates athlete skill from judge effects in a rigorous way.

For **head-to-head combat sports** (judo, taekwondo, wrestling): hierarchical BT, but the comparison graph will be classified-by-impairment and naturally disconnected — read §4 below first.

### 2. Geometric event-importance weighting

[1907.05082](https://arxiv.org/abs/1907.05082) — *How should we score athletes and candidates: geometric scoring rules.* Designed for the Olympic-style problem where one Worlds is worth more than five regional opens. Geometric scoring rules down-weight the long tail of low-importance results by construction — a finish at a tertiary event cannot dominate the rating no matter how good. The principled version of every hand-tuned federation point system.

### 3. Volatility handling — heavy tails + season decomposition

Three complementary papers:

- [2502.18206](https://arxiv.org/abs/2502.18206) — Robust Kalman filtering via normal variance mixtures. Student-t observation noise; 3σ days update the skill posterior much less than Gaussian noise would.
- [2405.17214](https://arxiv.org/abs/2405.17214) — Between- and within-season trajectories in elite athletic performance. Decomposes "peaking for Worlds" from "career arc"; bad finishes at the peak event get attributed to within-season variance, not declining skill.
- [2101.08175](https://arxiv.org/abs/2101.08175) — Bayesian GARCH for sports data. Models volatility itself as latent, so naturally-volatile athletes get wider posteriors rather than spurious certainty.

### 4. Disconnected-graph handling — partial identification

The most important and least obvious choice. Almost no off-the-shelf system handles it honestly.

- [2410.18272](https://arxiv.org/abs/2410.18272) — *Partially Identified Rankings from Pairwise Interactions.* When the comparison graph has multiple components or weak bridges, the right answer is a *set* of ratings consistent with the data, reported as intervals — not a single number. The honest answer to "Athlete A only races Europeans, Athlete B only races Americans, who is better?" is "between 3rd and 7th globally, with these specific candidates" — not a misleading point.
- [2511.03467](https://arxiv.org/abs/2511.03467) — *The Bradley–Terry Stochastic Block Model.* When disconnection is *partial* (a few cross-region matches per year), models regional pockets as blocks with partial pooling. Information flows between blocks through sparse cross-region matches; each block keeps its own scale.
- [2304.06821](https://arxiv.org/abs/2304.06821) — *Ranking from Pairwise Comparisons in General Graphs and Graphs with Locality.* Sample-complexity bounds — how many cross-component comparisons are required to recover a global ranking? Direct input to calendar design.
- [2207.01455](https://arxiv.org/abs/2207.01455) — *Dynamic Ranking and Translation Synchronization.* Components that connect *intermittently over time* (different eras / continents) get aligned via time-overlapping career arcs.
- [2002.08853](https://arxiv.org/abs/2002.08853) — *A General Pairwise Comparison Model for Extremely Sparse Networks.* BT-MLE consistency conditions under sparsity; tells you when the data is even adequate before fitting.

### 5. Calendar design — the upstream fix

[1207.6430](https://arxiv.org/abs/1207.6430) — *Optimal Data Collection For Informative Rankings Expose Well-Connected Graphs.* The cheapest way to get a good rating is to schedule matches that connect the graph — a handful of well-placed cross-region matches per year dominates clever statistical modelling on a badly-connected calendar. Send to whoever designs the federation calendar.

[1109.3701](https://arxiv.org/abs/1109.3701) — *Active Ranking using Pairwise Comparisons.* Active-learning view: which *next* matches reduce posterior uncertainty most? Useful for wildcard/invitation decisions.

## Federation case studies

- [1806.08259](https://arxiv.org/abs/1806.08259) — *Dynamic Network 3—0 FIFA Rankings.* Forensic critique of the old FIFA ranking + replacement design. The exploitability section is the key read — the old system rewarded *avoiding* strong opponents, the same pathology as athletes skipping each others' World Cups.
- [2201.00691](https://arxiv.org/abs/2201.00691) — FIFA ranking evaluation of the new Elo-based system. Comparable for a federation-scale redesign.
- [1705.05831](https://arxiv.org/abs/1705.05831) — ATP points system is predictively *worse* than Elo despite more data, because it isn't statistically motivated. The direct argument against hand-tuned point systems.
- [2411.02000](https://arxiv.org/abs/2411.02000) — Bayesian biathlon performance modelling. Worked Olympic-discipline example with the right ingredients.
- [2409.05714](https://arxiv.org/abs/2409.05714) — Dynamic ranking for the Men's Ice Hockey World (Junior) Championships. Single-benchmark-event forecasting; the Olympic shape.
- [2510.14723](https://arxiv.org/abs/2510.14723) — Bayesian Olympic medal table; cross-discipline aggregation for national strength.

## What this is not

Not a recommendation to deploy raw Elo, Glicko-2, or TrueSkill on Olympic data. They will produce confident-looking numbers that are quietly wrong: biased toward whichever region the athlete competes in most, brittle to single-event volatility, and silently averaging over disconnected comparison-graph components as if they were the same scale.

Not a single-line library install. Vanilla Glicko-2 is one config line; this stack is a PyMC / Stan / NumPyro programme on the order of 200–400 lines. The cost is real but one-time. The output is calibrated win-probabilities with honest uncertainty intervals — including the answer "we don't have enough data to rank these two", which off-the-shelf systems cannot produce.

Not a Paralympic-specific review. The arXiv-indexed literature is thin on Paralympic-specific rating; the recommendations above transfer at the statistical level but the impairment-class structure is a layer that needs its own follow-up search.

## Honest caveats

1. **Adversarial dynamics at federation scale.** If the ranking determines Olympic qualification, athletes and federations will optimise against it. Read the FIFA exploitability critique ([1806.08259](https://arxiv.org/abs/1806.08259)) before deploying.

2. **Compute is not the constraint.** A full posterior over a 500-athlete population with ~5 years of competition data is a 10-second NUTS run on a MacBook Pro M5 Max. No GPU. Build the right model.

3. **The retrieval that produced this review** has corpus-level Recall@100 ≈ 0.76 — ~24% of relevant work may be missed. Re-run this survey once Phase 2 dense + Phase 3 cross-encoder retrieval ships; the Paralympic gap in particular is the kind of semantic-rather-than-lexical miss that dense retrieval is built for. Provenance: [Rapid Literature Search for Sports AI](rapid-literature-search-for-sports-ai).
