---
title: Attention Mechanisms
tags:
- foundations
date: '2026-05-13'
lastmod: '2026-05-13'
---

# Attention Mechanisms

## Definition

> [!definition] Plain-language
> An **attention mechanism** is a learned function that lets each output position of a neural network compute a content-addressed weighted average over a set of input positions, where the weights come from a softmax over compatibility scores between a *query* vector emitted at the output position and *key* vectors emitted at the input positions. The averaged content is a third set of vectors — *values* — supplied by the same input positions. In a transformer this happens at every position simultaneously, against every (visible) position in the same sequence; "the model attends to the previous occurrence of `Smith`" is shorthand for "the query at the current position assigned high softmax weight to the key at the position where `Smith` appeared, and the value at that position was added into the residual stream."

**Formal.** Given queries $Q \in \mathbb{R}^{n \times d_k}$, keys $K \in \mathbb{R}^{n \times d_k}$, and values $V \in \mathbb{R}^{n \times d_v}$ derived from input $X$ via learned linear projections, *scaled dot-product attention* is

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V,
$$

with the softmax applied row-wise; for autoregressive decoding a causal mask $M$ with $M_{ij} = -\infty$ for $j > i$ and $0$ otherwise is added to $QK^\top/\sqrt{d_k}$ before the softmax so position $i$ cannot attend to future positions [\[19\]](#ref-19). *Multi-head attention* runs $h$ such operations in parallel on lower-dimensional projections of $X$ and concatenates the outputs through a learned output projection $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$, letting different heads specialise on different relational subspaces [\[19\]](#ref-19).

**Intuition.** Recurrent networks compress the whole past into a single state and read from it; attention instead *keeps* every past position around and routes information to the present position by content match. The $\sqrt{d_k}$ scaling exists because random dot products grow as $\mathcal{O}(d_k)$ in expectation and would otherwise push the softmax into saturation; the softmax exists because the model needs a differentiable, normalised distribution over positions rather than a hard pick; the multi-head split exists because forcing every relation through a single subspace loses signal. The cost is quadratic in sequence length — prefill is $\mathcal{O}(n^2 d)$ — but the constant factors and IO pattern can be reduced enormously with the right kernel [\[22\]](#ref-22). What people now call "transformer attention" is this single operation, stacked, masked, and tiled across an architecture that is otherwise just residual streams plus position-wise MLPs.

## Mechanism

Attention is a single, composable operation: each output position computes a content-addressed weighted average of value vectors, where the weights come from a softmax over compatibility scores between a query at that position and keys at every (visible) position in the context. Everything below — scaling, masking, multiple heads, the KV cache, the efficiency zoo — is engineering around that one operation.

### Scaled dot-product attention

Given an input sequence $X \in \mathbb{R}^{n \times d_{\text{model}}}$ of $n$ token vectors, three learned linear projections produce queries, keys, and values:

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V,
$$

with $W^Q, W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$. *Scaled dot-product attention* is then

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V,
$$

where the softmax is applied row-wise and the output has shape $\mathbb{R}^{n \times d_v}$ [\[19\]](#ref-19). The terms decompose as:

- **Queries $Q$** — what each position is looking for. Row $i$ of $Q$ is the query vector $q_i \in \mathbb{R}^{d_k}$ emitted by token $i$.
- **Keys $K$** — what each position advertises. The dot product $q_i \cdot k_j$ measures how relevant position $j$ is to position $i$.
- **Values $V$** — what each position contributes if attended to. The output at $i$ is $\sum_j \alpha_{ij} v_j$ with weights $\alpha_{ij}$ from the softmax row.
- **The $\sqrt{d_k}$ scaling** — without it, dot products grow as $\mathcal{O}(d_k)$ in expectation for unit-variance $q$ and $k$, pushing the softmax into saturating regions where gradients vanish. Dividing by $\sqrt{d_k}$ keeps the pre-softmax logits at roughly unit variance regardless of head dimension [\[19\]](#ref-19).
- **Row-wise softmax** — each query row $i$ produces a distribution over the $n$ key positions; the output at $i$ is a convex combination of the value rows.

The asymptotic cost is $\mathcal{O}(n^2 d_k)$ time and $\mathcal{O}(n^2)$ memory for the score matrix — the quadratic-in-$n$ regime that drives every later efficiency variant.

### Multi-head attention

A single attention operation forces all positional relationships through one $d_k$-dimensional subspace. *Multi-head attention* runs $h$ attention operations in parallel on lower-dimensional projections, then concatenates and re-projects:

$$
\mathrm{MHA}(X) = \mathrm{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O,
$$

where $\text{head}_i = \mathrm{Attention}(X W^Q_i, X W^K_i, X W^V_i)$ and each per-head projection maps $d_{\text{model}} \to d_k = d_v = d_{\text{model}} / h$, while $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$ recombines the heads [\[19\]](#ref-19). The total parameter count and FLOP count are essentially the same as a single $d_{\text{model}}$-wide head, but the heads are free to specialise — different heads attend to different types of relationships (syntactic vs. semantic, short-range vs. long-range, induction vs. copy) — and this *subspace decomposition* is what mech-interp papers later anchor on. The Anthropic transformer-circuits framework treats each head as an independent additive write to the residual stream, which is a direct consequence of the concat-then-project form above [\[19\]](#ref-19).

### Causal masking

Autoregressive decoders must not let position $i$ peek at position $j > i$. The fix is a *causal mask* $M \in \{0, -\infty\}^{n \times n}$ with $M_{ij} = -\infty$ for $j > i$ and $0$ otherwise. The masked attention is

$$
\mathrm{Attention}_{\text{causal}}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V.
$$

Adding $-\infty$ pre-softmax sends $e^{-\infty} = 0$ post-softmax, so masked positions receive exactly zero attention weight without requiring a separate index-tracking pass [\[19\]](#ref-19). Encoder attention drops $M$ (bidirectional); cross-attention in encoder-decoder models masks based on whether the source is finished, not on causality.

### KV-cache mechanics

At inference time the autoregressive decoder consumes prompt tokens (the *prefill* phase) and then emits one new token per step (the *decode* phase). Re-running attention over the entire context at every decode step would cost $\mathcal{O}(n^2 d)$ per token, but the keys and values for already-emitted tokens never change. The *KV cache* exploits this: after prefill, $K_{1:n}$ and $V_{1:n}$ are persisted; at decode step $n{+}1$ the model computes only the single new query $q_{n+1}$, the single new key $k_{n+1}$, and the single new value $v_{n+1}$, appends to the cache, and computes attention as $\mathrm{softmax}(q_{n+1} K_{1:n+1}^\top / \sqrt{d_k}) V_{1:n+1}$ [\[46\]](#ref-46).

The complexity shift is the central inference fact about transformers:

- **Prefill**: $\mathcal{O}(n^2 d)$ time, $\mathcal{O}(n^2)$ score memory, compute-bound on modern accelerators.
- **Decode (per token)**: $\mathcal{O}(n \cdot d)$ time and memory, **memory-bandwidth-bound** — the bottleneck is streaming $K_{1:n}$ and $V_{1:n}$ from HBM rather than the matmul itself.

This asymmetry is why serving systems batch decode steps across requests, why KV-cache size dominates inference VRAM at long context, and why MQA/GQA exist [\[20\]](#ref-20), [\[46\]](#ref-46).

### Variants and where they sit

- **Multi-Query Attention (MQA)** keeps $h$ query heads but shares a single $K$ and $V$ across all heads, cutting the KV cache by a factor of $h$ at modest quality cost [\[20\]](#ref-20). **Grouped-Query Attention (GQA)** generalises this to $g$ shared groups ($1 \le g \le h$); $g{=}h$ recovers full MHA, $g{=}1$ recovers MQA, $g{=}8$ is the production sweet spot adopted by Llama 2 70B / Llama 3 / Claude 3 / Gemma 2 [\[21\]](#ref-21).
- **FlashAttention** is an *exact* (not approximate) attention kernel that tiles $Q$, $K$, $V$ in on-chip SRAM and computes the softmax incrementally using the *online softmax* trick: maintain a running max $m$ and a running normaliser $\ell$ per query row, and rescale partial outputs as new tiles arrive so the result is mathematically identical to the unfused implementation [\[22\]](#ref-22). Tiling avoids materialising the $n \times n$ score matrix in HBM, dropping memory from $\mathcal{O}(n^2)$ to $\mathcal{O}(n)$ and yielding 2-4× wall-clock speedup at no accuracy cost. FlashAttention-2 reorders parallelism along the sequence dimension and reduces non-matmul FLOPs [\[23\]](#ref-23). Ring Attention shards the sequence across devices in a ring and overlaps K/V communication with compute, enabling million-token contexts [\[28\]](#ref-28).
- **Sparse and sliding-window patterns** restrict each query to a structured subset of keys: Sparse Transformer uses strided and fixed-stride patterns for $\mathcal{O}(n\sqrt{n})$ cost [\[36\]](#ref-36); Longformer combines a local sliding window with a small set of global tokens for $\mathcal{O}(n \cdot w)$ cost where $w$ is window size [\[39\]](#ref-39). Mistral 7B's sliding-window attention is the direct production descendant.
- **Low-rank and kernel approximations** rewrite the attention matrix to avoid the $n \times n$ multiplication. Linformer projects $K$ and $V$ along the *sequence* dimension to rank $k \ll n$, giving $\mathcal{O}(n)$ cost under the empirical observation that attention matrices are low-rank in practice [\[40\]](#ref-40). Linear-attention reformulates $\mathrm{softmax}(QK^\top)V$ as $\phi(Q)(\phi(K)^\top V)$ for a kernel feature map $\phi$, exposing the right-associative product and turning autoregressive attention into a constant-state RNN [\[41\]](#ref-41). Performer chooses $\phi$ as positive orthogonal random features to give an unbiased softmax approximation in $\mathcal{O}(n)$ time [\[42\]](#ref-42).
- **Positional encoding choices** inject sequence order, which the permutation-equivariant attention operator otherwise lacks. RoPE rotates each $(q, k)$ pair in 2-D subspaces by an angle proportional to absolute position, so the dot product $q_m \cdot k_n$ depends only on the relative offset $m - n$ [\[25\]](#ref-25); it is the open frontier standard (Llama, Mistral, Gemma, Qwen, DeepSeek) and the substrate for context-extension methods (YaRN, NTK-aware scaling). ALiBi instead replaces position embeddings with a per-head linear penalty $-m \cdot |i - j|$ on attention scores, enabling clean extrapolation to longer sequences than seen during training [\[26\]](#ref-26).

### Pseudocode: scaled dot-product multi-head attention

```pseudocode
   Input X ∈ ℝ^{n × d_model}        # n tokens, d_model channels
          │
          ▼
  ┌──────────────────────────────────────────────────┐
  │ STAGE 1: PROJECTIONS                             │
  │  for each head i ∈ {1, …, h}:                    │
  │    Q_i = X · W^Q_i        # shape (n, d_k)       │
  │    K_i = X · W^K_i        # shape (n, d_k)       │
  │    V_i = X · W^V_i        # shape (n, d_v)       │
  │  # d_k = d_v = d_model / h                       │
  │  ⇒ {Q_i, K_i, V_i}_{i=1..h}                      │
  └────────────────────┬─────────────────────────────┘
                       ▼
  ┌──────────────────────────────────────────────────┐
  │ STAGE 2: PER-HEAD ATTENTION                      │
  │  for each head i:                                │
  │    S_i = (Q_i · K_iᵀ) / √d_k     # scores (n, n) │
  │    if causal:                                    │
  │      S_i = S_i + M            # M_{jk} = −∞ if k > j │
  │    A_i = softmax(S_i, axis=−1)   # row-wise      │
  │    H_i = A_i · V_i               # head out (n, d_v) │
  │  ⇒ {H_i}_{i=1..h}                                │
  └────────────────────┬─────────────────────────────┘
                       ▼
  ┌──────────────────────────────────────────────────┐
  │ STAGE 3: CONCAT + OUTPUT PROJECTION              │
  │  H = concat(H_1, …, H_h, axis=−1)  # (n, h·d_v)  │
  │  Y = H · W^O                       # (n, d_model)│
  │  ⇒ Y                                             │
  └──────────────────────────────────────────────────┘

  # Decode-time variant (KV cache):
  #   on token t, X is a single row x_t ∈ ℝ^{1 × d_model}
  #   compute q_t, k_t, v_t per head
  #   K_cache_i ← concat(K_cache_i, k_t);  V_cache_i ← concat(V_cache_i, v_t)
  #   s_t = (q_t · K_cache_iᵀ) / √d_k
  #   a_t = softmax(s_t);  h_t = a_t · V_cache_i
  #   per-token cost: 𝒪(n · d)  ← memory-bandwidth bound on accelerators
```

A full transformer layer wraps this in pre- or post-norm and adds a residual connection: $y = x + \mathrm{MHA}(\mathrm{LayerNorm}(x))$ followed by the position-wise MLP — but those concerns belong to the transformer-architecture topic, not here.

## Historical lineage

Attention arrived in AI not as a single invention but as the convergence of three independent threads: cognitive science models of human selective attention, the neural memory-addressing literature of 2014, and the practical frustration of recurrent seq2seq engineers hitting a context-vector bottleneck. The table below traces the eleven years from first soft-attention paper to the post-attention hybrid frontier.

| Year | Milestone | Notes |
|---|---|---|
| 1980 | **Cognitive science precursor** | Treisman's feature-integration theory and Posner's spatial cueing (neither citable as ML papers) establish attention as a selective-routing concept in human perception. Researchers will later borrow the vocabulary directly. |
| 2014 | **Memory Networks** [\[30\]](#ref-30) | Weston, Chopra & Bordes propose read/write external memory addressed by learned embeddings. The model must "attend" to a slot; this is still discrete and non-differentiable, but the Q/K conceptual skeleton is present. |
| 2014 | **Neural Turing Machines** [\[31\]](#ref-31) | Graves, Wayne & Danihelka make memory addressing soft and differentiable via a combination of content-based and location-based attention. The first paper to use a fully differentiable read head — the direct precursor to the attention formulations that follow. |
| 2014 | **Bahdanau soft attention for NMT** [\[17\]](#ref-17), [\[32\]](#ref-32) | Bahdanau, Cho & Bengio replace the fixed context vector of encoder-decoder seq2seq with a learned alignment over all source hidden states. First practical demonstration that attention improves machine translation at scale. |
| 2015 | **Luong global & local attention** [\[18\]](#ref-18), [\[33\]](#ref-33) | Luong, Pham & Manning systematise attention variants: global vs. local, and multiplicative vs. additive scoring. The dot-product score introduced here becomes the dominant form in Vaswani et al. two years later. |
| 2017 | **"Attention Is All You Need"** [\[19\]](#ref-19), [\[34\]](#ref-34) | Vaswani et al. remove recurrence entirely. Multi-head scaled dot-product self-attention, positional encodings, stacked residual layers. Every frontier LLM since 2018 descends from this paper. The watershed. |
| 2018–2019 | **Transformer proliferation — BERT & GPT** [\[35\]](#ref-35) | Devlin et al. apply encoder-only transformers to masked-language modelling (BERT); Radford et al. apply decoder-only transformers to causal LM (GPT-1, GPT-2). The architecture spreads from translation to the entirety of NLP. |
| 2019–2020 | **Efficiency wave, part I: sparse & multi-query** [\[20\]](#ref-20), [\[36\]](#ref-36), [\[37\]](#ref-37) | Child et al. introduce strided+fixed sparse patterns (O(n√n)); Shazeer proposes multi-query attention (MQA) — sharing K/V across heads — to cut KV-cache bandwidth at decode time. |
| 2020 | **Efficiency wave, part II: linear, kernel, and low-rank** [\[38\]](#ref-38), [\[39\]](#ref-39), [\[40\]](#ref-40), [\[41\]](#ref-41), [\[42\]](#ref-42) | A burst of sub-quadratic proposals: Reformer (LSH), Longformer (sliding-window + global), Linformer (low-rank), linear attention (kernel feature maps), Performers (FAVOR+). None displace full attention at frontier scale, but the linear-attention RNN framing seeds the SSM revival. |
| 2021 | **Position encoding standardisation: RoPE & ALiBi** [\[25\]](#ref-25), [\[26\]](#ref-26), [\[43\]](#ref-43), [\[44\]](#ref-44) | RoPE wins adoption in Llama, Mistral, Gemma, Qwen, and DeepSeek; ALiBi is adopted by BLOOM and MPT. Both define the positional-encoding landscape for the next four years. |
| 2022 | **IO-aware kernels: FlashAttention** [\[22\]](#ref-22), [\[45\]](#ref-45) | Dao et al. reformulate exact softmax attention as a tiled online computation that stages reads/writes between SRAM and HBM. No approximation; 2–4× wall-clock speedup; linear (not quadratic) memory. The ChatGPT moment coincides. |
| 2022 | **Efficient inference cost model** [\[46\]](#ref-46) | Pope et al. develop the canonical analytical framework for transformer inference — arithmetic intensity, prefill vs. decode phases, KV-cache bandwidth. Shapes every subsequent serving-systems design. |
| 2023 | **Production patterns settle: GQA, FlashAttention-2, Ring Attention** [\[21\]](#ref-21), [\[23\]](#ref-23), [\[28\]](#ref-28), [\[47\]](#ref-47), [\[48\]](#ref-48), [\[49\]](#ref-49) | GQA generalises MQA with `g` K/V groups, recovering quality while keeping most memory savings — adopted by Llama-2-70B, Llama-3, and (per architectural disclosures) Claude 3. Ring Attention enables context windows of millions of tokens. |
| 2023 | **First credible non-attention contender: Mamba** [\[50\]](#ref-50) | Gu & Dao introduce selective state-space models with input-dependent A/B/C matrices. For the first time a non-attention architecture approaches transformer quality at language-modelling scale. |
| 2024 | **Hybrid era and frontier attention variants** [\[24\]](#ref-24), [\[51\]](#ref-51), [\[52\]](#ref-52), [\[53\]](#ref-53), [\[54\]](#ref-54) | Mamba-2 formalises the SSM–attention duality. Jamba interleaves transformer and Mamba layers with MoE. DeepSeek-V2 introduces Multi-head Latent Attention (MLA). FlashAttention-3 exploits Hopper's TMA, reaching 75% utilisation. The trajectory is no longer linear: pure-attention, SSM-attention hybrids, and attention-free SSMs now coexist at frontier scale. |

## State of the art

_As of 2026-05-13._

> [!important] State of the art summary
> As of mid-2026, production attention is a thin set of well-understood building blocks layered onto the decoder-only transformer: an IO-aware exact kernel (FlashAttention-2 on A100, FlashAttention-3 on Hopper), a KV-cache layout that is almost universally Grouped-Query Attention with MLA emerging on the DeepSeek-V2/V3 line, RoPE positional encoding with YaRN/NTK-aware extension for long context, and Ring/Striped sequence-parallel attention to push context windows past 1M tokens [\[21\]](#ref-21), [\[23\]](#ref-23), [\[25\]](#ref-25), [\[28\]](#ref-28). Sliding-window attention persists as an engineering shortcut in selected open models. Attention-sink stabilisation (anchor on the first few tokens) is now a standard streaming-inference trick.
>
> The post-attention frontier has consolidated around hybrid attention/SSM architectures. Pure-SSM models (Mamba, Mamba-2) and pure-linear-attention recurrents (RWKV-6, RetNet) reach competitive quality at 1-13B but fall behind on retrieval-heavy and in-context-learning benchmarks; the live consensus is that interleaved hybrids (Jamba, Samba-style stacks) win on the quality-vs-throughput Pareto frontier, while every frontier-scale (>100B) deployment still runs full attention with GQA/MLA. Lossless KV-cache compression (H2O, DuoAttention, ScissorHands, attention-sink streaming) is the most active inference-systems research area in 2025-2026.

### Production attention kernel

- **FlashAttention-3** is the production kernel on NVIDIA Hopper (H100) and Blackwell (B200/GB200) GPUs: asynchrony via warpgroup specialisation, FP8 support, and ~75% of peak FP16 throughput (1.5-2.0× over FlashAttention-2 on H100) [\[24\]](#ref-24), [\[55\]](#ref-55). FlashAttention-2 remains the default on Ampere (A100) and on consumer Ada cards where the warpgroup primitives FA-3 needs are absent [\[23\]](#ref-23). Evidence: `peer_reviewed` (NeurIPS 2024), confidence: `high`.
- The IO-aware reformulation introduced in FlashAttention-1 [\[22\]](#ref-22) is now the universal substrate — every production stack (PyTorch SDPA, TransformerEngine, vLLM, SGLang, TensorRT-LLM) dispatches to one of FA-2 / FA-3 / cuDNN's FA backend by default. Evidence: `blog_post`, confidence: `high`.

### Production KV-cache layout

- **Grouped-Query Attention (GQA)** is the de facto KV-cache layout across the open frontier: Llama-3 / Llama-3.1 / Llama-3.3 (8 KV heads), Mistral-Large, Qwen-2 / Qwen-2.5, Gemma-2, and Mixtral all ship GQA [\[21\]](#ref-21). Evidence: `official_doc`, confidence: `high`.
- **Multi-head Latent Attention (MLA)** is the DeepSeek line's KV-compression alternative: K and V are jointly compressed into a low-rank latent (per-token ~70 floats vs ~1 KB for vanilla MHA), then re-expanded inside the attention computation. MLA was introduced in DeepSeek-V2 [\[53\]](#ref-53), [\[56\]](#ref-56) and is retained in DeepSeek-V3 / R1; it gives a ~14× KV-cache reduction over MHA and ~4× over GQA at matched quality on the DeepSeek line. Evidence: `preprint`, confidence: `high`.
- Multi-Query Attention [\[20\]](#ref-20) is now rare in new releases; models that historically used MQA have largely migrated to GQA. Evidence: `blog_post`, confidence: `medium`.

### Long-context attention

- **Ring Attention** [\[28\]](#ref-28) and its **Striped** variant are the sequence-parallel substrate behind 1M+ token windows. Gemini 1.5 / 2.0 / 2.5 report context windows up to 2M tokens; Claude 4 / 4.5 ship 200K-token windows as standard and have demonstrated 1M for selected customers; Llama-3.1 / 3.3 list 128K. Evidence: `official_doc` for window sizes; `preprint` for the kernel mechanics, confidence: `high`.
- **YaRN** [\[27\]](#ref-27), [\[57\]](#ref-57) is the dominant RoPE-extension recipe for stretching pretrained context windows post-hoc — it combines NTK-aware base rescaling with attention-scaling and is used or referenced by Qwen-2 long-context variants, the Llama-3 64K → 128K extension, and Yi-200K. Evidence: `preprint` (ICLR 2024), confidence: `high`.
- **Sliding-window attention** is still in production: Mistral 7B (window=4096) [\[39\]](#ref-39), Gemma 2 (alternating sliding/full layers at 4096), and Phi-3 variants. Mistral-Large and Llama-3 dropped sliding-window in favour of full attention with RoPE/YaRN extension. Evidence: `official_doc`, confidence: `high`.
- **Attention-sink streaming** [\[58\]](#ref-58) — keep the first few tokens in the KV cache forever, evict the rest with a sliding window, and the model stays coherent across arbitrarily long streams. Now standard in streaming-inference stacks (StreamingLLM, vLLM's `--enable-prefix-caching`, NVIDIA Triton). Evidence: `peer_reviewed` (ICLR 2024), confidence: `high`.

### Position encoding

- **RoPE** [\[25\]](#ref-25) is universal in 2025-2026 open models (Llama-2/3/4, Mistral, Qwen, Gemma 2, DeepSeek, Yi, Phi-3) and is the assumed default in every YaRN/NTK-aware extension method [\[57\]](#ref-57). ALiBi [\[26\]](#ref-26) survives only in BLOOM-descended stacks and a handful of MPT/Falcon variants. Evidence: `peer_reviewed`, confidence: `high`.

### Post-attention contenders

- **Mamba-2** [\[51\]](#ref-51), [\[59\]](#ref-59) reframes selective state-space models as a structured-attention dual (the SSD framework) and ships 2.8× faster training than the original Mamba; competitive with similarly sized transformers at 1-7B but still behind on associative-recall benchmarks like the Phonebook / Needle-in-a-Haystack family. Evidence: `peer_reviewed` (ICML 2024), confidence: `high`.
- **Jamba** [\[52\]](#ref-52), [\[60\]](#ref-60) and **Jamba-1.5** interleave Mamba layers with attention (and MoE), and demonstrate the hybrid architecture wins on the quality-vs-KV-cache Pareto: Jamba-1.5-Large (398B total, 94B active) is the largest open hybrid model as of 2026 and matches or beats Llama-3-70B on most reasoning benchmarks at ≈4× longer effective context per GB of KV. Evidence: `preprint`, confidence: `medium`.
- **RWKV-6 ("Eagle/Finch")** [\[61\]](#ref-61) generalises RWKV's linear-attention recurrence to matrix-valued states and data-dependent time-decay; closes the gap with transformers at 7B on many language-modelling benchmarks but still trails on math and code. Evidence: `preprint`, confidence: `medium`.
- **RetNet** [\[62\]](#ref-62) — the retentive-network formulation with parallel/recurrent/chunkwise duals — has not been adopted by any major frontier or open-frontier lab as the primary architecture; cited primarily as a theoretical sibling of linear attention. Evidence: `preprint`, confidence: `medium`.
- Current consensus among practitioners and post-2024 survey work is that **hybrid attention + SSM stacks are the most promising sub-quadratic architecture class**; no pure non-attention model has matched a frontier dense transformer above 70B on retrieval-heavy benchmarks. Evidence: `blog_post`, confidence: `medium`.

### Lossless KV-cache compression

- **H2O / Heavy-Hitter Oracle** [\[63\]](#ref-63) keeps a budget of "heavy hitter" tokens (those with high accumulated attention mass) plus a sliding recent window; 2-10× KV reduction at flat-ish quality on most decoding tasks. Evidence: `peer_reviewed` (NeurIPS 2023), confidence: `high`.
- **DuoAttention** [\[64\]](#ref-64) partitions heads into **retrieval heads** (full KV cache) and **streaming heads** (constant-size sink + recent window), learned per model. Achieves 2.5× prefill speedup and 6.4× KV-memory reduction on Llama-3 at no quality loss. Evidence: `preprint`, confidence: `medium`.
- ScissorHands, SnapKV, and Quest are the other 2024-2025 entrants in this space; the field is converging on the observation that a small subset of "retrieval heads" carries most of the long-range work and the rest can be heavily compressed. Evidence: `preprint`, confidence: `medium`.

### Open problems and active research areas as of 2026

- **Hybrid architectures at frontier scale** — will a Jamba-style or Samba-style hybrid match dense full-attention at >100B and on RULER / NIAH-style retrieval? As of 2026-05 the answer is "close but not yet at GPT-5 / Claude 4.5 quality on long-context retrieval." Evidence: `preprint`, confidence: `medium`.
- **Lossless KV-cache compression** — production deployments are still on full GQA/MLA KV; H2O / DuoAttention / ScissorHands have not yet displaced full-cache decoding in major serving stacks (vLLM ships them as flags, not defaults). Evidence: `blog_post`, confidence: `high`.
- **Streaming + attention sinks** — attention-sink stabilisation is now baseline, but the theory of why sinks form (Darcet et al.'s register-tokens line, Xiao et al.'s positional-bias story) is still contested. Evidence: `peer_reviewed`, confidence: `medium`.
- **MLA generalisation** — whether MLA's joint low-rank K/V compression generalises beyond DeepSeek's training recipe is an open question; no non-DeepSeek model has shipped MLA in production as of 2026-05. Evidence: `preprint`, confidence: `medium`.

## Key papers

The attention literature splits into four eras: (i) the encoder-decoder attention bolt-ons of 2014–2015 that made neural machine translation viable; (ii) the 2017 transformer that made attention the entire architecture; (iii) the 2019–2023 wave of inference-efficiency variants (MQA, GQA, FlashAttention, RoPE/ALiBi/YaRN) that shaped every production LLM after GPT-3; and (iv) the 2022 mechanistic-interpretability work that recast attention heads as composable circuits inside the residual stream. The list below is the minimum-viable canon — read these in order and the rest of the field's papers are footnotes on them.

- **Bahdanau, Cho, Bengio (2014) — "Neural Machine Translation by Jointly Learning to Align and Translate"** [\[17\]](#ref-17) — first additive ("Bahdanau") attention mechanism; let an RNN decoder soft-search over the entire source sequence instead of compressing it to a fixed-length vector. The conceptual ancestor of every later attention variant.
- **Luong, Pham, Manning (2015) — "Effective Approaches to Attention-based Neural Machine Translation"** [\[18\]](#ref-18) — introduced multiplicative ("Luong") attention and the global-vs-local distinction; established the dot-product attention score that Vaswani et al. would scale up two years later.
- **Vaswani et al. (2017) — "Attention Is All You Need"** [\[19\]](#ref-19) — the foundational paper: removed recurrence entirely, introduced scaled dot-product attention, multi-head attention, sinusoidal positional encoding, and the encoder–decoder transformer block. Every frontier LLM since 2019 is a direct descendant.
- **Shazeer (2019) — "Fast Transformer Decoding: One Write-Head Is All You Need" (Multi-Query Attention)** [\[20\]](#ref-20) — first systematic attack on the KV-cache memory-bandwidth bottleneck at inference. Shared one K/V head across all query heads; lost a little quality but unlocked usable autoregressive decoding at scale.
- **Ainslie et al. (2023) — "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"** [\[21\]](#ref-21) — generalised MQA with an intermediate number of K/V groups and showed how to up-train an existing MHA checkpoint cheaply. GQA with `g=8` is the default in Llama-2/3, Mistral, and (per architectural disclosures) most current frontier decoders including Claude.
- **Dao et al. (2022) — "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** [\[22\]](#ref-22) — recognised that on GPU the attention bottleneck is HBM↔SRAM traffic, not FLOPs. Tiled the softmax to fit in SRAM and recomputed instead of materialising the N×N attention matrix; 2–4× speedup, no approximation, became the de-facto attention kernel within twelve months.
- **Dao (2023) — "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"** [\[23\]](#ref-23) — re-partitioned work across thread blocks and warps to push A100 utilisation to 50–73% of theoretical peak; this is the kernel that shipped inside most 2023–2024 training stacks.
- **Shah et al. (2024) — "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"** [\[24\]](#ref-24) — adds Hopper-specific asynchrony (warp-specialisation, TMA) and FP8 with incoherent processing; ~1.5–2× over FA2 on H100 and the current production kernel for frontier-scale training and serving as of mid-2026.
- **Su et al. (2021) — "RoFormer: Enhanced Transformer with Rotary Position Embedding" (RoPE)** [\[25\]](#ref-25) — replaced additive positional embeddings with a rotation applied directly to Q and K in each attention head. RoPE both encodes absolute position and recovers relative position in the dot product; it is the positional encoding inside Llama, Mistral, DeepSeek, Qwen, and (per Anthropic's architectural disclosures) Claude.
- **Press, Smith, Lewis (2021) — "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (ALiBi)** [\[26\]](#ref-26) — penalised attention scores by query-key distance instead of adding embeddings, enabling clean length extrapolation. ALiBi lost the production race to RoPE+YaRN but the paper is the cleanest framing of the extrapolation problem.
- **Peng et al. (2023) — "YaRN: Efficient Context Window Extension of Large Language Models"** [\[27\]](#ref-27) — compute-efficient recipe for extending a RoPE-trained model's context window by ~10× with 10× fewer tokens than previous methods. The pragmatic answer to "how do we ship a 128k-context Llama".
- **Liu, Zaharia, Abbeel (2023) — "Ring Attention with Blockwise Transformers for Near-Infinite Context"** [\[28\]](#ref-28) — distributed blockwise self-attention across devices with K/V communication overlapped with compute, breaking the single-device memory wall and enabling million-token training/inference. The backbone of the Gemini 1.5 and large-context-claim regime.
- **Olsson et al. (Anthropic, 2022) — "In-context Learning and Induction Heads"** [\[29\]](#ref-29) — identified a specific two-head circuit ("induction heads") that implements in-context pattern completion, and argued they are the mechanistic substrate of in-context learning. The cleanest demonstration that attention heads compose into interpretable algorithms, and the entry point to the Anthropic circuits programme.

## Key people and organisations

The attention-mechanism lineage spans three eras: the RNN-era pioneers who invented soft alignment (2014–2016), the "Attention Is All You Need" team who retired recurrence (2017), and the efficiency/long-context wave who made attention practical at scale (2019–2024).

### Individuals

| Person                    | Affiliation (2025–2026)                                                                    | Key contribution                                                                                                                                                                                                                  |
| ------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dzmitry Bahdanau**      | Microsoft Research (formerly MILA)                                                         | Soft, content-based attention for NMT [\[17\]](#ref-17), [\[65\]](#ref-65)                                                                                                                                                       |
| **KyungHyun Cho**         | NYU / Genentech (*affiliation uncertain as of 2026*)                                       | Co-author of Bahdanau attention and the GRU encoder-decoder [\[17\]](#ref-17), [\[65\]](#ref-65)                                                                                                                                 |
| **Yoshua Bengio**         | Prof., U. Montréal; MILA (stepped down as scientific director 2025); co-president, LawZero | Senior author of bahdanau-attention-2014; Turing Award 2018 [\[17\]](#ref-17), [\[65\]](#ref-65)                                                                                                                                 |
| **Ashish Vaswani**        | Co-founder & CEO, Essential AI (prev. Google Brain, Adept)                                 | Lead author of "Attention Is All You Need" [\[19\]](#ref-19), [\[66\]](#ref-66)                                                                                                                                                  |
| **Noam Shazeer**          | Google DeepMind (returned Aug 2024; prev. Character.AI co-founder)                         | Co-author of vaswani-attention-2017; introduced MQA, GeGLU [\[66\]](#ref-66), [\[67\]](#ref-67)                                                           |
| **Niki Parmar**           | Co-founder, Essential AI                                                                   | Co-author of vaswani-attention-2017; Image Transformer [\[19\]](#ref-19), [\[66\]](#ref-66)                                                                                                                                      |
| **Llion Jones**           | Co-founder, Sakana AI                                                                      | Co-author of vaswani-attention-2017 [\[19\]](#ref-19), [\[66\]](#ref-66)                                                                                                                                                         |
| **Aidan N. Gomez**        | Co-founder & CEO, Cohere                                                                   | Co-author of vaswani-attention-2017 [\[19\]](#ref-19), [\[66\]](#ref-66)                                                                                                                                                         |
| **Lukasz Kaiser**         | OpenAI (*as of 2026 uncertain*)                                                            | Co-author of vaswani-attention-2017; co-author of Performer [\[66\]](#ref-66), [\[68\]](#ref-68)                                                        |
| **Illia Polosukhin**      | Co-founder, NEAR Protocol                                                                  | Co-author of vaswani-attention-2017 [\[19\]](#ref-19), [\[66\]](#ref-66)                                                                                                                                                         |
| **Jakob Uszkoreit**       | Founder, Inceptive                                                                         | Co-author of vaswani-attention-2017; applied transformer attention to RNA design [\[19\]](#ref-19), [\[66\]](#ref-66)                                                                                                            |
| **Rewon Child**           | Independent / industry (prev. OpenAI)                                                      | First author of Sparse Transformers [\[36\]](#ref-36), [\[69\]](#ref-69)                                                                                                                                                         |
| **Alec Radford**          | Advisor, Thinking Machines Lab (left OpenAI Dec 2024)                                      | Co-author of Sparse Transformers; GPT-1/2/CLIP [\[36\]](#ref-36), [\[69\]](#ref-69)                                                                                                                                              |
| **Ilya Sutskever**        | Co-founder & CEO, Safe Superintelligence Inc. (SSI)                                        | Co-author of Sparse Transformers [\[36\]](#ref-36), [\[69\]](#ref-69)                                                                                                                                                            |
| **Iz Beltagy**            | Allen Institute for AI (AI2)                                                               | First author of Longformer [\[39\]](#ref-39), [\[70\]](#ref-70)                                                                                                                                                                  |
| **Krzysztof Choromanski** | Google Research                                                                            | Lead author of Performers (FAVOR+) [\[42\]](#ref-42), [\[68\]](#ref-68)                                                                                                                                                          |
| **Jianlin Su**            | Independent / Shengsheng AI                                                                | Sole first author of RoPE [\[25\]](#ref-25), [\[71\]](#ref-71)                                                                                                                                                                   |
| **Ofir Press**            | Princeton / Meta AI Research                                                               | First author of ALiBi [\[26\]](#ref-26), [\[72\]](#ref-72)                                                                                                                                                                       |
| **Tri Dao**               | Asst. Prof. CS, Princeton; Co-founder & Chief Scientist, Together AI                       | First author of FlashAttention and FlashAttention-2; co-author of Mamba [\[73\]](#ref-73), [\[74\]](#ref-74), [\[75\]](#ref-75) |
| **Albert Gu**             | CMU                                                                                        | First author of Mamba; S4/Mamba lineage [\[50\]](#ref-50), [\[75\]](#ref-75)                                                                                                                                                     |
| **Reiner Pope**           | Google DeepMind                                                                            | First author of canonical transformer-inference cost model [\[46\]](#ref-46), [\[76\]](#ref-76)                                                                                                                                  |
| **Jeff Dean**             | Google DeepMind (Chief Scientist)                                                          | Co-author of pope-efficient-inference-2022 [\[76\]](#ref-76)                                                                                                                                                                     |
| **Joshua Ainslie**        | Google Research                                                                            | Lead author of GQA [\[21\]](#ref-21), [\[77\]](#ref-77)                                                                                                                                                                          |
| **Hao Liu**               | UC Berkeley (PhD)                                                                          | First author of Ring Attention [\[28\]](#ref-28), [\[78\]](#ref-78)                                                                                                                                                              |
| **Pieter Abbeel**         | Prof. EECS, UC Berkeley; Co-founder, Covariant                                             | Co-author of Ring Attention [\[78\]](#ref-78)                                                                                                                                                                                    |
| **Chris Olah**            | Anthropic (interpretability research lead)                                                 | transformer-circuits.pub; induction heads; shaped Anthropic's interpretability programme                                                                                                                                          |
| **Nelson Elhage**         | Anthropic                                                                                  | A Mathematical Framework for Transformer Circuits                                                                                                                                                                                 |
| **Catherine Olsson**      | Anthropic                                                                                  | Co-author of induction-heads paper (2022)                                                                                                                                                                                         |
| **Neel Nanda**            | Google DeepMind (leads mech interp team; formerly Anthropic)                               | TransformerLens; induction-head phenomena                                                                                                                                                                                         |

### Organisations

| Organisation | Role |
|---|---|
| **Google Brain → Google DeepMind** | Where scaled dot-product attention was born (Vaswani 2017); still the single largest producer of attention-mechanism papers [\[66\]](#ref-66), [\[67\]](#ref-67), [\[76\]](#ref-76), [\[77\]](#ref-77) |
| **Anthropic** | Mechanistic interpretability of attention: transformer-circuits.pub; induction heads as a unit of in-context learning |
| **Princeton + Together AI (Tri Dao group)** | FlashAttention lineage [\[73\]](#ref-73), [\[74\]](#ref-74), [\[75\]](#ref-75) |
| **Stanford (Hazy Research / CRFM)** | Origins of FlashAttention; S4 and Mamba; hardware-aware algorithms paradigm |
| **OpenAI** | Sparse Transformers; GPT-* scaling of attention-based LLMs [\[36\]](#ref-36) |
| **Allen Institute for AI (AI2)** | Longformer and BigBird [\[39\]](#ref-39) |
| **CMU** | Mamba / S4 lineage (Albert Gu) [\[50\]](#ref-50) |
| **MILA / U. Montréal** | Foundational seq2seq and attention (Bahdanau, Cho, Bengio) [\[17\]](#ref-17) |
| **Cohere** | Aidan Gomez (vaswani-2017 co-author) leads; production transformer LLM APIs |
| **Essential AI** | Vaswani + Parmar co-founded; LLM/agent product startup |
| **Safe Superintelligence Inc. (SSI)** | Ilya Sutskever co-founded |

## Practical applications

Attention mechanisms are not one application but the shared architectural substrate of an entire application stack. The categories below organise the major deployment areas by which attention variant they rely on, what the engineering challenge is, and which papers ground the claim.

### LLM serving and inference

Production decoder-only LLMs (Claude, GPT, Llama, Mistral, Gemini) serve autoregressive text generation where the operational bottleneck is the **KV cache**: at decode time, streaming K and V tensors from HBM is the binding constraint, not arithmetic [\[46\]](#ref-46). Two attention variants directly address this:

- **Grouped-Query Attention (GQA)** with `g=8` K/V groups is the de facto serving layout in Llama-2/3, Mistral, Qwen-2, Gemma-2, and (per architectural disclosures) Claude 3+. It reduces KV-cache VRAM by a factor of `h/g` over full MHA with negligible quality loss, enabling larger batch sizes and longer contexts within fixed accelerator memory [\[21\]](#ref-21).
- **Multi-Query Attention (MQA)** — the predecessor, sharing one K/V head across all query heads — still appears in some older Falcon and StarCoder checkpoints but has been superseded by GQA in new releases [\[20\]](#ref-20).
- **PagedAttention / vLLM** virtualises the KV cache into non-contiguous physical blocks (analogous to OS virtual memory), eliminating fragmentation across variable-length requests and enabling high-throughput continuous batching in serving frameworks (vLLM, SGLang, TensorRT-LLM). This is the primary mechanism enabling commercially viable LLM API serving at the token throughputs required for products like the Claude API [\[91\]](#ref-91).
- **FlashAttention-2/3 kernels** are the universal prefill kernel on NVIDIA A100 (FA-2) and H100/Blackwell (FA-3): tiled SRAM computation eliminates the materialised N×N score matrix, giving 2–4× wall-clock speedup over naive attention at no quality cost [\[22\]](#ref-22), [\[23\]](#ref-23), [\[24\]](#ref-24).

### Long-context document QA and retrieval

Two competing paradigms handle documents that exceed a few thousand tokens: retrieval-augmented generation (RAG) chunks documents and retrieves relevant passages before passing them as a short context, and **long-context attention** simply ingests the entire document in a single forward pass.

- **RAG** reduces peak attention cost (smaller `n` → less quadratic cost) but introduces retrieval errors — the *lost-in-the-middle* effect shows that even when the retrieved chunk is inserted, LLMs exhibit substantially worse recall for information in the middle of the context relative to the ends. Attention pattern behaviour (U-shaped probability mass over position) is the direct mechanistic cause.
- **Long-context full attention** (128K–1M+ tokens) uses Ring Attention for distributed sequence-parallel computation: the sequence is sharded across devices in a ring topology, and K/V blocks are passed ring-buffer style with compute overlapping communication, breaking the single-device memory wall [\[28\]](#ref-28). Gemini 1.5/2.0 (1M context), Claude 4/4.5 (200K standard), and Llama-3.1/3.3 (128K) are production examples.
- **Context extension via YaRN** is the standard recipe for stretching a RoPE-trained model's context post-hoc — NTK-aware base rescaling plus attention temperature scaling, requiring only ~10% of the data needed by prior methods [\[27\]](#ref-27). Used in Qwen-2 long-context variants and Llama-3 128K extensions.

### Code completion and agentic coding

Code completion (GitHub Copilot, Claude Code, Cursor) and multi-turn agentic loops (Claude Code / Agents SDK) are the highest-stakes long-context applications today:

- Attention over a codebase context requires handling **long structured contexts** (system prompts, tool definitions, repository contents, prior tool calls) where the prefix recurs across every agent step. Without prompt caching, each tool call re-runs prefill over the full prefix — the quadratic cost FlashAttention was built to amortise. Anthropic's docs explicitly target this pattern, recommending breakpoints on `tools` and the last user message and `max_tokens: 0` pre-warming for agent servers with large tool catalogues [\[81\]](#ref-81).
- **Claude Sonnet** is Anthropic's current production recommendation for software-engineering agents: its 200K context window can hold an entire repository, and its attention architecture (GQA + RoPE per industry convention) handles the long, semi-structured prefill efficiently.
- The KV-cache budget is the binding constraint on how long a Claude Code agent loop can run before context-rotation logic must evict earlier turns; this is the reason vLLM-style continuous batching and prompt caching exist.

### Multimodal (vision–language) applications

Vision Transformers (ViT) apply the same self-attention mechanism to image patches, and cross-attention bridges image encoders to text decoders:

- **Vision Transformer (ViT)** divides an image into fixed-size patches, projects each to a token embedding, and applies standard multi-head self-attention. At scale this matches or exceeds CNN accuracy; the patch-token formulation makes the same FlashAttention kernels applicable to both image and text [\[92\]](#ref-92).
- **Cross-attention in image–text models** (GPT-4V, Claude vision, Gemini, LLaVA, Flamingo) binds visual tokens from a ViT encoder to the text decoder via cross-attention where queries come from the language model and keys/values from the image encoder — the same encoder-decoder cross-attention from Vaswani et al. [\[19\]](#ref-19) applied to a vision-language interface.
- Claude 3/4 vision capabilities use this cross-attention pattern; Anthropic has not published the specific ViT architecture, but the mechanistic structure (image tokens as keys/values, text queries) is consistent across all production vision-language models.

### Speech recognition and seq2seq

Whisper (OpenAI, 2022) demonstrated that encoder-decoder attention trained on large-scale weakly supervised audio data is sufficient for near-human speech recognition:

- **Whisper** uses a convolutional feature extractor feeding an encoder-decoder transformer; the encoder attends over log-Mel spectrogram frames with full self-attention, and the decoder cross-attends over encoder outputs while autoregressively predicting text tokens — a direct application of Vaswani et al.'s encoder-decoder architecture [\[19\]](#ref-19), [\[93\]](#ref-93).
- **Conformer** (Google, 2020) combines convolutional local feature extraction with self-attention for global context in a hybrid block, and is the dominant production speech architecture in streaming ASR (Google ASR, YouTube captions). The convolutional layers handle local acoustic features; the attention handles long-range prosodic and semantic dependencies.
- Both architectures rely on standard scaled dot-product attention; FlashAttention kernels apply directly, and the KV-cache pattern is the same as text decoders during beam search.

### Mechanistic interpretability tooling

Anthropic's interpretability programme and the wider mech-interp community treat **attention heads as the primary object of analysis**:

- **Induction-head probes** — Olsson et al. (2022) define a two-head induction circuit and identify it across models via attention-pattern probes (the key diagnostic is whether attention on token `B` given a `[A][B]...[A]` prefix correctly attends to the earlier `[B]` position). This is now standard in any mechanistic analysis of in-context learning [\[29\]](#ref-29).
- **Attribution patching / activation patching** — Given a clean and corrupted forward pass, attribution patching re-routes residual-stream activations at specific positions and layers to measure each attention head's causal contribution to the output logit. This is the workhorse technique behind IOI (Indirect Object Identification) circuits, mover-heads, and inhibitor-heads; TransformerLens (Neel Nanda's library) exposes this API over any decoder-style model.
- **QK / OV decomposition** splits each head's function into the attention-pattern part (QK circuit) and the value-move part (OV circuit), allowing independent study of *what each head attends to* versus *what it writes* — used in every Anthropic circuits paper since 2021.
- Scaling Monosemanticity [\[83\]](#ref-83) and Attribution Graphs [\[84\]](#ref-84) work on Claude 3 Sonnet and Claude 3.5 Haiku respectively, treating attention patterns as fixed inputs to the attribution graph but not decomposing individual head behaviour — a known gap the field is actively addressing.

### Production efficiency: sliding-window, sparse, and prompt caching

- **Sliding-window attention** (Mistral 7B: 4096-token window; Gemma-2: alternating sliding/full layers) limits each query to a local window of keys, reducing per-layer KV-cache to `O(w · d)` where `w` is the window size. Based on Longformer's local-attention formulation [\[39\]](#ref-39), and sufficient for tasks where relevant context is always local. Mistral-Large and Llama-3 dropped it in favour of full attention with RoPE/YaRN as hardware costs fell.
- **Prompt caching** (Claude API, `cache_control: {type: "ephemeral"}`) caches the processed prefix's K and V tensors; on subsequent requests they are loaded from cache instead of re-running prefill. Cache reads are priced at ~10% of base input tokens [\[81\]](#ref-81). This is the productisation of KV-cache reuse from the serving-systems literature [\[46\]](#ref-46).
- **Attention-sink streaming** keeps the first 1–4 sink tokens permanently in the KV cache while evicting the rest with a sliding window; this stabilises perplexity in arbitrarily long streaming contexts without growing the cache. Now a standard flag in vLLM (`--enable-prefix-caching`) and the default in StreamingLLM inference stacks [\[94\]](#ref-94).

## Claude / Anthropic relevance

Attention mechanisms are simultaneously the *substrate* Claude runs on and the *object of study* in Anthropic's mechanistic-interpretability programme. Five concrete connections:

- **Induction heads as the mechanistic basis of in-context learning.** Olsson et al. (2022) [\[29\]](#ref-29) define an induction head as a two-attention-head circuit — a "previous-token head" that copies token *i*'s identity into position *i+1*, composed with a head whose query at the second occurrence of `[A]` matches the key written by the previous-token head at the position just after the first `[A]`, so attention falls on `[B]`, and the OV circuit copies `[B]` into the logits[\[79\]](#ref-79). The paper's central claim is that induction heads "might constitute the mechanism for the actual majority of all in-context learning in large transformer models"[\[79\]](#ref-79), formed during a sharp loss-curve phase change. This is Anthropic's leading mechanistic hypothesis for *why* attention works at all in autoregressive LMs.
- **QK / OV decomposition as Anthropic's preferred attention-head abstraction.** Elhage et al. (2021) [\[80\]](#ref-80) reframe each attention head as two largely independent low-rank operators: the **QK circuit** $W_Q^\top W_K$ that produces the attention pattern, and the **OV circuit** $W_O W_V$ that determines how an attended-to token modifies the output[\[80\]](#ref-80). The framework also formalises head **composition** through keys, queries, and values — the algebraic precondition under which two-layer transformers can implement non-trivial in-context algorithms (e.g. induction). Every later transformer-circuits paper builds on this framework as its attention-side ground truth.
- **Prompt caching is the productisation of KV-cache reuse.** Claude's `cache_control: {type: "ephemeral"}` breakpoint (on `tools`, `system`, or `messages` blocks) caches the processed prefix's K and V tensors, then on subsequent requests loads them instead of re-running prefill [\[81\]](#ref-81). Cache reads are priced at 10% of base input ($0.10/MTok on Haiku 4.5, $0.30 on Sonnet 4.6, $0.50 on Opus 4.7) versus 1.25× base for the write — essentially the inference-cost equation from Pope et al. 2022 [\[46\]](#ref-46) productised: amortise the quadratic prefill across many decode-bound requests. Minimum cacheable prefix is 4096 tokens on Opus/Haiku and 2048 on Sonnet, reflecting block-tile granularity of the underlying attention kernel.
- **Long-context Claude reflects the long-context attention literature, but architecture is undisclosed.** All Claude 3+ models ship with a 200K-token context window standard (1M+ on request as of 2024) [\[82\]](#ref-82). Anthropic has *not* publicly disclosed Claude's attention architecture — the announcement explicitly omits whether GQA [\[21\]](#ref-21), MQA [\[20\]](#ref-20), RoPE [\[25\]](#ref-25), or sliding-window variants are used. Industry convention as of 2025 (Llama 3, Mistral, Gemma, DeepSeek all use GQA + RoPE) makes it overwhelmingly likely Claude does as well, but this is *inference*, not disclosure — flagged at `evidence_level: anecdotal` in `claude_anthropic_connections`.
- **Claude Code / Agent SDK long agentic runs are KV-cache-budget-bound.** In a multi-turn tool-use loop, the system prompt + tool definitions + earlier turns make up a growing prefix that recurs across every step. Without prompt caching, each tool call re-runs prefill over the full prefix — the quadratic cost the FlashAttention [\[22\]](#ref-22) / FlashAttention-2 [\[23\]](#ref-23) literature is built to amortise. Anthropic's docs explicitly target this pattern, recommending breakpoints on `tools` and the last user message, and `max_tokens: 0` pre-warming for agent servers with large tool catalogues [\[81\]](#ref-81). The KV-cache budget is, in practice, the binding constraint on how long a Claude Code agent loop can run before context-rotation logic must drop earlier turns.
- **Mech-interp work on Claude 3 Sonnet treats attention patterns as a fixed substrate.** Templeton et al.'s *Scaling Monosemanticity* (2024) [\[83\]](#ref-83) trains sparse autoencoders on Claude 3 Sonnet's residual stream and recovers interpretable directional features. The methodology deliberately *brackets* attention — the SAE sees the residual stream, which is the sum of all attention-head outputs at a layer, but does not decompose individual heads' contributions[\[83\]](#ref-83). The 2025 attribution-graphs / "biology of LLMs" work on Claude 3.5 Haiku makes this explicit: attribution graphs incorporate the original model's attention patterns as fixed inputs but "don't attempt to replace or interpret them" [\[84\]](#ref-84). Attention-head interpretation at Claude scale is therefore an explicitly *open* methodological frontier inside Anthropic's own programme.

## Limitations and open problems

> [!warning] Open problems
> Attention's two architectural costs — **quadratic prefill** in sequence length and a **linear-in-length KV cache** at decode — are the dominant bottlenecks at long context, and no approximation has matched full attention's quality at frontier scale [\[66\]](#ref-66), [\[67\]](#ref-67). Position-encoding extrapolation [\[71\]](#ref-71), [\[72\]](#ref-72), attention sinks under streaming inference [\[85\]](#ref-85), the *lost-in-the-middle* U-shape over long contexts [\[86\]](#ref-86), and the principled-length-generalisation problem [\[87\]](#ref-87) all remain empirically patched rather than solved. Mechanistic understanding of what attention learns at scale is still patchy.

### Architectural costs

- **Quadratic prefill cost in sequence length** — Self-attention computes an `n × n` similarity matrix per layer, giving prefill cost `O(n² · d)` [\[19\]](#ref-19). FlashAttention removes the materialised `n × n` matrix to win a *constant-factor* 2-4× wall-clock speedup and linear-in-`n` memory, but the asymptotic FLOP count is unchanged [\[22\]](#ref-22). FlashAttention-3 [\[88\]](#ref-88) pushes Hopper utilisation to ~75% (FP16) and introduces FP8 attention; that is the current best workaround, not a sub-quadratic algorithm. **Severity: high (fundamental, architectural).**
- **KV-cache memory at long context** — At decode time each layer stores `K, V ∈ ℝ^{n × h × d_head}`, so the cache grows `O(n · h · d)` per layer and dominates serving cost beyond a few thousand tokens [\[20\]](#ref-20). The current production mitigations are Multi-Query Attention (one shared K/V head, [\[67\]](#ref-67)) and Grouped-Query Attention with typically `g = 8` groups [\[21\]](#ref-21), which give most of MQA's memory win back at full-MHA quality; Multi-head Latent Attention (DeepSeek-V2/V3) compresses the cache further via a learned low-rank latent. None of these change the linear-in-`n` scaling. **Severity: high (drives the entire KV-compression literature).**

### Long-context failure modes

- **Length generalisation / train-short-test-long** — Vanilla absolute position embeddings extrapolate poorly [\[26\]](#ref-26). ALiBi was the first clean fix via linear-distance attention biases [\[26\]](#ref-26); RoPE is the current default and is extended at inference via NTK-aware scaling and YaRN [\[25\]](#ref-25). The deeper issue is task-level rather than encoding-level: Anil et al. show that even with capable pretrained LLMs and naive fine-tuning, models exhibit substantial length-generalisation deficits *independent of model scale* on reasoning tasks; scratchpad/chain-of-thought prompting partially closes the gap but does not eliminate it [\[87\]](#ref-87). **Severity: medium.**
- **Attention sinks / streaming inference** — Xiao et al. show that decoder attention heads concentrate a large fraction of probability mass on the *first few tokens* regardless of semantic relevance — the residual stream uses them as a `softmax` overflow valve [\[85\]](#ref-85). The consequence: naïve window-based streaming inference that evicts the earliest tokens collapses perplexity. The current best workaround is StreamingLLM — keep the first 1-4 tokens (the sinks) pinned in the cache while sliding the rest [\[85\]](#ref-85). **Severity: medium.**
- **Lost-in-the-middle / U-shape position bias** — Liu et al. document that long-context LLMs exhibit a pronounced U-shape over input position: accuracy on retrieval / multi-document QA is highest when the relevant information is at the start or end of the context, and *substantially* worse in the middle [\[86\]](#ref-86). The effect persists even on models explicitly trained for long context. Current workarounds — reranking retrieved chunks to put the most relevant near the extremes, query-aware reordering, and contrastive long-context training — are empirical rather than principled [\[86\]](#ref-86). **Severity: high for any long-context application (RAG, agentic, code).**

### Approximate attention

- **Approximate-attention quality gap** — Linear-attention [\[41\]](#ref-41), low-rank (Linformer, [\[40\]](#ref-40)), kernel-feature (Performer / FAVOR+, [\[68\]](#ref-68)), and sliding-window/sparse variants [\[69\]](#ref-69), [\[70\]](#ref-70) all break the quadratic cost asymptotically, but at frontier-LM scale they underperform full softmax attention on perplexity and downstream reasoning. The pragmatic response in 2024-2026 has been *hybrid* stacks — Mamba/SSM blocks interleaved with full-attention blocks ([\[75\]](#ref-75); Jamba; Zamba2) — rather than pure linear-attention models. **Severity: high (motivates the entire post-attention research programme).**

### Interpretability

- **Positional-encoding interpretability** — RoPE encodes position via per-frequency 2D rotations; YaRN, NTK-aware, and Position Interpolation extensions are *empirical* recipes for stretching the rotation frequencies at inference [\[25\]](#ref-25). There is no principled theory predicting which frequency-scaling schedule extrapolates cleanly to what context length, so practitioners search empirically per model. **Severity: medium.**
- **Mechanistic opacity at scale** — Despite landmark interpretability results — induction heads, IOI circuits, attribution graphs — the overwhelming majority of attention-head behaviour in frontier-scale models is uncharacterised. We can identify a few canonical circuits but cannot yet *predict* what a randomly-sampled head will compute, nor account for the bulk of validation-loss reduction across pretraining at the circuit level. **Severity: medium.**

> [!question] Open questions
> 1. Can we get attention's quality with a sub-quadratic asymptotic cost? Mamba-2 and hybrid attention/SSM stacks partially answer this at smaller scales; the frontier-scale answer is still open [\[50\]](#ref-50).
> 2. Is the KV-cache the *fundamental* bottleneck of autoregressive transformer inference, or just a current implementation choice that better attention algorithms (paged, latent, learned-eviction) could erase?
> 3. How do we make length-extrapolation principled rather than empirical? RoPE-extension recipes (YaRN, NTK) currently work without theory; ALiBi has theory but is no longer state-of-the-art [\[71\]](#ref-71), [\[72\]](#ref-72).
> 4. What is the right abstraction for *memory* in long-context models — attention, state-space recurrence, learned retrieval, or something not yet proposed? The lost-in-the-middle effect [\[86\]](#ref-86) and attention-sink dynamics [\[85\]](#ref-85) both suggest pure attention's memory model is mis-specified for very long contexts.

## Evaluation and benchmarks

Four clusters of benchmarks govern the attention-mechanisms field: (1) **long-context quality**, testing whether attention can actually use the context window it claims; (2) **efficient-attention quality**, testing whether sub-quadratic approximations pay the performance price they theoretically should; (3) **attention-head interpretability**, testing whether mech-interp methods correctly identify what individual heads compute; and (4) **throughput and memory efficiency**, testing the hardware-side reality of kernel implementations.

### Summary table (as of 2026-05-13)

| Benchmark | What it measures | Current best result | Confidence |
|---|---|---|---|
| **NIAH (Needle-in-a-Haystack)** | Single-fact retrieval from a distractor context; tests whether a model can locate one injected sentence anywhere in a long context. Known limitation: retrieval-only — passing NIAH does not imply multi-hop or aggregation ability [\[93\]](#ref-93), [\[95\]](#ref-95) | Near-perfect (≥99%) for frontier models (Claude 4, GPT-4o, Gemini 2.0) at 128K; scores degrade for open models above 64K | high |
| **RULER** | Extends NIAH to multi-needle retrieval, variable needle types, multi-hop tracing, and aggregation. 13 tasks across configurable context lengths [\[93\]](#ref-93), [\[95\]](#ref-95) | ~85–90% at 32K for best open models; frontier proprietary models approach 90% at 128K but degrade sharply beyond. Almost all 32K+ models fail above 64K on non-retrieval tasks. | high |
| **InfiniteBench** | First benchmark with mean context > 100K tokens; 12 tasks including long-context QA, summarisation, math, code, and retrieval in English and Chinese. Designed so that retrieving a small number of passages is insufficient — tasks require holistic context understanding [\[96\]](#ref-96) | GPT-4 and Claude models score ~35–45% on the hardest tasks (En.QA, En.MC); no model reaches 70% on the book-level QA tasks as of early 2026. Benchmark is not yet saturated. | medium |
| **BABILong** | Multi-hop reasoning tasks (20 bAbI sub-tasks) embedded in long natural-language distractors; scalable to 50M-token contexts. Key finding: popular LLMs effectively use only 10–20% of context <mark>[kuratov-babilong-2024]</mark> [\[97\]](#ref-97) | Recurrent memory transformers after fine-tuning achieve ~79.9% on single-fact QA at 50M tokens; standard transformer-based LLMs stall at ~60% on single-fact tasks and degrade rapidly on multi-hop sub-tasks at 128K. | high |
| **LOFT** | Benchmarks whether long-context LLMs can subsume retrieval systems, RAG pipelines, and SQL databases. Tasks require contexts up to millions of tokens <mark>[lee-loft-2024]</mark> [\[98\]](#ref-98) | Frontier LLMs (GPT-4-Turbo, Gemini 1.5 Pro) rival state-of-the-art retrieval/RAG systems on text tasks; compositional SQL-like tasks remain unsolved. | medium |
| **ZeroSCROLLS** | Zero-shot benchmark for long-text natural-language understanding (10 tasks including summarisation, QA, NLI). Aggregation tasks (e.g. counting positive reviews) are the hardest category <mark>[shaham-zeroscrolls-2023]</mark> [\[99\]](#ref-99) | GPT-4 / Claude models lead; aggregation sub-tasks still above naive baseline for all models. | medium |
| **Long Range Arena (LRA)** | Suite of 6 tasks (1K–16K tokens) for evaluating sub-quadratic efficient-attention variants: text classification, document retrieval, image classification (pixels as sequences), PathFinder spatial reasoning, ListOps, and retrieval. The de-facto standard for comparing Linformer, Performer, Longformer, Sparse Transformer, etc. <mark>[tay-lra-2020]</mark> [\[100\]](#ref-100) | Full-attention transformer is still competitive or best on most tasks; Performer, Longformer, and BigBird come close. Pathfinder-X and retrieval remain hard for all sub-quadratic models. Benchmark is *not* saturated for efficient variants, but increasingly less representative of frontier-scale production use. | high |
| **IOI (Indirect Object Identification)** | Mechanistic-interpretability micro-benchmark: does a 26-head attention circuit in GPT-2 small correctly identify the indirect object in sentences like "When Mary and John went to the store, John gave a drink to ___". Evaluated with faithfulness, completeness, and minimality metrics <mark>[wang-ioi-2022]</mark> [\[101\]](#ref-101) | The circuit discovered in GPT-2 small achieves ~50% faithfulness, ~90% completeness at the identified 26-head subset. The benchmark is designed to validate mechanistic circuit-discovery workflows rather than frontier-model performance. | high |
| **FlashAttention HBM-bandwidth utilisation** | Hardware-efficiency benchmark measuring what fraction of theoretical GPU peak throughput the attention kernel achieves, measured as FLOP/s attained vs. FLOP/s peak (FP16) or as HBM bandwidth saturation [\[22\]](#ref-22), [\[24\]](#ref-24) | FlashAttention-1: 25–40% of A100 peak. FlashAttention-2: 50–73% of A100 peak. FlashAttention-3 on H100: ~75% FP16 peak (75 TFLOP/s vs ~99.6 TFLOP/s theoretical). FP8 attention reaches ~1.2 PFLOP/s on H100. | high |
| **vLLM / serving-system throughput** | Tokens per second for batched decode across request loads; KV-cache memory per context window as a function of GQA group count. Measured with vLLM's PagedAttention against FasterTransformer / Orca baselines [\[46\]](#ref-46) | vLLM achieves 2–4× throughput over naive serving at same latency; GQA with `g=8` reduces KV-cache memory by ~8× relative to full MHA at matched quality. | medium |

### Methodological notes

**Long-context evals and what they actually test.** NIAH and RULER test *retrieval* from a distractor context, not comprehension. A model can pass NIAH with 99% accuracy by attending only to the first and last few tokens (attention-sink + recency bias) and still fail on multi-hop or aggregation tasks [\[93\]](#ref-93). This is the primary reason RULER was introduced: to separate retrieval-only capability from the more demanding multi-hop tracing and aggregation tasks. BABILong and InfiniteBench are harder still — they require genuine multi-step reasoning across distributed context, not just a proximity-to-endpoint heuristic.

**LRA and the efficient-attention gap.** LRA <mark>[tay-lra-2020]</mark> was the first systematic comparison of sub-quadratic attention variants and remains the standard citation for benchmarking Linformer, Performer, Longformer, and Sparse Transformer. Its tasks cap at 16K tokens, so it does not directly evaluate the 128K–2M regime that defines 2025–2026 frontier deployment. ZeroSCROLLS <mark>[shaham-zeroscrolls-2023]</mark> partially fills this gap on naturalistic tasks. The practical consequence: no efficient-attention variant that scores well on LRA has been shown to *match* full-attention transformers at frontier-scale pretraining on reasoning-heavy tasks; LRA scores are a necessary but not sufficient condition for production adoption.

**IOI and interpretability benchmarks.** The IOI task <mark>[wang-ioi-2022]</mark> introduced a rigorous evaluation protocol for mechanistic circuit claims: (a) *faithfulness* — does ablating the proposed circuit reduce model performance to near-chance? (b) *completeness* — does the circuit alone reproduce model performance? (c) *minimality* — is there a smaller circuit that is equally faithful? The induction-head detection benchmark [\[29\]](#ref-29) is complementary: it identifies heads by their attention-pattern signature (attending to the token one position after a previous occurrence of the current token) and validates via causal interventions. Both protocols are now standard in the Anthropic transformer-circuits programme and in the wider mech-interp community.

**Hardware benchmarks and the HBM story.** The FlashAttention papers [\[73\]](#ref-73), [\[74\]](#ref-74), [\[99\]](#ref-99) frame hardware efficiency as the *primary* metric for attention kernel comparison — because FLOPs are not the bottleneck. The relevant measurement is HBM bandwidth utilisation: what fraction of peak memory bandwidth is used when streaming K, V tensors from HBM to SRAM? FlashAttention-3 reaches ~75% of H100 FP16 peak, compared with the ~30–40% that unfused attention achieves.

**Saturation status summary.** NIAH is effectively saturated for frontier models at ≤128K. LRA is not saturated for efficient-attention variants but is no longer representative of frontier production use. RULER, InfiniteBench, BABILong, LOFT, and ZeroSCROLLS are live and discriminative as of 2026-05-13. IOI is a fixed-task benchmark that does not grow with model capability but serves as a methodological reference standard for circuit-discovery validation.

## Adjacent topics

- **Transformer Architecture** (topic_id: transformer-architecture) (relation: prerequisite) — Attention is the defining operation of the Transformer; every multi-head attention layer is embedded in a Transformer block, so understanding the broader residual-stream architecture is prerequisite to reasoning about attention's role in information routing.
- **Tokenisation** (topic_id: tokenisation) (relation: prerequisite) — Attention operates over token embeddings produced by the tokeniser; the granularity of the vocabulary and the byte/subword boundary choices directly shape what attention heads can align over and attend to.
- **Positional Encoding** (topic_id: positional-encoding) (relation: complement) — Because vanilla dot-product attention is permutation-invariant, it depends entirely on positional encodings (sinusoidal, learned, RoPE, ALiBi) to inject sequence order; the two mechanisms are co-designed.
- **Rotary Position Embedding (RoPE)** (topic_id: rope-positional-encoding) (relation: specialisation) — RoPE is the dominant production positional scheme inside modern attention layers; it encodes relative position by rotating query/key vectors, enabling length extrapolation while remaining a pure attention-layer modification.
- **Attention with Linear Biases (ALiBi)** (topic_id: alibi-positional-encoding) (relation: specialisation) — ALiBi replaces additive positional embeddings with a per-head linear bias on attention logits, providing zero-parameter length extrapolation; an alternative positional strategy applied within the attention score computation.
- **FlashAttention** (topic_id: flash-attention) (relation: complement) — FlashAttention reimplements the exact softmax attention computation with IO-aware tiling, achieving sub-quadratic memory access without approximation; the standard kernel used in all production attention layers.
- **FlashAttention-2** (topic_id: flash-attention-2) (relation: specialisation) — FlashAttention-2 refines parallelism and work partitioning over FlashAttention-1, improving GPU utilisation by ~2× and becoming the default CUDA implementation for attention in most frameworks.
- **FlashAttention-3** (topic_id: flash-attention-3) (relation: specialisation) — FlashAttention-3 extends memory-efficient attention to Hopper-generation GPUs via asynchronous pipelining and FP8 low-precision, representing the latest state of the art in exact-attention kernel optimisation.
- **Multi-Query Attention (MQA)** (topic_id: multi-query-attention) (relation: specialisation) — MQA collapses the key and value heads to a single shared head across all query heads, dramatically reducing KV-cache memory at the cost of some expressivity; a direct architectural variant of multi-head attention.
- **Grouped-Query Attention (GQA)** (topic_id: grouped-query-attention) (relation: specialisation) — GQA generalises MQA by grouping query heads to share a smaller set of key/value heads, offering a tunable trade-off between MHA expressivity and MQA memory efficiency; adopted by Llama-3, Mistral, and Gemma.
- **KV-Cache** (topic_id: kv-cache) (relation: complement) — The KV cache stores precomputed attention key and value tensors across decoding steps; its size, layout, and eviction policy are determined entirely by the attention head count and head dimension.
- **Prompt Caching** (topic_id: prompt-caching) (relation: application) — Prompt caching reuses previously computed KV tensors for repeated prefix tokens; a production inference optimisation that exploits the structure of the attention computation to amortise cost across requests.
- **State-Space Models (Mamba, S4)** (topic_id: state-space-models) (relation: alternative) — SSMs such as Mamba achieve sub-quadratic sequence modeling via learned recurrent state transitions rather than pairwise attention; the primary architectural alternative to attention for long-sequence tasks.
- **Linear Attention** (topic_id: linear-attention) (relation: alternative) — Linear attention approximates the softmax attention kernel with feature maps that decouple the query-key interaction, reducing complexity from O(n²) to O(n) and enabling recurrent-style inference.
- **Sliding Window Attention** (topic_id: sliding-window-attention) (relation: specialisation) — Sliding window (local) attention restricts each token to attending only within a fixed context window, reducing complexity to O(n·w); used in Mistral and Gemma architectures.
- **Ring Attention** (topic_id: ring-attention) (relation: complement) — Ring Attention distributes the attention computation across multiple devices arranged in a ring topology, enabling near-infinite context lengths by partitioning the sequence dimension across a device mesh.
- **YaRN Context-Window Extension** (topic_id: yarn-context-extension) (relation: complement) — YaRN extends the effective context window of RoPE-based models beyond their training length by rescaling and interpolating the rotary frequencies; directly applied to attention's positional encoding layer.
- **Mechanistic Interpretability Overview** (topic_id: mech-interp-overview) (relation: application) — Attention head analysis is a primary method in mechanistic interpretability; researchers decompose Q/K/V matrices and attention patterns to identify what information each head selects and routes through the residual stream.
- **Induction Heads** (topic_id: induction-heads) (relation: specialisation) — Induction heads are a specific two-head attention circuit responsible for in-context learning by implementing a prefix-matching lookup; the canonical example of a functional attention circuit identified via mech-interp analysis.
- **Attention Circuit Decomposition** (topic_id: attention-circuit-decomposition) (relation: application) — QK and OV decomposition of attention heads is the primary technique used in attribution-graph-based interpretability to trace information flow through multi-head attention.
- **Mixture-of-Experts** (topic_id: mixture-of-experts) (relation: alternative) — MoE routes tokens through sparse expert FFN layers rather than scaling attention, offering a different axis for increasing model capacity; hybrid models interleave MoE FFN blocks with standard attention layers.
- **Scaling Laws** (topic_id: scaling-laws) (relation: complement) — Empirical scaling laws govern how attention-based model performance varies with parameter count, training tokens, and compute; they motivated the shift from local/sparse attention to full attention at scale.

## Learning resources

### Essential starting points

- **"Attention? Attention!"** by Lilian Weng <mark>[weng-attention-blog]</mark> — Weng's comprehensive 2018 blog post traces every variant of attention from additive (Bahdanau) through self-attention to the full transformer, with clear notation and diagrams; the definitive single-page survey of the conceptual lineage before diving into papers.
- **"The Illustrated Transformer"** by Jay Alammar [\[100\]](#ref-100) — step-by-step animated walkthrough of multi-head self-attention, the encoder–decoder stack, and the QKV projection in vivid visual detail; the canonical first read for any newcomer to the transformer.
- **"Let's build GPT: from scratch, in code, spelled out"** by Andrej Karpathy [\[103\]](#ref-103) — Karpathy codes a working GPT (including scaled dot-product attention, multi-head attention, and causal masking) in ~90 minutes of live video; the fastest path from math notation to running PyTorch.

### Blog posts

- **"Attention? Attention!"** — Lilian Weng, *lilianweng.github.io*, Jun 2018. Source type: `blog`. Level: `intermediate`. Every attention family member (additive, dot-product, self, multi-head, key-value memory) is introduced with consistent notation and a mini-derivation <mark>[weng-attention-blog]</mark>.
- **"The Illustrated Transformer"** — Jay Alammar, *jalammar.github.io*, Jun 2018. Source type: `blog`. Level: `intro`. Visual, animation-driven tour of the self-attention operation and full encoder–decoder transformer; the most-linked single explainer in the field [\[100\]](#ref-100).
- **"The Illustrated GPT-2"** — Jay Alammar, *jalammar.github.io*, Aug 2019. Source type: `blog`. Level: `intermediate`. Extends the transformer explainer to decoder-only causal self-attention and shows how token generation works step-by-step, making KV-cache usage intuitive <mark>[alammar-illustrated-gpt2]</mark>.

### Videos

- **"Let's build GPT: from scratch, in code, spelled out"** — Andrej Karpathy, YouTube, Jan 2023. Source type: `video`. Level: `intermediate`. ~2 h walkthrough building a character-level GPT with multi-head causal self-attention from a blank Python file; includes the derivation of scaled dot-product attention from first principles [\[103\]](#ref-103).
- **"Attention Is All You Need — Paper Explained"** — Yannic Kilcher, YouTube, Jun 2020. Source type: `video`. Level: `intermediate`. Paper-reading walkthrough of "Attention Is All You Need" with live annotation of the equations; fills in derivation steps that the original paper elides <mark>[kilcher-attention-video]</mark>.

### Courses

- **Stanford CS336: Language Modeling from Scratch** — Stanford University, Spring 2024. Source type: `course`. Level: `advanced`. Full lecture + assignment sequence that has students implement FlashAttention, grouped-query attention, and RoPE from scratch, then train a small LM on a real cluster <mark>[stanford-cs336]</mark>.
- **fast.ai Practical Deep Learning** — Jeremy Howard & Sylvain Gugger, fast.ai. Source type: `course`. Level: `intro`. Code-first introduction to self-attention and transformers using Hugging Face; designed for practitioners who want to apply, not just understand, attention <mark>[fastai-practical-dl]</mark>.

### Code repositories

- **`karpathy/nanoGPT`** — Andrej Karpathy, GitHub. Source type: `code_repo`. Level: `intermediate`. ~300-line PyTorch implementation of GPT-2 with clean multi-head attention; the minimal readable reference that pairs with the "Let's build GPT" video <mark>[nanogpt-repo]</mark>.
- **`Dao-AILab/flash-attention`** — Tri Dao et al., GitHub. Source type: `code_repo`. Level: `advanced`. Reference CUDA implementation of FlashAttention 1/2/3; includes Triton kernels, benchmarks, and correctness tests <mark>[flash-attention-repo]</mark>.
- **`lucidrains/x-transformers`** — Phil Wang (lucidrains), GitHub. Source type: `code_repo`. Level: `intermediate`. Highly modular PyTorch library implementing ~30 attention variants (MHA, MQA, GQA, ALiBi, RoPE, linear attention, etc.) in a single codebase; ideal for ablation experiments <mark>[x-transformers-repo]</mark>.
- **`jessevig/bertviz`** — Jesse Vig, GitHub. Source type: `code_repo`. Level: `intro`. Interactive head-level and neuron-level attention visualiser for BERT and GPT-2 models; the fastest way to build intuition for head specialisation <mark>[bertviz-repo]</mark>.

## Personal synthesis

> [!quote] My take
> Attention is the one operation in modern deep learning that *looks* simple — a softmax over dot products, a weighted sum — and is endlessly deep when you actually probe it. The 2017 Vaswani recipe is conceptually trivial; the entire 2019-2026 research programme is engineering around the fact that this trivial recipe has a quadratic cost, a linear-in-length cache, a hardware-hostile IO pattern, no built-in notion of position, and — embarrassingly — no theory of *why* it works.

What I actually believe after this pass:

- **The attention-vs-SSM debate is mostly settled in favour of "and."** I started this topic thinking the interesting question was whether Mamba and friends would unseat transformers. Reading Mamba-2 [\[51\]](#ref-51), Jamba [\[52\]](#ref-52), and the Zoology / MQAR work [\[98\]](#ref-98) in sequence, the live consensus is much narrower: SSMs lose on associative recall, attention wins on associative recall, and the cheapest fix is to interleave layers. Pure-SSM frontier models look unlikely. Hybrid attention+SSM stacks are now the architectural question worth tracking.
- **The KV cache is the unit of inference economics.** Pope et al. 2022 [\[46\]](#ref-46) is the paper I keep returning to. Once you internalise the prefill-vs-decode split and the fact that decode is memory-bandwidth-bound, every other production decision — GQA [\[21\]](#ref-21), MLA [\[53\]](#ref-53), prompt caching, attention sinks [\[58\]](#ref-58), Ring Attention [\[28\]](#ref-28) — falls into place as a way to shrink, share, or compress the KV cache. The "attention mechanism" the research community talks about and the "attention mechanism" a serving engineer cares about are two different objects with the same name.
- **Position encoding is the most uncomfortable piece.** RoPE [\[25\]](#ref-25) won by adoption, but YaRN and NTK-aware extension are empirical recipes with no principled theory. ALiBi [\[26\]](#ref-26) has a cleaner theory and lost. Whatever replaces RoPE will need to be both better-grounded and at least as good at long-context extrapolation, and as of 2026 I don't see a contender.
- **Mechanistic interpretability is the most exciting bridge.** Induction heads [\[29\]](#ref-29) are the only specific computation attention performs that we have a *causal* account of. Everything else — the rest of the heads in every layer, what the OV circuits actually write, why some heads are sinks — is open. Anthropic's choice (Scaling Monosemanticity, biology of LLMs) to *bracket* attention and look at residual-stream features is honest about that gap. The next decade of interp on attention is going to be necessary, not luxury.

**Worked example — why prompt caching exists.** A Claude Code agent loop with a 30K-token system prompt + tool definitions runs 50 tool calls in a session. Without prompt caching, every call re-prefills 30K tokens at $\mathcal{O}(n^2 d)$ cost. With FlashAttention-2 / FlashAttention-3 [\[23\]](#ref-23), [\[24\]](#ref-24) the *constant factor* is great, but the asymptotic cost is unchanged: 50 calls × 30K-token prefill ≈ the prefill cost of a single 1.5M-token forward pass. Prompt caching [\[81\]](#ref-81) reloads the K/V tensors for the cached prefix at ~10% of base input token cost, so the same 50 tool calls cost ~1.5× a single 30K prefill rather than 50×. That entire pricing model — and the agentic-AI cost curve customers actually face — is a direct downstream consequence of the quadratic cost in the attention operator. The mechanism is small, but its economic shadow is enormous.

## Open questions

> [!question] Open questions
> 1. **Will a non-attention frontier model ever ship?** As of 2026, every >70B model in production runs full attention with GQA or MLA. The Mamba [\[50\]](#ref-50) / RWKV-6 [\[61\]](#ref-61) / Jamba [\[52\]](#ref-52) lineage is closing the gap at smaller scales, but no pure-SSM model has matched a dense transformer on retrieval-heavy benchmarks at frontier scale. Is this a fundamental limitation or a training-data / training-recipe gap that will close with scale?
> 2. **Can length-extrapolation be made principled?** YaRN and NTK-aware RoPE scaling work empirically; ALiBi has theory but lost the adoption war. Is there a positional encoding that is both principled and extrapolates cleanly to 10–100× the training context?
> 3. **What is the right abstraction for *memory* in long-context models?** The lost-in-the-middle U-shape [\[86\]](#ref-86) and attention-sink dynamics [\[58\]](#ref-58) both suggest pure attention's memory model is mis-specified for very long contexts. Retrieval-augmented attention, learned KV-eviction (H2O, [\[63\]](#ref-63); DuoAttention, [\[64\]](#ref-64)), and external state are all competing answers. None has won.
> 4. **Will Multi-head Latent Attention generalise beyond DeepSeek?** MLA [\[53\]](#ref-53) gives a striking ~14× KV reduction without quality loss on the DeepSeek line. As of 2026-05 no other frontier lab has shipped MLA in production. Is this a training-recipe sensitivity that will eventually transfer, or is MLA an architectural quirk that works only inside DeepSeek's specific training stack?
> 5. **How much of what attention learns is uninterpreted?** Induction heads [\[29\]](#ref-29) and a handful of named circuits cover a small fraction of the heads in a frontier model. Will mechanistic interpretability scale to characterise the rest, or will most heads remain "task-relevant but unexplained" — limited by the fundamental capacity gap between human analysis and the size of the residual-stream basis?

## Sources

[17] D. Bahdanau, K. Cho and Y. Bengio, “Neural Machine Translation by Jointly Learning to Align and Translate”, 2014. [Online]. Available: https://arxiv.org/abs/1409.0473. ^ref-17

[18] M.-T. Luong, H. Pham and C. D. Manning, “Effective Approaches to Attention-based Neural Machine Translation”, in EMNLP, 2015. [Online]. Available: https://arxiv.org/abs/1508.04025. ^ref-18

[19] A. Vaswani et al., “Attention Is All You Need”, 2017. [Online]. Available: https://arxiv.org/abs/1706.03762. ^ref-19

[20] N. Shazeer, “Fast Transformer Decoding: One Write-Head is All You Need”, 2019. [Online]. Available: https://arxiv.org/abs/1911.02150. ^ref-20

[21] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón and S. Sanghai, “GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints”, 2023. [Online]. Available: https://arxiv.org/abs/2305.13245. ^ref-21

[22] T. Dao, D. Y. Fu, S. Ermon, A. Rudra and C. Ré, “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness”, 2022. [Online]. Available: https://arxiv.org/abs/2205.14135. ^ref-22

[23] T. Dao, “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning”, 2023. [Online]. Available: https://arxiv.org/abs/2307.08691. ^ref-23

[24] J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani and T. Dao, “FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision”, in NeurIPS, 2024. [Online]. Available: https://arxiv.org/abs/2407.08608. ^ref-24

[25] J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen and Y. Liu, “RoFormer: Enhanced Transformer with Rotary Position Embedding”, 2021. doi: 10.1016/j.neucom.2023.127063. [Online]. Available: https://arxiv.org/abs/2104.09864 ^ref-25

[26] O. Press, N. A. Smith and M. Lewis, “Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation”, 2021. [Online]. Available: https://arxiv.org/abs/2108.12409. ^ref-26

[27] B. Peng, J. Quesnelle, H. Fan and E. Shippole, “YaRN: Efficient Context Window Extension of Large Language Models”, in ICLR, 2023. [Online]. Available: https://arxiv.org/abs/2309.00071. ^ref-27

[28] H. Liu, M. Zaharia and P. Abbeel, “Ring Attention with Blockwise Transformers for Near-Infinite Context”, 2023. [Online]. Available: https://arxiv.org/abs/2310.01889. ^ref-28

[29] C. Olsson et al., “In-context Learning and Induction Heads”, Transformer Circuits Thread. [Online]. Available: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html. ^ref-29

[30] J. Weston, S. Chopra and A. Bordes, “Memory Networks”, 2014. [Online]. Available: https://arxiv.org/abs/1410.3916. ^ref-30

[31] A. Graves, G. Wayne and I. Danihelka, “Neural Turing Machines”, 2014. [Online]. Available: https://arxiv.org/abs/1410.5401. ^ref-31

[32] D. Bahdanau, K. Cho and Y. Bengio, “Neural Machine Translation by Jointly Learning to Align and Translate”, 2014. [Online]. Available: https://arxiv.org/abs/1409.0473. ^ref-32

[33] M.-T. Luong, H. Pham and C. D. Manning, “Effective Approaches to Attention-based Neural Machine Translation”, in EMNLP, 2015. [Online]. Available: https://arxiv.org/abs/1508.04025. ^ref-33

[34] A. Vaswani et al., “Attention Is All You Need”, 2017. [Online]. Available: https://arxiv.org/abs/1706.03762. ^ref-34

[35] J. Devlin, M.-W. Chang, K. Lee and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”, 2018. [Online]. Available: https://arxiv.org/abs/1810.04805. ^ref-35

[36] R. Child, S. Gray, A. Radford and I. Sutskever, “Generating Long Sequences with Sparse Transformers”, 2019. [Online]. Available: https://arxiv.org/abs/1904.10509. ^ref-36

[37] N. Shazeer, “Fast Transformer Decoding: One Write-Head is All You Need”, 2019. [Online]. Available: https://arxiv.org/abs/1911.02150. ^ref-37

[38] N. Kitaev, L. Kaiser and A. Levskaya, “Reformer: The Efficient Transformer”, 2020. [Online]. Available: https://arxiv.org/abs/2001.04451. ^ref-38

[39] I. Beltagy, M. E. Peters and A. Cohan, “Longformer: The Long-Document Transformer”, 2020. [Online]. Available: https://arxiv.org/abs/2004.05150. ^ref-39

[40] S. Wang, B. Z. Li, M. Khabsa, H. Fang and H. Ma, “Linformer: Self-Attention with Linear Complexity”, 2020. [Online]. Available: https://arxiv.org/abs/2006.04768. ^ref-40

[41] A. Katharopoulos, A. Vyas, N. Pappas and F. Fleuret, “Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention”, 2020. [Online]. Available: https://arxiv.org/abs/2006.16236. ^ref-41

[42] K. Choromanski et al., “Rethinking Attention with Performers”, 2020. [Online]. Available: https://arxiv.org/abs/2009.14794. ^ref-42

[43] J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen and Y. Liu, “RoFormer: Enhanced Transformer with Rotary Position Embedding”, 2021. doi: 10.1016/j.neucom.2023.127063. [Online]. Available: https://arxiv.org/abs/2104.09864 ^ref-43

[44] O. Press, N. A. Smith and M. Lewis, “Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation”, 2021. [Online]. Available: https://arxiv.org/abs/2108.12409. ^ref-44

[45] T. Dao, D. Y. Fu, S. Ermon, A. Rudra and C. Ré, “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness”, 2022. [Online]. Available: https://arxiv.org/abs/2205.14135. ^ref-45

[46] R. Pope et al., “Efficiently Scaling Transformer Inference”, 2022. [Online]. Available: https://arxiv.org/abs/2211.05102. ^ref-46

[47] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón and S. Sanghai, “GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints”, 2023. [Online]. Available: https://arxiv.org/abs/2305.13245. ^ref-47

[48] T. Dao, “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning”, 2023. [Online]. Available: https://arxiv.org/abs/2307.08691. ^ref-48

[49] H. Liu, M. Zaharia and P. Abbeel, “Ring Attention with Blockwise Transformers for Near-Infinite Context”, 2023. [Online]. Available: https://arxiv.org/abs/2310.01889. ^ref-49

[50] A. Gu and T. Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces”, 2023. [Online]. Available: https://arxiv.org/abs/2312.00752. ^ref-50

[51] T. Dao and A. Gu, “Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality”, 2024. [Online]. Available: https://arxiv.org/abs/2405.21060. ^ref-51

[52] O. Lieber et al., “Jamba: A Hybrid Transformer-Mamba Language Model”, 2024. [Online]. Available: https://arxiv.org/abs/2403.19887. ^ref-52

[53] . DeepSeek-AI, “DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model”, 2024. [Online]. Available: https://arxiv.org/abs/2405.04434. ^ref-53

[54] J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani and T. Dao, “FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision”, in NeurIPS, 2024. [Online]. Available: https://arxiv.org/abs/2407.08608. ^ref-54

[55] J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani and T. Dao, “FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision”, in NeurIPS, 2024. [Online]. Available: https://arxiv.org/abs/2407.08608. ^ref-55

[56] . DeepSeek-AI, “DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model”, 2024. [Online]. Available: https://arxiv.org/abs/2405.04434. ^ref-56

[57] B. Peng, J. Quesnelle, H. Fan and E. Shippole, “YaRN: Efficient Context Window Extension of Large Language Models”, in ICLR, 2023. [Online]. Available: https://arxiv.org/abs/2309.00071. ^ref-57

[58] G. Xiao, Y. Tian, B. Chen, S. Han and M. Lewis, “Efficient Streaming Language Models with Attention Sinks”, 2024. [Online]. Available: https://arxiv.org/abs/2309.17453. ^ref-58

[59] T. Dao and A. Gu, “Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality”, 2024. [Online]. Available: https://arxiv.org/abs/2405.21060. ^ref-59

[60] O. Lieber et al., “Jamba: A Hybrid Transformer-Mamba Language Model”, 2024. [Online]. Available: https://arxiv.org/abs/2403.19887. ^ref-60

[61] B. Peng, D. Goldstein, Q. Anthony, A. Albalak and S. Biderman, “Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence”, 2024. [Online]. Available: https://arxiv.org/abs/2404.05892. ^ref-61

[62] Y. Sun et al., “Retentive Network: A Successor to Transformer for Large Language Models”, 2023. [Online]. Available: https://arxiv.org/abs/2307.08621. ^ref-62

[63] Z. Zhang et al., “H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models”, 2023. [Online]. Available: https://arxiv.org/abs/2306.14048. ^ref-63

[64] G. Xiao et al., “DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads”, 2024. [Online]. Available: https://arxiv.org/abs/2410.10819. ^ref-64

[65] D. Bahdanau, K. Cho and Y. Bengio, “Neural Machine Translation by Jointly Learning to Align and Translate”, 2014. [Online]. Available: https://arxiv.org/abs/1409.0473. ^ref-65

[66] A. Vaswani et al., “Attention Is All You Need”, 2017. [Online]. Available: https://arxiv.org/abs/1706.03762. ^ref-66

[67] N. Shazeer, “Fast Transformer Decoding: One Write-Head is All You Need”, 2019. [Online]. Available: https://arxiv.org/abs/1911.02150. ^ref-67

[68] K. Choromanski et al., “Rethinking Attention with Performers”, 2020. [Online]. Available: https://arxiv.org/abs/2009.14794. ^ref-68

[69] R. Child, S. Gray, A. Radford and I. Sutskever, “Generating Long Sequences with Sparse Transformers”, 2019. [Online]. Available: https://arxiv.org/abs/1904.10509. ^ref-69

[70] I. Beltagy, M. E. Peters and A. Cohan, “Longformer: The Long-Document Transformer”, 2020. [Online]. Available: https://arxiv.org/abs/2004.05150. ^ref-70

[71] J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen and Y. Liu, “RoFormer: Enhanced Transformer with Rotary Position Embedding”, 2021. doi: 10.1016/j.neucom.2023.127063. [Online]. Available: https://arxiv.org/abs/2104.09864 ^ref-71

[72] O. Press, N. A. Smith and M. Lewis, “Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation”, 2021. [Online]. Available: https://arxiv.org/abs/2108.12409. ^ref-72

[73] T. Dao, D. Y. Fu, S. Ermon, A. Rudra and C. Ré, “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness”, 2022. [Online]. Available: https://arxiv.org/abs/2205.14135. ^ref-73

[74] T. Dao, “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning”, 2023. [Online]. Available: https://arxiv.org/abs/2307.08691. ^ref-74

[75] A. Gu and T. Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces”, 2023. [Online]. Available: https://arxiv.org/abs/2312.00752. ^ref-75

[76] R. Pope et al., “Efficiently Scaling Transformer Inference”, 2022. [Online]. Available: https://arxiv.org/abs/2211.05102. ^ref-76

[77] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón and S. Sanghai, “GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints”, 2023. [Online]. Available: https://arxiv.org/abs/2305.13245. ^ref-77

[78] H. Liu, M. Zaharia and P. Abbeel, “Ring Attention with Blockwise Transformers for Near-Infinite Context”, 2023. [Online]. Available: https://arxiv.org/abs/2310.01889. ^ref-78

[79] C. Olsson et al., “In-context Learning and Induction Heads”, Transformer Circuits Thread. [Online]. Available: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html. ^ref-79

[80] N. Elhage et al., “A Mathematical Framework for Transformer Circuits”, 2021. [Online]. Available: https://transformer-circuits.pub/2021/framework/index.html. ^ref-80

[81] [Online]. Available: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching. ^ref-81

[82] [Online]. Available: https://www.anthropic.com/news/claude-3-family. ^ref-82

[83] [Online]. Available: https://transformer-circuits.pub/2024/scaling-monosemanticity/. ^ref-83

[84] [Online]. Available: https://transformer-circuits.pub/2025/attribution-graphs/biology.html. ^ref-84

[85] G. Xiao, Y. Tian, B. Chen, S. Han and M. Lewis, “Efficient Streaming Language Models with Attention Sinks”, 2024. [Online]. Available: https://arxiv.org/abs/2309.17453. ^ref-85

[86] N. F. Liu et al., “Lost in the Middle: How Language Models Use Long Contexts”, 2023. [Online]. Available: https://arxiv.org/abs/2307.03172. ^ref-86

[87] C. Anil et al., “Exploring Length Generalization in Large Language Models”, 2022. [Online]. Available: https://arxiv.org/abs/2207.04901. ^ref-87

[88] J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani and T. Dao, “FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision”, in NeurIPS, 2024. [Online]. Available: https://arxiv.org/abs/2407.08608. ^ref-88

[89] A. Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale”, 2020. [Online]. Available: https://arxiv.org/abs/2010.11929. ^ref-89

[91] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey and I. Sutskever, “Robust Speech Recognition via Large-Scale Weak Supervision”, 2022. [Online]. Available: https://arxiv.org/abs/2212.04356. ^ref-91

[92] G. Xiao, Y. Tian, B. Chen, S. Han and M. Lewis, “Efficient Streaming Language Models with Attention Sinks”, 2024. [Online]. Available: https://arxiv.org/abs/2309.17453. ^ref-92

[93] C.-P. Hsieh et al., “RULER: What's the Real Context Size of Your Long-Context Language Models?”, 2024. [Online]. Available: https://arxiv.org/abs/2404.06654. ^ref-93

[94] Y. Bai et al., “LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding”, 2023. [Online]. Available: https://arxiv.org/abs/2308.14508. ^ref-94

[95] Y. Bai et al., “LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks”, 2024. [Online]. Available: https://arxiv.org/abs/2412.15204. ^ref-95

[96] X. Zhang et al., “∞Bench: Extending Long Context Evaluation Beyond 100K Tokens”, 2024. [Online]. Available: https://arxiv.org/abs/2402.13718. ^ref-96

[97] N. F. Liu et al., “Lost in the Middle: How Language Models Use Long Contexts”, 2023. [Online]. Available: https://arxiv.org/abs/2307.03172. ^ref-97

[98] S. Arora et al., “Zoology: Measuring and Improving Recall in Efficient Language Models”, 2023. [Online]. Available: https://arxiv.org/abs/2312.04927. ^ref-98

[99] J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani and T. Dao, “FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision”, in NeurIPS, 2024. [Online]. Available: https://arxiv.org/abs/2407.08608. ^ref-99

[100] [Online]. Available: https://jalammar.github.io/illustrated-transformer/. ^ref-100

[101] [Online]. Available: https://www.youtube.com/watch?v=eMlx5fFNoYc. ^ref-101

[102] [Online]. Available: https://nlp.seas.harvard.edu/annotated-transformer/. ^ref-102

[103] [Online]. Available: https://www.youtube.com/watch?v=kCc8FmEb1nY. ^ref-103

[104] [Online]. Available: https://web.stanford.edu/~jurafsky/slp3/. ^ref-104

[105] [Online]. Available: https://web.stanford.edu/class/cs224n/. ^ref-105

[106] [Online]. Available: https://transformer-circuits.pub/. ^ref-106

[107] [Online]. Available: https://tridao.me/. ^ref-107

[90] [^90]: GitHub, "GitHub Copilot," *GitHub*, 2024. [Online]. Available: https://github.com/features/copilot ^ref-90
