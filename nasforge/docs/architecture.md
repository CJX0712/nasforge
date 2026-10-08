# NasForge Architecture

This document describes the internal architecture of NasForge: the search space
(fixed cell DAG), the hand-written operators and one-shot supernet, the DarFuse
differentiable search algorithm (first- and second-order), weight inheritance,
and the project's error-code taxonomy.

Author: **晨星 (CJX0712)**.

---

## 1. The fixed cell DAG

NasForge searches over a **DARTS-style cell** with a small, *fixed* topology.
The topology is defined in `nasforge/spaces/search_space.py`
(`SearchSpace`) and is deliberately tiny so the full pipeline finishes in
well under a minute on one CPU core.

| Parameter | Value | Meaning |
| --- | --- | --- |
| `n_input` | 2 | Two cell inputs = the outputs of the previous two cells (nodes `0` and `1`). |
| `n_nodes` | 3 | Three internal (learnable) nodes. |
| `edges_per_node` | 2 | Each internal node pulls **2 edges** from earlier nodes. |
| **total edges** | **6** | `n_nodes × edges_per_node = 6` edges, indexed `0..5`. |

Edges are constructed in `SearchSpace.__post_init__`: for internal node `j`, the
two incoming edges come from `j-1` and `j-2` (the two immediately preceding
nodes), guaranteeing `src < dst` (acyclic). The resulting edge list is:

```
edge 0: (0 -> 2)
edge 1: (1 -> 2)
edge 2: (0 -> 3)
edge 3: (2 -> 3)
edge 4: (1 -> 4)
edge 5: (3 -> 4)
```

A node's output is the **sum** of its two incoming transformed edges. This fixed
DAG is the *macro* structure; only the *operator on each edge* is searched.

### 1.1 Edge = softmax mixture over 6 primitives

Every edge is a **mixed operator**: a softmax over the 6 primitives in
`nasforge/ops/registry.py` (`PRIMITIVE_ORDER`):

| Index | Primitive | Type |
| --- | --- | --- |
| 0 | `skip_connect` | identity (param-free) |
| 1 | `conv_1x1` | 1×1 convolution |
| 2 | `conv_3x3` | 3×3 convolution |
| 3 | `conv_5x5` | 5×5 convolution |
| 4 | `avg_pool` | 2×2 average pooling (param-free) |
| 5 | `max_pool` | 2×2 max pooling (param-free) |

The architecture parameters `alpha` have shape `(n_edges, n_ops) = (6, 6)`. For
an edge `e`, the mixed output is:

```
x_e = Σ_k  softmax(alpha[e, :])[k] · op_k(x)
```

A **discrete genotype** collapses each edge to a single primitive via `argmax`
(`SearchSpace.discretize`), which is what gets independently evaluated.

Convolutions are weight-shared across all edges that use the same primitive:
there is one set of weights per primitive family (`conv_1x1`, `conv_3x3`,
`conv_5x5`), reused by every edge — this is the one-shot weight-sharing
approximation.

---

## 2. The one-shot supernet

`nasforge/spaces/network.py` (`Network`) defines the supernet that is trained
once and shared by all edges:

```
stem (3x3 conv, 3 -> C)        ← C = channels = 8
  └─ Cell #1 (mixed, alpha)    ← 6 mixed edges
       └─ Cell #2 (mixed, alpha) ← same alpha reused across cells
            └─ global average pool over (H, W)
                 └─ head (C -> n_classes)
```

- The `stem` is a 3×3 convolution mapping the 3 input channels to `C=8`.
- `n_cells = 2` mixed cells are stacked; the same continuous `alpha` is reused
  by every cell (standard DARTS amortisation).
- Each cell's internal node activation is the sum of its two mixed incoming
  edges, with the two cell inputs taken from the previous two cell outputs
  (or the stem for the first cell).
- A **global average pool** collapses spatial dimensions, and a linear `head`
  produces logits.
- Loss is mean softmax cross-entropy; `Network.backward` returns the gradient
  w.r.t. logits so the caller can chain any loss.

`Network` also exposes `flat_weights` / `set_weights`, giving a single vector
view of every parameter — this is what the DarFuse-2nd unrolled gradient
machinery needs.

### 2.1 Supernet parameter breakdown

`Genotype.param_count(channels, num_classes, stem_channels)` counts the discrete
network parameters:

- stem: `3 · stem_channels · 3 · 3 + stem_channels`
- each `conv_*` edge contributes `channels·channels·k·k + channels` (k = 1, 3, 5)
- head: `channels·num_classes + num_classes`

Skip-connect and pooling are parameter-free.

---

## 3. DarFuse differentiable search (first-order)

`nasforge/search/darts.py` implements the shared supernet training loop. Each
search step:

1. Train the shared weights for `budget.epochs` epochs with SGD
   (`w_lr=0.05`, `momentum=0.9`, `weight_decay=1e-4`).
2. Compute `g_val = ∇_α L_val` and `g_train = ∇_α L_train` (architecture
   gradients via the mixed-cell backward).
3. Update `alpha` with the **DarFuse gradient**:

```
g_alpha = g_val - grl_scale * g_train - alpha_wd * alpha
alpha  := alpha - alpha_lr * g_alpha
```

- **GRL (Gradient Reversal Layer)** — `grl(g_train)` negates and scales the
  *training* architecture gradient (`grl_scale=1.0` exactly negates it, verified
  by invariant G3). Reversing `g_train` decouples architecture learning from
  weight fitting: the search is pushed to follow validation (generalisation)
  while *un*-following training (overfitting).
- **architecture weight decay** — `alpha_wd * alpha` (`alpha_wd=1e-3`) regularises
  `alpha` for stability.
- Defaults: `alpha_lr=0.005`, `xi=0.01` (used only by the 2nd-order variant).

After the final step, `alpha` is discretized (`argmax` per edge) into a
`Genotype`.

---

## 4. Weight inheritance (one-shot evaluation)

Weight inheritance is the third DarFuse refinement
(`nasforge/training/trainer.py::train_discrete`). Instead of training the found
discrete genotype **from scratch**, it is **fine-tuned starting from the
one-shot supernet's shared weights**:

- The discrete cell reuses the already-trained `conv_1x1 / conv_3x3 / conv_5x5`
  weights that the supernet learned, restricted to the selected edges.
- This typically yields a much higher, more faithful accuracy than from-scratch
  training would, and it is what the `SearchResult.best_val_acc` /
  `best_test_acc` report after a DarFuse run.

This is the "one-shot" promise: a single supernet training phase produces a
discrete architecture that is immediately usable.

---

## 5. DarFuse-2nd (DARTS second-order unrolled)

`nasforge/search/darts_second_order.py` implements the **unrolled implicit
architecture gradient**. The first-order gradient `∇_α L_val(w)` ignores how the
*weights* move when `α` changes. The second-order (unrolled) gradient
linearises the validation loss at the virtual weights
`w' = w − ξ ∇_w L_train`:

```
∇_α L_val(w') ≈ ∇_α L_val(w) − ξ · ∇_α( ∇_w L_train · ∇_w L_val )
```

- `∇_α L_val(w)` is the ordinary first-order validation architecture gradient.
- The **cross term** `∇_α(∇_w L_train · ∇_w L_val)` is the directional
  derivative of the dot product of the two weight-grad vectors w.r.t. `α`.
- In NasForge it is computed **exactly by finite differences in `α`** while the
  validation weight-gradient vector `∇_w L_val` is held fixed:

```
for i in alpha_indices:               # 6 edges × 6 ops = 36 indices
    d+ = ( ∇_w L_train(α+ε·e_i) · g_w_val )
    d- = ( ∇_w L_train(α-ε·e_i) · g_w_val )
    cross[i] = (d+ - d-) / (2ε)
g_alpha_va = g_alpha_va − xi · cross
```

This is the faithful DARTS-2nd approximation with hand-written VJPs and no
autograd. It is **strictly more accurate** than the first-order gradient, which
is what invariants **G2** and **G12** verify. The cost is CPU time: 36 `alpha`
indices each require two extra weight-gradient evaluations — see
[known_limitations.md](known_limitations.md).

---

## 6. Baselines and zero-cost proxies

All baselines share the `search(dataset, budget, seed, **kw) -> SearchResult`
contract (`nasforge/search/__init__.py`):

- **`evolution_search`** — regularized (aging) evolution (Real et al., AAAI'19):
  tournament selection, single-edge mutation, independent evaluation, oldest
  member dropped.
- **`random_search`** — uniform random sampling + independent evaluation.
- **`zerocost_search`** — scores `budget.max_steps` random genotypes with
  training-free proxies, then trains the top composite score.

Zero-cost proxies (`nasforge/search/naswiz_metrics.py`, NAS-WiZ / Zero-Cost-NAS,
Abdelfattah et al., ICLR'21) at a single random init on one minibatch:

| Proxy | Definition |
| --- | --- |
| `n_params` | Total parameter count (complexity prior). |
| `grad_norm` | Frobenius norm of the weight-gradient of the loss (signal). |
| `snip` | `|Σ_p grad_p · p|` (SNIP single-shot importance). |
| `logits_var` | Variance of the logits across samples (expressivity). |

The composite ranker used by `zerocost_search` is
`snip + grad_norm + logits_var − 1e-4·n_params` (higher = better). Proxy scores
are stored in `SearchResult.one_shot_scores` and used for the rank-correlation
invariant (G12).

---

## 7. Error-code taxonomy

All exceptions derive from `nasforge.core.errors.NasForgeError` so callers can
catch one base class. Codes are segmented by subsystem:

| Range | Subsystem | Example classes |
| --- | --- | --- |
| **E1xx** | Configuration / validation | `ConfigError` (100), `SeedError` (110) |
| **E2xx** | Data layer | `DataError` (200) |
| **E3xx** | Search space / operators | `SpaceError` (300), `OperatorError` (310) |
| **E4xx** | Search algorithms | `SearchError` (400), `DiscretizationError` (410) |
| **E5xx** | Evaluation / training | `EvaluationError` (500), `OneShotError` (510), `TrainingError` (520) |

CLI exit codes mirror these families: `EXIT_OK=0`, `EXIT_CONFIG=2`,
`EXIT_DATA=3`, `EXIT_SEARCH=4`, `EXIT_EVAL=5`.
