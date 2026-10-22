# Other Biological Networks: Candidates for the Graph Zoo

Survey for task 5. Question: which biological network topologies, beyond the mammalian /
avian / fish respiratory graphs already in the zoo, are worth simulating as amplifier or
suppressor candidates?

Sources are PubMed (DOIs linked inline) plus one Physical Review Letters paper. All
literature claims below are attributed; the ranking and the zoo-fit arguments are mine.

## 1. What the zoo already covers, topologically

Stripped of biology, the current factories span a narrow slice of graph space:

| factory | topological type | circuit rank (independent cycles) |
|---|---|---|
| `mammalian_lung_graph` | balanced tree | 0 |
| `avian_graph` | one long cycle through parallel rods | `n_rods` |
| `fish_graph` | comb / caterpillar (tree) | 0 |
| `line_graph`, `star_graph` | trees | 0 |
| `cycle_graph` | single cycle | 1 |
| `grid_graph` | planar lattice | `(w-1)(h-1)` |
| `complete_graph` | clique | maximal |
| `random_connected_graph` | Erdos-Renyi-ish | tunable |

Three of the four biological factories are **acyclic**. The one structure the zoo cannot
currently produce is the one most common in real transport anatomy: a **reticulate mesh,
with many nested loops at several length scales**. That gap is what should drive the
candidate ranking, and it is exactly what the new cycle metrics (task 6) will measure.

## 2. The result that reframes the question

The single most relevant paper is Kuo, Nombela-Arrieta and Carja 2024, which is doing the
same thing this thesis does, one level down in scale. They build cellular spatial networks
of **bone marrow stem cell niches** from imaging data and find them to be **strong
suppressors of selection**, delaying mutation accumulation; suppression weakens as stem
cell population size drops ([DOI](https://doi.org/10.1038/s41467-024-48617-2)).

Two consequences for this project:

- It is an existence proof that "read a real anatomical structure as a graph, ask whether
  it amplifies or suppresses" yields a publishable answer. Useful precedent to cite.
- It also stakes out the tissue-scale claim. This thesis' novelty is at the **organ**
  scale (whole respiratory topologies), and the framing should say so explicitly.

A methodological warning that must be stated before adding any candidate: Hindersin and
Traulsen showed that almost all undirected random graphs are **amplifiers under
Birth-death updating but suppressors under death-Birth updating**
([DOI](https://doi.org/10.1371/journal.pcbi.1004437)). The verdict is a property of the
graph *and the update rule together*, not the graph alone. This repo's `MoranProcess`
implements fitness-weighted reproduction followed by random-neighbour replacement, i.e.
**Birth-death**, which is the amplifier-prone rule. Every result here is conditional on
that choice, and a reviewer will ask. Relatedly, Pattni, Overton and Sharkey show the star
can *inhibit* adaptive spread once individuals die naturally
([DOI](https://doi.org/10.1016/j.jtbi.2021.110648)), and Sharma and Traulsen show
suppressors of fixation can beat amplifiers on long-run mean fitness once mutation is
recurrent rather than one-shot ([DOI](https://doi.org/10.1073/pnas.2205424119)).

## 3. Ranked candidates

### Rank 1: Reticulate leaf venation (dicot), contrasted with parallel venation (monocot)

**Why first.** It fills the exact hole in section 1: a planar mesh whose defining feature
is a high density of closed loops at nested scales. Katifori, Szollosi and Magnasco showed
these loops are not incidental but optimal, arising under two independent pressures,
resilience to random vein damage and fluctuating sparse load
([DOI](https://doi.org/10.1103/PhysRevLett.104.048704)).

That gives the thesis its sharpest available argument. If reticulate venation turns out to
be an amplifier, the loops are explained by transport physics and evolutionary
amplification is a **side effect**; if a suppressor, there is a genuine trade-off between
damage resilience and mutation control. Either way the result is interpretable, which is
not true of every candidate.

**Bonus structure.** Monocots (parallel veins, low circuit rank) versus dicots (reticulate,
high circuit rank) is a within-system contrast of the same kind as avian versus mammalian
lung, and it varies the one property the new metrics measure. A single parametric factory
can produce both ends by a loop-density knob.

**Constructibility.** High. A planar hierarchical mesh with a tunable loop density is a
straightforward generator; no external data needed.

### Rank 2: Insect tracheal system

**Why.** It completes the comparative respiratory set (mammal, bird, fish, insect), which
is the thesis' own framing, and it is topologically distinct from all three in a way that
matters theoretically: it is **multi-source**. Air enters through many spiracles along the
body rather than one trachea, then branches to tracheoles.

This matters because the repo has already established that single-source digraphs collapse
selection entirely: the directed mammalian tree, directed star, and directed line all give
rho = 1/N at every r, since a mutant fixates only if born at the unique source. A digraph
with *k* sources is a different object and its behaviour is not covered by that argument.
That makes the insect trachea the most theoretically informative directed candidate
available.

Adult insects also add anastomoses and air sacs, so the adult topology carries loops the
larval one does not, giving another within-system contrast
([Royal Society Interface 2026](https://royalsocietypublishing.org/rsif/article/23/234/20250420/478900/Architecture-of-the-insect-tracheal-system-driven);
review of tracheal branching morphogenesis: [DOI](https://doi.org/10.1242/dev.014498)).

**Constructibility.** Medium. The large-scale structure is stereotyped and the fine
structure stochastic, so a factory would take spiracle count, trunk length and branching
depth as parameters. Quantitative whole-network graph data is thinner than for leaves;
expect to build a stylised generator rather than trace a real specimen.

### Rank 3: Bone marrow stem cell niche / intestinal crypt

**Why.** Direct benchmark against Kuo et al. 2024 above. If your pipeline reproduces their
suppression finding on a niche-like structure, that validates the whole apparatus against
an independent published result, which is worth a methods figure on its own.

Hindersin, Werner, Dingli and Traulsen give the theoretical companion: in small
niche-sized systems, amplification or suppression can flip on **subtle** architectural
changes, and the right architecture depends on the distribution of mutant fitness effects
([DOI](https://doi.org/10.1186/s13062-016-0140-7)).

**Caveat.** Not a respiratory organ, and this is the one candidate where the finding is
already published. Frame it as validation, not discovery.

**Constructibility.** Medium. Small dense clusters joined sparsely; a two-level generator
(clique size, number of niches, inter-niche connectivity).

### Rank 4: Vascular / capillary networks

**Why lower.** Well characterised and biologically compelling, but topologically it is
close to leaf venation (looped, hierarchical, near-planar at the capillary bed), so the
marginal information over rank 1 is small. Worth adding only if you want a mammalian
looped structure to pair against the mammalian tree.

### Rank 5: Fungal mycelium / *Physarum polycephalum*

**Why lower.** Genuinely loopy and adaptive, and there is quantitative anastomosis-rate
imaging ([PMC7035296](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7035296/)), but the
topology is **dynamic**, re-wiring in response to load. There is no canonical static
parametric form, so any factory would be an arbitrary snapshot. Note there is EGT work on
time-varying networks: switching temporal networks are usually *less* amplifying than
either static network composing them
([DOI](https://doi.org/10.1007/s00285-023-01987-5)). That is a whole second thesis, not an
extra zoo entry.

### Rank 6: *C. elegans* connectome

**Why lowest, but cheap.** A real, small, directed, fully published adjacency matrix, so
it costs a data file rather than a generator. Small-world with hubs, which is a distinct
regime from everything else here. But it is not a transport organ and does not fit the
respiratory framing; use it only as an out-of-domain control if a reviewer asks whether
the method says anything sensible about non-transport biology.

## 4. Recommendation

Add **leaf venation** first: it is the only candidate that is simultaneously easy to
generate, topologically absent from the zoo, exercises the new cycle metrics, and carries
a published functional explanation for its loops that sets up a real trade-off argument.

Add **insect trachea** second, as the multi-source directed case, which is where the
existing single-source rho = 1/N result stops applying.

Treat **bone marrow niche** as a validation target rather than a discovery target.

Defer vascular, mycelium and connectome.

## 5. Open question to settle before building any of these

Every result in this repo is conditional on Birth-death updating (section 2). Before
investing in new topologies, it is worth running the existing zoo under death-Birth as
well, because the literature says the amplifier/suppressor verdict can invert. If it does
invert for the respiratory graphs, that is itself a finding, and it changes how every new
candidate should be reported.

---

*Literature retrieved from PubMed unless otherwise noted. Corpus searched August 2026.*
