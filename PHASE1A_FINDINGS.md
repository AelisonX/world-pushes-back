# Phase 1A Findings

## Title

**Phase 1A — Action-Grid Cohort Feasibility**

---

## Status

**PHASE1A_GRID_FEASIBLE**

The preregistered exploratory 3 × 3 diagnostic-action grid passed the Phase 1A feasibility gates.

No action-value metric was computed.

No action was ranked.

No action winner was selected.

---

# 1. Research question

Before asking:

> Which legal physical action reveals more information about the future transition?

Phase 1A asked:

> Can the proposed diagnostic actions be compared on a sufficiently broad and interpretable common population of hidden worlds?

The answer under the current frozen simulator and paired identification design is:

> Yes.

---

# 2. Frozen target

The target remained fixed at:

```text
45 degrees
```

for every diagnostic action.

The diagnostic action did not redefine the future-transition label.

Labels remained:

```text
MATERIAL_YIELD
SUPPORT_SLIP
```

with the frozen target near-tie rule:

```text
epsilon_tie = 0.10
```

---

# 3. Target-valid population

The frozen Phase 1A run generated:

```text
5,000 target-valid pairs
10,000 target-valid worlds
```

Each pair contained one world from each target class.

The paired class-balance invariant was preserved.

---

# 4. Universal legal cohort

A pair entered the universal cohort only if both twins were legal under all nine candidate diagnostic actions.

Result:

```text
universal legal pair fraction = 0.9618
```

Therefore:

```text
4,809 / 5,000 pairs
```

remained legal across the complete action grid.

Only:

```text
3.82%
```

of otherwise target-valid pairs were removed by universal legality.

The preregistered redesign trigger was:

```text
universal legal pair fraction < 0.50
```

This trigger did not fire.

---

# 5. Per-action legality

Observed legal-pair fractions were:

```text
A1  3 N × 30°   1.0000
A2  3 N × 45°   1.0000
A3  3 N × 60°   1.0000

A4  5 N × 30°   0.9998
A5  5 N × 45°   1.0000
A6  5 N × 60°   0.9998

A7  7 N × 30°   0.9670
A8  7 N × 45°   0.9980
A9  7 N × 60°   0.9928
```

The higher-force actions produced most of the legality loss.

This is consistent with the expected physical structure of the grid.

---

# 6. Main legality bottleneck

The strongest cohort restriction came from:

```text
A7 = 7 N × 30°
```

Its legal-pair fraction was:

```text
0.9670
```

Removing A7 from the universal requirement increased universal retention by:

```text
0.0302 absolute
```

or:

```text
3.02 percentage points
```

The preregistered single-action bottleneck trigger was:

```text
leave-one-out gain >= 0.20
```

Therefore A7 did not qualify as a redesign-level bottleneck.

---

# 7. Leave-one-out results

Observed leave-one-out gains were:

```text
A1  0.0000
A2  0.0000
A3  0.0000
A4  0.0000
A5  0.0000
A6  0.0000
A7  0.0302
A8  0.0000
A9  0.0046
```

No individual candidate action disproportionately defined the universal population under the frozen redesign criterion.

---

# 8. Bayes numerical adequacy

The validated targeted importance-sampling strategy was checked separately for each candidate action.

All nine candidate actions passed the frozen ESS gate.

Observed p05 ESS fractions:

```text
            SUPPORT_SLIP   MATERIAL_YIELD

A1          0.917932       0.928111
A2          0.844800       0.856331
A3          0.743088       0.746372

A4          0.790859       0.766227
A5          0.587413       0.653347
A6          0.502151       0.536023

A7          0.597457       0.611869
A8          0.580842       0.604507
A9          0.489144       0.505401
```

Frozen minimum p05 ESS:

```text
0.05
```

All actions remained comfortably above that requirement.

No per-action AUC was computed.

---

# 9. Redesign triggers

The frozen Phase 1A runner reported:

```text
triggered = []
count = 0
```

Therefore none of the preregistered redesign conditions fired.

These included checks for:

- excessive universal-cohort loss
- single-action cohort domination
- excessive transition-before-probe rejection
- strong hidden-parameter shift
- strong target-margin distortion
- per-action Bayes ESS failure

---

# 10. What Phase 1A establishes

Under the current simulator and paired identification population:

> the nine exploratory diagnostic actions can be compared on a broad common legal cohort without severe population truncation under the preregistered feasibility criteria.

The action grid therefore survives the design-feasibility stage.

---

# 11. What Phase 1A does not establish

Phase 1A does not establish:

- which action is most informative
- whether stronger probes are better
- whether any action improves real-world robot behavior
- whether the current grid is optimal
- whether information gain justifies physical risk
- whether the result generalizes beyond the current simulator
- superiority over conventional robotic sensing or control

Those questions require later phases.

---

# 12. Important interpretation boundary

The universal cohort is a controlled comparison population.

It is not an operational prior.

A real agent does not know in advance whether the hidden world belongs to the universal legal cohort.

Future policy analysis must therefore retain the distinction between:

```text
information conditional on legal comparison
```

and:

```text
risk of attempting the action in the original world population
```

---

# 13. Phase 1B gate

Phase 1A satisfies the preregistered feasibility requirements needed to design Phase 1B.

Phase 1B may now freeze:

```text
information metric
same-world action comparison
noise coupling
policy-risk view
estimator
fresh seeds
confirmatory acceptance rules
```

Only after that freeze may action information value be inspected.

---

# 14. Scientific progression

The project has now moved through:

```text
Can mismatch be observed?
        ↓
Can different physical futures be identified?
        ↓
Can the numerical reference be trusted?
        ↓
Can multiple legal actions be compared fairly?
        ↓
YES — Phase 1A
```

The next research question is:

> Given the same hidden world, which legal action changes what the agent can know before the physical transition occurs?

---

# Working conclusion

The proposed action grid did not collapse under its own safety constraints.

Across 5,000 paired identification worlds, 96.18% remained jointly legal under all nine candidate actions, no preregistered redesign trigger fired, and all per-action Bayes ESS checks passed.

Phase 1A therefore supports proceeding to a preregistered Phase 1B action-value comparison.