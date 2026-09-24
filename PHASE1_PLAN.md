# Phase 1 Plan

## Title

**Phase 1 — Diagnostic Action Value**

---

## Status

**PLANNING ONLY**

No Phase 1 confirmatory experiment has been run.

No action ranking is currently claimed.

This document defines the research logic that must be resolved before a Phase 1 manifest is frozen.

---

# 1. Motivation

Phase 0 and Phase 0.5 established a conditional result:

> In the current toy simulator, safe tool-side observations can contain information about which physical transition will occur first when the world exposes a measurable compliant-motion precursor.

Phase 0.5b additionally established that the Bayes numerical approximation can be made adequate using targeted importance sampling without changing the underlying scientific posterior.

The next question is no longer:

> Is there information?

It is:

> Which legal physical action reveals the most useful information under a fixed safety budget?

This is the **Diagnostic Action Value** problem.

---

# 2. Core research question

For the same hidden physical world, different legal actions may produce different observations.

Phase 1 asks:

> Given the same hidden world and the same future-transition target, how much useful information does each legal physical action reveal before consuming the allowed safety budget?

The comparison must be counterfactual.

The same world should be evaluated under multiple candidate actions.

This prevents action comparisons from being confounded by different world populations.

---

# 3. Target variable

The target remains:

```text
MATERIAL_YIELD
vs
SUPPORT_SLIP
```

The target is defined by the same counterfactual future-transition rule used in Phase 0.5:

> Which transition would occur first under monotonically increasing load at the fixed nominal reference direction?

The diagnostic action must not redefine the target merely because it uses a different force direction.

The target therefore remains anchored to a fixed nominal continuation.

Tentative reference:

```text
reference angle = 45 degrees
```

This must be frozen before confirmatory execution.

---

# 4. Fundamental design rule

## Same world, different action

Every candidate action must be evaluated on the same hidden physical world.

For one world:

```text
world W
```

we evaluate:

```text
W + action A
W + action B
W + action C
...
```

The hidden physical parameters remain unchanged.

Only the diagnostic action changes.

This enables genuine counterfactual comparison:

> What would we have learned if we had acted differently in the same world?

---

# 5. No magical probe action

Phase 1 must not introduce a generic operation such as:

```text
PROBE
```

that directly queries hidden state.

Every action must be an ordinary physical intervention.

Examples may include:

- lower-force push
- higher-force push
- shallower force angle
- steeper force angle
- lateral nudge
- unload-and-hold
- repeated low-force interaction

However, Phase 1 should begin with actions already representable by the current simulator before adding temporal or multi-step mechanics.

---

# 6. Phase 1A scope

The first Phase 1 experiment should remain minimal.

Phase 1A should compare a small grid of one-step physical actions defined by:

```text
force magnitude
force angle
```

A candidate exploratory grid may include values such as:

```text
force:
3 N
5 N
7 N

angle:
30 degrees
45 degrees
60 degrees
```

This grid is not yet frozen.

The final action set must be selected before confirmatory execution.

No action should be added merely because it produced an attractive exploratory result.

---

# 7. Universal legal cohort

A major Phase 1 risk is action-dependent sample selection.

For example:

```text
Action A may be safe in 90% of worlds.
Action B may be safe only in 40%.
```

Comparing A and B on their separate surviving cohorts would create a confound.

Therefore Phase 1 should use a **universal legal cohort**.

A world enters the main action-comparison dataset only if:

> Every candidate action in the frozen action set is legal and remains within the frozen safety constraints.

This means the world must satisfy the safety requirements for:

```text
Action A
AND
Action B
AND
Action C
...
```

The same worlds are then used for every action comparison.

---

# 8. Safety budget

The safety budget must be defined before action ranking.

At minimum, every diagnostic action must satisfy:

```text
probe occurs before first physical transition
rho < rho_safe_max
normal force > 0
support model remains below clipped regime
```

The current Phase 0.5 safety boundary is:

```text
rho_safe_max = 0.80
```

Phase 1 should initially inherit this limit unless there is a documented reason to change it before the Phase 1 manifest is frozen.

A candidate action that violates the safety budget is not considered informative.

It is illegal.

---

# 9. Safety is not a reward bonus

Phase 1 should not hide safety inside an arbitrary weighted score such as:

```text
utility =
information
- lambda * risk
```

unless a meaningful value of `lambda` can be justified independently.

Instead, the preferred structure is:

```text
1. define the safety budget
2. reject actions outside the budget
3. compare information among legal actions
```

Secondary analyses may show information-versus-risk tradeoffs.

The primary comparison should remain interpretable.

---

# 10. Observation model

The legal observation remains:

```text
Fx
Fy
tip_dx
tip_dy
```

No privileged hidden field becomes legal in Phase 1.

The following remain unavailable to the decision system:

- static friction
- material yield strength
- true transition thresholds
- rho
- future label
- hidden support parameters

Research code may retain those fields for audit and provenance.

---

# 11. Noise handling

Action comparisons must not gain artificial advantages from inconsistent noise handling.

For same-world action comparison, the Phase 1 implementation should use controlled randomization.

Preferred design:

> The same world receives reproducibly generated measurement-noise realizations under each action, with the noise distribution identical across actions.

The implementation should avoid accidentally making one action easier because it received a different noise model.

Whether the exact numerical noise realization should be shared across actions or independently drawn from a common frozen seed schedule must be decided before the manifest is frozen.

Both options must be documented explicitly.

---

# 12. Primary information metric

The preferred Phase 1 primary metric is:

**Expected Information Gain**

Conceptually:

```text
information before action
minus
expected uncertainty after observing the action
```

Equivalent interpretation:

> How much does observing the result of action `a` reduce uncertainty about the future-transition label?

For the binary target:

```text
Y ∈ {
    MATERIAL_YIELD,
    SUPPORT_SLIP
}
```

Phase 1 seeks:

```text
I(Y ; O | action)
```

where:

```text
O = legal observation
```

This provides an action-specific measure of information.

---

# 13. Why AUC is not enough

ROC-AUC remains useful as a secondary metric.

However, Phase 1 is not merely asking whether a classifier can separate two classes.

It is asking:

> Which intervention produces more informative observations?

Two actions can have similar AUC while producing different posterior uncertainty.

Therefore Phase 1 should report both:

```text
information-value metric
classification metric
```

Possible secondary metrics include:

- ROC-AUC
- log loss
- Brier score
- posterior entropy
- false-negative rate at fixed false-positive rate

---

# 14. Bayes reference

Phase 1 should continue using the validated Bayes numerical machinery from Phase 0.5b.

The targeted importance proposal may be used as a computational method.

However:

> The proposal must not change the scientific prior.

Importance correction must remain explicit.

The Phase 0.5b result established that the targeted proposal can improve numerical efficiency while preserving the posterior.

That result does not automatically validate every Phase 1 action.

Bayes numerical adequacy should still be checked under the new action set.

---

# 15. Same-world counterfactual comparison

For every admitted hidden world:

```text
W_i
```

Phase 1 should produce observations under every candidate action:

```text
O_i,A
O_i,B
O_i,C
...
```

The future-transition label remains:

```text
Y_i
```

The core dataset therefore has a structure like:

```text
world_id
label
action_id
Fx
Fy
tip_dx
tip_dy
safety metadata
```

World identity must be preserved.

Train/test splitting must occur by world or paired-world identity.

The same hidden world must never appear in both training and test sets under different actions.

---

# 16. Pair structure

The paired identification design from Phase 0.5 should be retained unless a documented analysis shows that Phase 1 requires a different structure.

Each pair should continue to contain:

```text
one MATERIAL_YIELD world
one SUPPORT_SLIP world
```

with shared nuisance variables where appropriate.

Action comparison then occurs inside the same paired identification framework.

This maintains class balance without pretending that balanced identification data represent operational prevalence.

---

# 17. Operational prior remains separate

The operational prior must remain separate from the paired identification prior.

Phase 0.5 found that the default operational distribution was approximately:

```text
MATERIAL_YIELD ≈ 98.9%
SUPPORT_SLIP   ≈ 1.1%
```

Phase 1 action-value estimates under the balanced identification prior must not be described as natural-world action frequencies or natural-world utility.

If operational weighting is later used, it must be reported separately.

---

# 18. Candidate action comparison

For each frozen legal action, Phase 1 should report at least:

```text
action ID
force magnitude
force angle
sample count
information gain
ROC-AUC
posterior uncertainty
safety statistics
Bayes numerical adequacy
```

Safety statistics may include:

- maximum rho
- median rho
- p95 rho
- minimum safety margin
- fraction rejected before universal-cohort construction

---

# 19. Primary action-selection rule

Under a fixed frozen safety budget:

> Select the action with the highest validated information value among legal actions.

However, Phase 1 must not declare a universal best action.

The result is conditional on:

- the toy physical model
- the action set
- the identification prior
- the noise model
- the safety budget
- the observation model

The correct claim structure is:

> Under this frozen experimental design, action A produced more information than actions B and C.

Not:

> Action A is the optimal robotic probing action.

---

# 20. Action dominance

Phase 1 should also test whether any action is dominated.

Action A is dominated by Action B if:

```text
B provides at least as much information
AND
B consumes no more safety budget
```

with at least one strict improvement.

Dominated actions may be removed from later policy experiments.

This creates an interpretable action frontier without inventing an arbitrary scalar utility.

---

# 21. Important controls

Phase 1 should include controls capable of detecting fake action value.

At minimum:

## Same-action control

Compare an action against an exact duplicate of itself.

Expected result:

```text
no meaningful action-value difference
```

## Label shuffle

Randomize labels while preserving the world/action grouping.

Expected result:

```text
information advantage disappears
```

## Rigid null

Repeat action comparisons under rigid support.

Expected result:

```text
no strong compliant-motion action advantage
```

## Noise control

Verify that an apparent action advantage is not produced solely by different noise treatment.

---

# 22. Leakage prevention

The following are forbidden:

- choosing actions after inspecting confirmatory test performance
- filtering worlds differently for each action in the primary comparison
- using privileged rho as an action feature
- splitting different actions from the same world across train and test
- selecting information metrics after viewing results
- changing the safety budget after action outcomes are inspected
- changing action magnitudes after confirmatory performance is known

---

# 23. Exploratory and confirmatory separation

Phase 1 should use two stages.

## Phase 1A — Exploratory action screen

Purpose:

- test implementation
- examine action behavior
- identify obviously redundant or illegal actions
- validate information metrics
- validate Bayes numerical adequacy

No final action claim is made.

## Phase 1B — Frozen comparison

Before Phase 1B:

- freeze action set
- freeze seeds
- freeze safety budget
- freeze information metric
- freeze Bayes method
- freeze train/test split protocol
- freeze acceptance and interpretation rules

Only Phase 1B may support confirmatory action-value claims.

---

# 24. Phase 1 success condition

Phase 1 succeeds if:

> At least two legal physical actions produce meaningfully different information value on the same hidden worlds under the same frozen safety budget.

A useful result may be:

```text
Action A > Action B
```

or:

```text
No meaningful action-value difference exists.
```

Both are valid scientific outcomes.

---

# 25. Phase 1 negative result

Phase 1 should be considered scientifically successful even if:

```text
all legal actions provide approximately the same information
```

That result would imply:

> Action selection may not provide enough diagnostic benefit to justify a more complex active policy.

In that case, Phase 2 should not be built merely because active diagnosis sounds interesting.

---

# 26. Phase 1 kill condition

Phase 1 should stop or reformulate if:

1. action comparisons require action-specific cohorts
2. legal actions provide no reproducible information difference
3. information gain appears only near unsafe thresholds
4. Bayes numerical adequacy cannot be restored under the action set
5. action value depends entirely on privileged hidden state
6. apparent gains disappear under same-world controls

---

# 27. Phase 2 gate

Phase 2 — Active Diagnosis should begin only if Phase 1 establishes:

> Different legal actions have reproducibly different information value under the same safety constraints.

If that gate passes, Phase 2 may ask:

> Can an agent choose the more informative action online?

If it does not pass, there is no reason to build an action-selection policy.

---

# 28. Implementation order

Phase 1 implementation should proceed in this order:

```text
1. define ActionSpec representation
2. define candidate action grid
3. implement universal legal cohort
4. implement same-world multi-action observations
5. verify world-level split integrity
6. implement information-value metric
7. implement Bayes action evaluation
8. implement safety summaries
9. implement controls
10. run exploratory Phase 1A only
11. inspect diagnostics
12. freeze Phase 1B manifest
13. run independent confirmatory comparison
```

Do not skip directly to policy learning.

---

# 29. Current unresolved decisions

Before Phase 1A begins, the following must be resolved:

### Action grid

Exact force magnitudes and angles.

### Noise coupling

Whether same-world counterfactual actions share the same noise realization or use independently seeded noise draws from the same distribution.

### Primary information metric

Exact implementation of expected information gain.

### Safety summaries

Which action-level safety statistics are mandatory.

### Bayes proposal

Whether the Phase 0.5b targeted proposal remains numerically adequate across all candidate actions.

These are design decisions, not results.

They must be resolved before confirmatory freezing.

---

# 30. Claims Phase 1 may eventually support

If validated, Phase 1 may support statements such as:

> Under the frozen toy-model conditions, some legal physical actions provide more information about the future transition than others while remaining within the same safety budget.

Phase 1 must still not claim:

- real-world optimal probing
- universal robot action selection
- demonstrated task-level safety improvement
- autonomous policy superiority
- general embodied intelligence

---

# Working principle

Phase 0 asked:

> Can the world tell us what is about to happen?

Phase 1 asks:

> Which safe action makes the world tell us more?