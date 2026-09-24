# Phase 1A Feasibility Plan

## Title

**Phase 1A — Action-Grid Cohort Feasibility**

---

## Status

**PLANNING / PRE-RESULT FREEZE**

No Phase 1 action-value result has been inspected.

This phase does not rank actions.

Its purpose is to determine whether the proposed action grid can support a fair Phase 1B comparison.

---

# 1. Research purpose

Phase 1 asks:

> Which legal physical action provides the most useful information about the future transition under a fixed safety budget?

Before measuring information value, Phase 1A asks a simpler question:

> Can the proposed actions be compared on a sufficiently broad and interpretable common set of hidden worlds?

Phase 1A therefore evaluates legality, cohort selection, hidden-parameter shift, and Bayes numerical adequacy only.

It does not evaluate which action is more informative.

---

# 2. Frozen target definition

The future-transition target is fixed independently of the diagnostic action.

For every hidden world:

```text
Y = first transition under monotonically increasing force
    at the fixed nominal reference angle of 45 degrees
```

Possible labels:

```text
MATERIAL_YIELD
SUPPORT_SLIP
```

Diagnostic action angle must not redefine the target.

This prevents the correct answer from changing when the action changes.

---

# 3. Target near-tie rule

Near-tie filtering is evaluated once per world at the frozen target angle:

```text
45 degrees
```

Normalized margin:

```text
abs(F_material - F_slip)
--------------------------------
min(F_material, F_slip)
```

Frozen minimum:

```text
epsilon_tie = 0.10
```

Diagnostic action angle must not change near-tie eligibility.

---

# 4. Exploratory candidate action grid

Phase 1A will evaluate the following nine one-step diagnostic actions:

```text
3 N × 30°
3 N × 45°
3 N × 60°

5 N × 30°
5 N × 45°
5 N × 60°

7 N × 30°
7 N × 45°
7 N × 60°
```

These actions are exploratory candidates.

Passing Phase 1A does not automatically freeze them for Phase 1B.

---

# 5. Action identifiers

Use stable IDs:

```text
A1 = 3 N × 30°
A2 = 3 N × 45°
A3 = 3 N × 60°

A4 = 5 N × 30°
A5 = 5 N × 45°
A6 = 5 N × 60°

A7 = 7 N × 30°
A8 = 7 N × 45°
A9 = 7 N × 60°
```

---

# 6. Physical interpretation of the grid

For the current model:

```text
Fx = F * cos(theta)
Fy = F * sin(theta)
```

Lower angles increase horizontal loading.

Higher angles increase downward loading.

Therefore:

```text
7 N × 30°
```

is expected to be the strongest support-slip-side stressor.

And:

```text
7 N × 60°
```

is expected to be among the strongest material-loading stressors.

These expectations are model consequences, not Phase 1 results.

---

# 7. Target-generation rule

Identification pairs must be generated using the frozen target angle:

```text
45 degrees
```

Every accepted pair contains:

```text
one MATERIAL_YIELD world
one SUPPORT_SLIP world
```

under the 45-degree target definition.

Diagnostic actions are evaluated only after the pair has been generated.

No diagnostic action may alter pair labels.

---

# 8. Pair structure

The Phase 0.5 paired identification design is retained.

Within each pair, twins share nuisance variables where appropriate.

They differ in the failure-margin variables required to produce opposite labels.

Universal-cohort filtering operates at the complete-pair level.

If either twin fails universal legality, the entire pair is excluded from the primary universal cohort.

This preserves paired class balance.

---

# 9. Action legality

For one world and one diagnostic action, the action is legal only if all frozen conditions hold.

## A. Pre-transition condition

```text
action force < first transition force
```

under the diagnostic action's actual angle.

## B. Support safety condition

```text
rho < 0.80
```

where:

```text
rho =
abs(Fx)
-----------------------
mu_s * (m*g + Fy)
```

## C. Normal-force condition

```text
m*g + Fy > 0
```

This condition is retained for model integrity even if it is expected to be automatically satisfied by the current downward-force grid.

## D. Clipping condition

The action must remain below the clipped support regime.

---

# 10. Universal legal cohort

The primary Phase 1B comparison is intended to use a universal legal cohort.

A complete identification pair enters that cohort only if:

> both twins are legal under all nine candidate actions.

Conceptually:

```text
pair legal under A1
AND A2
AND A3
...
AND A9
```

Only then may it enter the future primary paired action comparison.

---

# 11. Why universal legality is diagnostic rather than sufficient

Universal legality solves one problem:

> different actions are not compared on different surviving populations.

But it can introduce another:

> the retained worlds may become unusually stable against both support slip and material yield.

Phase 1A therefore measures how strongly universal filtering changes the identification population.

---

# 12. Action-specific cohorts

Phase 1A must also report action-specific legal cohorts.

For each action:

```text
fraction of pairs legal under that action
```

These cohorts are secondary diagnostic views.

They are not interchangeable with the universal cohort.

---

# 13. Policy-view principle

A real agent does not know in advance that a world belongs to the universal cohort.

Therefore Phase 1B must eventually retain a separate policy-level view in which unsafe outcomes are counted rather than silently removed.

Phase 1A does not yet compute policy value.

It only records the legality quantities required to support that later analysis.

---

# 14. Rejection reasons

For every action and pair, record all applicable rejection reasons.

Possible reasons include:

```text
SUPPORT_RHO_LIMIT
TRANSITION_BEFORE_OR_AT_PROBE
NONPOSITIVE_NORMAL_FORCE
CLIPPED_SUPPORT_REGIME
```

Multiple reasons may coexist.

Do not retain only the first reason.

---

# 15. Fixed rejection precedence

For summary tables that require one primary rejection category, use this frozen precedence:

```text
1. TRANSITION_BEFORE_OR_AT_PROBE
2. SUPPORT_RHO_LIMIT
3. NONPOSITIVE_NORMAL_FORCE
4. CLIPPED_SUPPORT_REGIME
```

The complete overlap matrix must still be reported separately.

---

# 16. Mandatory per-action outputs

For each action report:

```text
action_id
force_N
angle_deg
legal_pair_fraction
legal_world_fraction
primary rejection counts
all-reason overlap counts
rho minimum
rho median
rho p95
rho maximum
first-transition force margin minimum
first-transition force margin median
first-transition force margin p05
```

No information-value metric is allowed in Phase 1A.

---

# 17. Transition-margin diagnostic

For each diagnostic action define:

```text
action transition margin =
(first_transition_force - action_force)
----------------------------------------
first_transition_force
```

for legal worlds.

This measures how close the action comes to observing the transition itself.

Phase 1A must report at least:

```text
minimum
p05
median
```

per action.

A strong future information result must not be interpreted as useful diagnosis if the action merely sits immediately below failure.

---

# 18. Universal-cohort outputs

Report:

```text
total generated pairs
target-valid pairs
universally legal pairs
universal pair fraction
universal world fraction
```

Also report the number of pairs excluded by:

```text
exactly 1 action
2 actions
3 actions
...
all 9 actions
```

---

# 19. Leave-one-out bottleneck diagnostic

For every candidate action:

> recompute the universal cohort after removing only that action.

Report:

```text
universal fraction with all 9 actions
universal fraction without A1
...
universal fraction without A9
```

Define:

```text
leave_one_out_gain(action)
=
universal_fraction_without_action
-
universal_fraction_all_actions
```

This is the primary bottleneck diagnostic.

No action is removed automatically from the grid based on this number alone.

---

# 20. Legality nesting

Determine whether legality sets are nested.

For example:

```text
legal(A7) subset of legal(A4)
```

may occur if one action is strictly more demanding under the current model.

Report all strong or exact nesting relationships.

This helps identify redundant actions.

---

# 21. Hidden-parameter shift

Compare the universal cohort against the original paired identification population.

Mandatory hidden parameters:

```text
static_friction
material_yield_strength
container_mass
contact_area
contact_stiffness
support_gain
kinetic_friction
```

For each parameter report at least:

```text
original mean
universal-cohort mean
original median
universal-cohort median
standardized mean difference
```

The goal is not inference.

The goal is to detect strong cohort truncation.

---

# 22. Target-margin shift

Compare the 45-degree target-margin distribution before and after universal filtering.

Report:

```text
original p05
original median
universal p05
universal median
fraction near epsilon_tie
```

Universal filtering must not silently convert the experiment into a dataset containing only unusually easy target separations.

---

# 23. Class balance

Because complete opposite-label pairs are retained or rejected together, the universal paired identification cohort should remain:

```text
50% MATERIAL_YIELD
50% SUPPORT_SLIP
```

Phase 1A must assert this invariant.

Any imbalance indicates an implementation error.

---

# 24. Noise

Phase 1A does not require noisy observations for action-value inference.

Legality must always be computed from true physical quantities.

Measurement noise must never affect cohort membership.

---

# 25. Future Phase 1B noise rule

The planned primary Phase 1B design is:

```text
common random numbers across actions within the same world
```

Meaning each world receives one standardized noise draw shared across its counterfactual actions.

A robustness analysis should later use independent action-specific noise draws.

Phase 1A does not use noise to choose the action grid.

---

# 26. Bayes numerical gate

The Phase 0.5b targeted proposal is not assumed to remain adequate for all Phase 1 actions.

For every candidate action, Phase 1A must evaluate Bayes ESS using the frozen numerical method.

Report for both classes:

```text
p05 ESS fraction
median ESS fraction
```

Frozen adequacy thresholds remain:

```text
p05 >= 0.05
median >= 0.20
```

This diagnostic uses ESS only.

Do not report per-action Bayes AUC during Phase 1A.

---

# 27. Phase 1A forbidden outputs

The following must not be computed or inspected during the feasibility stage:

```text
expected information gain
mutual information
per-action ROC-AUC
per-action classifier accuracy
per-action log loss
per-action Brier score
action winner
action ranking
best action
```

Phase 1A is a design diagnostic, not an action-value experiment.

---

# 28. Fresh randomness

Phase 1A must use fresh seeds not used by:

- Phase 0.5 confirmatory Attempt 1
- Phase 0.5b validation
- exploratory Bayes repair studies

The exact Phase 1A seeds must be frozen before execution.

---

# 29. Pre-registered redesign triggers

The candidate action grid must be reconsidered before Phase 1B if any of the following occur.

## Trigger A — Universal cohort too small

Redesign if:

```text
universal legal pair fraction < 0.50
```

Rationale:

A primary analysis that discards more than half of otherwise target-valid identification pairs is judged too strongly conditioned on universal action legality for the initial Phase 1 design.

---

## Trigger B — Single-action bottleneck

Redesign if removing one action increases the universal legal-pair fraction by:

```text
>= 0.20 absolute
```

Example:

```text
all actions        = 0.55
without action A7  = 0.77
gain               = 0.22
```

This indicates that one action is disproportionately defining the population.

---

## Trigger C — Transition-before-probe dominance

Redesign an action if:

```text
>= 20% of target-valid pairs
```

are rejected because either twin reaches a physical transition before or at that diagnostic action.

This indicates that the action is functioning too much like outcome revelation rather than safe diagnosis.

---

## Trigger D — Strong hidden-parameter shift

Flag redesign if any major hidden parameter shows:

```text
absolute standardized mean difference >= 0.50
```

between the original paired identification population and the universal cohort.

Parameters of particular concern:

```text
static_friction
material_yield_strength
contact_stiffness
support_gain
```

---

## Trigger E — Target-margin distortion

Redesign if the universal cohort changes the median 45-degree target margin by:

```text
>= 50% relative
```

compared with the original target-valid paired population.

---

## Trigger F — Bayes numerical inadequacy

Redesign the numerical method or action grid if any candidate action fails:

```text
p05 ESS fraction >= 0.05
median ESS fraction >= 0.20
```

for either class.

Do not rank actions while numerical adequacy differs materially across actions.

---

# 30. Interpretation of triggers

A redesign trigger does not mean:

> the action is scientifically bad.

It means:

> the proposed Phase 1B comparison would be difficult to interpret fairly under the current design.

Redesign occurs before any information-value result is viewed.

---

# 31. Possible redesign directions

If Phase 1A fails, possible redesigns include:

- removing an extreme action
- narrowing the force range
- narrowing the angle range
- deliberately retaining matched-Fx action pairs
- replacing the force-angle grid with a rho-matched design
- separating support-risk and material-risk action families

No redesign choice should use Phase 1 information-value results.

---

# 32. Matched-Fx diagnostic pairs

Two approximate matched-horizontal-force comparisons are intentionally retained:

```text
3 N × 30°  vs  5 N × 60°

5 N × 45°  vs  7 N × 60°
```

These pairs have similar horizontal force but different vertical loading.

They may later help isolate the effect of normal loading from horizontal loading.

Phase 1A must not interpret their information value.

---

# 33. Phase 1B gate

Phase 1B may be designed only after Phase 1A establishes:

1. the final action set
2. acceptable cohort retention
3. interpretable hidden-parameter shift
4. acceptable transition margins
5. adequate Bayes ESS for every retained action
6. explicit treatment of action risk outside the universal cohort

Only then may the project freeze:

```text
information metric
action-value estimator
noise coupling
seeds
comparison protocol
confirmatory acceptance rules
```

---

# 34. Valid Phase 1A outcomes

All of the following are scientifically valid outcomes:

```text
the 3×3 grid is feasible
the grid is too aggressive
one action dominates cohort rejection
the universal cohort is too selective
the Bayes proposal fails for some actions
the grid should be reparameterized
```

Phase 1A does not need to "pass."

Its purpose is to prevent an uninterpretable Phase 1B experiment.

---

# Working conclusion

Before asking:

> Which action reveals more?

Phase 1A asks:

> Are these actions even being compared on a world population that still means what we think it means?