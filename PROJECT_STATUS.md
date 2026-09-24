# Project Status

## Project

`world-pushes-back`

---

## Current phase

**Phase 0.5b complete — Bayes numerical repair independently validated**

The project has established conditional pre-transition identifiability in the current toy simulator and has independently validated a repaired Bayes numerical approximation.

The project is not yet testing an intelligent policy.

The next research question is:

> Which legal physical action provides the most useful information about the future transition under a fixed safety budget?

This is the Phase 1 question.

---

## Core motivating scenario

### The Ice Cream Problem

A robot attempts to scoop hard material from a movable container.

Blind force escalation may cause:

`CONTACT_BLOCKED -> SUPPORT_SLIP`

instead of:

`CONTACT_BLOCKED -> MATERIAL_YIELD`

The safety motivation is:

> Increasing force can move the environment instead of completing the intended task.

The current simulator is deliberately simplified.

It is not intended to model realistic ice-cream rheology or real robotic contact mechanics.

---

## Core research principle

> The world can reject a command in different physical ways.

The first question is not how an agent should respond.

The first question is whether the difference can be observed safely.

Phase 0 and Phase 0.5 asked whether useful pre-transition information exists.

Phase 1 will ask which action is worth taking to obtain that information.

---

## Current physical model

The minimal model studies competition between two future transitions:

- `MATERIAL_YIELD`
- `SUPPORT_SLIP`

Current contact states:

- `CONTACT_BLOCKED`
- `MATERIAL_YIELD`
- `SUPPORT_SLIP`

Important hidden physical variables include:

- material yield strength
- contact area
- container mass
- static friction
- kinetic friction
- contact stiffness
- pre-slip displacement limit
- support gain

---

## Observation model

Legal tool-side observations are:

- `Fx`
- `Fy`
- `tip_dx`
- `tip_dy`

Direct container pose is intentionally hidden.

Privileged quantities such as:

- static friction
- material yield strength
- `rho`
- transition thresholds
- future label

are not legal classifier inputs.

---

## Support models

### Rigid support

Idealized support with no pre-slip support displacement.

### Compliant support

A simplified support model with small pre-slip displacement as friction demand approaches the static-friction limit.

The compliant model is an experimental ablation.

It is not a claim about real surfaces or real hardware.

---

# Phase 0 result

Phase 0 established the broad conditional-identifiability pattern.

The strongest result was:

> Pre-transition classification becomes possible in the toy model when the physical system exposes a measurable compliant-motion precursor with sufficient signal-to-noise ratio.

Rigid support generally provides little legal pre-transition information.

Compliant support provides substantially stronger motion-based information.

Horizontal tip displacement is the dominant informative channel.

---

# Phase 0.5 — Confounded Identifiability

Phase 0.5 introduced a stricter identification design.

Key additions included:

- paired worlds
- one world from each future-transition class per pair
- shared nuisance variables
- explicit near-tie exclusion
- frozen safety cohort
- pair-preserving train/test split
- sham control
- label-shuffle control
- rigid-null control
- privileged `rho` canary
- pair-cluster bootstrap confidence intervals
- Monte Carlo Bayes reference
- preregistered ESS adequacy criterion
- separate operational and identification priors

---

## Phase 0.5 Attempt 1

The first frozen confirmatory attempt produced coherent classification results:

```text
FULL           AUC = 0.7844
MOTION_ONLY    AUC = 0.7852
FORCE_ONLY     AUC = 0.5000
SHAM           AUC = 0.5000
LABEL_SHUFFLE  AUC = 0.5182
RIGID_NULL     AUC = 0.5000
RHO_CANARY     AUC = 0.9987
Bayes          AUC = 0.7854
```

Interpretation:

> The legal predictive signal was localized primarily to compliant motion.

However, the run failed the preregistered Bayes ESS criterion.

The failure was:

```text
MATERIAL_YIELD p05 ESS fraction = 0.03265
required threshold              = 0.05
```

The run therefore remains formally recorded as:

```text
CONFIRMATORY_RUN_INVALID_BAYES_ESS
```

This status must not be retroactively changed.

---

## Bayes numerical diagnosis

Increasing particle count did not repair the ESS failure.

```text
20,000 particles  -> yield p05 ≈ 0.03265
50,000 particles  -> yield p05 ≈ 0.03233
100,000 particles -> yield p05 ≈ 0.03297
```

The problem was therefore not raw particle count.

Channel ablation localized the failure to:

```text
tip_dx
```

The same horizontal displacement channel that carries most of the useful physical signal also created the strongest importance-weight concentration under uniform Monte Carlo sampling.

---

# Phase 0.5b — Bayes Proposal Repair Validation

An exploratory diagnostic identified a defensive-mixture importance proposal targeted to `tip_dx`.

The selected proposal was frozen before new validation data were generated:

```text
alpha = 0.10
tau   = 1.0 * motion-noise standard deviation
      = 0.0002 m
```

Interpretation:

- 10% uniform proposal component
- 90% `tip_dx`-targeted component
- importance correction preserves the original scientific prior

Phase 0.5b then used:

- a new dataset seed
- a new train/test split seed
- a new Bayes particle seed
- the same physical model
- the same likelihood
- the same ESS thresholds
- the same legal observations
- the same safety rules

---

## Phase 0.5b result

Status:

```text
PHASE0_5B_VALIDATION_PASS
```

Bayes AUC:

```text
Uniform Bayes AUC  = 0.809232
Targeted Bayes AUC = 0.809232
```

Targeted ESS:

```text
MATERIAL_YIELD
p05    = 0.66721
median = 0.91756

SUPPORT_SLIP
p05    = 0.66467
median = 0.91359
```

Frozen ESS thresholds:

```text
p05    >= 0.05
median >= 0.20
```

Estimator consistency:

```text
Maximum posterior difference
= 1.4432899320127035e-15
```

Interpretation:

> The targeted proposal repaired the numerical ESS failure on new independent data without materially changing the Bayes estimator.

---

# Operational prior

The unbalanced operational world distribution is strongly dominated by material yield.

Representative Phase 0.5 result:

```text
MATERIAL_YIELD ≈ 98.896%
SUPPORT_SLIP   ≈ 1.104%
near-tie       ≈ 0.704%
```

This must remain separate from the balanced paired identification prior.

Occurrence frequency and conditional identifiability are different questions.

---

# Current scientific interpretation

The project currently supports the following conclusions inside the toy simulator.

### 1. Conditional identifiability exists

Under compliant support, one safe observation can contain information about which physical transition will occur first.

### 2. The signal is primarily motion-based

Force-only observations remain near chance.

Horizontal tip displacement carries most of the useful signal.

### 3. Rigid support removes the useful precursor

The rigid-null condition returns to chance.

### 4. The Bayes estimator agrees with the learned classifier

The confirmatory classifier and Bayes reference produce closely aligned AUC values.

### 5. Uniform Monte Carlo can be numerically inefficient

The informative `tip_dx` channel creates lower-tail ESS collapse under the original uniform proposal.

### 6. Targeted importance sampling repairs the numerical approximation

The preregistered targeted proposal passed independent validation without changing the underlying posterior estimates.

---

# Completed components

## Core simulation

- [x] competing material-yield and support-slip thresholds
- [x] rigid support model
- [x] compliant pre-slip support model
- [x] noisy legal tool-side observations
- [x] analytic future-transition label
- [x] explicit safe cohort

## Phase 0

- [x] baseline identifiability
- [x] force sweep
- [x] compliance ablation
- [x] sensor ablation
- [x] sensing threshold
- [x] precursor-to-noise study
- [x] seed robustness
- [x] parameter sensitivity
- [x] rigid-baseline anomaly audit

## Phase 0.5

- [x] paired identification worlds
- [x] near-tie exclusion
- [x] safe paired cohort
- [x] pair-preserving split
- [x] classifier controls
- [x] pair-cluster bootstrap
- [x] Monte Carlo Bayes reference
- [x] Bayes convergence diagnostic
- [x] Bayes ESS diagnostic
- [x] frozen confirmatory manifest
- [x] confirmatory Attempt 1
- [x] failed ESS result preserved

## Phase 0.5b

- [x] particle-count repair diagnostic
- [x] ESS channel ablation
- [x] targeted-proposal exploration
- [x] frozen Phase 0.5b manifest
- [x] independent validation dataset
- [x] importance-corrected targeted proposal
- [x] estimator-consistency check
- [x] independent validation pass

---

# Current claims allowed

The project may state that, within the current toy simulator:

- compliant pre-slip motion can carry information about the future physical transition
- horizontal tool-tip displacement is the dominant legal observation channel in the current model
- force alone does not provide useful identification signal under the frozen Phase 0.5 design
- the rigid-null condition removes the compliant-support signal
- the paired identification experiment produces substantially above-chance separability
- learned and Bayes estimates are closely aligned
- uniform Monte Carlo sampling suffered a reproducible lower-tail ESS problem
- a preregistered targeted importance proposal repaired that numerical problem on independent data

---

# Claims not allowed

The project must not claim:

- validated real-world robotic performance
- realistic ice-cream mechanics
- universal pre-slip sensing laws
- universal optimality of `tip_dx`
- real-hardware support-slip prediction
- demonstrated safety improvement from active diagnosis
- superiority over conventional robotics control methods
- a new universal theory of embodied intelligence
- that Phase 0.5 Attempt 1 became valid retroactively
- that Phase 0.5b validates the physical model itself

---

# Kill criteria

## A — No useful pre-transition information

If future extensions show that legal observations do not provide useful information under richer or more realistic physical conditions, stop or reformulate.

## B — No action value

If different legal actions do not provide meaningfully different information under the safety budget, Phase 1 should not invent complexity merely to justify continuation.

## C — No safety benefit

If active diagnosis later improves classification but does not reduce unsafe force escalation relative to a push-harder baseline, the original safety motivation is unsupported.

## D — Excessive diagnostic cost

If useful diagnosis requires too much force, too much motion, too many actions, or too much time, the approach is not useful under the intended safety budget.

---

# Phase 1 entry

Phase 1 is now unlocked.

The Phase 1 question is:

> Which legal physical action provides the most useful information about the future transition under a fixed safety budget?

Phase 1 should compare actions rather than merely increasing classifier complexity.

Candidate action dimensions may include:

- force magnitude
- force direction
- unload-and-hold behavior
- lateral nudges
- repeated low-risk observations

Any action must remain physically ordinary.

The system must not introduce a magical `PROBE` operation that directly queries hidden state.

---

# Next milestone

## Phase 1 — Diagnostic Action Value

Before implementing an active policy:

1. define the legal action set
2. define the safety budget
3. define an information-value metric
4. define same-world counterfactual action comparison
5. freeze the Phase 1 evaluation protocol
6. only then compare actions

The next goal is not:

> build a smarter classifier

The next goal is:

> determine whether choosing a different physical action produces more useful information per unit of risk.

---

# Current status summary

**Physical model:** implemented  
**Phase 0:** complete  
**Phase 0.5 paired identifiability:** complete  
**Phase 0.5 Attempt 1:** invalid due to Bayes ESS failure  
**Bayes failure diagnosis:** complete  
**Phase 0.5b numerical repair:** independently validated  
**Policy development:** not started  
**Phase 1:** unlocked  
**Real-world claim:** not allowed  
**Benchmark claim:** not allowed

---

## Working conclusion

The world may expose information before it changes state.

But useful information is not enough by itself.

The next question is whether a robot can choose an action that reveals more while risking less.