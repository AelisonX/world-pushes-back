# world-pushes-back

A small experimental sandbox for testing whether safe tool-side interaction can reveal which physical constraint is likely to fail first.

The motivating question is simple:

> When the world resists a command, can a robot tell *why* before pushing harder makes the situation worse?

---

# Motivating scenario

## The Ice Cream Problem

Imagine a robot trying to scoop very hard ice cream from a movable container.

A naive strategy is:

```text
blocked -> increase force
```

But stronger force can produce two different physical outcomes:

```text
CONTACT_BLOCKED -> MATERIAL_YIELD
```

or:

```text
CONTACT_BLOCKED -> SUPPORT_SLIP
```

The tool may penetrate the material.

Or the entire container may move instead.

The safety problem is therefore:

> Blind force escalation can move the environment instead of completing the intended task.

This repository does not attempt to model realistic ice-cream rheology.

The scenario is a deliberately simple test case for competing physical thresholds.

---

# Research principle

> The world can reject a command in different physical ways.

The first question is not how an agent should respond.

The first question is whether the difference can be observed safely.

Only after that question is answered does it make sense to ask which action a robot should choose.

---

# Current research status

```text
Phase 0    — Identifiability                         COMPLETE
Phase 0.5  — Confounded Identifiability              COMPLETE
Attempt 1  — Bayes confirmatory reference            INVALID: ESS
Phase 0.5b — Bayes numerical repair validation       PASS
Phase 1    — Diagnostic Action Value                  UNLOCKED
```

The project is not yet testing an intelligent policy.

---

# Physical model

The current toy model contains two competing future transitions:

- `MATERIAL_YIELD`
- `SUPPORT_SLIP`

The physical world contains hidden variables including:

- material yield strength
- contact area
- container mass
- static friction
- kinetic friction
- contact stiffness
- support gain

The robot does not directly observe those quantities.

---

# Legal observations

The official tool-side observation is:

```text
Fx
Fy
tip_dx
tip_dy
```

Direct container pose is intentionally hidden.

Privileged quantities such as:

```text
static friction
material yield strength
rho
transition thresholds
future label
```

are not legal model inputs.

---

# Rigid and compliant support

Two support models are used.

## Rigid support

Idealized support with no pre-slip support motion.

Under this model, safe observations contain little reliable information about how close the system is to support slip.

## Compliant support

A simplified model in which small horizontal pre-slip motion increases as friction demand approaches the support-slip threshold.

This model is an experimental ablation.

It is not a claim about real surface mechanics.

---

# Phase 0

Phase 0 asked:

> Is useful pre-transition information present at all?

The main result was conditional.

When the physical system exposes a measurable compliant-motion precursor, future transition type becomes distinguishable before the transition occurs.

When the support is rigid, the useful signal largely disappears.

The dominant informative observation channel is:

```text
tip_dx
```

or horizontal tool-tip displacement.

Force-only observations provide little useful identification information under the frozen Phase 0.5 design.

---

# Phase 0.5 — Confounded Identifiability

Phase 0.5 introduced a stricter experimental design.

Important controls included:

- paired worlds
- one future-transition class per twin
- shared nuisance variables
- near-tie exclusion
- safe-cohort filtering
- pair-preserving train/test split
- sham control
- label shuffle
- rigid-null control
- privileged `rho` canary
- pair-cluster bootstrap confidence intervals
- Monte Carlo Bayes reference
- preregistered Bayes ESS criterion
- separate operational and identification priors

---

## Phase 0.5 confirmatory Attempt 1

The frozen confirmatory run produced:

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

The pattern strongly localized the predictive signal to compliant motion.

However, the run failed the preregistered Bayes effective-sample-size criterion.

The failure was:

```text
MATERIAL_YIELD p05 ESS fraction = 0.03265
required threshold              = 0.05
```

Therefore the run remains formally recorded as:

```text
CONFIRMATORY_RUN_INVALID_BAYES_ESS
```

That result is not retroactively changed.

---

# Bayes failure diagnosis

Increasing particle count did not repair the problem.

```text
20,000 particles  -> yield p05 ≈ 0.03265
50,000 particles  -> yield p05 ≈ 0.03233
100,000 particles -> yield p05 ≈ 0.03297
```

The problem was therefore not insufficient raw particle count.

A channel-ablation diagnostic localized the problem to:

```text
tip_dx
```

Results:

```text
FULL          -> FAIL
MOTION_ONLY   -> FAIL
TIP_DX_ONLY   -> FAIL

FORCE_ONLY    -> PASS
TIP_DY_ONLY   -> PASS
```

The same channel that carried most of the useful physical signal was also the channel that caused the strongest importance-weight concentration.

---

# Phase 0.5b — Bayes numerical repair

An exploratory diagnostic identified a targeted defensive-mixture importance proposal.

The proposal was then frozen before new validation data were generated.

Frozen proposal:

```text
alpha = 0.10
tau   = 0.0002 m
```

Interpretation:

```text
10% uniform proposal
90% tip_dx-targeted proposal
targeting width = 1 × motion-noise standard deviation
```

Importance correction preserves the original scientific prior.

---

## Independent validation

Phase 0.5b used:

- a new dataset seed
- a new split seed
- a new Bayes particle seed
- the same physical model
- the same observation model
- the same likelihood
- the same ESS thresholds
- the same safety rules

Result:

```text
PHASE0_5B_VALIDATION_PASS
```

Bayes AUC:

```text
Uniform  = 0.809232
Targeted = 0.809232
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

Required thresholds:

```text
p05    >= 0.05
median >= 0.20
```

Maximum posterior difference between uniform and targeted estimators:

```text
1.4432899320127035e-15
```

Interpretation:

> The targeted proposal repaired the numerical ESS problem without materially changing the Bayes estimator.

---

# Operational prevalence

The default unbalanced world distribution strongly favors material yield.

Representative result:

```text
MATERIAL_YIELD ≈ 98.896%
SUPPORT_SLIP   ≈ 1.104%
near-tie       ≈ 0.704%
```

This is deliberately separate from the balanced identification prior.

Occurrence and identifiability are different questions.

---

# What the project currently supports

Within the current toy simulator:

- compliant pre-slip motion can contain information about the future transition
- horizontal tool-tip displacement carries most of that useful signal
- force-only observations remain near chance
- rigid support removes the useful compliant-motion signal
- the paired identification design produces substantially above-chance separability
- the learned classifier and Bayes reference agree closely
- uniform Monte Carlo sampling can suffer severe lower-tail ESS collapse
- a targeted importance proposal can repair that numerical problem without changing the scientific posterior

---

# What the project does not claim

This repository does not currently demonstrate:

- real-world robotic performance
- realistic ice-cream mechanics
- real-hardware support-slip prediction
- universal sensing thresholds
- universal optimality of `tip_dx`
- demonstrated task-safety improvement
- superiority over conventional robotics control
- a universal theory of embodied intelligence
- a general robotics benchmark

Phase 0.5b validates the numerical repair.

It does not validate the physical model itself.

---

# Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run regression tests:

```bash
pytest -q
```

Run the original Phase 0 pipeline:

```bash
python run_phase0.py
```

Run the frozen Phase 0.5 confirmatory experiment:

```bash
python run_phase0_5_confirmatory.py
```

Run Bayes repair diagnostics:

```bash
python phase0_5_bayes_repair.py
python phase0_5_bayes_ess_ablation.py
python phase0_5_bayes_targeted_proposal.py
```

Run the independent Phase 0.5b validation:

```bash
python run_phase0_5b_validation.py
```

The GitHub Actions workflow runs the complete research pipeline automatically.

---

# Key documents

```text
PROJECT_STATUS.md
PHASE0_FINDINGS.md
PHASE0_5_PLAN.md
PHASE0_5_MANIFEST.json
PHASE0_5B_MANIFEST.json
PHASE0_5B_FINDINGS.md
RESULTS.md
```

---

# Phase 1

Phase 1 is now unlocked.

The next research question is:

> Which legal physical action provides the most useful information about the future transition under a fixed safety budget?

This phase is called:

**Diagnostic Action Value**

The goal is not to build a more complicated classifier.

The goal is to compare ordinary physical actions according to how much useful information they provide relative to their physical risk.

Possible action dimensions include:

- force magnitude
- force direction
- lateral nudges
- unload-and-hold behavior
- repeated low-risk observations

There is no magical `PROBE` action.

Every diagnostic action must be an ordinary physical intervention.

---

# Phase 1 entry rule

Before running Phase 1 experiments:

1. define the legal action set
2. define the safety budget
3. define the information-value metric
4. define same-world counterfactual action comparison
5. freeze the evaluation protocol
6. only then run action comparisons

---

# Working conclusion

The world may expose information before it changes state.

But useful information is not enough by itself.

The estimator must know where to look.

The next question is whether the robot can choose an action that reveals more while risking less.