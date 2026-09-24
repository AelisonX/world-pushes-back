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
Phase 0.5b — Bayes numerical repair + sampler audit  PASS
Phase 1A   — Action-Grid Cohort Feasibility          COMPLETE
Phase 1B   — Diagnostic Action Value                  NOT YET RUN
```

Phase 1A tested whether nine candidate diagnostic actions could still be compared on a broad common population of hidden worlds before any action-value ranking was allowed.

Result:

```text
5,000 identification pairs
10,000 simulated worlds

4,809 pairs legal under all nine actions
universal retention = 0.9618

redesign triggers = 0
```

The grid passed the frozen Phase 1A feasibility criteria.

High retention also suggests that the current action grid is conservative: selection bias was not strongly stressed, and Phase 1B may find that the candidate actions are too similar in information value to produce a meaningful contrast.

See:

[PHASE1A_FINDINGS.md](PHASE1A_FINDINGS.md)

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

Importance correction preserves the empirical class-conditional target distribution used by the Bayes reference.

---

## Realized sampler validation

The initial Phase 0.5b study showed that the targeted proposal had much better theoretical ESS characteristics.

A later prospective audit then tested actual finite-sample categorical draws from that proposal.

Result:

```text
PHASE0_5B_REALIZED_SAMPLER_AUDIT_PASS
```

Targeted realized ESS:

```text
MATERIAL_YIELD
p05    ≈ 0.690
median ≈ 0.923

SUPPORT_SLIP
p05    ≈ 0.719
median ≈ 0.920
```

Frozen thresholds:

```text
p05    >= 0.05
median >= 0.20
```

Targeted posterior error against the exact full empirical-bank reference was also small:

```text
p95 absolute error ≈ 0.00158
maximum error      ≈ 0.00522
```

The original Phase 0.5 Attempt 1 remains invalid.

The later numerical repair does not retroactively change that historical result.

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

# Phase 1A — Action-Grid Cohort Feasibility

Before comparing which physical action reveals more information, Phase 1A asked a narrower question:

> Can the candidate actions be compared on the same hidden worlds, or does action legality silently rewrite the sample?

The frozen exploratory grid contained nine actions:

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

These are simulator forces, not calibrated real-world ice-cream forces.

The future-transition target remained fixed at the nominal 45-degree continuation angle for every diagnostic action.

In other words:

> changing the probe angle did not change the scoring target.

---

## Phase 1A result

The frozen run generated:

```text
5,000 identification pairs
10,000 simulated worlds
```

Universal legality required both twins in a pair to remain legal under all nine actions.

Result:

```text
4,809 universally legal pairs
universal retention = 96.18%
```

Per-action legal-pair fractions ranged from:

```text
96.70% to 100%
```

The most restrictive grid cell was:

```text
7 N × 30°
```

Removing it increased universal retention by only:

```text
3.02 percentage points
```

The frozen single-action bottleneck threshold was:

```text
20 percentage points
```

All nine per-action Bayes ESS numerical checks passed.

Hidden-parameter and target-margin shift diagnostics stayed below the frozen redesign thresholds; exact values are retained in the Phase 1A findings artifact.

No preregistered redesign trigger fired.

Therefore:

```text
PHASE1A_GRID_FEASIBLE
```

This does not mean the current grid is difficult.

The high retention itself suggests that the grid is relatively conservative.

That matters for Phase 1B: nine actions can all be legal and still be too similar to produce a meaningful information-value contrast.

See:

[PHASE1A_FINDINGS.md](PHASE1A_FINDINGS.md)

---

# What the project currently supports

Within the current toy simulator:

- compliant pre-slip motion can contain information about the future transition
- horizontal tool-tip displacement carries most of that useful signal
- force-only observations remain near chance under the frozen Phase 0.5 design
- rigid support removes the useful compliant-motion signal
- the paired identification design produces substantially above-chance separability
- the learned classifier and Bayes reference agree closely
- uniform Monte Carlo sampling can suffer severe lower-tail ESS collapse
- the targeted importance proposal works as a finite-sample sampler over the audited empirical particle prior
- the current nine-action Phase 1A grid retains 96.18% of target-valid pairs under universal legality
- no frozen Phase 1A cohort-redesign trigger fired

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
- an optimal diagnostic action
- an action-information ranking

Phase 1A validates the feasibility of the comparison design under the current toy simulator.

It does not establish which action is most informative.

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

Run the Phase 0.5b validation:

```bash
python run_phase0_5b_validation.py
```

Run the realized importance-sampler audit:

```bash
python phase0_5b_sampler_audit.py
```

Run Phase 1A action-grid feasibility:

```bash
python phase1a_feasibility.py
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
PHASE0_5B_SAMPLER_AUDIT.md
PHASE0_5B_SAMPLER_AUDIT_MANIFEST.json
PHASE0_5B_SAMPLER_AUDIT_FINDINGS.md
PHASE1_PLAN.md
PHASE1A_FEASIBILITY_PLAN.md
PHASE1A_FEASIBILITY_MANIFEST.json
PHASE1A_FINDINGS.md
RESULTS.md
```

---

# Phase 1B

Phase 1A has established that the current candidate action grid can be compared on a broad common legal cohort under the frozen feasibility rules.

The next research question is:

> Given the same hidden world, which legal physical action changes what the agent can know before the physical transition occurs?

Phase 1B will require freezing, before inspecting action-value results:

1. the primary information-value metric
2. the same-world action comparison
3. noise coupling
4. the policy-risk view
5. the estimator
6. fresh seeds
7. confirmatory acceptance criteria
8. a minimum meaningful action-information difference

If observed differences fall below that frozen minimum effect, the interpretation will be:

> the current action grid is too similar to distinguish meaningfully

not:

> the physical world contains no useful constraint information

No Phase 1B action-value comparison has been run yet.

---

# Working conclusion

The world may expose information before it changes state.

But useful information is not enough by itself.

Before comparing actions, the experiment must also make sure that the actions are still being tested on meaningfully comparable worlds.

Phase 1A passed that feasibility check.

The next question is whether different legal actions actually reveal meaningfully different information.