# Phase 0.5 Plan

## Status

**Phase 0.5 — Confounded Identifiability**

Phase 1 remains blocked.

Phase 0 established that the simulator, tests, experiment runner, artifact generation, and CI pipeline function as intended.

Phase 0 did **not** establish a scientific identifiability result.

The strongest Phase 0 separability results were largely produced by a simulator structure in which slip proximity was written too directly into horizontal tool-tip displacement.

Phase 0.5 exists to remove that shortcut before any active diagnostic-action study begins.

---

## 1. Research question

Phase 0.5 asks:

> When tool-tip displacement no longer directly encodes support-slip proximity, how much information about the counterfactual future transition remains in one safe observation?

Possible future transitions remain:

- `MATERIAL_YIELD`
- `SUPPORT_SLIP`

The purpose is not to demonstrate high classification accuracy.

The purpose is to construct a valid single-observation identifiability problem that can fail honestly.

---

## 2. Phase 0 reinterpretation

The Phase 0 verdict is:

> **Infrastructure verified. Scientific identifiability result not established.**

Phase 0 demonstrated:

- the simulator executes correctly
- the regression tests and CI pipeline work
- classifier feature scaling matters
- the corrected friction-demand ratio is internally consistent
- the analysis pipeline can recover a signal deliberately placed into the observation model

Phase 0 did not demonstrate:

- real-world pre-slip identifiability
- general tool-side observability of support slip
- a realistic pre-slip friction law
- meaningful active diagnostic value
- a real-world sensor requirement
- a universal precursor-to-noise scaling law

The Phase 0 SNR sweep is retained as a simulator sanity check rather than a scientific finding.

---

## 3. Core Phase 0 shortcut

The current compliant model contains approximately:

```text
tip_dx
=
contact_deformation
+
support_precursor
+
measurement_noise
```

where:

```text
contact_deformation
=
Fx / k_contact
```

and:

```text
support_precursor
=
L * rho
```

for the safe, unsaturated region.

The friction-demand ratio is:

```text
            |Fx|
rho = -----------------
      mu_s * (m*g + Fy)
```

In Phase 0:

- `k_contact` was fixed across worlds
- support precursor gain was effectively fixed
- support motion was the main additional horizontal-motion source

This made `tip_dx` too directly informative about slip proximity.

Phase 0.5 must remove this structural shortcut.

---

## 4. Phase 0.5 hidden world state

The minimum Phase 0.5 hidden state contains:

- container mass `m`
- static friction coefficient `mu_s`
- material yield strength
- contact area
- pre-slip displacement scale `L`
- contact/tool stiffness `k_contact`
- support precursor gain `g_support`

`k_contact` and `g_support` are newly treated as world-specific nuisance variables.

Kinetic friction is not part of the official Phase 0.5 identifiability model because Phase 0.5 studies the state before the first transition.

It may remain in existing data structures for compatibility, but it must not be treated as an informative Phase 0.5 latent variable.

---

## 5. Observation model

The official horizontal tool-tip model will have the form:

```text
tip_dx
=
delta_contact
+
delta_support
+
motion_noise
```

with:

```text
delta_contact
=
Fx / k_contact
```

and:

```text
delta_support
=
g_support * L * rho
```

for official samples that remain below the support-slip boundary.

The official observation remains:

```text
o =
(
    Fx,
    Fy,
    tip_dx,
    tip_dy
)
```

The diagnostic classifier may receive only these legal observations and the known probe action.

It must not receive:

- `m`
- `mu_s`
- material yield parameters
- `k_contact`
- `g_support`
- `rho`
- transition thresholds
- future label `Y`

except in explicit privileged controls.

---

## 6. Friction-demand ratio

The Phase 0.5 friction-demand ratio remains:

```text
            |Fx|
rho = -----------------
      mu_s * (m*g + Fy)
```

under the current convention that positive `Fy` presses downward and increases normal force.

Required physical properties:

- increasing `m` lowers `rho`
- increasing `mu_s` lowers `rho`
- increasing downward `Fy` lowers `rho`
- increasing `|Fx|` raises `rho`

Official Phase 0.5 samples must maintain positive normal force:

```text
m*g + Fy > 0
```

Contact-loss actions are out of scope for Phase 0.5.

---

## 7. No official clipping

The existing toy support law contains clipping at:

```text
rho >= 1
```

Official Phase 0.5 observations must never reach that region.

The confirmatory Phase 0.5 dataset will impose a frozen safe upper bound:

```text
rho < rho_safe_max < 1
```

The exact value of `rho_safe_max` will be selected during exploratory development and frozen before the confirmatory run.

The official pipeline must contain a regression/property test proving that no confirmatory observation reaches the clipped regime.

If an official sample enters the clipped regime, the run is invalid.

---

## 8. Counterfactual label

The future-transition label must be independent of the diagnostic probe.

For each initial world:

```text
Y =
first transition that would occur
under a fixed reference continuation
from the original pre-probe state
```

The reference continuation is:

> monotonically increase applied force at the fixed nominal angle.

Because the current toy model has analytic transition thresholds, Phase 0.5 does not need to simulate this ramp numerically.

Define:

```text
F_material
=
analytic material-yield threshold
```

and:

```text
F_slip
=
analytic support-slip threshold
```

Then:

```text
if F_material < F_slip:
    Y = MATERIAL_YIELD

if F_slip < F_material:
    Y = SUPPORT_SLIP
```

The diagnostic probe may change the observation.

It must not redefine `Y`.

This prevents the diagnostic action from creating the event it claims to diagnose.

> Do not let the arsonist start the fire and then claim credit for diagnosing a fire.

---

## 9. Near ties

Worlds in which the two thresholds are nearly identical are intrinsically ambiguous.

Define a normalized transition margin such as:

```text
margin
=
abs(F_material - F_slip)
/
min(F_material, F_slip)
```

A frozen tie margin:

```text
epsilon_tie
```

will be selected during exploratory development.

Confirmatory binary-identification results will include only worlds satisfying:

```text
margin >= epsilon_tie
```

The fraction of excluded near-tie worlds must be reported separately.

This exclusion changes the identification prior and must not be confused with the operational prior.

---

## 10. Two different priors

Phase 0.5 must keep two distributions separate.

### Operational prior

Used only to answer:

> How often does each future transition occur under the default world distribution?

This prior preserves natural occurrence frequencies.

It must not be balanced for classification.

### Identification prior

Used to answer:

> When both future transitions are physically plausible and nuisance variables overlap, how distinguishable are they from one safe observation?

This distribution may deliberately use matching or paired generation.

Its class prevalence must never be interpreted as real-world occurrence frequency.

Operational prevalence and identification performance must always be reported separately.

---

## 11. Paired identification worlds

The working Phase 0.5 identification design uses paired worlds.

Each pair shares nuisance variables such as:

- mass
- contact area
- `L`
- `k_contact`
- `g_support`
- official probe action
- observation-noise model

The pair differs only in variables required to produce opposite counterfactual future transitions, initially expected to involve:

- `mu_s`
- material yield strength

The generator must not deterministically assign one numerical ordering to one label.

Pair construction should use a shared latent center plus randomized perturbation and rejection until:

```text
Twin A -> MATERIAL_YIELD
Twin B -> SUPPORT_SLIP
```

under the same counterfactual reference continuation.

The classifier must never be told that two observations are twins.

Every pair receives a `pair_id`.

---

## 12. Pair-level data splitting

Twin worlds are statistically dependent.

Therefore:

- train/test splitting must occur by `pair_id`
- no pair may cross train and test
- bootstrap confidence intervals must resample pairs, not individual observations

This requirement applies even if each individual world contributes only one observation.

---

## 13. Safe cohort

Phase 0.5 must not generate an observation and then silently discard worlds that became unsafe.

The official identification dataset must be defined as safe **before** classification evaluation.

All official worlds must satisfy the frozen safety constraints for the official probe.

At minimum:

```text
normal_force > 0
rho < rho_safe_max
probe occurs before either transition
```

Unsafe worlds belong to a different experimental question and are reserved for later Phase 1 safety analysis.

Phase 0.5 therefore studies:

> identifiability inside a fixed safe cohort

rather than comparing different survivor populations across increasingly aggressive probes.

---

## 14. Confounder ranges

The exact ranges for:

- `k_contact`
- `g_support`

must not be chosen to manufacture a desired classifier score.

An exploratory confounder-strength study may be used to identify ranges in which:

- contact deformation overlaps support-induced deformation
- support gain varies meaningfully across worlds
- neither source trivially dominates every observation
- the task is neither algebraically trivial nor destroyed only by overwhelming sensor noise

The exploratory range selection must be completed before the confirmatory Phase 0.5 manifest is frozen.

The final ranges must then remain unchanged for the official run.

No fixed “10x” range is assumed in advance without justification.

---

## 15. Noise model

The official Phase 0.5 v0.1 noise model will remain:

- fixed across worlds
- class-independent
- explicitly documented
- known to the Bayes reference

World-specific sensor-noise variation is deferred to a later robustness experiment.

This keeps the first confounded experiment focused on:

- random contact stiffness
- random support precursor gain

rather than mixing several new nuisance mechanisms simultaneously.

---

## 16. Official feature conditions

Phase 0.5 must evaluate at least:

### FORCE_ONLY

```text
Fx
Fy
```

Purpose:

Test whether the fixed probe-force channels themselves contain class information.

### MOTION_ONLY

```text
tip_dx
tip_dy
```

Purpose:

Measure how much motion information survives after the new confounders are introduced.

### FULL

```text
Fx
Fy
tip_dx
tip_dy
```

Purpose:

Measure the best legal single-observation classifier under the official observation model.

---

## 17. Negative and positive controls

### SHAM

Observation consists only of matched noise.

Expected:

```text
ROC-AUC approximately 0.5
```

Purpose:

Verify that the full evaluation pipeline produces chance when no signal exists.

### LABEL SHUFFLE

Shuffle future-transition labels while preserving observations.

Expected:

```text
ROC-AUC approximately 0.5
```

Purpose:

Detect evaluation leakage or broken splitting.

### RIGID NULL

Set:

```text
g_support = 0
```

while using the same matched identification design.

Expected:

No support-specific motion signal.

If the rigid null remains strongly predictive, the causal path must be explained before scientific interpretation continues.

### RHO CANARY

Explicitly provide privileged `rho` as a feature.

Expected:

Strong recoverability of future-transition structure.

Purpose:

Confirm that the classification/evaluation pipeline can detect deliberately privileged information.

`Y` itself must not be used as the canary because that would test almost nothing.

---

## 18. Rigid-baseline anomaly

Phase 0 rigid cells occasionally exceeded chance substantially.

Examples included approximately:

```text
8 N  -> 0.657
20 N -> 0.700
```

The current rigid simulator contains no:

- support preslip
- contact-solver jitter
- gradual yield precursor
- physical micro-slip

At fixed action and fixed contact stiffness:

```text
Fx
Fy
```

are fixed by the probe plus class-independent noise, and:

```text
tip_dx
=
Fx / k_contact
+
noise
```

Therefore hidden world parameters cannot create observation-level class separation unless an actual causal path exists in the implementation.

The Phase 0.5 diagnostic must therefore first test:

- minority-class sample counts
- confidence intervals
- label shuffle
- rigid class-conditional observation distributions
- code paths from label/hidden state to observation generation

If no observation-level dependency exists, the Phase 0 rigid anomaly should be attributed to finite-sample variance rather than given a physical interpretation.

---

## 19. Official metrics

Phase 0.5 will avoid a metrics zoo.

Mandatory confirmatory outputs:

1. sample count `n`
2. ROC-AUC
3. 95% confidence interval for ROC-AUC
4. false-negative rate at `FPR = 0.10`
5. Bayes-reference AUC
6. label-shuffle result
7. sham result
8. canary result

For paired identification data, confidence intervals must use pair-cluster bootstrap resampling.

Balanced accuracy, mutual information, Brier score, calibration, and PR-AUC are not required for the first confirmatory Phase 0.5 run.

Operational-prior performance may later use prevalence-sensitive metrics, but those results must remain separate from identification-prior results.

---

## 20. Bayes reference

The Bayes observer receives exactly the same legal observation as the learned model:

```text
o =
(
    Fx,
    Fy,
    tip_dx,
    tip_dy
)
```

plus the known probe action.

It does not directly receive hidden world state.

Hidden variables are integrated out under the exact identification distribution used for evaluation.

Conceptually:

```text
P(Y | o, a)
∝
integral
    I[Y(z) = Y]
    p(o | z, a)
    p(z)
dz
```

The first implementation should use:

- exact simulator likelihood where available
- Monte Carlo integration over hidden worlds

Particle count must not be chosen by arbitrary convention.

Instead, the Bayes approximation must pass a convergence study.

Candidate particle counts may increase progressively, for example:

```text
1k
5k
10k
50k
100k
```

until:

- posterior estimates stabilize
- Bayes AUC stabilizes
- effective sample size is adequate

The convergence criterion must be frozen before the confirmatory run.

A learned classifier exceeding an approximate Bayes estimate is not automatically evidence of leakage.

It becomes an alarm only if the difference exceeds validated Monte Carlo and statistical uncertainty.

---

## 21. Bayes validation

Before using the Bayes reference as an authority, validate it on a simplified case whose posterior or expected ranking behavior is known.

The Bayes implementation must demonstrate:

- convergence with particle count
- stable AUC
- adequate effective sample size
- reproducibility under fixed seed

If the approximation cannot be validated, Phase 0.5 cannot use the Bayes result as a ceiling.

---

## 22. Exploratory versus confirmatory work

Phase 0.5 has two explicit stages.

### Exploratory stage

May be used to:

- choose reasonable `k_contact` variation
- choose reasonable `g_support` variation
- inspect safe `rho` bounds
- determine tie margin
- test Bayes convergence
- debug paired-world generation
- debug controls

Exploratory numerical results must not be presented as confirmatory Phase 0.5 findings.

### Confirmatory stage

Begins only after the experiment manifest is frozen and committed.

Any change to frozen design variables after seeing the official result invalidates confirmatory status and requires a new versioned experiment.

---

## 23. Confirmatory manifest

Before the first official Phase 0.5 run, freeze:

- operational prior
- identification prior
- counterfactual label definition
- reference continuation
- tie rule
- `rho_safe_max`
- `k_contact` distribution
- `g_support` distribution
- noise model
- official probe action
- paired-world generator
- pair-level split protocol
- Bayes-reference method
- Bayes convergence criterion
- mandatory metrics
- control definitions
- redirect / failure criteria
- random seeds or dataset-generation protocol

Artifacts should also record:

- code commit SHA
- manifest hash
- dataset seed
- pair/world IDs
- sampled hidden parameters for reproducibility

---

## 24. Phase 0.5 success criterion

Phase 0.5 success does **not** mean high classifier accuracy.

Phase 0.5 succeeds if it establishes a valid single-observation experiment in which:

- direct shortcut channels have been removed
- nuisance variables overlap
- the safety cohort is explicit
- the counterfactual label is independent of the diagnostic observation
- train/test separation is valid
- null controls behave correctly
- privileged canary behaves correctly
- Bayes reference is validated
- uncertainty is reported honestly

The final scientific result may show:

- high remaining information
- moderate remaining information
- almost no remaining information

All three are acceptable outcomes.

---

## 25. Valid Phase 1 entry condition

Phase 1 may begin when:

- Phase 0.5 controls pass
- no known privileged shortcut remains
- the identification prior is explicit
- the counterfactual label is stable
- the Bayes reference is credible
- the same underlying worlds can later be exposed to different legal diagnostic actions

High single-observation AUC is **not** required.

A weak single-observation result may provide the strongest motivation for Phase 1.

Phase 1 will ask whether a deliberately chosen second action adds information relative to a sham action under a fixed safety budget.

---

## 26. Failure categories

### Implementation failure

Examples:

- shuffle above chance
- sham above chance
- canary unrecoverable
- pair leakage across train/test
- invalid manifest provenance
- official sample enters unsafe or clipped region

Result:

> Fix implementation. Do not interpret scientifically.

### Simulator-design failure

Examples:

- rigid null remains predictive without a legitimate causal path
- paired generator embeds label signatures
- nuisance distributions do not overlap
- Bayes reference cannot be validated

Result:

> Redesign Phase 0.5 before interpretation.

### Scientifically null result

Example:

```text
legal single-observation AUC approximately chance
```

with valid controls.

Result:

> One safe observation is insufficient.

This is a valid scientific outcome and does not kill the project.

### Valid Phase 1 entry

Controls pass and the remaining uncertainty is well characterized.

Result:

> Study whether another legal action reduces uncertainty.

### Project-level kill condition

Reserved for a later active-diagnosis experiment.

The deeper project would lose its motivation if, after shortcut removal and under valid safety constraints:

> informative diagnostic actions provide no meaningful information advantage over sham actions on the same worlds.

Phase 0.5 alone cannot establish this project-level kill condition.

---

## 27. Implementation order

No Phase 0.5 implementation code should be written before this plan is frozen.

After plan approval, implementation should proceed in small commits.

Tentative order:

1. Phase 0 diagnostic logging for rigid anomaly
2. world and pair identifiers
3. world-specific `k_contact`
4. world-specific `g_support`
5. analytic counterfactual labels
6. paired identification generator
7. explicit safe-cohort validation
8. pair-level train/test splitting
9. official controls
10. AUC + pair-bootstrap CI
11. Bayes reference
12. Bayes convergence tests
13. experiment-manifest freeze
14. official confirmatory run

Each commit should have one main reason.

---

## Working principle

Phase 0 verified that the pipeline can recover information deliberately placed into the simulator.

Phase 0.5 asks whether useful information survives after that direct shortcut is removed.

If the signal disappears, that is information.

If the signal survives, the burden is to show that it survives through legitimate observable physics rather than a new shortcut.

Only then should the project ask whether another physical action is worth taking.