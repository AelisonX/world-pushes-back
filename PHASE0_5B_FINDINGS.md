# Phase 0.5b Findings

## Status

**Bayes Repair Validation: PASS**

Phase 0.5b independently validated the preregistered targeted importance-sampling repair selected after the failed Phase 0.5 confirmatory attempt.

The original Phase 0.5 Attempt 1 remains invalid because its preregistered Bayes ESS criterion failed.

Phase 0.5b does not retroactively change that result.

Instead, it establishes that the Bayes numerical approximation can be repaired without changing:

- the physical model
- the transition labels
- the probe
- the safe cohort
- the noise model
- the legal observations
- the likelihood
- the class prior
- the ESS thresholds

---

## Background

The Phase 0.5 confirmatory experiment produced coherent classifier evidence:

```text
FULL           AUC = 0.7844
MOTION_ONLY    AUC = 0.7852
FORCE_ONLY     AUC = 0.5000
SHAM           AUC = 0.5000
LABEL_SHUFFLE  AUC = 0.5182
RIGID_NULL     AUC = 0.5000
RHO_CANARY     AUC = 0.9987
```

The predictive signal was therefore localized primarily to the compliant motion channel.

The corresponding Bayes reference produced:

```text
Bayes AUC = 0.7854
```

However, the confirmatory run failed the preregistered Bayes ESS criterion.

The failure was specifically concentrated in the lower tail of the `MATERIAL_YIELD` class.

---

## Phase 0.5 ESS failure

The frozen criterion required, for each class:

```text
p05 ESS fraction >= 0.05
median ESS fraction >= 0.20
```

Phase 0.5 Attempt 1 produced:

```text
SUPPORT_SLIP
p05    = 0.14582
median = 0.49909

MATERIAL_YIELD
p05    = 0.03265
median = 0.63266
```

The `MATERIAL_YIELD` p05 value therefore failed the frozen threshold.

The run was correctly recorded as:

```text
CONFIRMATORY_RUN_INVALID_BAYES_ESS
```

---

## Particle-count repair test

A numerical repair study tested whether simply increasing the number of Bayes particles would repair the ESS failure.

Results:

```text
Particles | MATERIAL_YIELD p05
------------------------------
20,000    | 0.03265
50,000    | 0.03233
100,000   | 0.03297
```

The ESS fraction remained essentially unchanged.

Interpretation:

> The failure was not caused by insufficient raw particle count.

Increasing computation alone did not repair the sampling inefficiency.

---

## Channel-ablation diagnosis

The Bayes likelihood was then decomposed by observation channel.

Results:

```text
FULL          yield_p05 = 0.03265   FAIL
FORCE_ONLY    yield_p05 = 1.00000   PASS
MOTION_ONLY   yield_p05 = 0.03265   FAIL
TIP_DX_ONLY   yield_p05 = 0.03244   FAIL
TIP_DY_ONLY   yield_p05 = 0.69259   PASS
```

Interpretation:

> The lower-tail ESS collapse was driven specifically by the horizontal tool-tip displacement channel, `tip_dx`.

This is the same channel that carries most of the predictive information in the compliant-support identification experiment.

The informative channel was therefore also the channel most difficult for the original uniform Monte Carlo proposal to approximate efficiently.

---

## Exploratory targeted proposal

An exploratory defensive-mixture proposal was evaluated using the already observed Phase 0.5 held-out set.

The selected proposal was:

```text
A0.10_T1.0
```

Meaning:

```text
10% uniform prior proposal
90% tip_dx-targeted proposal
targeting width = 1.0 * motion noise standard deviation
```

The exploratory diagnostic produced:

```text
SUPPORT_SLIP p05    = 0.68880
MATERIAL_YIELD p05  = 0.68798
```

Both comfortably exceeded the frozen threshold of:

```text
0.05
```

Because this proposal was selected after inspecting the original held-out set, those exploratory results were not treated as confirmatory evidence.

A new independent validation was therefore preregistered as Phase 0.5b.

---

## Phase 0.5b design

Phase 0.5b froze the targeted proposal before generating new validation data.

The validation used:

```text
new dataset seed
new split seed
new Bayes particle seed
same physical model
same likelihood
same ESS thresholds
same observation model
same safety rules
```

The selected proposal remained fixed at:

```text
alpha = 0.10
tau   = 0.0002 m
```

No proposal search was performed on the Phase 0.5b validation data.

---

## Phase 0.5b result

The independent validation passed.

```text
PHASE0_5B_VALIDATION_PASS
```

Bayes ROC-AUC:

```text
Uniform Bayes AUC  = 0.809232
Targeted Bayes AUC = 0.809232
```

The targeted proposal therefore changed numerical allocation without changing the inferred ranking performance.

---

## Uniform ESS on new independent data

The original uniform proposal again showed the same weakness.

```text
MATERIAL_YIELD
p05    = 0.02159
median = 0.65746

SUPPORT_SLIP
p05    = 0.17161
median = 0.49717
```

The uniform proposal again failed because:

```text
MATERIAL_YIELD p05 < 0.05
```

This independently reproduced the original numerical pathology.

---

## Targeted ESS on new independent data

The preregistered targeted proposal produced:

```text
MATERIAL_YIELD
p05    = 0.66721
median = 0.91756

SUPPORT_SLIP
p05    = 0.66467
median = 0.91359
```

Both classes passed the frozen criteria by a wide margin.

---

## Estimator consistency

The targeted proposal was importance-corrected back to the original scientific prior.

The resulting Bayes predictions were numerically consistent with the uniform estimator.

```text
Uniform AUC  = 0.809232
Targeted AUC = 0.809232
```

Maximum posterior difference:

```text
1.4432899320127035e-15
```

Interpretation:

> The targeted proposal improved Monte Carlo efficiency without materially changing the Bayes estimator.

This supports interpreting the repair as computational rather than scientific.

---

## Main conclusion

Phase 0.5b supports the following claim:

> The Bayes ESS failure in Phase 0.5 Attempt 1 was caused by inefficient uniform sampling around the informative `tip_dx` channel, not by insufficient particle count or by disagreement between the learned classifier and the underlying Bayes estimator.

A preregistered tip-dx-targeted defensive-mixture proposal repaired the ESS failure on new independent data while preserving the original posterior estimates.

---

## What Phase 0.5b does not establish

Phase 0.5b does not establish that:

- the toy physical model is realistic
- the simulated compliance law occurs on real hardware
- real robots can predict support slip using this method
- the targeted proposal is universally optimal
- `tip_dx` will be the dominant channel in richer contact models
- active probing improves task safety
- Phase 0.5 Attempt 1 should be retroactively considered valid

The original failed confirmatory attempt remains part of the record.

---

## Current scientific interpretation

The combined Phase 0.5 and Phase 0.5b evidence supports three separate conclusions.

### 1. Conditional identifiability exists in the toy model

Under compliant support, one safe observation contains meaningful information about which physical transition will occur first.

### 2. The information is primarily carried by horizontal pre-slip motion

Force-only observations remain at chance.

Removing support compliance also removes the predictive signal.

### 3. Numerical inference must respect the geometry of the informative observation channel

Uniform Monte Carlo sampling can become inefficient even when the estimator itself is valid.

Targeted importance sampling can recover adequate effective sample size without changing the underlying scientific prior.

---

## Phase 1 implication

Phase 0.5b removes the Bayes numerical-adequacy blocker.

The project may now proceed to Phase 1 under a limited interpretation.

Phase 1 should ask:

> Which legal physical action provides the most useful information about the future transition under a fixed safety budget?

Phase 1 must not assume that stronger probing is automatically better.

The next problem is therefore not merely classification.

It is:

**Diagnostic Action Value**

---

## Working conclusion

The world may expose useful pre-transition structure.

The useful structure may be subtle.

And even when the information exists, an inference method can still fail if it spends its computational budget in the wrong parts of the latent state space.

Phase 0.5b therefore adds a second lesson to the original project principle:

> The world must expose information.

and:

> The estimator must know where to look for it.