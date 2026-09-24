# Phase 0.5b Realized Sampler Audit Findings

## Status

**PASS**

The frozen tip-dx-targeted defensive proposal has now been tested using actual finite-sample categorical draws from the proposal distribution.

This audit closes the numerical gap identified after the original Phase 0.5b validation.

---

# 1. Why this audit was required

Phase 0.5 Attempt 1 failed its preregistered Bayes ESS criterion.

Subsequent diagnostics showed that:

- increasing uniform particle count did not repair the failure
- the ESS collapse was driven primarily by the informative `tip_dx` channel
- a defensive proposal targeted on `tip_dx` had much better theoretical ESS characteristics

Phase 0.5b then reproduced that theoretical improvement on independent data.

However, the Phase 0.5b implementation evaluated the complete empirical particle bank rather than actually sampling finite particle sets from the targeted proposal.

Therefore Phase 0.5b established:

- proposal-quality improvement
- algebraic importance-correction consistency

but did not yet establish:

- realized finite-sample importance-sampler performance

This audit tested that missing claim.

---

# 2. Frozen audit design

Proposal:

```text
alpha = 0.10
tau   = 0.0002 m
```

Target:

```text
uniform empirical class-conditional particle prior
```

Particle bank:

```text
20,000 total particles
10,000 per class
```

Primary realized sample budget:

```text
20,000 sampled particles per class
```

Sampler:

```text
Categorical(q)
with replacement
```

Replicates:

```text
10
```

Evaluation subset:

```text
200 held-out observations
```

Historical Phase 0.5 and Phase 0.5b artifacts were not modified.

---

# 3. Audit result

```text
PHASE0_5B_REALIZED_SAMPLER_AUDIT_PASS
```

All frozen primary numerical gates passed.

---

# 4. Targeted realized ESS

## MATERIAL_YIELD

```text
minimum = 0.417478
p05     = 0.690219
median  = 0.923471
mean    = 0.892319
p95     = 0.989408
maximum = 0.993804
```

## SUPPORT_SLIP

```text
minimum = 0.551536
p05     = 0.719086
median  = 0.919747
mean    = 0.882917
p95     = 0.974292
maximum = 0.984589
```

Frozen ESS criteria:

```text
p05    >= 0.05
median >= 0.20
```

Both classes passed comfortably.

---

# 5. Uniform finite-sample control

The realized uniform sampler reproduced the earlier numerical pathology.

## MATERIAL_YIELD

```text
minimum = 0.000417
p05     = 0.026867
median  = 0.680138
mean    = 0.591633
```

The MATERIAL_YIELD p05 ESS remained below the frozen threshold.

## SUPPORT_SLIP

```text
minimum = 0.026553
p05     = 0.179528
median  = 0.500817
mean    = 0.481485
```

This control supports the earlier diagnosis:

> the numerical problem was not fixed merely by performing finite sampling.

The targeted allocation materially changes sampling efficiency.

---

# 6. Posterior accuracy

The realized targeted sampler was compared against the exact full empirical-bank posterior reference.

Absolute posterior error:

```text
minimum = 0.000000008
p05     = 0.0000114
median  = 0.000311
mean    = 0.000473
p95     = 0.001582
maximum = 0.005215
```

Frozen tolerances:

```text
p95 absolute error <= 0.01
maximum absolute error <= 0.05
```

Both criteria passed.

---

# 7. Replicate stability

Posterior standard deviation across sampler replicates:

```text
minimum = 0.000000083
p05     = 0.0000608
median  = 0.000489
mean    = 0.000580
p95     = 0.001375
maximum = 0.002010
```

The targeted sampler therefore showed low replicate-to-replicate posterior variation at the frozen particle budget.

---

# 8. Sanity checks

The audit also verified:

```text
proposal normalization = PASS
importance identity     = PASS
targeted realized ESS   = PASS
posterior accuracy      = PASS
```

The importance-identity check remains an algebraic sanity test.

The realized sampler result comes from actual categorical sampling and is separate from that identity.

---

# 9. What is now supported

The project may now state:

> The frozen tip-dx-targeted defensive proposal was validated as a finite-sample importance sampler over the fixed empirical class-conditional particle prior used by the Bayes reference.

This statement is supported by:

- actual categorical sampling
- realized ESS
- independent sampler replicates
- comparison with a full-bank reference
- a finite-sample uniform control
- preregistered posterior-error tolerances

---

# 10. What remains limited

The result does not establish:

- exact integration over the continuous identification prior
- universal validity for arbitrary diagnostic actions
- validity for Phase 1 actions without per-action numerical checks
- real-world robot performance
- superiority over conventional robotic estimation or control methods

The finite particle bank remains an empirical approximation to the underlying identification prior.

---

# 11. Historical record

The following historical result remains unchanged:

```text
Phase 0.5 Attempt 1:
CONFIRMATORY_RUN_INVALID_BAYES_ESS
```

The later sampler repair does not retroactively validate that run.

Phase 0.5b remains part of the historical numerical-repair sequence.

This audit adds the previously missing realized finite-sample validation.

---

# 12. Phase 1 consequence

The targeted proposal can now be treated as a validated numerical starting point for future Bayes calculations over the same Phase 0.5-style empirical prior.

However, Phase 1 introduces new diagnostic actions.

Therefore:

> numerical adequacy must still be checked separately for every retained Phase 1 action.

The Phase 1A feasibility stage may now proceed with its preregistered per-action ESS gate.

---

# Current numerical chain

The numerical history is now:

```text
Phase 0.5 Attempt 1
        ↓
uniform ESS failure
        ↓
particle-count increase does not repair it
        ↓
tip_dx identified as concentration source
        ↓
targeted proposal designed
        ↓
independent theoretical proposal validation
        ↓
realized finite-sample sampler audit
        ↓
PASS
```

---

# Working conclusion

The original numerical warning was real.

The repair was not merely algebraic.

When particles are actually drawn from the targeted proposal, the sampler retains high realized ESS and closely reproduces the full empirical-bank posterior at the frozen particle budget.