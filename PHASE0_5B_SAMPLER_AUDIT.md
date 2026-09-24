# Phase 0.5b Sampler Audit

## Title

**Phase 0.5b — Realized Importance-Sampler Audit**

---

## Status

**AUDIT PLANNING**

This audit does not invalidate or replace the historical Phase 0.5 or Phase 0.5b records.

It clarifies what Phase 0.5b actually established and defines the additional numerical test required before reusing the targeted proposal as a validated finite-sample sampler.

---

# 1. Historical context

Phase 0.5 Attempt 1 was invalid under its preregistered Bayes ESS criterion.

The failure was preserved.

Subsequent diagnostics found:

1. increasing the uniform particle count did not repair the ESS failure
2. ESS concentration was driven primarily by the informative `tip_dx` channel
3. a defensive proposal targeted on `tip_dx` strongly improved the theoretical ESS fraction

The selected exploratory proposal was:

```text
alpha = 0.10
tau = 0.0002 m
```

Phase 0.5b then evaluated that frozen proposal on new independent data.

---

# 2. What Phase 0.5b validly established

Phase 0.5b established that, on an independent dataset:

- the original uniform proposal reproduced the previously observed ESS pathology
- the frozen targeted proposal had substantially better theoretical ESS characteristics
- the importance-correction algebra preserved the same finite-bank evidence target

These remain valid findings.

---

# 3. Implementation audit finding

The Phase 0.5b implementation evaluated the targeted proposal over the complete empirical particle bank.

For each particle:

```text
target = p
proposal = q
importance factor = p / q
```

The evidence calculation used:

```text
sum over full bank:
    q * likelihood * p / q
```

Therefore:

```text
q cancels algebraically
```

and the result reduces exactly to the empirical-prior evidence over the complete bank.

This explains why the targeted and uniform posterior estimates matched to numerical precision.

---

# 4. Interpretation correction

The Phase 0.5b posterior-consistency result should therefore be interpreted as:

> an algebraic consistency check of importance correction over the full empirical particle bank

rather than:

> validation of realized finite-sample importance sampling from q

The previous posterior equality is not evidence that a finite sample drawn from the targeted proposal has low Monte Carlo error.

---

# 5. ESS interpretation correction

The targeted ESS quantity used in the exploratory proposal study and Phase 0.5b was an expected or asymptotic ESS fraction computed over the complete empirical proposal distribution.

It was not the realized ESS obtained from a finite random draw of targeted particles.

Therefore:

```text
high theoretical ESS
```

does not yet prove:

```text
high realized ESS in a finite targeted sample
```

---

# 6. Empirical-bank target clarification

The current targeted proposal is defined over a finite class-conditional empirical particle bank.

Strictly interpreted, the proposal targets:

```text
the empirical discrete approximation to the identification prior
```

not the continuous original identification prior exactly.

The empirical particle bank remains a Monte Carlo approximation to that original prior.

This distinction must be explicit in future documentation.

---

# 7. What this audit does NOT invalidate

This audit does not invalidate:

- the Phase 0.5 classifier results
- the control results
- the finding that compliant motion carries the predictive signal
- the rigid-null result
- the diagnosis that `tip_dx` drives the ESS concentration
- the finding that increasing uniform particle count did not solve the ESS problem
- the observation that the targeted proposal has much better theoretical ESS characteristics on new data

It changes the strength of the numerical-validation claim only.

---

# 8. Corrected numerical claim

The strongest currently supported statement is:

> The frozen tip-dx-targeted defensive proposal reproduced improved theoretical ESS characteristics on new independent data while preserving the same finite-bank evidence target algebraically.

The project should not yet claim:

> The realized finite-sample targeted importance sampler has been independently validated.

That requires an additional audit.

---

# 9. Audit research question

The sampler audit asks:

> When particles are actually drawn from the frozen targeted proposal q, does finite-sample importance sampling achieve adequate realized ESS and stable posterior estimation relative to the full-bank reference?

---

# 10. Frozen proposal under audit

The audit must use the existing frozen proposal:

```text
alpha = 0.10
tau = 0.0002 m
```

No proposal tuning is permitted.

The audit exists to test the proposal already selected.

---

# 11. Frozen target distribution

Within each class, the target distribution is:

```text
uniform over the fixed empirical particle bank
```

The audit therefore tests finite-sample importance sampling for the empirical approximation used by Phase 0.5b.

It does not claim exact continuous-prior integration.

---

# 12. Proposal distribution

For each observation and class:

```text
q_i
=
alpha * p_i
+
(1 - alpha) * k_i
```

where:

```text
p_i = 1 / N
```

and:

```text
k_i
```

is the normalized Gaussian similarity kernel over particle `tip_dx` means:

```text
exp(
    -0.5
    * ((particle_tip_dx - observed_tip_dx) / tau)^2
)
```

---

# 13. Realized sampling requirement

The audit must actually draw particle indices from:

```text
Categorical(q)
```

with replacement.

The finite sample must not evaluate every particle in the bank.

This is the key distinction from Phase 0.5b.

---

# 14. Importance weights

For each sampled particle:

```text
w_i
=
likelihood_i
*
p_i
/
q_i
```

The evidence estimator is:

```text
mean(w_i)
```

up to the same likelihood scaling convention used for numerical stability.

---

# 15. Realized ESS

For each observation and class:

```text
ESS
=
(sum w)^2
/
sum(w^2)
```

Normalized ESS fraction:

```text
ESS_fraction
=
ESS / n_samples
```

This realized quantity is the primary numerical diagnostic.

---

# 16. Reference posterior

The reference posterior must be computed using the full empirical particle bank.

This reference is deterministic conditional on:

- the particle bank
- the observation
- the frozen noise model

The sampled estimator is compared against this full-bank empirical reference.

---

# 17. Fresh randomness

The sampler audit must use fresh random seeds not used in:

- Phase 0.5 Attempt 1
- particle-count diagnostics
- channel ablation
- proposal selection
- Phase 0.5b validation
- Phase 1A

The sampler RNG must be separate from the particle-bank RNG.

---

# 18. Fixed particle bank

The audit should use one fixed empirical particle bank per run.

Sampling variability should come from proposal draws, not from silently regenerating the prior bank for every observation.

This isolates sampler performance from prior-bank variation.

---

# 19. Sample size

The primary audit should use:

```text
20,000 sampled particles
```

to match the historical Bayes particle budget.

If computationally practical, secondary diagnostics may include:

```text
5,000
10,000
20,000
50,000
```

These are convergence diagnostics only.

No sample-size choice may be tuned based on desired pass/fail outcome.

---

# 20. Replication

A single realized draw is insufficient for sampler validation.

The audit should repeat targeted sampling with multiple independent sampler seeds.

Recommended minimum:

```text
10 replicates
```

per evaluated observation or per evaluation subset.

If full per-observation replication is computationally expensive, the audit may use a frozen representative subset.

Any subset must be selected before sampler results are inspected.

---

# 21. Primary outputs

For each class report:

```text
realized ESS fraction p05
realized ESS fraction median
realized ESS fraction minimum
```

Across observations report:

```text
absolute posterior error p50
absolute posterior error p95
absolute posterior error maximum
```

Across sampler replicates report:

```text
posterior standard deviation
evidence relative error
```

---

# 22. ESS criterion

Retain the historical adequacy thresholds:

```text
p05 ESS fraction >= 0.05
median ESS fraction >= 0.20
```

These thresholds are evaluated on realized finite-sample ESS.

---

# 23. Posterior-accuracy criterion

The sampler audit must define a finite-sample posterior tolerance before execution.

A candidate tolerance may be:

```text
p95 absolute posterior error <= 0.01
```

and:

```text
maximum absolute posterior error <= 0.05
```

These values are not yet frozen by this document unless explicitly adopted in an audit manifest.

The tolerance must be frozen before audit execution.

---

# 24. Evidence-bias diagnostic

For each class and observation, compare:

```text
sampled evidence estimate
```

against:

```text
full-bank empirical evidence
```

Report signed and absolute relative error where numerically stable.

A systematic signed error indicates estimator or implementation failure.

---

# 25. Required controls

## Uniform-sampling control

Actually sample uniformly from the same empirical bank.

This verifies that realized targeted sampling improves numerical efficiency relative to realized uniform sampling.

## Exact-full-bank reference

Evaluate the complete empirical bank.

This is the numerical reference, not a sampled method.

## Proposal-normalization check

Assert:

```text
sum(q) = 1
```

within numerical tolerance.

## Importance-identity check

Verify numerically that:

```text
E_q[p * likelihood / q]
```

matches the full-bank empirical target when evaluated exactly.

This remains an algebra sanity check, not the sampler result.

---

# 26. Failure conditions

The targeted sampler fails validation if any of the following occur:

1. realized ESS fails the historical p05 or median thresholds
2. posterior error exceeds the frozen tolerance
3. evidence estimates show systematic bias
4. proposal normalization fails
5. sampled results depend pathologically on sampler seed
6. targeted sampling is not materially better than uniform sampling under the same finite-sample budget

---

# 27. Interpretation if the audit passes

If the audit passes, the project may claim:

> The frozen tip-dx-targeted proposal was validated as a finite-sample importance sampler over the empirical class-conditional particle prior used by the Bayes reference.

This still does not mean:

> exact integration over the continuous original identification prior

unless that is separately demonstrated.

---

# 28. Interpretation if the audit fails

If the audit fails:

- Phase 0.5 historical scientific results remain unchanged
- Phase 0.5b theoretical proposal diagnosis remains informative
- the targeted sampler must not be reused as a validated Phase 1 numerical method
- a new proposal or sampling strategy would require a separate preregistered numerical study

No historical result should be rewritten to pretend the failure did not occur.

---

# 29. Phase 1 implication

Phase 1A may proceed with non-Bayes cohort-feasibility diagnostics independently.

However:

> per-action Bayes ESS gating should not rely on the Phase 0.5b targeted proposal as a validated realized sampler until this audit is complete.

The numerical audit should therefore be completed before any Phase 1B action-value comparison that depends on Bayes posterior estimates.

---

# 30. Historical record rule

Do not modify old result artifacts to erase the previous interpretation.

Instead:

- preserve original artifacts
- add this audit
- update current documentation prospectively
- state clearly what was learned and what remains unresolved

---

# Working conclusion

Phase 0.5b showed that the targeted proposal looks numerically promising and is algebraically consistent.

This audit asks the missing question:

> Does it still work when we actually draw the particles instead of summing over all of them?