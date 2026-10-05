# Phase 1B Findings — LOCAL DRAFT, awaiting human review

Date: 2026-10-05 (Australia/Brisbane). Baseline: `main` / `4f6377c`. No commit or push performed. No new public claim is authorised.

## Result

**PHASE1B_DIAGNOSTIC_ACTION_DIFFERENCE**

The two frozen repetitions passed the specified cohort, numerical and control gates. Thirty-one of the 36 action contrasts had simultaneous intervals beyond the predeclared 0.02-bit meaningful-effect threshold in the same direction in both repetitions. Every such contrast retained its sign and a point difference above 0.02 bits under independent-action noise, +/-5% noise SD, and the frozen interior subset. This is not a declaration that all nine actions differ meaningfully.

Allowed local interpretation: under this frozen toy-model design and predictive information lower-bound estimator, some legal diagnostic actions reveal meaningfully different amounts of information about the fixed future-transition target. Exact continuous-prior mutual information, universal optimal action selection, policy benefit and real hardware are not established.

## Frozen design and provenance

`PHASE1B_PLAN.md` and `PHASE1B_MANIFEST.json` were saved and hashed before implementation and before inspecting rankings. Neither was changed after freezing. Their recorded hashes and all provenance-tracked source and experiment-output hashes are in `phase1b_provenance.json`. Existing Phase 0/0.5/0.5b/1A findings, README, PROJECT_STATUS and Control Ledger are unchanged. The closed 51st State candidate is not implemented.

The Phase 1A action set, 45-degree nominal target, near-tie rule and universal complete-pair legality logic were retained. Fresh datasets were regenerated prospectively for independent confirmation, as documented before running; no action-specific cohort was substituted. Unlike the Phase 1A numerical screen, the Bayes training banks also use universal legality for all nine actions.

The primary score is `mean(1 + log2 q(Y|O,action))` in bits. The primary score is a finite-sample estimate of a predictive-information lower-bound quantity; it is not an exact mutual-information estimate and the finite-sample point estimate is not itself guaranteed to be a strict lower bound. Negative scores are permitted. ROC-AUC, natural-log loss, Brier and posterior entropy are secondary only. Full empirical-bank evaluation determines primary scores; targeted samples are numerical audit only.

## Common cohorts and held-out split

| Repetition | Target-valid pairs | Universal pairs | Retention | Train pairs | Test pairs | Interior test retention |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 5000 | 4782 | 0.9564 | 2869 | 1913 | 0.8024 |
| R2 | 5000 | 4820 | 0.9640 | 2892 | 1928 | 0.8045 |

Train/test assignment is at complete-pair level, before evaluating actions. Both twins and all nine actions of a world stay in one split. Pair-ID and full hidden-parameter signature overlap checks passed. Primary inference has 3,826 held-out world rows in R1 and 3,856 in R2; rows are not treated as independent bootstrap units. Original-population per-action rejections, universal rho/margins and parameter shifts are in `phase1b_results.json`; they are not operational prevalence estimates.

## Primary per-action estimates

| Action | Force / angle | R1 bits (marginal 95% CI) | R2 bits (marginal 95% CI) |
|---|---|---|---|
| A1 | 3 N / 30 deg | 0.170040 [0.158936, 0.180922] | 0.172035 [0.161648, 0.181670] |
| A2 | 3 N / 45 deg | 0.134414 [0.125288, 0.143752] | 0.135845 [0.126878, 0.143870] |
| A3 | 3 N / 60 deg | 0.082687 [0.076352, 0.089255] | 0.083243 [0.077285, 0.088755] |
| A4 | 5 N / 30 deg | 0.252416 [0.238448, 0.265837] | 0.254905 [0.241676, 0.267164] |
| A5 | 5 N / 45 deg | 0.218499 [0.205203, 0.231080] | 0.221665 [0.209683, 0.233090] |
| A6 | 5 N / 60 deg | 0.157180 [0.146655, 0.167606] | 0.159658 [0.149890, 0.168774] |
| A7 | 7 N / 30 deg | 0.296713 [0.282383, 0.310783] | 0.297983 [0.283719, 0.311437] |
| A8 | 7 N / 45 deg | 0.269419 [0.255107, 0.283396] | 0.272791 [0.259289, 0.285316] |
| A9 | 7 N / 60 deg | 0.212594 [0.199580, 0.225094] | 0.217018 [0.205159, 0.228210] |

These individual CIs are marginal. Acceptance uses the simultaneous intervals over all 36 paired action contrasts, not overlap of these marginal intervals.

Example accepted contrast, A7 minus A1:
- R1: 0.126673 bits; simultaneous 95% CI [0.115416, 0.137930].
- R2: 0.125948 bits; simultaneous 95% CI [0.114540, 0.137356].

Both intervals exceed 0.02 bits. This example is reported from the already multiplicity-adjusted full contrast family; it does not justify declaring A7 universally optimal. Five contrasts did not meet the meaningful-effect acceptance rule:
- A1 vs A6
- A2 vs A6
- A4 vs A8
- A5 vs A9
- A7 vs A8

Failure to exceed the threshold in these five comparisons is not proof of equality. No force/angle, metric, threshold or sample population was changed to rescue the experiment.

## Controls

| Control | R1 | R2 | Frozen interpretation |
|---|---|---|---|
| Same-action A5 duplicate | 0.0 max log-posterior difference | 0.0 | PASS, identical inputs/noise recomputed |
| Label shuffle | span 0.000699 bits | span 0.000574 bits | PASS, below 0.02; class banks rebuilt after pairwise orientation shuffle |
| Rigid null | approximately 0 bits | approximately 0 bits | PASS, no compliant-motion advantage |
| Noise / interior checks | PASS | PASS | All 31 accepted contrasts retain sign and point effect >0.02 |
| World/pair split integrity | PASS | PASS | No action or twin crosses train/test |

The primary noise is shared across twins and actions; robustness uses independent action-specific vectors with identical channel variances, and separately +/-5% noise scales with matching likelihoods. Bootstrap resamples complete test pairs jointly across actions (2,000 draws). The 95th percentile of maximum centred contrast deviations defines a simultaneous interval radius. Robustness point estimates are not themselves separate confirmatory confidence intervals.

## Bayes numerical adequacy

| Numerical diagnostic (worst over actions/classes) | R1 | R2 | Frozen gate |
|---|---:|---:|---:|
| Realized ESS p05 | 0.353241 | 0.382511 | >=0.05 |
| Realized ESS median | 0.833049 | 0.819308 | >=0.20 |
| Sampled posterior p95 absolute error | 0.002440 | 0.002770 | <=0.01 |
| Sampled posterior maximum error | 0.005909 | 0.006426 | <=0.05 |
| Bank-half mean primary difference (bits) | 0.000390 | 0.000342 | <=0.005 |
| Bank-half posterior p95 error | 0.012031 | 0.011214 | <=0.02 |
| Bank-half posterior maximum error | 0.015144 | 0.015971 | <=0.10 |

All nine actions and both classes passed. The audit uses 50 held-out world rows per action, five sampler repetitions and 20,000 importance-corrected categorical draws per class, with alpha=0.10 and tau=0.0002. Exact full-bank posteriors are the computational reference. The class codes in raw JSON are 0=MATERIAL_YIELD, 1=SUPPORT_SLIP.

## Tests and artifact checks

- Full suite: `python -m pytest -q` — 136 passed (including 10 new Phase 1B tests).
- Saved-array integrity: regenerated per-pair scores from saved log posteriors match exactly; identity/split/labels and report means match saved artifacts.
- Frozen decision recomputation agrees with the stored status.
- Fresh confirmatory seeds are not used by the new unit fixtures.
- Local run only: no GitHub CI run, no commit, no push.

## Red-team boundaries and unresolved limits

- This is a simplified model whose compliant precursor explicitly changes with physical loading. Force terms cancel between classes in the likelihood, so force is not directly rewarded in the metric, but larger force can still change physical signal-to-noise. This result is conditional on that model and noise assumption.
- The reference target stays at 45 degrees; diagnostic angles change only observations and action legality.
- Hidden parameters are used only for simulation, target generation, cohort/safety audit and split-integrity checks, never as legal observation features.
- Balanced paired identification worlds are artificial and do not represent the operational prior. Universal legality is researcher-conditioned, not known by an online agent.
- High universal retention does not prove absence of population selection. All action comparisons use the same retained population; interior robustness retains about 80% of held-out pairs.
- Numerical adequacy is established for the frozen empirical bank and audited rows. It is not proof of exact integration over the continuous physical prior. Predictive lower-bound ranking could differ from true mutual-information ranking if posterior approximation errors are action-dependent.
- Bootstrap intervals condition on the training bank. Two independent repetitions and bank-half checks test sensitivity but are not a complete uncertainty interval over all possible banks or model choices.
- The interior robustness score uses the original universal-bank posterior, not a newly defined target or re-fitted interior prior; it is not the interior population's exact MI.
- Secondary classifier-style metrics do not establish intervention utility, safety improvement, policy benefit or a real-world robotics result.
- No model/grid/metric/meaningful-effect redesign was performed after looking at results.

## Human review stop

The local Phase 1B result passes the frozen acceptance rules. Review the metric choice, 0.02-bit rationale, conditional-prior limits and full controls before authorising a commit. README and PROJECT_STATUS have deliberately not been advanced to a new public claim. STOP: no commit or push until human review.

## Generated-artifact governance

The four R1/R2 worlds JSON and records NPZ files existed at experiment completion and were intentionally excluded from normal Git history. They remain local; exact-copy checksums, sizes and regeneration requirements are recorded separately in [PHASE1B_ARTIFACTS.json](PHASE1B_ARTIFACTS.json). The frozen experiment can be regenerated with the recorded source, seeds and dependency versions; see the metadata's historical text-byte profiles because Git newline conversion affects the raw-byte freeze checks.
