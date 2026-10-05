# Phase 1B — Diagnostic Action Value: pre-result design freeze

Status: FROZEN_BEFORE_ACTION_VALUE_RUN. Date: 2026-10-05 (Australia/Brisbane).
Repository inspected at main / 4f6377c; clean working tree; no pre-existing Phase 1B implementation. Historical documents are not rewritten. PROJECT_STATUS is older than the Phase 1A findings; PHASE1A_FINDINGS and its frozen manifest govern the Phase 1A gate.

## Population and target

Inherit the nine A1–A9 actions, fixed 45-degree nominal target, epsilon_tie=0.10, rho<0.80, positive normal force, pre-transition and unclipped conditions from Phase 1A. Generate 5,000 fresh target-valid pairs for each of two independent repetitions with fresh frozen seeds. This rebuild is intentional confirmatory independence, not a change in sampling distribution or legality. Reuse Phase 1A generation and legality functions. A complete twin pair is admitted only if both worlds pass every action. Bank/training and held-out evaluation pairs are split 60/40 once; neither twins nor actions cross the boundary. Independent repetitions have separate namespace IDs and seeds.

The Bayes bank uses only the same universal population's training pairs, with one member of each class per pair. Unlike the Phase 1A per-action numerical screen, no action-specific bank or evaluation filtering is allowed. Report retention, per-action original-population rejections, universal safety and parameter/target-margin shifts; inherit Phase 1A cohort redesign gates. This is a balanced conditional identification experiment, not operational prevalence, online safety certification or policy value. Original-population rejection fractions are retained as the policy-risk view; no policy is implemented.

## Primary quantity: held-out predictive information lower bound in bits

For each observation, score g=1+log2(q(Y|O,a)); average twins to obtain a per-pair score. q is the Gaussian-likelihood posterior with equal class prior over the empirical universal training bank. The population expectation is I(Y;O|a) minus the expected conditional KL discrepancy between true and fitted posteriors. Thus it is a lower bound, equalling mutual information only for a correct posterior. A finite held-out estimate can be negative and must not be clipped. It is NOT claimed to be exact true-model EIG or a direct mutual-information measurement. This proper log score penalises confident mistakes, unlike simply reporting low posterior entropy.

Full-bank log-sum-exp evidence is used for primary scoring, avoiding sampler randomness in the ranking. All four legal channels remain in the Gaussian likelihood. Action-dependent force means are known constants, equal in both classes; their likelihood factors cancel, not a reward for force magnitude. No hidden state is supplied as an observation. Future labels are assigned once at 45 degrees, never from a probe result at another angle.

Secondary, fixed before execution: ROC-AUC, log loss (natural logarithms), Brier score, mean posterior entropy (bits). They cannot override the primary result. No learned classifier or model selection is required.

## Pairing, noise and uncertainty

Primary: one four-channel Gaussian noise vector per twin pair, shared across twins (as in Phase 0.5) and all counterfactual actions. Its marginal distribution has force SD 0.20 N, motion SD 0.0002 m. This common-random-number coupling reduces comparison variance without changing any action's marginal distribution. Robustness: independent action-specific vectors from the same frozen seed schedule, still shared by twins. Two new-seed repetitions assess world and bank variation. Noise SD multipliers 0.95 and 1.05 use correctly matched likelihoods, are robustness only, and never change the target or grid.

Resample complete held-out pairs jointly across all nine actions (2,000 bootstrap draws). For the 36 action contrasts, use a common 95th-percentile maximum absolute centred bootstrap deviation as a simultaneous two-sided familywise interval radius; report marginal percentile intervals for individual action values. These are conditional on the empirical training bank; they do not include all continuous-prior/model uncertainty. Repetitions and independent bank-half diagnostics address bank sensitivity separately.

## Meaningful difference and decision rules

Delta=0.02 bits, fixed before results: 2% of the initial one-bit uncertainty, a deliberately modest toy-study resolution, not a calibrated robotics utility threshold.

SUCCESS requires at least one identical oriented action contrast whose simultaneous interval lies beyond +0.02 (or -0.02) in BOTH independent repetitions, with all gates passing. It must retain sign and point difference >0.02 under independent-action noise, SD multipliers 0.95/1.05 and a common interior subset (every action rho<0.60 and transition margin>=0.20). Bank references remain the primary universal prior in the interior diagnostic; report this as a robustness score on an interior subset, not the subset's true MI. This conservatively rejects an advantage confined to boundary worlds. If any required sensitivity changes sign or fails the meaningful threshold, stop; do not switch winning contrast after observing robustness.

NEGATIVE requires all 36 simultaneous intervals to be contained within [-0.02,+0.02] in both repetitions and controls/gates pass: CURRENT ACTION GRID TOO SIMILAR TO DISTINGUISH MEANINGFULLY. Otherwise INCONCLUSIVE, not absence of information. No rescue by moving delta or grid. A tie in trivial contrasts is not required to preserve a noisy total ranking.

## Controls and numerical gates

- Same action: independently recompute A5 with identical inputs/noise. All posterior differences <=1e-12; primary difference zero.
- Label shuffle: independently random within-pair orientation swaps in bank and evaluation; one shuffled label per world across all actions. Rebuild class banks, do not merely score a true-label posterior against shuffled answers. Require maximum positive information and max action span <0.02 bits. Negative finite predictive scores are permitted.
- Rigid null: same admitted pairs and actions, support_model=rigid for bank and evaluation; max absolute primary value and span <0.02 bits. Since twins share stiffness, class bank distributions should agree.
- Noise control: independent-action schedule and +/-5% SD perturbations as above.
- World split: immutable action-wide assignment by pair_id; disjoint train/test pair IDs and full parameter signatures; twins and every action remain together.
- Bayes realized sampler audit: 50 evenly spaced held-out world rows/action, 5 repeats, 20,000 categorical draws/class, alpha=0.10, tau=0.0002. Reuse importance-corrected Phase 0.5b sampler. Every action/class: p05 ESS>=0.05 and median>=0.20. Posterior sampled-versus-full-bank p95 absolute error<=0.01, maximum<=0.05. These computational tolerances are stricter than the 0.02-bit resolution only when propagated to a mean; also report the primary-score errors on the audited rows.
- Full-bank stability: split training pairs into two disjoint halves, preserve class balance; score the same held-out rows with both. Each half versus full-bank mean primary difference <=0.005 bits; p95 absolute posterior difference <=0.02 and maximum<=0.10. This is a finite-bank sensitivity check, not proof that the continuous prior is integrated exactly.
- Gate cohort retention>=0.50; leave-one-out gain<0.20; transition rejection<0.20; inherited parameter SMD<0.50 and median target-margin relative shift<0.50. Interior subset must retain >=50% of held-out pairs.

Gates are evaluated before reporting action-value tables. Any failed numerical/control/cohort gate stops the run and writes a STOP artifact without an accepted ranking. No retries with changed criteria. Test fixtures must not use the frozen confirmatory seeds.

## Governance

Only local files. No push, no commit, no changes to Control Ledger or the closed 51st State candidate. Record manifest/source hashes, dependency versions, actual admitted IDs and split IDs, controls, per-pair scores and findings. README/PROJECT_STATUS stay unchanged until integrity is established; even then any proposed update remains local and explicitly awaiting human review. Stop for human review before commit/push.
