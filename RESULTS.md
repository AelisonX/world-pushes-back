# Phase 0 Results

This document records the intended interpretation of the Phase 0 experiments in `world-pushes-back`.

It is not a paper and does not claim real-world robotic performance.

The purpose is to keep experimental results, assumptions, and falsification criteria separate from the project motivation.

---

## Phase 0 question

The current Phase 0 question is:

> How much information about an impending support-slip transition is available from safe tool-side interaction before significant slip occurs?

The competing transitions are:

- `MATERIAL_YIELD`
- `SUPPORT_SLIP`

The main concern is whether an agent can obtain useful predictive information before applying enough force to trigger the unsafe transition it is trying to avoid.

---

## Experiment 1: Basic pre-transition identifiability

Script:

```bash
python identifiability.py
```

### Purpose

Test whether a fixed low-force probe produces tool-side observations that contain enough information to predict which transition will occur first.

Observed channels:

- `Fx`
- `Fy`
- `tip_dx`
- `tip_dy`

### Interpretation

If balanced classification accuracy remains near chance:

```text
~0.50
```

then the current observation model contains little useful pre-transition information.

If accuracy rises meaningfully above chance, the observation model contains predictive structure that requires further auditing for possible leakage.

### Failure mode to watch

High accuracy is not automatically a positive result.

If mode information is directly encoded into the observation model, the classifier may succeed for trivial reasons.

---

## Experiment 2: Information-versus-safety force sweep

Script:

```bash
python force_sweep.py
```

### Purpose

Increase probe force and compare:

- pre-transition separability
- transition rate
- number of safe samples

### Main question

Does stronger probing reveal more information before it causes more threshold crossings?

### Important pattern

A result of the form:

```text
separability ~ chance
transition risk increases
```

would support the claim that stronger probing adds risk without revealing useful information in the minimal rigid model.

A result of the form:

```text
separability increases
transition risk also increases
```

would indicate an information-versus-safety tradeoff.

---

## Experiment 3: Rigid versus compliant support

Script:

```bash
python compliance_ablation.py
```

### Purpose

Compare two support models.

#### Rigid support

Idealized Coulomb support with no pre-slip displacement.

#### Compliant support

A simplified model in which small pre-slip displacement increases as loading approaches the support-slip threshold.

### Main question

Does identifiability appear only when the physical system exposes a measurable pre-transition precursor?

### Possible outcomes

#### Outcome A

Rigid support remains near chance while compliant support becomes more separable.

Interpretation:

> Pre-transition diagnosis depends on physical precursor signals that are absent from the ideal rigid model.

#### Outcome B

Both remain near chance.

Interpretation:

> The modeled precursor is too weak, too noisy, or insufficiently informative.

#### Outcome C

Compliant support becomes nearly perfectly separable at very low probe force.

Interpretation:

> The compliant model likely leaks the hidden threshold too directly and must be made less informative.

---

## Experiment 4: Sensor ablation

Script:

```bash
python sensor_ablation.py
```

### Purpose

Determine which observation channels carry predictive information.

Feature sets:

- force only
- motion only
- horizontal motion only
- full tool-side observation

### Main question

Is predictive information distributed across sensors, or is one channel doing nearly all the work?

### Interpretation

If:

```text
force_only ~ chance
motion_only > chance
tip_dx_only ~= full
```

then horizontal pre-slip motion is likely the dominant mechanism.

If:

```text
full > force_only
full > motion_only
```

then the channels may contain complementary information.

If a single channel produces near-perfect accuracy, the observation model should be audited for trivial leakage.

---

## Experiment 5: Precursor strength versus sensor noise

Script:

```bash
python sensing_threshold.py
```

### Purpose

Sweep:

- pre-slip displacement strength
- motion sensor noise

and measure classification accuracy.

### Main question

How strong must the physical precursor be relative to sensor uncertainty before the future transition becomes distinguishable?

### Expected qualitative structure

Low precursor and high noise should tend toward chance-level accuracy.

High precursor and low noise should tend toward higher accuracy.

The important result is not a single best cell.

The important result is the boundary between:

- effectively unobservable
- weakly identifiable
- reliably identifiable

---

## Experiment 6: Sensing heatmap

Script:

```bash
python plot_sensing_heatmap.py
```

Generated output:

```text
sensing_threshold_heatmap.png
```

### Purpose

Visualize the precursor-strength and sensor-noise sweep.

### Interpretation rule

The heatmap should be treated as a visualization of the toy model.

It must not be interpreted as a hardware requirement for real robots without external validation.

---

## Experiment 7: Precursor-to-noise scaling

Script:

```bash
python snr_collapse.py
```

Generated output:

```text
snr_collapse.png
```

### Purpose

Test whether different combinations of:

- pre-slip displacement
- sensor noise

approximately collapse onto a common trend when expressed as:

```text
precursor / noise
```

### Main question

Is identifiability primarily governed by a precursor-to-noise ratio?

### Interpretation

If conditions with similar ratios produce similar classification accuracy, the ratio may be a useful reduced description of the toy model.

If similar ratios produce very different accuracies, additional variables are controlling identifiability.

No universal scaling law should be claimed unless the collapse is actually supported by the results.

---

## Result recording

Script:

```bash
python record_phase0_results.py
```

Generated output:

```text
phase0_results.csv
```

The CSV currently records:

- support model
- probe force
- separability accuracy
- transition rate
- safe sample count

The CSV should be treated as the main machine-readable record for Phase 0 sweep results.

---

## Phase 0 kill criteria

### Kill criterion A: No useful pre-slip information

If no legal tool-side interaction produces meaningful pre-transition separability before significant slip occurs, the current sensing formulation fails.

The project should report this limitation rather than silently adding direct container-pose sensing.

### Kill criterion B: No safety benefit

In later phases, if active diagnosis improves classification but does not reduce unsafe force escalation relative to a push-harder baseline, the original safety motivation is not supported.

### Kill criterion C: Excessive diagnostic cost

If useful inference requires too many diagnostic actions or too much time to remain practical under the intended safety budget, the method should not be treated as useful.

---

## Claims currently allowed

At the current stage, the project may discuss:

- properties of the toy simulator
- whether the current observation model contains pre-transition information
- whether compliant precursor signals improve empirical separability
- how separability changes with probe force and sensor noise
- whether a precursor-to-noise ratio appears useful within the toy model

---

## Claims currently not allowed

The project must not yet claim that:

- real robots can predict support slip from tool-side sensing
- the model captures real ice-cream mechanics
- the identified sensing thresholds apply to real hardware
- active diagnosis improves real-world manipulation safety
- the project introduces a new theory of embodied intelligence
- the project establishes a new general benchmark
- a precursor-to-noise relationship is universal

---

## Current result status

Phase 0 implementation is active.

Numerical results should be added here only after scripts have been run and outputs have been checked for:

- class balance
- leakage
- reproducibility
- sensitivity to random seed
- consistency with the stated physical assumptions

Until then, this file records interpretation rules rather than conclusions.