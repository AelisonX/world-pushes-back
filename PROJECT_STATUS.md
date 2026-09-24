# Project Status

## Project

`world-pushes-back`

## Current phase

**Phase 0 — Identifiability**

The project is currently testing whether impending physical threshold transitions can be inferred from safe tool-side interaction before significant support slip occurs.

The project is not yet testing an intelligent policy.

The current question comes first:

> Is there enough information in the physical interaction to justify building a policy at all?

---

## Core motivating scenario

### The Ice Cream Problem

A robot attempts to scoop hard material from a movable container.

Blind force escalation may cause:

`CONTACT_BLOCKED -> SUPPORT_SLIP`

instead of:

`CONTACT_BLOCKED -> MATERIAL_YIELD`

The safety motivation is simple:

> Increasing force can move the environment instead of completing the intended task.

---

## Current physical model

The minimal model studies competition between two thresholds:

- material yield
- support slip

Current contact states:

- `CONTACT_BLOCKED`
- `MATERIAL_YIELD`
- `SUPPORT_SLIP`

Current hidden physical parameters include:

- material yield strength
- contact area
- container mass
- static friction
- kinetic friction
- applied force
- force direction

---

## Current observation model

Default tool-side observations:

- `Fx`
- `Fy`
- `tip_dx`
- `tip_dy`

Direct container pose is intentionally hidden.

Sensor noise is included.

---

## Current support models

### Rigid

Idealized Coulomb support with no pre-slip displacement.

### Compliant

A simplified support model that produces small pre-slip motion as loading approaches the support-slip threshold.

The compliant model is used as an ablation, not as a claim about real hardware.

---

## Current research questions

### Q1

Can a safe low-force probe predict whether:

- `MATERIAL_YIELD`
- or `SUPPORT_SLIP`

will occur first?

### Q2

Does increasing probe force reveal more information before it increases threshold-crossing risk?

### Q3

Does pre-slip support compliance create useful predictive information that is absent in an ideal rigid model?

### Q4

Which sensor channels carry that information?

### Q5

How strong must the physical precursor be relative to sensor noise before pre-transition diagnosis becomes feasible?

### Q6

Are the conclusions stable across:

- random seeds
- physical parameter ranges

---

## Completed components

### Core simulation

- [x] `physics.py`
- [x] competing material-yield and support-slip thresholds
- [x] safe pre-transition probe model
- [x] rigid support model
- [x] compliant pre-slip support model
- [x] noisy tool-side observations

### Phase 0 experiments

- [x] `identifiability.py`
- [x] `force_sweep.py`
- [x] `compliance_ablation.py`
- [x] `sensor_ablation.py`
- [x] `sensing_threshold.py`
- [x] `plot_sensing_heatmap.py`
- [x] `snr_collapse.py`
- [x] `seed_robustness.py`
- [x] `parameter_sensitivity.py`

### Results and reproducibility

- [x] `record_phase0_results.py`
- [x] `phase0_results.csv` generation
- [x] `RESULTS.md`
- [x] `requirements.txt`
- [x] `test_physics.py`
- [x] experiment manifest
- [x] manifest validator
- [x] one-command Phase 0 runner

### Documentation

- [x] README research framing
- [x] kill criteria
- [x] reproducibility instructions
- [x] related-work positioning
- [x] deliberate scope exclusions

---

## Current experiment IDs

The machine-readable experiment registry is stored in:

`experiments.json`

Current experiment sequence:

- `E0` — basic pre-transition identifiability
- `E1` — information-versus-safety force sweep
- `E2` — rigid-versus-compliant support
- `E3` — sensor ablation
- `E4` — minimum sensing requirement
- `E5` — sensing heatmap
- `E6` — precursor-to-noise scaling
- `E7` — Phase 0 results recording
- `E8` — seed robustness
- `E9` — parameter sensitivity

---

## Regression guards

Current tests protect assumptions including:

- increasing static friction raises the support-slip threshold
- increasing container mass raises the support-slip threshold
- increasing material strength raises the material-yield threshold
- increasing contact area raises the material-yield threshold
- rigid support does not generate artificial pre-slip support motion
- compliant support motion increases near the slip threshold
- pre-slip support motion remains bounded

Run:

```bash
pytest -q
```

---

## Full Phase 0 run

The current Phase 0 pipeline can be executed with:

```bash
python run_phase0.py
```

The runner:

1. validates the experiment manifest
2. runs regression tests
3. runs the main Phase 0 experiments
4. generates visualizations
5. checks robustness
6. records final Phase 0 results
7. stops if a step fails

---

## Kill criteria

### A — No useful pre-slip information

If no legal tool-side interaction produces meaningful predictive information before significant slip occurs, stop or reformulate.

Do not silently solve the problem by exposing direct container pose.

### B — No safety benefit

If future active diagnosis improves classification but does not reduce unsafe force escalation relative to a push-harder baseline, the original safety motivation is unsupported.

### C — Excessive diagnostic cost

If useful inference requires too many actions, too much time, or too much physical risk, the method is not useful under the intended safety budget.

---

## Current claims allowed

The project may currently discuss:

- behavior of the toy simulator
- empirical pre-transition separability
- rigid versus compliant support differences
- sensor-channel contribution
- precursor-versus-noise effects
- seed robustness
- parameter-range sensitivity

---

## Current claims not allowed

The project must not currently claim:

- validated real-world robotic performance
- realistic ice-cream mechanics
- universal sensing thresholds
- a new theory of embodied intelligence
- a new general robotics benchmark
- demonstrated safety improvement from active diagnosis
- real-hardware support-slip prediction

---

## Deliberate exclusions

Currently out of scope:

- geometric jam
- rotational degrees of freedom
- multi-point contact
- realistic material rheology
- temperature-dependent material models
- ROS
- MuJoCo
- Isaac Sim
- Gazebo
- reinforcement learning
- LLM agents
- active-inference branding
- synthetic digestive systems

These should not be added merely because they are interesting.

They require a reason generated by Phase 0 results.

---

## Next milestone

### Phase 0 result consolidation

Before moving to Phase 1:

1. run the full Phase 0 pipeline
2. inspect generated results
3. verify robustness
4. update `RESULTS.md` with actual numbers
5. decide whether the Identifiability Gate passes

Only if the gate passes should the project move to:

**Phase 1 — Diagnostic Action Value**

Phase 1 will ask:

> Which legal physical action provides the most useful information under a fixed safety budget?

---

## Current status summary

**Research framing:** frozen for Phase 0  
**Physics model:** implemented  
**Identifiability experiments:** implemented  
**Robustness checks:** implemented  
**Result recording:** implemented  
**Policy development:** not started  
**Benchmark claim:** not allowed yet  
**Phase 1:** locked until Phase 0 results justify continuation