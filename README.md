# world-pushes-back

A small experimental sandbox for testing whether tool-side interaction can reveal impending physical constraint transitions under a safety budget.

## Motivating scenario

### The Ice Cream Problem

Imagine a robot trying to scoop very hard ice cream from a movable container.

A naive response to resistance is:

`blocked -> increase force`

But increasing force does not guarantee that the scoop will penetrate the material.

The applied load may instead exceed the support friction of the container, causing the entire container to slide.

The motivating safety problem is therefore:

> Blind force escalation can move the environment instead of completing the intended manipulation.

This repository does not attempt to model realistic ice-cream rheology.

The ice-cream scenario is a simple physical story for studying competing thresholds during contact.

---

## Phase 0 research question

Before building an agent or a policy, we ask a simpler question:

> How much information about an impending support-slip transition is available from safe tool-side interaction before significant slip occurs?

This is an identifiability question.

The project does not assume that the answer is positive.

If the minimal physical model makes the transition impossible to infer before it occurs, that is itself a useful Phase 0 result.

---

## Physical competition

The first version studies two competing physical thresholds:

1. material yield
2. support slip

As applied load increases, one threshold is reached first.

Conceptually:

`CONTACT_BLOCKED`

then either:

`MATERIAL_YIELD`

or:

`SUPPORT_SLIP`

If material yield occurs first, the tool begins to penetrate the material.

If support slip occurs first, the container begins to move.

The central question is whether tool-side observations contain enough information to estimate which transition is approaching before unsafe force escalation occurs.

---

## Contact-mode state machine

Version 0.1 uses three contact-mode states:

- `CONTACT_BLOCKED`
- `MATERIAL_YIELD`
- `SUPPORT_SLIP`

These are states, not fixed episode labels.

A single episode may transition from:

`CONTACT_BLOCKED -> MATERIAL_YIELD`

or:

`CONTACT_BLOCKED -> SUPPORT_SLIP`

The model therefore treats contact mode as a state variable rather than a permanent cause class.

---

## Hidden physical parameters

The current minimal parameter set includes:

- material yield strength
- contact area
- container mass
- static friction coefficient
- kinetic friction coefficient
- applied force
- force direction

These parameters determine the thresholds at which material yield or support slip occurs.

Tool stiffness may be added later if required, but it is not part of the first minimal model.

---

## Observation space

The default condition provides tool-side observations only.

Candidate observations include:

- `Fx`
- `Fy`
- incremental tool-tip motion
- sensor noise

The reference frame for tool-tip motion must be explicitly defined in the simulator.

Direct container pose or velocity is not available in the default condition.

This is intentional.

The project asks whether useful information about support stability can be extracted without directly observing the container.

---

## Action space

Phase 0 does not use a generic `PROBE` action.

All diagnostic interactions must be ordinary physical actions.

Initial candidate actions include:

- `LOW_FORCE_PUSH`
- `LATERAL_NUDGE`
- `UNLOAD_AND_HOLD`
- `STOP`

The purpose of an action may be either:

- task progress
- information gathering
- risk reduction

A diagnostic action is therefore an intervention on the physical system, not a direct query for the hidden state.

---

## Why identifiability comes first

A policy cannot reliably infer a hidden physical condition if the observation model contains no information that distinguishes it.

Before building an active diagnosis policy, this repository first asks:

> Do different threshold regimes actually produce distinguishable tool-side observation sequences under legal actions?

If they do not, no downstream policy should be credited for solving the problem.

This is the Phase 0 gate.

---

## Phase 0 experiment

For each legal action:

1. sample physical parameters from predefined ranges
2. simulate tool-side observation sequences
3. separate episodes by resulting transition
4. train a simple classifier on the observation sequences
5. evaluate on held-out episodes
6. report empirical separability

The initial metric is:

**cross-validated pairwise classification accuracy**

This is deliberately simple.

The first goal is to determine whether discriminative information exists at all.

Information-theoretic metrics can be added later if needed.

---

## Required experimental assumptions

Any identifiability result depends on the experimental setup.

The following must therefore be stated explicitly:

- observation horizon
- control timestep
- force limits
- parameter sampling ranges
- sensor noise model
- action magnitudes
- initial conditions
- coordinate reference frames
- container-pose visibility
- safety budget

Without these assumptions, an identifiability result is not interpretable.

---

## Safety budget

The project is not only interested in classification accuracy.

Diagnostic interaction itself can be risky.

A useful method must operate under measurable limits such as:

- maximum applied force
- maximum container displacement
- maximum number of diagnostic actions
- optional keep-out-zone constraints

The project therefore treats information gathering as a constrained physical process.

---

## Core tradeoff

In an idealized Coulomb-friction system, a container below the static-friction threshold may reveal very little about how close it is to slipping.

For example, a small safe push may produce the same observable response for:

- a highly stable container
- a container very close to the slip threshold

This creates a possible tension:

> More informative probing may require approaching the same physical threshold that safe behavior is trying to avoid.

This information-versus-safety tradeoff is a central Phase 0 question.

---

## Pre-slip compliance ablation

The project compares two simplified support models.

### Rigid support

Idealized Coulomb support with no pre-slip support motion.

Before the static-friction threshold is crossed, the support does not reveal how close it is to slipping.

### Compliant support

A simplified model in which small pre-slip displacement increases as loading approaches the support-slip threshold.

This model is not intended as a complete description of real support mechanics.

It is an explicit ablation used to ask:

> If the physical world produces a measurable precursor before a transition, how strong must that precursor be relative to sensor noise before the transition becomes identifiable?

---

## Sensor ablation

The project also tests which observation channels contain predictive information.

Compared feature sets include:

- force only
- motion only
- horizontal tool-tip motion only
- full tool-side observation

This helps distinguish genuine multi-channel information from cases where one sensor channel trivially reveals the answer.

---

## Minimum sensing requirement

A dedicated sweep varies:

- pre-slip displacement strength
- motion sensor noise

The goal is to estimate the sensing regime in which pre-transition classification becomes feasible.

A precursor that is much smaller than sensor noise may be effectively unobservable.

A precursor that is much larger than sensor noise may make the transition easy to identify.

The project therefore studies the ratio:

`physical precursor / sensor noise`

as a possible governing quantity.

---

## Kill criteria

### Kill criterion A: No useful pre-slip information

If no legal tool-side action can distinguish an impending support-slip transition meaningfully above chance before significant container motion occurs, stop and report that limitation.

Do not silently fix the problem by adding direct container-pose sensing.

### Kill criterion B: No safety benefit

If later active diagnosis improves transition inference but does not reduce unsafe force escalation compared with a push-harder baseline, the original safety motivation is not supported.

### Kill criterion C: Excessive diagnostic cost

If useful inference requires so many diagnostic actions that the interaction becomes impractically slow or violates the safety budget, the method is not useful under the intended setting.

---

## Planned phases

### Phase 0: Identifiability

Can impending threshold transitions be distinguished from tool-side interaction at all?

### Phase 1: Diagnostic action value

Which legal physical actions provide the most useful information?

### Phase 2: Active diagnosis

Can an agent choose informative actions under a safety and action budget?

### Phase 3: Safety consequence

Does active diagnosis reduce unsafe force escalation compared with a naive push-harder baseline?

---

## Related work positioning

This repository does not claim novelty for:

- feedback control
- command-observation mismatch
- forward models
- Kalman innovation
- disturbance observers
- fault detection and isolation
- interactive perception
- system identification
- contact detection
- model-based control

The narrower goal is to build an inspectable toy system for studying whether safe physical interaction can reveal competing contact-threshold transitions under limited sensing.

---

## Running the project

### 1. Install dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Run regression tests

```bash
pytest -q
```

The tests check properties such as:

- higher static friction increases the support-slip threshold
- a heavier container increases the support-slip threshold
- stronger material increases the material-yield threshold
- larger contact area increases the material-yield threshold
- rigid support produces no artificial pre-slip support motion
- compliant pre-slip motion grows as the slip threshold is approached

### 3. Run the basic identifiability test

```bash
python identifiability.py
```

This tests whether safe pre-transition tool-side observations contain enough information to predict which threshold will be reached first:

- `MATERIAL_YIELD`
- `SUPPORT_SLIP`

### 4. Run the information-versus-safety force sweep

```bash
python force_sweep.py
```

This sweeps probe force and reports:

- pre-transition separability
- transition rate
- number of safe samples

The purpose is to test whether stronger probing produces useful information before increasing transition risk.

### 5. Compare rigid and compliant support models

```bash
python compliance_ablation.py
```

This compares:

- ideal rigid Coulomb support
- simplified compliant pre-slip support

The experiment asks whether measurable physical precursors are required for pre-transition identifiability.

### 6. Run the sensor ablation

```bash
python sensor_ablation.py
```

This compares:

- force only
- motion only
- horizontal tool-tip motion only
- full tool-side observation

The purpose is to identify which sensor channels actually carry predictive information.

### 7. Sweep precursor strength versus sensor noise

```bash
python sensing_threshold.py
```

This tests how classification accuracy changes as:

- pre-slip displacement strength changes
- motion sensor noise changes

### 8. Generate the sensing heatmap

```bash
python plot_sensing_heatmap.py
```

This creates:

```text
sensing_threshold_heatmap.png
```

The heatmap visualizes classification accuracy across precursor-strength and sensor-noise conditions.

### 9. Test precursor-to-noise scaling

```bash
python snr_collapse.py
```

This creates:

```text
snr_collapse.png
```

The experiment tests whether identifiability approximately follows the ratio:

`pre-slip precursor / sensor noise`

### 10. Record Phase 0 results

```bash
python record_phase0_results.py
```

This creates:

```text
phase0_results.csv
```

The CSV stores:

- support model
- probe force
- separability accuracy
- transition rate
- safe sample count

---

## Reproducibility notes

Current experiments use fixed random seeds where practical.

The simulator is intentionally simple so that:

- physical assumptions remain inspectable
- hidden state and agent-visible observations remain separate
- experimental results can be traced to explicit modeling choices

Generated results should be treated as properties of the current toy model, not as claims about real robotic hardware or real ice-cream mechanics.

---

## Deliberate exclusions

Version 0.1 does not include:

- geometric jam modes
- rotational degrees of freedom
- multi-point contact
- realistic ice-cream rheology
- temperature-dependent material models
- ROS
- MuJoCo
- Isaac Sim
- Gazebo
- reinforcement learning
- LLM agents
- active-inference branding
- synthetic digestive systems

These may be reconsidered only if the Phase 0 model demonstrates that the core problem is worth extending.

---

## Current status

Design-freeze Phase 0 implementation.

No benchmark claim yet.

No policy claim yet.

No claim of general embodied intelligence.

No claim that the hidden physical transition is identifiable in advance.

The current implementation first tests whether the problem itself is solvable under the stated sensing and safety assumptions.

---

## Working principle

> The world can reject a command in different physical ways.

The first question is not how an agent should respond.

The first question is whether the difference can be observed safely.