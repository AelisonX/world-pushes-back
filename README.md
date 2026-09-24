# world-pushes-back

A small experimental benchmark for physical mismatch attribution under partial observability.

An agent may observe that an action did not produce the expected result. The harder problem is determining why.

> An error signal tells you something went wrong. Attribution tells you what.

## Research question

When several hidden physical causes can produce similar tool-side observations, can an agent actively probe the environment, identify the likely cause, and choose a safe corrective action?

## Example 001: The Ice Cream Problem

Imagine a robot trying to scoop very hard ice cream from a movable freezer.

From the robot's tool-side sensors, several different physical situations may look similar:

- the material is too hard
- the scoop angle is poor
- the scoop hits the container wall
- the container is sliding
- the tool is bending

In all of these cases, the robot may initially observe:

- high reaction force
- little scoop displacement

A naive policy might respond with:

`blocked -> push harder`

That can make the situation worse.

The purpose of this benchmark is not to model realistic ice-cream scooping in full detail.

It is to test whether an agent can distinguish among hidden causes of failure and choose a safer response.

## Hidden causes

The first version uses a small discrete cause set:

- `HARD_MATERIAL`
- `BAD_ANGLE`
- `WALL_CONTACT`
- `CONTAINER_SLIDING`
- `TOOL_BENDING`

These causes are intentionally designed to produce overlapping observations.

A single observation may therefore be insufficient to identify the cause.

## Partial observability

By default, the agent receives only tool-side observations:

- reaction force
- tool displacement
- tool deformation

The agent does not directly observe the container position.

Sensor noise may also be added.

This means that similar observations can correspond to different hidden world states.

## Diagnostic actions

The agent may choose actions such as:

- `PUSH(F, theta)`
- `PROBE`
- `CHANGE_ANGLE`
- `REDUCE_CONTACT_AREA`
- `BRACE_CONTAINER`
- `STOP`

Some actions are useful mainly for completing the task.

Others are useful because they reveal information.

This allows the benchmark to test whether an agent can use physical interaction as a diagnostic probe.

## Attribution loop

The intended loop is:

1. perform an action
2. observe the tool-side response
3. compare expected and observed behavior
4. maintain candidate causes
5. choose a diagnostic probe if uncertainty remains
6. update the estimated cause
7. choose a corrective action
8. stop if safety constraints would be violated

The commanded-versus-observed mismatch is treated as evidence, not as the final research target.

## Minimal physical model

The first version will use a quasi-static 2D model with a small parameter set:

- applied force `F`
- force angle `theta`
- contact area `A`
- material yield strength `sigma_y`
- container mass `m`
- static friction coefficient `mu_s`
- kinetic friction coefficient `mu_k`
- tool stiffness `k`
- tool load limit
- `braced` state
- sensor noise

The model is deliberately simplified.

It is a diagnosis benchmark, not a high-fidelity ice-cream simulator.

## Example physical interaction

A container may begin to slide when the horizontal force component exceeds the available static friction.

A simplified condition is:

`F * cos(theta) > mu_s * (m * g + F * sin(theta))`

This means that changing the force angle can affect whether the container remains stable.

A corrective action can therefore be physically meaningful rather than purely heuristic.

## Policies

Version 0.1 will compare three simple policies.

### 1. Push-harder baseline

A deliberately naive policy:

`blocked -> increase force`

This baseline demonstrates how blind force escalation can create unnecessary constraint violations.

### 2. Heuristic policy

A fixed-rule policy that reacts to patterns in the observations.

For example:

- high tool deformation -> reduce force or stop
- high force with low displacement -> change angle
- suspected container motion -> brace before retrying

### 3. Probe-then-act policy

A diagnostic policy that:

1. maintains candidate causes
2. chooses a probe action
3. updates cause likelihoods
4. attributes the mismatch
5. selects a corrective action

The goal is not merely to act successfully, but to identify why the original action failed.

## Safety constraints

Task completion is not sufficient.

The benchmark will track measurable constraints such as:

- maximum contact force
- maximum container displacement
- maximum tool deformation
- keep-out-zone violations
- tool overload

A policy may therefore fail even if it eventually obtains the target.

## Evaluation metrics

The first evaluation will report:

- cause attribution accuracy
- task success rate
- constraint violation rate
- maximum applied force
- maximum container displacement
- number of actions
- number of diagnostic probes

## Ablation

A simple ablation will compare:

### Tool-side sensing only

The agent observes:

- reaction force
- tool displacement
- tool deformation

### Tool-side sensing + container position

The agent also receives the container position.

This tests how much the attribution problem depends on partial observability.

## Related work

This project is not intended to introduce command-observation mismatch as a new concept.

Relevant areas include:

- forward models and efference copy
- feedback control
- Kalman-filter innovation
- disturbance observers
- fault detection and isolation
- interactive perception
- contact and collision detection
- system identification
- model-based control
- safe and constrained control

The purpose of this repository is narrower:

to build a small, understandable benchmark for physical cause attribution under ambiguous contact observations.

## Falsifiable target

The benchmark is intended to support claims of the form:

> Under tool-side-only sensing, blind force escalation produces more constraint violations than a probe-then-act policy, while the probe-then-act policy improves hidden-cause attribution.

Exact numerical results will be reported only after experiments are implemented.

## Status

Early experimental sandbox.

No claim of human-like physical understanding.

No claim that successful task completion implies safe or intelligent behavior.

No claim that mismatch detection itself is novel.

## Working principle

The world may not respond as commanded.

The useful question is why.