# world-pushes-back

Tiny experiments in embodied intelligence for agents that must deal with a world that does not always respond as commanded.

## Core idea

Embodied intelligence is not just the ability to issue actions.

An embodied agent should notice when the physical world responds differently from what it expected, infer why, and adapt safely.

Basic loop:

- command
- world responds
- observe
- compare expected vs observed
- infer hidden constraint
- adapt

## Research question

When an action encounters resistance, can an embodied agent infer what is resisting it instead of simply increasing force?

## Example 001: The Ice Cream Problem

Imagine a robot trying to scoop very hard ice cream from a movable freezer.

A naive policy might be:

`blocked -> push harder`

But increasing force may not move the scoop.

Instead, the freezer itself may slide across the floor.

The important question is therefore not only:

> Can the robot scoop the ice cream?

It is:

> Can the robot identify what part of the physical system is limiting the action?

Possible causes include:

- the ice cream is too hard
- the scoop angle is poor
- the scoop hits the container wall
- the freezer is too light
- floor friction is too low
- the tool is bending
- the material suddenly breaks or softens

Different causes require different actions.

## Commanded vs observed delta

Example:

Commanded:

`Scoop downward: 4.0 cm`

Observed:

`Scoop downward: 0.4 cm`

`Freezer sideways: 18.0 cm`

The agent commanded motion of the scoop.

The world produced motion of the freezer.

That mismatch is information.

## First toy model

The first simulator will use a small set of variables:

- `robot_force`
- `material_resistance`
- `scoop_depth`
- `container_mass`
- `floor_friction`
- `container_stabilized`

Possible outcomes:

- `SUCCESS`
- `SCOOP_BLOCKED`
- `CONTAINER_SLID`
- `TOOL_OVERLOAD`

The first deliberately bad policy will be:

`if blocked: push harder`

We will then build a safer policy that reacts to observed resistance and unexpected motion.

## Safety constraints

Task success is not enough.

A useful policy should respect constraints such as:

1. do not exceed safe force limits
2. do not move the container dangerously
3. do not overload the tool
4. adapt when expected and observed motion differ
5. obtain the target only if the action remains safe

## Why this matters

A powerful robot that blindly increases force may complete a local task while destabilizing the surrounding environment.

Embodied intelligence therefore requires reasoning about:

- the agent's own force
- material resistance
- tool properties
- object mass
- friction
- contact
- unexpected motion
- changing constraints

The world is not a passive background.

It pushes back.

## Future examples

Possible future scenarios include:

- movable furniture
- stuck doors
- deformable objects
- tool bending
- uncertain contact
- internal-state simulation
- biomimetic digestive systems

These examples explore the same general question:

> How should an embodied agent update its behavior when reality disagrees with its command?

## Status

Early experimental sandbox.

No claim of human-like physical understanding.

No claim that successful task completion implies safe or intelligent behavior.