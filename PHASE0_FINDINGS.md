# Phase 0 Findings

## Status

**Identifiability Gate: PARTIAL PASS**

Phase 0 provides evidence that an impending support-slip transition can become empirically distinguishable from a material-yield transition when the physical system exposes a measurable pre-slip motion precursor with sufficient signal-to-noise ratio.

This result is conditional.

The current project does **not** establish general pre-slip identifiability for real robots, real containers, or real contact mechanics.

---

## Phase 0 question

The Phase 0 research question is:

> How much information about an impending support-slip transition is available from safe tool-side interaction before significant slip occurs?

The competing future transitions are:

- `MATERIAL_YIELD`
- `SUPPORT_SLIP`

The central concern is whether useful information is available before force escalation causes the unsafe transition itself.

---

## 1. Baseline rigid support

Under the rigid-support model, tool-side pre-transition observations generally remain close to chance-level separability.

Representative sensor-ablation results:

### 5 N probe

```text
force_only     ≈ 0.50
motion_only    ≈ 0.49
tip_dx_only    ≈ 0.43
full           ≈ 0.50
```

### 11 N probe

```text
force_only     ≈ 0.56
motion_only    ≈ 0.51
tip_dx_only    ≈ 0.54
full           ≈ 0.52
```

Interpretation:

> In the current rigid model, safe tool-side probing does not expose a strong and reliable pre-slip precursor.

This is consistent with the original concern that ideal static friction may provide little direct information about distance to the slip threshold before that threshold is approached.

---

## 2. Compliant support

After feature standardization and the physical-model audit, compliant support produced substantially higher empirical separability than rigid support.

Representative compliance-ablation results:

```text
Probe force | Rigid | Compliant
--------------------------------
2 N         | 0.485 | 0.727
5 N         | 0.515 | 0.788
8 N         | 0.657 | 0.829
11 N        | 0.487 | 0.821
14 N        | 0.500 | 1.000
17 N        | 0.529 | 0.941
20 N        | 0.700 | 0.900
```

At larger probe forces, some conditions lacked enough balanced safe samples for reliable classification.

Interpretation:

> When the model includes measurable pre-slip support motion, future transition type becomes substantially more separable from tool-side observations.

This does not prove that real supports behave according to the simulated compliance law.

It shows that the existence and detectability of a physical precursor can change the identifiability of the task.

---

## 3. Sensor ablation

The most informative channel in the compliant-support condition is horizontal tool-tip motion.

### 5 N probe

```text
RIGID
force_only     ≈ 0.50
motion_only    ≈ 0.49
tip_dx_only    ≈ 0.43
full           ≈ 0.50

COMPLIANT
force_only     ≈ 0.50
motion_only    ≈ 0.86
tip_dx_only    ≈ 0.86
full           ≈ 0.88
```

### 11 N probe

```text
RIGID
force_only     ≈ 0.56
motion_only    ≈ 0.51
tip_dx_only    ≈ 0.54
full           ≈ 0.52

COMPLIANT
force_only     ≈ 0.56
motion_only    ≈ 0.87
tip_dx_only    ≈ 0.87
full           ≈ 0.87
```

Interpretation:

> The compliant-model gain is primarily carried by motion sensing, especially horizontal tip displacement, rather than force sensing alone.

This supports treating pre-slip motion as the modeled diagnostic precursor.

---

## 4. Sensing threshold

The minimum-sensing experiment varied:

- pre-slip displacement magnitude
- motion-sensor noise

while using balanced future-transition classes.

Representative results:

```text
Precursor | Noise | Accuracy
----------------------------
0.10 mm   | 1.00 mm | 0.486
0.10 mm   | 0.05 mm | 0.644
0.25 mm   | 0.05 mm | 0.824
0.50 mm   | 0.05 mm | 0.934
1.00 mm   | 0.10 mm | 0.934
2.00 mm   | 0.20 mm | 0.934
4.00 mm   | 0.50 mm | 0.908
```

Interpretation:

> Identifiability improves as the modeled physical precursor becomes large relative to sensor noise.

---

## 5. Precursor-to-noise scaling

The precursor-to-noise ratio produced a strong monotonic trend in the current toy model.

Representative results:

```text
SNR    Accuracy
---------------
0.10   0.486
0.50   0.518
1.00   0.572
2.00   0.644
5.00   0.824
10.00  0.934
20.00  0.968
40.00  0.976
80.00  0.976
```

Multiple precursor/noise combinations with the same ratio produced similar classification accuracy.

Examples:

```text
SNR = 1

0.10 mm / 0.10 mm -> 0.572
0.50 mm / 0.50 mm -> 0.572
1.00 mm / 1.00 mm -> 0.572
```

and:

```text
SNR = 10

0.50 mm / 0.05 mm -> 0.934
1.00 mm / 0.10 mm -> 0.934
2.00 mm / 0.20 mm -> 0.934
```

Interpretation:

> Within the current simulator, precursor-to-noise ratio is a useful reduced description of sensing difficulty.

This is not a universal scaling law.

It is a property observed in the current simplified model.

---

## 6. Information-versus-safety pattern

Increasing probe force increases the rate at which physical transitions are triggered.

Representative transition rates:

```text
2 N   -> 0.000
5 N   -> 0.000
8 N   -> ~0.03
11 N  -> ~0.14
14 N  -> ~0.30
17 N  -> ~0.48
20 N  -> ~0.63
23 N  -> ~0.75
26 N  -> ~0.84
```

Interpretation:

> Stronger probing carries increasing physical risk.

The project therefore cannot treat higher-force diagnosis as free information gathering.

The central Phase 0 tradeoff remains:

> More interaction may reveal more information, but may also consume the safety budget required to obtain it.

---

## 7. Natural class imbalance

Under the default parameter distribution, future transitions are highly imbalanced.

One baseline run produced approximately:

```text
MATERIAL_YIELD: 5945
SUPPORT_SLIP:     55
```

This revealed an important distinction between two separate questions.

### Occurrence

How often does each transition arise under a chosen world distribution?

### Identifiability

Given examples of both transition types, can tool-side observations distinguish them?

Balanced sampling is used for separability experiments.

Natural frequencies must therefore not be inferred from balanced-class classification results.

---

## 8. Model audit

Phase 0 underwent two important corrections before this findings document was frozen.

### Audit A — Feature scaling

Early classifiers combined force values measured in newtons with displacement values measured in fractions of a millimeter without standardization.

After introducing:

```text
StandardScaler -> LogisticRegression
```

the compliant motion precursor became clearly visible in the classification results.

Interpretation:

> Earlier rigid/compliant similarity was partly a numerical-scale artifact.

---

### Audit B — Pre-slip load definition

The initial compliant model used a mixed quantity:

```text
horizontal force / total-force slip threshold
```

This was replaced with a dimensionless friction-demand ratio:

```text
horizontal friction demand
--------------------------
static-friction capacity
```

Conceptually:

```text
|Fx|
--------------------
mu_s * (m*g + Fy)
```

This aligns the modeled pre-slip precursor with proximity to the static-friction limit.

The Phase 0 pipeline continued to pass after this correction, and the compliant-support identifiability pattern remained.

Interpretation:

> The main qualitative result survived a physically more consistent reformulation of the pre-slip load variable.

---

## 9. Current interpretation

The strongest Phase 0 result is:

> In this toy model, future support-slip risk becomes empirically distinguishable before transition when the physical system exposes a measurable pre-slip motion precursor and that precursor is sufficiently large relative to sensor noise.

The corresponding negative result is equally important:

> Under the idealized rigid-support condition, safe tool-side observations provide little reliable pre-slip information.

Together, these results suggest that diagnosability depends not only on inference method, but on whether the physical system exposes informative pre-transition structure.

---

## 10. Identifiability Gate verdict

### Verdict

**PARTIAL PASS**

### Why it passes

The project demonstrated that:

- some physically motivated pre-transition signals can carry useful information
- motion sensing can distinguish future transition type in the compliant toy model
- the effect survives feature standardization
- the effect survives a correction to the pre-slip load definition
- sensing performance varies systematically with precursor-to-noise ratio

### Why it is not a full pass

The project has not demonstrated that:

- real supports expose the same precursor
- real robot tool-tip measurements contain equivalent information
- the precursor model is realistic
- the result generalizes beyond the current parameterization
- active diagnostic actions improve actual task safety
- the same signal exists under richer contact mechanics

---

## 11. Claims now supported

The project may now state that, within the current toy simulator:

- rigid and compliant support models differ strongly in pre-transition identifiability
- horizontal motion carries most of the useful compliant-model signal
- classification performance improves with precursor-to-noise ratio
- stronger probing increases transition risk
- natural transition frequencies and balanced-class identifiability are separate experimental questions
- Phase 0 supports conditional continuation into diagnostic-action research

---

## 12. Claims still not supported

The project must not claim that:

- real robots can predict support slip using this method
- the simulated compliance law represents real surfaces
- the reported SNR thresholds are hardware requirements
- the project has demonstrated real-world safety improvement
- active diagnosis is superior to conventional control methods
- the project introduces a universal law of embodiment
- the current classifier is an optimal estimator
- the Ice Cream Problem is itself a new robotics problem class

---

## 13. Phase 1 entry condition

Phase 1 may begin only under the following interpretation:

> The purpose is not to prove that pre-slip information always exists.

The purpose is:

> Given a world in which informative pre-transition signals may exist, which legal physical action obtains the most useful information under a limited safety budget?

Phase 1 therefore studies:

**Diagnostic Action Value**

rather than simply increasing classifier complexity.

---

## Working conclusion

The world does not always reveal how it will fail.

But when physical systems expose measurable precursors, interaction may reveal which constraint is about to dominate.

The engineering problem is therefore not only:

> What should the robot do?

It is also:

> What can the robot learn safely before the world changes state?