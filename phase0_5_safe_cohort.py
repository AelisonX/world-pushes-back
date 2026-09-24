from dataclasses import dataclass
import math

from phase0_5_pairs import (
    IdentificationPair,
    generate_identification_pair,
)
from physics import (
    Action,
    support_load_fraction,
    predict_next_transition,
)


DEFAULT_PROBE_FORCE = 5.0
DEFAULT_PROBE_ANGLE_DEG = 45.0

# Exploratory only.
#
# Confirmatory Phase 0.5 will freeze this value before the
# official run.
DEFAULT_RHO_SAFE_MAX = 0.80


@dataclass
class PairSafetyCheck:
    pair_id: int
    rho_a: float
    rho_b: float
    blocked_a: bool
    blocked_b: bool
    rho_safe_a: bool
    rho_safe_b: bool
    safe_pair: bool


@dataclass
class SafePairSample:
    pair: IdentificationPair
    action: Action
    rho_a: float
    rho_b: float


def true_force_components(
    action: Action,
) -> tuple[
    float,
    float,
]:
    """
    Return noiseless horizontal and vertical force components.
    """

    theta = math.radians(
        action.angle_deg
    )

    true_fx = (
        action.force
        * math.cos(theta)
    )

    true_fy = (
        action.force
        * math.sin(theta)
    )

    return (
        true_fx,
        true_fy,
    )


def world_is_still_blocked(
    params,
    action: Action,
) -> bool:
    """
    Check analytically whether the probe force remains below
    both possible transition thresholds.
    """

    (
        _,
        material_force,
        slip_force,
    ) = predict_next_transition(
        params,
        action.angle_deg,
    )

    first_transition_force = min(
        material_force,
        slip_force,
    )

    return (
        action.force
        < first_transition_force
    )


def evaluate_pair_safety(
    pair: IdentificationPair,
    action: Action,
    rho_safe_max: float = DEFAULT_RHO_SAFE_MAX,
) -> PairSafetyCheck:
    """
    Evaluate whether both twins are safe under the same action.

    Official Phase 0.5 observations must satisfy:

        rho < rho_safe_max < 1

    and must remain below both transition thresholds.

    If either twin fails, the entire pair is unsafe.
    """

    if not (
        0.0
        < rho_safe_max
        < 1.0
    ):
        raise ValueError(
            "rho_safe_max must be between 0 and 1"
        )

    (
        true_fx,
        true_fy,
    ) = true_force_components(
        action
    )

    rho_a = (
        support_load_fraction(
            params=pair.world_a,
            true_fx=true_fx,
            true_fy=true_fy,
        )
    )

    rho_b = (
        support_load_fraction(
            params=pair.world_b,
            true_fx=true_fx,
            true_fy=true_fy,
        )
    )

    blocked_a = (
        world_is_still_blocked(
            pair.world_a,
            action,
        )
    )

    blocked_b = (
        world_is_still_blocked(
            pair.world_b,
            action,
        )
    )

    rho_safe_a = (
        rho_a
        < rho_safe_max
    )

    rho_safe_b = (
        rho_b
        < rho_safe_max
    )

    safe_pair = (
        blocked_a
        and blocked_b
        and rho_safe_a
        and rho_safe_b
    )

    return PairSafetyCheck(
        pair_id=pair.pair_id,
        rho_a=rho_a,
        rho_b=rho_b,
        blocked_a=blocked_a,
        blocked_b=blocked_b,
        rho_safe_a=rho_safe_a,
        rho_safe_b=rho_safe_b,
        safe_pair=safe_pair,
    )


def generate_safe_pair_sample(
    pair_id: int,
    action: Action | None = None,
    rho_safe_max: float = DEFAULT_RHO_SAFE_MAX,
    max_attempts: int = 10_000,
) -> SafePairSample:
    """
    Generate one matched identification pair that is safe for
    the same probe action on both twins.
    """

    if action is None:
        action = Action(
            force=DEFAULT_PROBE_FORCE,
            angle_deg=DEFAULT_PROBE_ANGLE_DEG,
        )

    for _ in range(
        max_attempts
    ):
        pair = (
            generate_identification_pair(
                pair_id=pair_id,
                angle_deg=action.angle_deg,
            )
        )

        safety = (
            evaluate_pair_safety(
                pair=pair,
                action=action,
                rho_safe_max=rho_safe_max,
            )
        )

        if not safety.safe_pair:
            continue

        return SafePairSample(
            pair=pair,
            action=action,
            rho_a=safety.rho_a,
            rho_b=safety.rho_b,
        )

    raise RuntimeError(
        "Could not generate a safe identification pair"
    )


def main():
    import random

    random.seed(
        42
    )

    action = Action(
        force=DEFAULT_PROBE_FORCE,
        angle_deg=DEFAULT_PROBE_ANGLE_DEG,
    )

    print(
        "=== Phase 0.5 safe paired cohort ==="
    )

    print(
        "probe_force_N=",
        action.force,
    )

    print(
        "probe_angle_deg=",
        action.angle_deg,
    )

    print(
        "rho_safe_max=",
        DEFAULT_RHO_SAFE_MAX,
    )

    for pair_id in range(
        5
    ):
        sample = (
            generate_safe_pair_sample(
                pair_id=pair_id,
                action=action,
            )
        )

        print()

        print(
            f"pair_id={sample.pair.pair_id}"
        )

        print(
            "A:",
            sample.pair.label_a.value,
            "rho=",
            round(
                sample.rho_a,
                4,
            ),
        )

        print(
            "B:",
            sample.pair.label_b.value,
            "rho=",
            round(
                sample.rho_b,
                4,
            ),
        )


if __name__ == "__main__":
    main()