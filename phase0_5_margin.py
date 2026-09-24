from dataclasses import dataclass
import math

from phase0_5_pairs import (
    IdentificationPair,
)
from physics import (
    PhysicalParams,
    predict_next_transition,
)


# Exploratory only.
#
# This is NOT yet the frozen confirmatory value.
DEFAULT_EPSILON_TIE = 0.10


@dataclass
class WorldMarginCheck:
    material_force: float
    slip_force: float
    normalized_margin: float
    separated: bool


@dataclass
class PairMarginCheck:
    pair_id: int
    margin_a: float
    margin_b: float
    separated_a: bool
    separated_b: bool
    separated_pair: bool


def normalized_threshold_margin_from_forces(
    material_force: float,
    slip_force: float,
) -> float:
    """
    Return normalized separation between the two candidate
    transition thresholds.

    Definition:

        abs(F_material - F_slip)
        ------------------------
        min(F_material, F_slip)

    A value near zero means the two candidate futures are
    effectively tied.

    Larger values mean the first transition is more clearly
    separated from the alternative.
    """

    if material_force <= 0:
        raise ValueError(
            "material_force must be positive"
        )

    if slip_force <= 0:
        raise ValueError(
            "slip_force must be positive"
        )

    if (
        math.isinf(material_force)
        and math.isinf(slip_force)
    ):
        return 0.0

    if (
        math.isinf(material_force)
        or math.isinf(slip_force)
    ):
        return float(
            "inf"
        )

    denominator = min(
        material_force,
        slip_force,
    )

    return (
        abs(
            material_force
            - slip_force
        )
        / denominator
    )


def evaluate_world_margin(
    params: PhysicalParams,
    angle_deg: float,
    epsilon_tie: float = DEFAULT_EPSILON_TIE,
) -> WorldMarginCheck:
    """
    Evaluate whether one world's two candidate transition
    thresholds are sufficiently separated.
    """

    if not (
        0.0
        < epsilon_tie
        < 1.0
    ):
        raise ValueError(
            "epsilon_tie must be between 0 and 1"
        )

    (
        _,
        material_force,
        slip_force,
    ) = predict_next_transition(
        params,
        angle_deg,
    )

    margin = (
        normalized_threshold_margin_from_forces(
            material_force=material_force,
            slip_force=slip_force,
        )
    )

    separated = (
        margin
        >= epsilon_tie
    )

    return WorldMarginCheck(
        material_force=material_force,
        slip_force=slip_force,
        normalized_margin=margin,
        separated=separated,
    )


def evaluate_pair_margin(
    pair: IdentificationPair,
    angle_deg: float,
    epsilon_tie: float = DEFAULT_EPSILON_TIE,
) -> PairMarginCheck:
    """
    Require both twins to have clearly separated candidate
    transition thresholds.

    If either twin is near-tied, the entire pair is excluded.
    """

    check_a = (
        evaluate_world_margin(
            params=pair.world_a,
            angle_deg=angle_deg,
            epsilon_tie=epsilon_tie,
        )
    )

    check_b = (
        evaluate_world_margin(
            params=pair.world_b,
            angle_deg=angle_deg,
            epsilon_tie=epsilon_tie,
        )
    )

    separated_pair = (
        check_a.separated
        and check_b.separated
    )

    return PairMarginCheck(
        pair_id=pair.pair_id,
        margin_a=(
            check_a.normalized_margin
        ),
        margin_b=(
            check_b.normalized_margin
        ),
        separated_a=(
            check_a.separated
        ),
        separated_b=(
            check_b.separated
        ),
        separated_pair=(
            separated_pair
        ),
    )


def main():
    examples = [
        (
            10.0,
            10.001,
        ),
        (
            10.0,
            11.0,
        ),
        (
            10.0,
            15.0,
        ),
    ]

    print(
        "=== Phase 0.5 normalized transition margin ==="
    )

    print(
        "epsilon_tie=",
        DEFAULT_EPSILON_TIE,
    )

    for (
        material_force,
        slip_force,
    ) in examples:
        margin = (
            normalized_threshold_margin_from_forces(
                material_force=material_force,
                slip_force=slip_force,
            )
        )

        print(
            "material=",
            material_force,
            "slip=",
            slip_force,
            "margin=",
            round(
                margin,
                6,
            ),
            "separated=",
            margin
            >= DEFAULT_EPSILON_TIE,
        )


if __name__ == "__main__":
    main()