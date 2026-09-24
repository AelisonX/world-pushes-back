from dataclasses import dataclass
import random

from phase0_5_worlds import (
    sample_phase0_5_world,
)
from physics import (
    NextTransition,
    PhysicalParams,
    predict_next_transition,
)


@dataclass
class IdentificationPair:
    pair_id: int
    world_a: PhysicalParams
    world_b: PhysicalParams
    label_a: NextTransition
    label_b: NextTransition


def clone_with_failure_parameters(
    base: PhysicalParams,
    static_friction: float,
    material_yield_strength: float,
) -> PhysicalParams:
    """
    Copy one Phase 0.5 world while changing only the
    failure-margin variables.

    Shared nuisance variables remain identical between twins.
    """

    return PhysicalParams(
        material_yield_strength=(
            material_yield_strength
        ),
        contact_area=(
            base.contact_area
        ),
        container_mass=(
            base.container_mass
        ),
        static_friction=(
            static_friction
        ),
        kinetic_friction=(
            base.kinetic_friction
        ),
        contact_stiffness=(
            base.contact_stiffness
        ),
        pre_slip_displacement_limit=(
            base.pre_slip_displacement_limit
        ),
        support_gain=(
            base.support_gain
        ),
    )


def generate_identification_pair(
    pair_id: int,
    angle_deg: float = 45.0,
    max_attempts: int = 10_000,
) -> IdentificationPair:
    """
    Generate two matched Phase 0.5 worlds with opposite
    counterfactual future transitions.

    Twins share nuisance variables.

    They differ only in:

    - static friction
    - material yield strength

    The A/B ordering is randomized so one side does not
    deterministically correspond to one label.
    """

    for _ in range(
        max_attempts
    ):
        base = (
            sample_phase0_5_world()
        )

        friction_center = random.uniform(
            0.25,
            0.50,
        )

        friction_delta = random.uniform(
            0.03,
            0.15,
        )

        yield_center = random.uniform(
            100_000,
            160_000,
        )

        yield_delta = random.uniform(
            10_000,
            45_000,
        )

        low_friction = max(
            0.05,
            friction_center
            - friction_delta,
        )

        high_friction = (
            friction_center
            + friction_delta
        )

        low_yield = max(
            1.0,
            yield_center
            - yield_delta,
        )

        high_yield = (
            yield_center
            + yield_delta
        )

        candidate_slip = (
            clone_with_failure_parameters(
                base=base,
                static_friction=(
                    low_friction
                ),
                material_yield_strength=(
                    high_yield
                ),
            )
        )

        candidate_yield = (
            clone_with_failure_parameters(
                base=base,
                static_friction=(
                    high_friction
                ),
                material_yield_strength=(
                    low_yield
                ),
            )
        )

        (
            slip_label,
            _,
            _,
        ) = predict_next_transition(
            candidate_slip,
            angle_deg,
        )

        (
            yield_label,
            _,
            _,
        ) = predict_next_transition(
            candidate_yield,
            angle_deg,
        )

        if (
            slip_label
            != NextTransition.SUPPORT_SLIP
        ):
            continue

        if (
            yield_label
            != NextTransition.MATERIAL_YIELD
        ):
            continue

        if random.random() < 0.5:
            world_a = (
                candidate_slip
            )

            world_b = (
                candidate_yield
            )

            label_a = (
                slip_label
            )

            label_b = (
                yield_label
            )

        else:
            world_a = (
                candidate_yield
            )

            world_b = (
                candidate_slip
            )

            label_a = (
                yield_label
            )

            label_b = (
                slip_label
            )

        return IdentificationPair(
            pair_id=pair_id,
            world_a=world_a,
            world_b=world_b,
            label_a=label_a,
            label_b=label_b,
        )

    raise RuntimeError(
        "Could not generate an opposite-label "
        "identification pair"
    )


def shared_nuisance_signature(
    params: PhysicalParams,
) -> tuple:
    """
    Return nuisance variables that should match exactly
    within an identification pair.
    """

    return (
        params.contact_area,
        params.container_mass,
        params.kinetic_friction,
        params.contact_stiffness,
        params.pre_slip_displacement_limit,
        params.support_gain,
    )


def main():
    random.seed(
        42
    )

    print(
        "=== Phase 0.5 paired identification worlds ==="
    )

    for pair_id in range(
        5
    ):
        pair = (
            generate_identification_pair(
                pair_id=pair_id,
            )
        )

        print()

        print(
            f"pair_id={pair.pair_id}"
        )

        print(
            "A:",
            pair.label_a.value,
            "mu_s=",
            round(
                pair.world_a.static_friction,
                3,
            ),
            "yield=",
            round(
                pair.world_a.material_yield_strength,
                1,
            ),
        )

        print(
            "B:",
            pair.label_b.value,
            "mu_s=",
            round(
                pair.world_b.static_friction,
                3,
            ),
            "yield=",
            round(
                pair.world_b.material_yield_strength,
                1,
            ),
        )

        print(
            "shared k_contact=",
            round(
                pair.world_a.contact_stiffness,
                1,
            ),
            "shared support_gain=",
            round(
                pair.world_a.support_gain,
                3,
            ),
        )


if __name__ == "__main__":
    main()