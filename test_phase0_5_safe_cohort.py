import random

from phase0_5_pairs import (
    IdentificationPair,
)
from phase0_5_safe_cohort import (
    DEFAULT_RHO_SAFE_MAX,
    evaluate_pair_safety,
    generate_safe_pair_sample,
    true_force_components,
)
from physics import (
    Action,
    NextTransition,
    PhysicalParams,
)


def make_world(
    static_friction: float,
    material_yield_strength: float,
    container_mass: float = 20.0,
) -> PhysicalParams:
    return PhysicalParams(
        material_yield_strength=(
            material_yield_strength
        ),
        contact_area=0.0001,
        container_mass=container_mass,
        static_friction=static_friction,
        kinetic_friction=0.25,
        contact_stiffness=20_000.0,
        support_gain=1.0,
    )


def test_true_force_components_match_45_degree_probe():
    action = Action(
        force=10.0,
        angle_deg=45.0,
    )

    fx, fy = (
        true_force_components(
            action
        )
    )

    assert (
        abs(
            fx
            - fy
        )
        < 1e-12
    )


def test_generated_safe_pair_is_safe_on_both_twins():
    random.seed(
        42
    )

    sample = (
        generate_safe_pair_sample(
            pair_id=1,
        )
    )

    safety = (
        evaluate_pair_safety(
            pair=sample.pair,
            action=sample.action,
        )
    )

    assert (
        safety.safe_pair
    )

    assert (
        safety.blocked_a
    )

    assert (
        safety.blocked_b
    )

    assert (
        safety.rho_a
        < DEFAULT_RHO_SAFE_MAX
    )

    assert (
        safety.rho_b
        < DEFAULT_RHO_SAFE_MAX
    )


def test_generated_pair_keeps_same_action_for_both_twins():
    random.seed(
        42
    )

    sample = (
        generate_safe_pair_sample(
            pair_id=2,
            action=Action(
                force=5.0,
                angle_deg=45.0,
            ),
        )
    )

    safety = (
        evaluate_pair_safety(
            pair=sample.pair,
            action=sample.action,
        )
    )

    assert (
        safety.safe_pair
    )

    assert (
        sample.action.force
        == 5.0
    )

    assert (
        sample.action.angle_deg
        == 45.0
    )


def test_pair_is_rejected_when_one_twin_exceeds_rho_budget():
    safe_world = (
        make_world(
            static_friction=0.60,
            material_yield_strength=80_000,
            container_mass=20.0,
        )
    )

    unsafe_world = (
        make_world(
            static_friction=0.05,
            material_yield_strength=180_000,
            container_mass=5.0,
        )
    )

    pair = IdentificationPair(
        pair_id=10,
        world_a=safe_world,
        world_b=unsafe_world,
        label_a=(
            NextTransition.MATERIAL_YIELD
        ),
        label_b=(
            NextTransition.SUPPORT_SLIP
        ),
    )

    safety = (
        evaluate_pair_safety(
            pair=pair,
            action=Action(
                force=5.0,
                angle_deg=45.0,
            ),
            rho_safe_max=0.80,
        )
    )

    assert (
        safety.rho_safe_a
    )

    assert (
        not safety.rho_safe_b
    )

    assert (
        not safety.safe_pair
    )


def test_pair_is_rejected_when_one_twin_has_already_transitioned():
    weak_world = (
        make_world(
            static_friction=0.60,
            material_yield_strength=20_000,
            container_mass=20.0,
        )
    )

    normal_world = (
        make_world(
            static_friction=0.60,
            material_yield_strength=180_000,
            container_mass=20.0,
        )
    )

    pair = IdentificationPair(
        pair_id=11,
        world_a=weak_world,
        world_b=normal_world,
        label_a=(
            NextTransition.MATERIAL_YIELD
        ),
        label_b=(
            NextTransition.MATERIAL_YIELD
        ),
    )

    safety = (
        evaluate_pair_safety(
            pair=pair,
            action=Action(
                force=5.0,
                angle_deg=45.0,
            ),
            rho_safe_max=0.80,
        )
    )

    assert (
        not safety.blocked_a
    )

    assert (
        safety.blocked_b
    )

    assert (
        not safety.safe_pair
    )


def test_invalid_rho_safe_max_is_rejected():
    random.seed(
        42
    )

    sample = (
        generate_safe_pair_sample(
            pair_id=12,
        )
    )

    for invalid_value in [
        0.0,
        1.0,
        -0.1,
        1.1,
    ]:
        try:
            evaluate_pair_safety(
                pair=sample.pair,
                action=sample.action,
                rho_safe_max=invalid_value,
            )
        except ValueError as error:
            assert (
                "rho_safe_max"
                in str(error)
            )
        else:
            raise AssertionError(
                "Expected invalid rho_safe_max to fail"
            )


def test_multiple_safe_pairs_all_respect_budget():
    random.seed(
        42
    )

    for pair_id in range(
        20
    ):
        sample = (
            generate_safe_pair_sample(
                pair_id=pair_id,
            )
        )

        assert (
            sample.rho_a
            < DEFAULT_RHO_SAFE_MAX
        )

        assert (
            sample.rho_b
            < DEFAULT_RHO_SAFE_MAX
        )