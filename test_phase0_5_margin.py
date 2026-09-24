from phase0_5_margin import (
    evaluate_world_margin,
    normalized_threshold_margin_from_forces,
)
from physics import (
    PhysicalParams,
    material_transition_force,
    slip_transition_force,
)


def make_world(
    material_yield_strength: float = 120_000.0,
    contact_area: float = 0.0001,
    container_mass: float = 20.0,
    static_friction: float = 0.35,
) -> PhysicalParams:
    return PhysicalParams(
        material_yield_strength=(
            material_yield_strength
        ),
        contact_area=(
            contact_area
        ),
        container_mass=(
            container_mass
        ),
        static_friction=(
            static_friction
        ),
        kinetic_friction=0.25,
        contact_stiffness=20_000.0,
        support_gain=1.0,
    )


def test_equal_thresholds_have_zero_margin():
    margin = (
        normalized_threshold_margin_from_forces(
            material_force=10.0,
            slip_force=10.0,
        )
    )

    assert (
        margin
        == 0.0
    )


def test_normalized_margin_matches_definition():
    margin = (
        normalized_threshold_margin_from_forces(
            material_force=10.0,
            slip_force=12.0,
        )
    )

    assert (
        abs(
            margin
            - 0.20
        )
        < 1e-12
    )


def test_margin_is_symmetric():
    forward = (
        normalized_threshold_margin_from_forces(
            material_force=10.0,
            slip_force=12.0,
        )
    )

    reverse = (
        normalized_threshold_margin_from_forces(
            material_force=12.0,
            slip_force=10.0,
        )
    )

    assert (
        abs(
            forward
            - reverse
        )
        < 1e-12
    )


def test_tiny_difference_is_near_tie():
    margin = (
        normalized_threshold_margin_from_forces(
            material_force=10.0,
            slip_force=10.001,
        )
    )

    assert (
        margin
        < 0.001
    )


def test_large_difference_is_clearly_separated():
    margin = (
        normalized_threshold_margin_from_forces(
            material_force=10.0,
            slip_force=15.0,
        )
    )

    assert (
        margin
        == 0.5
    )


def test_infinite_alternative_is_fully_separated():
    margin = (
        normalized_threshold_margin_from_forces(
            material_force=10.0,
            slip_force=float(
                "inf"
            ),
        )
    )

    assert (
        margin
        == float(
            "inf"
        )
    )


def test_two_infinite_thresholds_are_not_treated_as_separated():
    margin = (
        normalized_threshold_margin_from_forces(
            material_force=float(
                "inf"
            ),
            slip_force=float(
                "inf"
            ),
        )
    )

    assert (
        margin
        == 0.0
    )


def test_nonpositive_threshold_is_rejected():
    try:
        normalized_threshold_margin_from_forces(
            material_force=0.0,
            slip_force=10.0,
        )
    except ValueError as error:
        assert (
            "material_force"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected nonpositive material force to fail"
        )


def test_world_margin_uses_analytic_thresholds():
    world = make_world()

    check = (
        evaluate_world_margin(
            params=world,
            angle_deg=45.0,
            epsilon_tie=0.10,
        )
    )

    expected_material = (
        material_transition_force(
            world,
            angle_deg=45.0,
        )
    )

    expected_slip = (
        slip_transition_force(
            world,
            angle_deg=45.0,
        )
    )

    assert (
        abs(
            check.material_force
            - expected_material
        )
        < 1e-12
    )

    assert (
        abs(
            check.slip_force
            - expected_slip
        )
        < 1e-12
    )


def test_world_separation_matches_epsilon_rule():
    world = make_world()

    check = (
        evaluate_world_margin(
            params=world,
            angle_deg=45.0,
            epsilon_tie=0.10,
        )
    )

    assert (
        check.separated
        ==
        (
            check.normalized_margin
            >= 0.10
        )
    )


def test_invalid_epsilon_tie_is_rejected():
    world = make_world()

    for invalid_value in [
        0.0,
        1.0,
        -0.1,
        1.1,
    ]:
        try:
            evaluate_world_margin(
                params=world,
                angle_deg=45.0,
                epsilon_tie=invalid_value,
            )
        except ValueError as error:
            assert (
                "epsilon_tie"
                in str(error)
            )
        else:
            raise AssertionError(
                "Expected invalid epsilon_tie to fail"
            )