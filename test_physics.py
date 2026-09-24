from physics import (
    Action,
    PhysicalParams,
    compute_pre_slip_support_motion,
    material_transition_force,
    run_safe_probe,
    slip_transition_force,
    support_load_fraction,
)


def make_params(
    container_mass=20.0,
    static_friction=0.35,
    material_yield_strength=120_000,
    contact_area=0.0001,
    contact_stiffness=20_000.0,
    support_gain=1.0,
):
    return PhysicalParams(
        material_yield_strength=(
            material_yield_strength
        ),
        contact_area=contact_area,
        container_mass=container_mass,
        static_friction=static_friction,
        kinetic_friction=0.25,
        contact_stiffness=(
            contact_stiffness
        ),
        support_gain=(
            support_gain
        ),
    )


def test_higher_static_friction_raises_slip_threshold():
    low_friction = make_params(
        static_friction=0.20,
    )

    high_friction = make_params(
        static_friction=0.60,
    )

    low_threshold = (
        slip_transition_force(
            low_friction,
            angle_deg=45.0,
        )
    )

    high_threshold = (
        slip_transition_force(
            high_friction,
            angle_deg=45.0,
        )
    )

    assert (
        high_threshold
        > low_threshold
    )


def test_heavier_container_raises_slip_threshold():
    light = make_params(
        container_mass=5.0,
    )

    heavy = make_params(
        container_mass=30.0,
    )

    light_threshold = (
        slip_transition_force(
            light,
            angle_deg=45.0,
        )
    )

    heavy_threshold = (
        slip_transition_force(
            heavy,
            angle_deg=45.0,
        )
    )

    assert (
        heavy_threshold
        > light_threshold
    )


def test_stronger_material_raises_yield_threshold():
    weak = make_params(
        material_yield_strength=80_000,
    )

    strong = make_params(
        material_yield_strength=180_000,
    )

    weak_threshold = (
        material_transition_force(
            weak,
            angle_deg=45.0,
        )
    )

    strong_threshold = (
        material_transition_force(
            strong,
            angle_deg=45.0,
        )
    )

    assert (
        strong_threshold
        > weak_threshold
    )


def test_larger_contact_area_raises_yield_threshold():
    small_area = make_params(
        contact_area=0.00005,
    )

    large_area = make_params(
        contact_area=0.00015,
    )

    small_threshold = (
        material_transition_force(
            small_area,
            angle_deg=45.0,
        )
    )

    large_threshold = (
        material_transition_force(
            large_area,
            angle_deg=45.0,
        )
    )

    assert (
        large_threshold
        > small_threshold
    )


def test_rigid_support_uses_world_contact_stiffness():
    params = make_params(
        contact_stiffness=10_000.0,
    )

    result = run_safe_probe(
        params=params,
        action=Action(
            force=5.0,
            angle_deg=45.0,
        ),
        support_model="rigid",
        force_noise_std=0.0,
        motion_noise_std=0.0,
    )

    expected_local_dx = (
        result.observation.fx
        / params.contact_stiffness
    )

    assert (
        abs(
            result.observation.tip_dx
            - expected_local_dx
        )
        < 1e-12
    )


def test_lower_contact_stiffness_increases_local_tip_motion():
    soft = make_params(
        contact_stiffness=10_000.0,
    )

    stiff = make_params(
        contact_stiffness=40_000.0,
    )

    action = Action(
        force=5.0,
        angle_deg=45.0,
    )

    soft_result = run_safe_probe(
        params=soft,
        action=action,
        support_model="rigid",
        force_noise_std=0.0,
        motion_noise_std=0.0,
    )

    stiff_result = run_safe_probe(
        params=stiff,
        action=action,
        support_model="rigid",
        force_noise_std=0.0,
        motion_noise_std=0.0,
    )

    assert (
        abs(
            soft_result.observation.tip_dx
        )
        >
        abs(
            stiff_result.observation.tip_dx
        )
    )


def test_nonpositive_contact_stiffness_is_rejected():
    params = make_params(
        contact_stiffness=0.0,
    )

    try:
        run_safe_probe(
            params=params,
            action=Action(
                force=5.0,
                angle_deg=45.0,
            ),
            support_model="rigid",
            force_noise_std=0.0,
            motion_noise_std=0.0,
        )
    except ValueError as error:
        assert (
            "contact_stiffness"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected nonpositive contact stiffness to fail"
        )


def test_support_load_fraction_increases_with_horizontal_demand():
    params = make_params()

    low_demand = (
        support_load_fraction(
            params=params,
            true_fx=2.0,
            true_fy=5.0,
        )
    )

    high_demand = (
        support_load_fraction(
            params=params,
            true_fx=10.0,
            true_fy=5.0,
        )
    )

    assert (
        high_demand
        > low_demand
    )


def test_support_load_fraction_decreases_with_mass():
    light = make_params(
        container_mass=5.0,
    )

    heavy = make_params(
        container_mass=30.0,
    )

    light_fraction = (
        support_load_fraction(
            params=light,
            true_fx=10.0,
            true_fy=5.0,
        )
    )

    heavy_fraction = (
        support_load_fraction(
            params=heavy,
            true_fx=10.0,
            true_fy=5.0,
        )
    )

    assert (
        heavy_fraction
        < light_fraction
    )


def test_support_load_fraction_decreases_with_static_friction():
    low_friction = make_params(
        static_friction=0.20,
    )

    high_friction = make_params(
        static_friction=0.60,
    )

    low_fraction = (
        support_load_fraction(
            params=low_friction,
            true_fx=10.0,
            true_fy=5.0,
        )
    )

    high_fraction = (
        support_load_fraction(
            params=high_friction,
            true_fx=10.0,
            true_fy=5.0,
        )
    )

    assert (
        high_fraction
        < low_fraction
    )


def test_support_load_fraction_decreases_with_downward_force():
    params = make_params()

    low_downward_force = (
        support_load_fraction(
            params=params,
            true_fx=10.0,
            true_fy=1.0,
        )
    )

    high_downward_force = (
        support_load_fraction(
            params=params,
            true_fx=10.0,
            true_fy=10.0,
        )
    )

    assert (
        high_downward_force
        < low_downward_force
    )


def test_compliant_support_motion_increases_with_load_fraction():
    far_from_slip = (
        compute_pre_slip_support_motion(
            load_fraction=0.10,
            pre_slip_displacement_limit=0.002,
            direction=1.0,
        )
    )

    near_slip = (
        compute_pre_slip_support_motion(
            load_fraction=0.90,
            pre_slip_displacement_limit=0.002,
            direction=1.0,
        )
    )

    assert (
        abs(near_slip)
        > abs(far_from_slip)
    )


def test_pre_slip_support_motion_is_bounded():
    displacement = (
        compute_pre_slip_support_motion(
            load_fraction=2.0,
            pre_slip_displacement_limit=0.002,
            direction=1.0,
        )
    )

    assert (
        abs(displacement)
        <= 0.002
    )


def test_pre_slip_support_motion_preserves_direction():
    positive = (
        compute_pre_slip_support_motion(
            load_fraction=0.5,
            pre_slip_displacement_limit=0.002,
            direction=1.0,
        )
    )

    negative = (
        compute_pre_slip_support_motion(
            load_fraction=0.5,
            pre_slip_displacement_limit=0.002,
            direction=-1.0,
        )
    )

    assert positive > 0
    assert negative < 0


def test_higher_support_gain_increases_compliant_tip_motion():
    low_gain = make_params(
        support_gain=0.5,
    )

    high_gain = make_params(
        support_gain=2.0,
    )

    action = Action(
        force=5.0,
        angle_deg=45.0,
    )

    low_result = run_safe_probe(
        params=low_gain,
        action=action,
        support_model="compliant",
        force_noise_std=0.0,
        motion_noise_std=0.0,
    )

    high_result = run_safe_probe(
        params=high_gain,
        action=action,
        support_model="compliant",
        force_noise_std=0.0,
        motion_noise_std=0.0,
    )

    assert (
        abs(
            high_result.observation.tip_dx
        )
        >
        abs(
            low_result.observation.tip_dx
        )
    )


def test_support_gain_does_not_affect_rigid_support():
    low_gain = make_params(
        support_gain=0.5,
    )

    high_gain = make_params(
        support_gain=2.0,
    )

    action = Action(
        force=5.0,
        angle_deg=45.0,
    )

    low_result = run_safe_probe(
        params=low_gain,
        action=action,
        support_model="rigid",
        force_noise_std=0.0,
        motion_noise_std=0.0,
    )

    high_result = run_safe_probe(
        params=high_gain,
        action=action,
        support_model="rigid",
        force_noise_std=0.0,
        motion_noise_std=0.0,
    )

    assert (
        abs(
            low_result.observation.tip_dx
            - high_result.observation.tip_dx
        )
        < 1e-12
    )

    assert (
        abs(
            low_result.observation.tip_dy
            - high_result.observation.tip_dy
        )
        < 1e-12
    )


def test_contact_stiffness_and_support_gain_act_independently():
    soft_low_gain = make_params(
        contact_stiffness=10_000.0,
        support_gain=0.5,
    )

    stiff_high_gain = make_params(
        contact_stiffness=40_000.0,
        support_gain=2.0,
    )

    action = Action(
        force=5.0,
        angle_deg=45.0,
    )

    soft_low_result = run_safe_probe(
        params=soft_low_gain,
        action=action,
        support_model="compliant",
        force_noise_std=0.0,
        motion_noise_std=0.0,
    )

    stiff_high_result = run_safe_probe(
        params=stiff_high_gain,
        action=action,
        support_model="compliant",
        force_noise_std=0.0,
        motion_noise_std=0.0,
    )

    assert (
        soft_low_result.observation.tip_dy
        != stiff_high_result.observation.tip_dy
    )

    assert (
        soft_low_gain.contact_stiffness
        != stiff_high_gain.contact_stiffness
    )

    assert (
        soft_low_gain.support_gain
        != stiff_high_gain.support_gain
    )


def test_nonpositive_support_gain_is_rejected():
    params = make_params(
        support_gain=0.0,
    )

    try:
        run_safe_probe(
            params=params,
            action=Action(
                force=5.0,
                angle_deg=45.0,
            ),
            support_model="compliant",
            force_noise_std=0.0,
            motion_noise_std=0.0,
        )
    except ValueError as error:
        assert (
            "support_gain"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected nonpositive support gain to fail"
        )