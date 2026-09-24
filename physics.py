from dataclasses import dataclass
from enum import Enum
import math
import random


GRAVITY = 9.81


class NextTransition(str, Enum):
    MATERIAL_YIELD = "MATERIAL_YIELD"
    SUPPORT_SLIP = "SUPPORT_SLIP"


@dataclass
class PhysicalParams:
    material_yield_strength: float
    contact_area: float
    container_mass: float
    static_friction: float
    kinetic_friction: float

    # Generic local tool/contact stiffness.
    #
    # Phase 0 used a fixed 20,000 N/m value.
    # Phase 0.5 promotes this to a world parameter so local
    # deformation can vary independently of support-slip proximity.
    contact_stiffness: float = 20_000.0

    # Maximum tiny support displacement allowed before slip.
    #
    # This is NOT full container sliding.
    # It represents small pre-slip compliance from things such as
    # rubber feet, mounts, wheels, or deformable support contact.
    pre_slip_displacement_limit: float = 0.002


@dataclass
class Action:
    force: float
    angle_deg: float


@dataclass
class Observation:
    fx: float
    fy: float
    tip_dx: float
    tip_dy: float


@dataclass
class ProbeResult:
    observation: Observation
    still_blocked: bool
    material_transition_force: float
    slip_transition_force: float
    next_transition: NextTransition


def add_noise(
    value: float,
    std: float,
) -> float:
    return (
        value
        + random.gauss(
            0.0,
            std,
        )
    )


def material_transition_force(
    params: PhysicalParams,
    angle_deg: float,
) -> float:
    """
    Total applied force required to reach material yield
    at a fixed force angle.
    """

    theta = math.radians(
        angle_deg
    )

    vertical_fraction = math.sin(
        theta
    )

    if vertical_fraction <= 0:
        return float(
            "inf"
        )

    yield_force = (
        params.material_yield_strength
        * params.contact_area
    )

    return (
        yield_force
        / vertical_fraction
    )


def slip_transition_force(
    params: PhysicalParams,
    angle_deg: float,
) -> float:
    """
    Total applied force required to reach support slip.

    Slip begins when horizontal friction demand exceeds
    available static-friction capacity:

        F*cos(theta)
        >
        mu_s * (m*g + F*sin(theta))
    """

    theta = math.radians(
        angle_deg
    )

    denominator = (
        math.cos(theta)
        - (
            params.static_friction
            * math.sin(theta)
        )
    )

    if denominator <= 0:
        return float(
            "inf"
        )

    return (
        params.static_friction
        * params.container_mass
        * GRAVITY
        / denominator
    )


def predict_next_transition(
    params: PhysicalParams,
    angle_deg: float,
) -> tuple[
    NextTransition,
    float,
    float,
]:
    material_force = material_transition_force(
        params,
        angle_deg,
    )

    slip_force = slip_transition_force(
        params,
        angle_deg,
    )

    if material_force <= slip_force:
        next_transition = (
            NextTransition.MATERIAL_YIELD
        )
    else:
        next_transition = (
            NextTransition.SUPPORT_SLIP
        )

    return (
        next_transition,
        material_force,
        slip_force,
    )


def support_load_fraction(
    params: PhysicalParams,
    true_fx: float,
    true_fy: float,
) -> float:
    """
    Return static-friction demand divided by capacity.

    demand:
        abs(Fx)

    capacity:
        mu_s * normal_force

    where:

        normal_force = m*g + Fy

    Under the model used here:

        load_fraction < 1
            support remains below the ideal slip threshold

        load_fraction >= 1
            ideal support-slip threshold has been reached

    The returned value is not clipped so tests and diagnostics
    can inspect overload conditions explicitly.
    """

    normal_force = (
        params.container_mass
        * GRAVITY
        + true_fy
    )

    if normal_force <= 0:
        return float(
            "inf"
        )

    friction_capacity = (
        params.static_friction
        * normal_force
    )

    if friction_capacity <= 0:
        return float(
            "inf"
        )

    return (
        abs(true_fx)
        / friction_capacity
    )


def compute_pre_slip_support_motion(
    load_fraction: float,
    pre_slip_displacement_limit: float,
    direction: float,
) -> float:
    """
    Simple pre-slip support-compliance model.

    Tiny support displacement grows as friction demand approaches
    static-friction capacity.

    This is an explicit toy-model assumption for the compliant
    ablation. It is NOT a claim that real static friction always
    produces this linear displacement law.
    """

    if (
        load_fraction <= 0
        or math.isnan(load_fraction)
    ):
        return 0.0

    bounded_fraction = min(
        load_fraction,
        1.0,
    )

    displacement = (
        pre_slip_displacement_limit
        * bounded_fraction
    )

    if direction == 0:
        return 0.0

    return math.copysign(
        displacement,
        direction,
    )


def run_safe_probe(
    params: PhysicalParams,
    action: Action,
    force_noise_std: float = 0.20,
    motion_noise_std: float = 0.0002,
    support_model: str = "rigid",
) -> ProbeResult:
    """
    Apply a probe before either transition occurs.

    support_model:
        "rigid"
            Idealized Coulomb support.
            No pre-slip support motion.

        "compliant"
            Support produces a tiny modeled pre-slip displacement
            as static-friction demand approaches capacity.

    The agent receives only noisy tool-side observations.

    Hidden transition thresholds are returned for researcher
    evaluation only.
    """

    if support_model not in {
        "rigid",
        "compliant",
    }:
        raise ValueError(
            "support_model must be "
            "'rigid' or 'compliant'"
        )

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

    (
        next_transition,
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

    still_blocked = (
        action.force
        < first_transition_force
    )

    # Generic local contact compliance.
    #
    # Phase 0.5 treats stiffness as a world-specific nuisance
    # parameter rather than a globally fixed constant.
    if params.contact_stiffness <= 0:
        raise ValueError(
            "contact_stiffness must be positive"
        )

        local_tip_dx = (
        true_fx
        / params.contact_stiffness
    )

    local_tip_dy = (
        -true_fy
        / params.contact_stiffness
    )

        support_dx = 0.0

    if support_model == "compliant":
        load_fraction = (
            support_load_fraction(
                params=params,
                true_fx=true_fx,
                true_fy=true_fy,
            )
        )

        support_dx = (
            compute_pre_slip_support_motion(
                load_fraction=load_fraction,
                pre_slip_displacement_limit=(
                    params.pre_slip_displacement_limit
                ),
                direction=true_fx,
            )
        )

    # World-frame tool-tip displacement contains:
    #
    # 1. local contact compliance
    # 2. modeled support motion transmitted through contact
    tip_dx = (
        local_tip_dx
        + support_dx
    )

    tip_dy = (
        local_tip_dy
    )

    observation = Observation(
        fx=add_noise(
            true_fx,
            force_noise_std,
        ),
        fy=add_noise(
            true_fy,
            force_noise_std,
        ),
        tip_dx=add_noise(
            tip_dx,
            motion_noise_std,
        ),
        tip_dy=add_noise(
            tip_dy,
            motion_noise_std,
        ),
    )

    return ProbeResult(
        observation=observation,
        still_blocked=still_blocked,
        material_transition_force=(
            material_force
        ),
        slip_transition_force=(
            slip_force
        ),
        next_transition=(
            next_transition
        ),
    )


if __name__ == "__main__":
    random.seed(
        42
    )

    params = PhysicalParams(
        material_yield_strength=120_000,
        contact_area=0.0001,
        container_mass=20.0,
        static_friction=0.35,
        kinetic_friction=0.25,
    )

    probe = Action(
        force=5.0,
        angle_deg=45.0,
    )

    for support_model in [
        "rigid",
        "compliant",
    ]:
        result = run_safe_probe(
            params,
            probe,
            support_model=support_model,
        )

        print(
            f"=== "
            f"{support_model.upper()} "
            f"SUPPORT ==="
        )

        print(
            "Still blocked:",
            result.still_blocked,
        )

        print(
            "Next transition:",
            result.next_transition.value,
        )

        print(
            "Fx:",
            round(
                result.observation.fx,
                3,
            ),
            "N",
        )

        print(
            "Fy:",
            round(
                result.observation.fy,
                3,
            ),
            "N",
        )

        print(
            "tip_dx:",
            round(
                result.observation.tip_dx,
                6,
            ),
            "m",
        )

        print(
            "tip_dy:",
            round(
                result.observation.tip_dy,
                6,
            ),
            "m",
        )

        print()