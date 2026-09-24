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


def add_noise(value: float, std: float) -> float:
    return value + random.gauss(0.0, std)


def material_transition_force(
    params: PhysicalParams,
    angle_deg: float,
) -> float:
    """
    Total applied force required to reach material yield
    at a fixed force angle.
    """

    theta = math.radians(angle_deg)
    vertical_fraction = math.sin(theta)

    if vertical_fraction <= 0:
        return float("inf")

    yield_force = (
        params.material_yield_strength
        * params.contact_area
    )

    return yield_force / vertical_fraction


def slip_transition_force(
    params: PhysicalParams,
    angle_deg: float,
) -> float:
    """
    Total applied force required to reach support slip.

    Slip begins when:

    F*cos(theta)
    >
    mu_s * (m*g + F*sin(theta))
    """

    theta = math.radians(angle_deg)

    denominator = (
        math.cos(theta)
        - params.static_friction * math.sin(theta)
    )

    if denominator <= 0:
        return float("inf")

    return (
        params.static_friction
        * params.container_mass
        * GRAVITY
        / denominator
    )


def predict_next_transition(
    params: PhysicalParams,
    angle_deg: float,
) -> tuple[NextTransition, float, float]:
    material_force = material_transition_force(
        params,
        angle_deg,
    )

    slip_force = slip_transition_force(
        params,
        angle_deg,
    )

    if material_force <= slip_force:
        next_transition = NextTransition.MATERIAL_YIELD
    else:
        next_transition = NextTransition.SUPPORT_SLIP

    return (
        next_transition,
        material_force,
        slip_force,
    )


def compute_pre_slip_support_motion(
    true_fx: float,
    slip_force: float,
    pre_slip_displacement_limit: float,
) -> float:
    """
    Simple pre-slip support-compliance model.

    The closer horizontal loading gets to the support-slip
    threshold, the larger the tiny support displacement becomes.

    This is an explicit modeling assumption for the compliant
    ablation. It is not part of ideal Coulomb friction.
    """

    if slip_force <= 0 or math.isinf(slip_force):
        return 0.0

    load_fraction = min(
        abs(true_fx) / slip_force,
        1.0,
    )

    displacement = (
        pre_slip_displacement_limit
        * load_fraction
    )

    return math.copysign(
        displacement,
        true_fx,
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
            Idealized Coulomb support. No pre-slip motion.

        "compliant"
            Support produces a tiny pre-slip displacement
            that grows as loading approaches the slip threshold.

    The agent receives only noisy tool-side observations.

    Hidden transition thresholds are returned for researcher
    evaluation only.
    """

    if support_model not in {
        "rigid",
        "compliant",
    }:
        raise ValueError(
            "support_model must be 'rigid' or 'compliant'"
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
    # This is intentionally independent of the hidden
    # material/slip thresholds.
    contact_stiffness = 20_000.0

    local_tip_dx = (
        true_fx / contact_stiffness
    )

    local_tip_dy = (
        -true_fy / contact_stiffness
    )

    support_dx = 0.0

    if support_model == "compliant":
        support_dx = compute_pre_slip_support_motion(
            true_fx=true_fx,
            slip_force=slip_force,
            pre_slip_displacement_limit=(
                params.pre_slip_displacement_limit
            ),
        )

    # World-frame tool-tip displacement contains both:
    #
    # 1. local contact compliance
    # 2. any tiny support motion transmitted through contact
    tip_dx = (
        local_tip_dx
        + support_dx
    )

    tip_dy = local_tip_dy

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
        material_transition_force=material_force,
        slip_transition_force=slip_force,
        next_transition=next_transition,
    )


if __name__ == "__main__":
    random.seed(42)

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
            f"=== {support_model.upper()} SUPPORT ==="
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