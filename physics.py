from dataclasses import dataclass
from enum import Enum
import math
import random


GRAVITY = 9.81


class ContactMode(str, Enum):
    CONTACT_BLOCKED = "CONTACT_BLOCKED"
    MATERIAL_YIELD = "MATERIAL_YIELD"
    SUPPORT_SLIP = "SUPPORT_SLIP"


@dataclass
class PhysicalParams:
    material_yield_strength: float
    contact_area: float
    container_mass: float
    static_friction: float
    kinetic_friction: float


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
class StepResult:
    mode: ContactMode
    observation: Observation
    material_yield_threshold: float
    support_slip_threshold: float


def add_noise(value: float, std: float) -> float:
    return value + random.gauss(0.0, std)


def resolve_step(
    params: PhysicalParams,
    action: Action,
    force_noise_std: float = 0.25,
    motion_noise_std: float = 0.0005,
) -> StepResult:
    """
    Resolve one simplified quasi-static contact step.

    angle_deg:
        0 degrees = horizontal push
        90 degrees = straight downward push

    The simulator internally knows the true thresholds.
    The agent only receives tool-side observations.
    """

    theta = math.radians(action.angle_deg)

    fx = action.force * math.cos(theta)
    fy = action.force * math.sin(theta)

    material_yield_threshold = (
        params.material_yield_strength * params.contact_area
    )

    support_normal_force = (
        params.container_mass * GRAVITY + fy
    )

    support_slip_threshold = (
        params.static_friction * support_normal_force
    )

    material_yields = fy >= material_yield_threshold
    support_slips = abs(fx) >= support_slip_threshold

    if support_slips and not material_yields:
        mode = ContactMode.SUPPORT_SLIP

    elif material_yields and not support_slips:
        mode = ContactMode.MATERIAL_YIELD

    elif material_yields and support_slips:
        yield_ratio = (
            fy / material_yield_threshold
            if material_yield_threshold > 0
            else float("inf")
        )

        slip_ratio = (
            abs(fx) / support_slip_threshold
            if support_slip_threshold > 0
            else float("inf")
        )

        if yield_ratio >= slip_ratio:
            mode = ContactMode.MATERIAL_YIELD
        else:
            mode = ContactMode.SUPPORT_SLIP

    else:
        mode = ContactMode.CONTACT_BLOCKED

    # Simplified tool-tip motion model.
    #
    # These are deliberately crude placeholders.
    # The important distinction is that the agent receives
    # noisy tool-side motion rather than hidden thresholds.

    if mode == ContactMode.CONTACT_BLOCKED:
        tip_dx = 0.0005
        tip_dy = 0.0005

    elif mode == ContactMode.MATERIAL_YIELD:
        tip_dx = 0.001
        tip_dy = -0.010

    elif mode == ContactMode.SUPPORT_SLIP:
        tip_dx = 0.010
        tip_dy = -0.001

    else:
        raise ValueError(f"Unknown mode: {mode}")

    observation = Observation(
        fx=add_noise(fx, force_noise_std),
        fy=add_noise(fy, force_noise_std),
        tip_dx=add_noise(tip_dx, motion_noise_std),
        tip_dy=add_noise(tip_dy, motion_noise_std),
    )

    return StepResult(
        mode=mode,
        observation=observation,
        material_yield_threshold=material_yield_threshold,
        support_slip_threshold=support_slip_threshold,
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

    action = Action(
        force=30.0,
        angle_deg=60.0,
    )

    result = resolve_step(params, action)

    print("True mode:", result.mode.value)
    print()
    print("Agent observation:")
    print("Fx:", round(result.observation.fx, 3), "N")
    print("Fy:", round(result.observation.fy, 3), "N")
    print("tip_dx:", round(result.observation.tip_dx, 5), "m")
    print("tip_dy:", round(result.observation.tip_dy, 5), "m")
    print()
    print("Researcher-only hidden values:")
    print(
        "Material yield threshold:",
        round(result.material_yield_threshold, 2),
        "N",
    )
    print(
        "Support slip threshold:",
        round(result.support_slip_threshold, 2),
        "N",
    )