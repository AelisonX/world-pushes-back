from dataclasses import dataclass
from enum import Enum
import math


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
class StepResult:
    mode: ContactMode
    normal_force: float
    horizontal_force: float
    material_yield_threshold: float
    support_slip_threshold: float


def resolve_step(params: PhysicalParams, action: Action) -> StepResult:
    """
    Resolve one simplified quasi-static contact step.

    angle_deg:
        0 degrees = horizontal push
        90 degrees = straight downward push

    The model compares two thresholds:

    1. Material yield:
       normal force must exceed material_yield_strength * contact_area

    2. Support slip:
       horizontal force must exceed available static friction
    """

    theta = math.radians(action.angle_deg)

    horizontal_force = action.force * math.cos(theta)
    downward_force = action.force * math.sin(theta)

    material_yield_threshold = (
        params.material_yield_strength * params.contact_area
    )

    support_normal_force = (
        params.container_mass * GRAVITY + downward_force
    )

    support_slip_threshold = (
        params.static_friction * support_normal_force
    )

    material_yields = downward_force >= material_yield_threshold
    support_slips = abs(horizontal_force) >= support_slip_threshold

    if support_slips and not material_yields:
        mode = ContactMode.SUPPORT_SLIP

    elif material_yields and not support_slips:
        mode = ContactMode.MATERIAL_YIELD

    elif material_yields and support_slips:
        # Both thresholds are crossed in this simplified single-step model.
        # For now, choose the transition with the smaller normalized margin.
        yield_ratio = (
            downward_force / material_yield_threshold
            if material_yield_threshold > 0
            else float("inf")
        )

        slip_ratio = (
            abs(horizontal_force) / support_slip_threshold
            if support_slip_threshold > 0
            else float("inf")
        )

        if yield_ratio >= slip_ratio:
            mode = ContactMode.MATERIAL_YIELD
        else:
            mode = ContactMode.SUPPORT_SLIP

    else:
        mode = ContactMode.CONTACT_BLOCKED

    return StepResult(
        mode=mode,
        normal_force=downward_force,
        horizontal_force=horizontal_force,
        material_yield_threshold=material_yield_threshold,
        support_slip_threshold=support_slip_threshold,
    )


if __name__ == "__main__":
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

    print("Mode:", result.mode.value)
    print("Normal force:", round(result.normal_force, 2), "N")
    print("Horizontal force:", round(result.horizontal_force, 2), "N")
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