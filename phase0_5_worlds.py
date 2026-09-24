import math
import random

from physics import PhysicalParams


DEFAULT_CONTACT_STIFFNESS_MIN = 10_000.0
DEFAULT_CONTACT_STIFFNESS_MAX = 40_000.0


def sample_log_uniform(
    low: float,
    high: float,
) -> float:
    """
    Sample a positive value uniformly in log space.

    This is useful for physical scale parameters such as stiffness,
    where multiplicative variation is more natural than additive
    variation.
    """

    if low <= 0:
        raise ValueError(
            "low must be positive"
        )

    if high <= low:
        raise ValueError(
            "high must be greater than low"
        )

    log_low = math.log(
        low
    )

    log_high = math.log(
        high
    )

    return math.exp(
        random.uniform(
            log_low,
            log_high,
        )
    )


def sample_phase0_5_world(
    contact_stiffness_min: float = (
        DEFAULT_CONTACT_STIFFNESS_MIN
    ),
    contact_stiffness_max: float = (
        DEFAULT_CONTACT_STIFFNESS_MAX
    ),
) -> PhysicalParams:
    """
    Generate one exploratory Phase 0.5 world.

    At this stage the only new randomized nuisance variable is
    contact_stiffness.

    The numerical stiffness range is exploratory and is NOT yet a
    frozen confirmatory Phase 0.5 range.
    """

    contact_stiffness = (
        sample_log_uniform(
            contact_stiffness_min,
            contact_stiffness_max,
        )
    )

    return PhysicalParams(
        material_yield_strength=random.uniform(
            80_000,
            180_000,
        ),
        contact_area=random.uniform(
            0.00005,
            0.00015,
        ),
        container_mass=random.uniform(
            5.0,
            30.0,
        ),
        static_friction=random.uniform(
            0.15,
            0.60,
        ),
        kinetic_friction=random.uniform(
            0.10,
            0.50,
        ),
        contact_stiffness=(
            contact_stiffness
        ),
    )


def main():
    random.seed(
        42
    )

    print(
        "=== Phase 0.5 exploratory world generator ==="
    )

    print(
        "contact_stiffness_N_per_m"
    )

    for _ in range(10):
        world = (
            sample_phase0_5_world()
        )

        print(
            f"{world.contact_stiffness:.3f}"
        )


if __name__ == "__main__":
    main()