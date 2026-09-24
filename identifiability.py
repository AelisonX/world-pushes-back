import random
from collections import Counter

from physics import (
    Action,
    ContactMode,
    PhysicalParams,
    resolve_step,
)


NUM_SAMPLES = 1000


def sample_params() -> PhysicalParams:
    """
    Randomly sample a small physical world.

    These ranges are intentionally simple.
    They are not intended to represent realistic ice cream yet.
    """

    return PhysicalParams(
        material_yield_strength=random.uniform(80_000, 180_000),
        contact_area=random.uniform(0.00005, 0.00015),
        container_mass=random.uniform(5.0, 30.0),
        static_friction=random.uniform(0.15, 0.60),
        kinetic_friction=random.uniform(0.10, 0.50),
    )


def run_low_force_push(params: PhysicalParams):
    """
    Apply one fixed low-force diagnostic action.
    """

    action = Action(
        force=20.0,
        angle_deg=45.0,
    )

    return resolve_step(params, action)


def collect_results(num_samples: int = NUM_SAMPLES):
    results = []

    for _ in range(num_samples):
        params = sample_params()
        result = run_low_force_push(params)

        results.append(
            {
                "mode": result.mode,
                "normal_force": result.normal_force,
                "horizontal_force": result.horizontal_force,
                "yield_threshold": result.material_yield_threshold,
                "slip_threshold": result.support_slip_threshold,
            }
        )

    return results


def summarize(results):
    counts = Counter(item["mode"] for item in results)

    print("=== Mode counts ===")

    for mode in ContactMode:
        print(
            f"{mode.value}: "
            f"{counts.get(mode, 0)}"
        )

    print()

    print("=== Example observations ===")

    for mode in ContactMode:
        examples = [
            item for item in results
            if item["mode"] == mode
        ][:5]

        print(f"\n{mode.value}")

        if not examples:
            print("  No samples")
            continue

        for item in examples:
            print(
                "  "
                f"normal={item['normal_force']:.2f} N, "
                f"horizontal={item['horizontal_force']:.2f} N, "
                f"yield_threshold={item['yield_threshold']:.2f} N, "
                f"slip_threshold={item['slip_threshold']:.2f} N"
            )


def main():
    random.seed(42)

    results = collect_results()
    summarize(results)


if __name__ == "__main__":
    main()