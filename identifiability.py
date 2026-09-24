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
    return PhysicalParams(
        material_yield_strength=random.uniform(80_000, 180_000),
        contact_area=random.uniform(0.00005, 0.00015),
        container_mass=random.uniform(5.0, 30.0),
        static_friction=random.uniform(0.15, 0.60),
        kinetic_friction=random.uniform(0.10, 0.50),
    )


def run_low_force_push(params: PhysicalParams):
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
                "fx": result.observation.fx,
                "fy": result.observation.fy,
                "tip_dx": result.observation.tip_dx,
                "tip_dy": result.observation.tip_dy,
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

    print("=== Example agent observations ===")

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
                f"Fx={item['fx']:.2f} N, "
                f"Fy={item['fy']:.2f} N, "
                f"tip_dx={item['tip_dx']:.5f} m, "
                f"tip_dy={item['tip_dy']:.5f} m"
            )


def main():
    random.seed(42)

    results = collect_results()
    summarize(results)


if __name__ == "__main__":
    main()