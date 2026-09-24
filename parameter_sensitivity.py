import random
import statistics

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from physics import (
    Action,
    PhysicalParams,
    run_safe_probe,
)


NUM_SAMPLES = 5000

PROBE_FORCE = 11.0
PROBE_ANGLE = 45.0

PARAMETER_SCENARIOS = {
    "baseline": {
        "material_yield_strength": (80_000, 180_000),
        "contact_area": (0.00005, 0.00015),
        "container_mass": (5.0, 30.0),
        "static_friction": (0.15, 0.60),
        "kinetic_friction": (0.10, 0.50),
    },

    "light_containers": {
        "material_yield_strength": (80_000, 180_000),
        "contact_area": (0.00005, 0.00015),
        "container_mass": (2.0, 10.0),
        "static_friction": (0.15, 0.60),
        "kinetic_friction": (0.10, 0.50),
    },

    "heavy_containers": {
        "material_yield_strength": (80_000, 180_000),
        "contact_area": (0.00005, 0.00015),
        "container_mass": (20.0, 50.0),
        "static_friction": (0.15, 0.60),
        "kinetic_friction": (0.10, 0.50),
    },

    "low_friction": {
        "material_yield_strength": (80_000, 180_000),
        "contact_area": (0.00005, 0.00015),
        "container_mass": (5.0, 30.0),
        "static_friction": (0.05, 0.25),
        "kinetic_friction": (0.03, 0.20),
    },

    "high_friction": {
        "material_yield_strength": (80_000, 180_000),
        "contact_area": (0.00005, 0.00015),
        "container_mass": (5.0, 30.0),
        "static_friction": (0.50, 0.90),
        "kinetic_friction": (0.35, 0.75),
    },

    "soft_material": {
        "material_yield_strength": (40_000, 100_000),
        "contact_area": (0.00005, 0.00015),
        "container_mass": (5.0, 30.0),
        "static_friction": (0.15, 0.60),
        "kinetic_friction": (0.10, 0.50),
    },

    "hard_material": {
        "material_yield_strength": (160_000, 260_000),
        "contact_area": (0.00005, 0.00015),
        "container_mass": (5.0, 30.0),
        "static_friction": (0.15, 0.60),
        "kinetic_friction": (0.10, 0.50),
    },
}


def sample_from_range(value_range):
    low, high = value_range

    return random.uniform(
        low,
        high,
    )


def sample_params(
    scenario,
) -> PhysicalParams:
    return PhysicalParams(
        material_yield_strength=sample_from_range(
            scenario["material_yield_strength"]
        ),
        contact_area=sample_from_range(
            scenario["contact_area"]
        ),
        container_mass=sample_from_range(
            scenario["container_mass"]
        ),
        static_friction=sample_from_range(
            scenario["static_friction"]
        ),
        kinetic_friction=sample_from_range(
            scenario["kinetic_friction"]
        ),
    )


def collect_samples(
    scenario,
    support_model,
):
    action = Action(
        force=PROBE_FORCE,
        angle_deg=PROBE_ANGLE,
    )

    samples = []
    transition_count = 0

    for _ in range(NUM_SAMPLES):
        params = sample_params(
            scenario
        )

        result = run_safe_probe(
            params=params,
            action=action,
            support_model=support_model,
        )

        if not result.still_blocked:
            transition_count += 1
            continue

        samples.append(
            {
                "label": result.next_transition.value,
                "features": [
                    result.observation.fx,
                    result.observation.fy,
                    result.observation.tip_dx,
                    result.observation.tip_dy,
                ],
            }
        )

    transition_rate = (
        transition_count
        / NUM_SAMPLES
    )

    return (
        samples,
        transition_rate,
    )


def balance_samples(samples):
    grouped = {}

    for sample in samples:
        grouped.setdefault(
            sample["label"],
            [],
        ).append(sample)

    if len(grouped) < 2:
        return []

    target_size = min(
        len(group)
        for group in grouped.values()
    )

    if target_size < 50:
        return []

    balanced = []

    for group in grouped.values():
        balanced.extend(
            random.sample(
                group,
                target_size,
            )
        )

    random.shuffle(
        balanced
    )

    return balanced


def estimate_accuracy(samples):
    balanced = balance_samples(
        samples
    )

    if not balanced:
        return None

    x = [
        item["features"]
        for item in balanced
    ]

    y = [
        item["label"]
        for item in balanced
    ]

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(
        max_iter=2000,
    )

    model.fit(
        x_train,
        y_train,
    )

    predictions = model.predict(
        x_test
    )

    return accuracy_score(
        y_test,
        predictions,
    )


def run_scenario(
    name,
    scenario,
    support_model,
):
    samples, transition_rate = collect_samples(
        scenario=scenario,
        support_model=support_model,
    )

    accuracy = estimate_accuracy(
        samples
    )

    return {
        "name": name,
        "support_model": support_model,
        "accuracy": accuracy,
        "transition_rate": transition_rate,
        "safe_samples": len(samples),
    }


def main():
    random.seed(42)

    print(
        "=== Parameter Sensitivity ==="
    )

    print(
        f"Probe: {PROBE_FORCE:.1f} N "
        f"at {PROBE_ANGLE:.1f} degrees"
    )

    print()

    print(
        "scenario         | support    | accuracy | transition_rate | safe_samples"
    )

    print(
        "-----------------------------------------------------------------------"
    )

    results = []

    for name, scenario in PARAMETER_SCENARIOS.items():
        for support_model in [
            "rigid",
            "compliant",
        ]:
            random.seed(42)

            result = run_scenario(
                name=name,
                scenario=scenario,
                support_model=support_model,
            )

            results.append(
                result
            )

            if result["accuracy"] is None:
                accuracy_text = "N/A"
            else:
                accuracy_text = (
                    f"{result['accuracy']:.3f}"
                )

            print(
                f"{name:16} | "
                f"{support_model:10} | "
                f"{accuracy_text:8} | "
                f"{result['transition_rate']:15.3f} | "
                f"{result['safe_samples']:12d}"
            )

    print()

    valid_accuracies = [
        result["accuracy"]
        for result in results
        if result["accuracy"] is not None
    ]

    if valid_accuracies:
        print(
            "Overall accuracy spread:"
        )

        print(
            f"  mean = "
            f"{statistics.mean(valid_accuracies):.3f}"
        )

        print(
            f"  min  = "
            f"{min(valid_accuracies):.3f}"
        )

        print(
            f"  max  = "
            f"{max(valid_accuracies):.3f}"
        )


if __name__ == "__main__":
    main()