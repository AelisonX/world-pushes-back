import random

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from physics import (
    Action,
    PhysicalParams,
    run_safe_probe,
)


NUM_SAMPLES_PER_FORCE = 4000

PROBE_FORCES = [
    2.0,
    5.0,
    8.0,
    11.0,
    14.0,
    17.0,
    20.0,
    23.0,
    26.0,
    29.0,
    32.0,
]


def sample_params() -> PhysicalParams:
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
    )


def collect_force_data(
    probe_force: float,
    num_samples: int = NUM_SAMPLES_PER_FORCE,
):
    safe_samples = []
    transition_count = 0

    action = Action(
        force=probe_force,
        angle_deg=45.0,
    )

    for _ in range(num_samples):
        params = sample_params()

        result = run_safe_probe(
            params,
            action,
        )

        if not result.still_blocked:
            transition_count += 1
            continue

        safe_samples.append(
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
        transition_count / num_samples
    )

    return safe_samples, transition_rate


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

    if target_size < 20:
        return []

    balanced = []

    for group in grouped.values():
        balanced.extend(
            random.sample(
                group,
                target_size,
            )
        )

    random.shuffle(balanced)

    return balanced


def estimate_accuracy(samples):
    balanced = balance_samples(
        samples,
    )

    if not balanced:
        return None

    x = [
        sample["features"]
        for sample in balanced
    ]

    y = [
        sample["label"]
        for sample in balanced
    ]

    x_train, x_test, y_train, y_test = train_test_split(
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
        x_test,
    )

    return accuracy_score(
        y_test,
        predictions,
    )


def run_sweep():
    print(
        "=== Information vs Safety Sweep ==="
    )

    print(
        "force_N | separability | transition_rate | safe_samples"
    )

    print(
        "--------------------------------------------------------"
    )

    for probe_force in PROBE_FORCES:
        samples, transition_rate = collect_force_data(
            probe_force
        )

        accuracy = estimate_accuracy(
            samples
        )

        if accuracy is None:
            accuracy_text = "N/A"
        else:
            accuracy_text = f"{accuracy:.3f}"

        print(
            f"{probe_force:7.1f} | "
            f"{accuracy_text:12} | "
            f"{transition_rate:15.3f} | "
            f"{len(samples):12d}"
        )


def main():
    random.seed(42)

    run_sweep()


if __name__ == "__main__":
    main()