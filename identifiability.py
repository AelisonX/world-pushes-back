import random
from collections import Counter

from physics import (
    Action,
    NextTransition,
    PhysicalParams,
    run_safe_probe,
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split


NUM_SAMPLES = 6000


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


def collect_safe_probe_data(
    num_samples: int = NUM_SAMPLES,
):
    probe = Action(
        force=5.0,
        angle_deg=45.0,
    )

    results = []

    for _ in range(num_samples):
        params = sample_params()

        result = run_safe_probe(
            params,
            probe,
        )

        # Phase 0 only studies observations obtained
        # before either physical transition occurs.
        if not result.still_blocked:
            continue

        results.append(
            {
                "label": result.next_transition.value,
                "fx": result.observation.fx,
                "fy": result.observation.fy,
                "tip_dx": result.observation.tip_dx,
                "tip_dy": result.observation.tip_dy,
            }
        )

    return results


def summarize_labels(results):
    counts = Counter(
        item["label"]
        for item in results
    )

    print("=== Future transition labels ===")

    for transition in NextTransition:
        print(
            f"{transition.value}: "
            f"{counts.get(transition.value, 0)}"
        )

    print()


def balance_dataset(results):
    grouped = {}

    for item in results:
        grouped.setdefault(
            item["label"],
            [],
        ).append(item)

    available = [
        len(group)
        for group in grouped.values()
    ]

    if len(available) < 2:
        raise RuntimeError(
            "Need at least two transition classes."
        )

    target_size = min(available)

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


def build_dataset(results):
    x = []
    y = []

    for item in results:
        x.append(
            [
                item["fx"],
                item["fy"],
                item["tip_dx"],
                item["tip_dy"],
            ]
        )

        y.append(
            item["label"]
        )

    return x, y


def evaluate_separability(results):
    balanced = balance_dataset(
        results,
    )

    x, y = build_dataset(
        balanced,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    classifier = LogisticRegression(
        max_iter=2000,
    )

    classifier.fit(
        x_train,
        y_train,
    )

    predictions = classifier.predict(
        x_test,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print(
        "=== Pre-transition empirical separability ==="
    )

    print(
        f"Balanced accuracy: {accuracy:.3f}"
    )

    print(
        "Chance baseline: 0.500"
    )

    print()

    print(
        classification_report(
            y_test,
            predictions,
            digits=3,
        )
    )


def main():
    random.seed(42)

    results = collect_safe_probe_data()

    summarize_labels(
        results,
    )

    evaluate_separability(
        results,
    )


if __name__ == "__main__":
    main()