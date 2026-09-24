import random
from collections import Counter

from physics import (
    Action,
    ContactMode,
    PhysicalParams,
    resolve_step,
)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


NUM_SAMPLES = 3000


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
                "mode": result.mode.value,
                "fx": result.observation.fx,
                "fy": result.observation.fy,
                "tip_dx": result.observation.tip_dx,
                "tip_dy": result.observation.tip_dy,
            }
        )

    return results


def summarize_modes(results):
    counts = Counter(item["mode"] for item in results)

    print("=== Mode counts ===")

    for mode in ContactMode:
        print(
            f"{mode.value}: "
            f"{counts.get(mode.value, 0)}"
        )

    print()


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
        y.append(item["mode"])

    return x, y


def evaluate_separability(results):
    x, y = build_dataset(results)

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

    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print("=== Empirical separability ===")
    print(f"Accuracy: {accuracy:.3f}")
    print()

    print("=== Classification report ===")
    print(
        classification_report(
            y_test,
            predictions,
            digits=3,
        )
    )


def main():
    random.seed(42)

    results = collect_results()

    summarize_modes(results)

    evaluate_separability(results)


if __name__ == "__main__":
    main()