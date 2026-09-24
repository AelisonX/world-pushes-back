import random

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from physics import (
    Action,
    PhysicalParams,
    run_safe_probe,
)


NUM_SAMPLES = 6000

PROBE_FORCES = [
    5.0,
    11.0,
    17.0,
    23.0,
]

FEATURE_SETS = {
    "force_only": [
        "fx",
        "fy",
    ],
    "motion_only": [
        "tip_dx",
        "tip_dy",
    ],
    "tip_dx_only": [
        "tip_dx",
    ],
    "full": [
        "fx",
        "fy",
        "tip_dx",
        "tip_dy",
    ],
}


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


def collect_samples(
    probe_force: float,
    support_model: str,
):
    action = Action(
        force=probe_force,
        angle_deg=45.0,
    )

    samples = []

    for _ in range(NUM_SAMPLES):
        params = sample_params()

        result = run_safe_probe(
            params=params,
            action=action,
            support_model=support_model,
        )

        if not result.still_blocked:
            continue

        observation = result.observation

        samples.append(
            {
                "label": result.next_transition.value,
                "fx": observation.fx,
                "fy": observation.fy,
                "tip_dx": observation.tip_dx,
                "tip_dy": observation.tip_dy,
            }
        )

    return samples


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


def build_xy(
    samples,
    feature_names,
):
    x = []
    y = []

    for sample in samples:
        x.append(
            [
                sample[name]
                for name in feature_names
            ]
        )

        y.append(
            sample["label"]
        )

    return x, y


def evaluate_feature_set(
    samples,
    feature_names,
):
    x, y = build_xy(
        samples,
        feature_names,
    )

    model = LogisticRegression(
        max_iter=2000,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        model,
        x,
        y,
        cv=cv,
        scoring="accuracy",
    )

    return (
        float(np.mean(scores)),
        float(np.std(scores)),
    )


def run_ablation(
    support_model: str,
):
    print()
    print(
        f"=== {support_model.upper()} SUPPORT ==="
    )

    for probe_force in PROBE_FORCES:
        print()
        print(
            f"Probe force: {probe_force:.1f} N"
        )

        samples = collect_samples(
            probe_force=probe_force,
            support_model=support_model,
        )

        balanced = balance_samples(
            samples
        )

        if not balanced:
            print(
                "Not enough balanced samples."
            )
            continue

        print(
            f"Balanced samples: {len(balanced)}"
        )

        for (
            feature_set_name,
            feature_names,
        ) in FEATURE_SETS.items():
            (
                mean_accuracy,
                std_accuracy,
            ) = evaluate_feature_set(
                samples=balanced,
                feature_names=feature_names,
            )

            print(
                f"{feature_set_name:14} "
                f"accuracy="
                f"{mean_accuracy:.3f} "
                f"+/- {std_accuracy:.3f}"
            )


def main():
    print(
        "=== Sensor Ablation ==="
    )

    random.seed(42)

    run_ablation(
        support_model="rigid"
    )

    random.seed(42)

    run_ablation(
        support_model="compliant"
    )


if __name__ == "__main__":
    main()