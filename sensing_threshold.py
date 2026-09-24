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


NUM_SAMPLES = 5000

PROBE_FORCE = 11.0

PRE_SLIP_LIMITS = [
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.002,
    0.004,
]

MOTION_NOISE_LEVELS = [
    0.00005,
    0.0001,
    0.0002,
    0.0005,
    0.001,
]


def sample_params(
    pre_slip_displacement_limit: float,
) -> PhysicalParams:
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
        pre_slip_displacement_limit=(
            pre_slip_displacement_limit
        ),
    )


def collect_samples(
    pre_slip_displacement_limit: float,
    motion_noise_std: float,
):
    action = Action(
        force=PROBE_FORCE,
        angle_deg=45.0,
    )

    samples = []

    for _ in range(NUM_SAMPLES):
        params = sample_params(
            pre_slip_displacement_limit
        )

        result = run_safe_probe(
            params=params,
            action=action,
            support_model="compliant",
            motion_noise_std=motion_noise_std,
        )

        if not result.still_blocked:
            continue

        samples.append(
            {
                "label": result.next_transition.value,
                "tip_dx": result.observation.tip_dx,
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


def evaluate_accuracy(samples):
    balanced = balance_samples(
        samples
    )

    if not balanced:
        return None

    x = [
        [item["tip_dx"]]
        for item in balanced
    ]

    y = [
        item["label"]
        for item in balanced
    ]

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


def run_sweep():
    print(
        "=== Minimum Sensing Requirement Sweep ==="
    )

    print(
        f"Probe force: {PROBE_FORCE:.1f} N"
    )

    print()

    print(
        "pre_slip_mm | noise_mm | accuracy | std"
    )

    print(
        "------------------------------------------"
    )

    for pre_slip_limit in PRE_SLIP_LIMITS:
        for noise_level in MOTION_NOISE_LEVELS:
            random.seed(42)

            samples = collect_samples(
                pre_slip_displacement_limit=(
                    pre_slip_limit
                ),
                motion_noise_std=noise_level,
            )

            result = evaluate_accuracy(
                samples
            )

            if result is None:
                accuracy_text = "N/A"
                std_text = "N/A"
            else:
                mean_accuracy, std_accuracy = result

                accuracy_text = (
                    f"{mean_accuracy:.3f}"
                )

                std_text = (
                    f"{std_accuracy:.3f}"
                )

            print(
                f"{pre_slip_limit * 1000:11.3f} | "
                f"{noise_level * 1000:8.3f} | "
                f"{accuracy_text:8} | "
                f"{std_text}"
            )


def main():
    run_sweep()


if __name__ == "__main__":
    main()