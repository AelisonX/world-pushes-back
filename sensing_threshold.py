import random

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from physics import (
    Action,
    NextTransition,
    PhysicalParams,
    run_safe_probe,
)


TARGET_SAMPLES_PER_CLASS = 250

MAX_ATTEMPTS = 100_000

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

    grouped = {
        NextTransition.MATERIAL_YIELD.value: [],
        NextTransition.SUPPORT_SLIP.value: [],
    }

    attempts = 0

    while attempts < MAX_ATTEMPTS:
        attempts += 1

        if all(
            len(group) >= TARGET_SAMPLES_PER_CLASS
            for group in grouped.values()
        ):
            break

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

        label = result.next_transition.value

        if (
            len(grouped[label])
            >= TARGET_SAMPLES_PER_CLASS
        ):
            continue

        grouped[label].append(
            {
                "label": label,
                "tip_dx": result.observation.tip_dx,
            }
        )

    samples = (
        grouped[
            NextTransition.MATERIAL_YIELD.value
        ]
        +
        grouped[
            NextTransition.SUPPORT_SLIP.value
        ]
    )

    counts = {
        label: len(group)
        for label, group in grouped.items()
    }

    return (
        samples,
        counts,
        attempts,
    )


def evaluate_accuracy(samples):
    labels = {
        sample["label"]
        for sample in samples
    }

    if len(labels) < 2:
        return None

    class_counts = {}

    for sample in samples:
        label = sample["label"]

        class_counts[label] = (
            class_counts.get(
                label,
                0,
            )
            + 1
        )

    smallest_class = min(
        class_counts.values()
    )

    if smallest_class < 5:
        return None

    x = [
        [item["tip_dx"]]
        for item in samples
    ]

    y = [
        item["label"]
        for item in samples
    ]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
        ),
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
        float(
            np.mean(scores)
        ),
        float(
            np.std(scores)
        ),
    )


def run_sweep():
    print(
        "=== Minimum Sensing Requirement Sweep ==="
    )

    print(
        f"Probe force: {PROBE_FORCE:.1f} N"
    )

    print(
        f"Target samples per class: "
        f"{TARGET_SAMPLES_PER_CLASS}"
    )

    print()

    print(
        "pre_slip_mm | noise_mm | accuracy | std | "
        "yield_n | slip_n | attempts"
    )

    print(
        "---------------------------------------------------------------"
    )

    for pre_slip_limit in PRE_SLIP_LIMITS:
        for noise_level in MOTION_NOISE_LEVELS:
            random.seed(42)

            (
                samples,
                counts,
                attempts,
            ) = collect_samples(
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
                (
                    mean_accuracy,
                    std_accuracy,
                ) = result

                accuracy_text = (
                    f"{mean_accuracy:.3f}"
                )

                std_text = (
                    f"{std_accuracy:.3f}"
                )

            yield_count = counts[
                NextTransition.MATERIAL_YIELD.value
            ]

            slip_count = counts[
                NextTransition.SUPPORT_SLIP.value
            ]

            print(
                f"{pre_slip_limit * 1000:11.3f} | "
                f"{noise_level * 1000:8.3f} | "
                f"{accuracy_text:8} | "
                f"{std_text:5} | "
                f"{yield_count:7d} | "
                f"{slip_count:6d} | "
                f"{attempts:8d}"
            )


def main():
    run_sweep()


if __name__ == "__main__":
    main()