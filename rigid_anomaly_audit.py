import random

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from physics import (
    Action,
    NextTransition,
    PhysicalParams,
    run_safe_probe,
)


NUM_SAMPLES = 12_000

PROBE_FORCES = [
    5.0,
    8.0,
    11.0,
    14.0,
    17.0,
    20.0,
]

BOOTSTRAP_REPEATS = 2000

SEED = 42


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


def collect_rigid_samples(
    probe_force: float,
):
    action = Action(
        force=probe_force,
        angle_deg=45.0,
    )

    samples = []

    unsafe_count = 0

    for world_id in range(
        NUM_SAMPLES
    ):
        params = sample_params()

        result = run_safe_probe(
            params=params,
            action=action,
            support_model="rigid",
        )

        if not result.still_blocked:
            unsafe_count += 1
            continue

        samples.append(
            {
                "world_id": world_id,
                "label": result.next_transition.value,
                "legal": [
                    result.observation.fx,
                    result.observation.fy,
                    result.observation.tip_dx,
                    result.observation.tip_dy,
                ],
                "force_only": [
                    result.observation.fx,
                    result.observation.fy,
                ],
                "motion_only": [
                    result.observation.tip_dx,
                    result.observation.tip_dy,
                ],
                "privileged": [
                    params.container_mass,
                    params.static_friction,
                    params.material_yield_strength,
                    params.contact_area,
                ],
            }
        )

    transition_rate = (
        unsafe_count
        / NUM_SAMPLES
    )

    return (
        samples,
        transition_rate,
    )


def class_counts(samples):
    counts = {
        NextTransition.MATERIAL_YIELD.value: 0,
        NextTransition.SUPPORT_SLIP.value: 0,
    }

    for sample in samples:
        counts[
            sample["label"]
        ] += 1

    return counts


def build_xy(
    samples,
    feature_key,
):
    x = np.asarray(
        [
            sample[feature_key]
            for sample in samples
        ],
        dtype=float,
    )

    y = np.asarray(
        [
            1
            if (
                sample["label"]
                == NextTransition.SUPPORT_SLIP.value
            )
            else 0
            for sample in samples
        ],
        dtype=int,
    )

    return (
        x,
        y,
    )


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
    )


def out_of_fold_scores(
    x,
    y,
):
    unique, counts = np.unique(
        y,
        return_counts=True,
    )

    if len(unique) < 2:
        return None

    minority_count = int(
        counts.min()
    )

    if minority_count < 5:
        return None

    n_splits = min(
        5,
        minority_count,
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    probabilities = cross_val_predict(
        make_model(),
        x,
        y,
        cv=cv,
        method="predict_proba",
    )

    return probabilities[
        :,
        1,
    ]


def bootstrap_auc_ci(
    y,
    scores,
):
    rng = np.random.default_rng(
        SEED
    )

    auc_values = []

    positive_indices = np.where(
        y == 1
    )[0]

    negative_indices = np.where(
        y == 0
    )[0]

    if (
        len(positive_indices) == 0
        or len(negative_indices) == 0
    ):
        return None

    for _ in range(
        BOOTSTRAP_REPEATS
    ):
        sampled_positive = rng.choice(
            positive_indices,
            size=len(
                positive_indices
            ),
            replace=True,
        )

        sampled_negative = rng.choice(
            negative_indices,
            size=len(
                negative_indices
            ),
            replace=True,
        )

        sampled_indices = np.concatenate(
            [
                sampled_positive,
                sampled_negative,
            ]
        )

        bootstrap_y = y[
            sampled_indices
        ]

        bootstrap_scores = scores[
            sampled_indices
        ]

        auc_values.append(
            roc_auc_score(
                bootstrap_y,
                bootstrap_scores,
            )
        )

    lower = float(
        np.percentile(
            auc_values,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            auc_values,
            97.5,
        )
    )

    return (
        lower,
        upper,
    )


def evaluate_feature_set(
    samples,
    feature_key,
):
    x, y = build_xy(
        samples,
        feature_key,
    )

    scores = out_of_fold_scores(
        x,
        y,
    )

    if scores is None:
        return None

    auc = float(
        roc_auc_score(
            y,
            scores,
        )
    )

    ci = bootstrap_auc_ci(
        y,
        scores,
    )

    return (
        auc,
        ci,
    )


def evaluate_shuffle(
    samples,
):
    x, y = build_xy(
        samples,
        "legal",
    )

    rng = np.random.default_rng(
        SEED
    )

    shuffled_y = rng.permutation(
        y
    )

    scores = out_of_fold_scores(
        x,
        shuffled_y,
    )

    if scores is None:
        return None

    return float(
        roc_auc_score(
            shuffled_y,
            scores,
        )
    )


def summarize_observations(
    samples,
):
    grouped = {
        NextTransition.MATERIAL_YIELD.value: [],
        NextTransition.SUPPORT_SLIP.value: [],
    }

    for sample in samples:
        grouped[
            sample["label"]
        ].append(
            sample["legal"]
        )

    summaries = {}

    for label, rows in grouped.items():
        if not rows:
            summaries[label] = None
            continue

        array = np.asarray(
            rows,
            dtype=float,
        )

        summaries[label] = {
            "mean": np.mean(
                array,
                axis=0,
            ),
            "std": np.std(
                array,
                axis=0,
            ),
        }

    return summaries


def format_result(
    result,
):
    if result is None:
        return "N/A"

    auc, ci = result

    if ci is None:
        return (
            f"AUC={auc:.3f}, CI=N/A"
        )

    lower, upper = ci

    return (
        f"AUC={auc:.3f}, "
        f"95% CI=[{lower:.3f}, {upper:.3f}]"
    )


def run_force(
    probe_force: float,
):
    random.seed(
        SEED
    )

    (
        samples,
        transition_rate,
    ) = collect_rigid_samples(
        probe_force
    )

    counts = class_counts(
        samples
    )

    legal_result = evaluate_feature_set(
        samples,
        "legal",
    )

    force_result = evaluate_feature_set(
        samples,
        "force_only",
    )

    motion_result = evaluate_feature_set(
        samples,
        "motion_only",
    )

    privileged_result = evaluate_feature_set(
        samples,
        "privileged",
    )

    shuffled_auc = evaluate_shuffle(
        samples
    )

    summaries = summarize_observations(
        samples
    )

    print()
    print(
        "=" * 72
    )

    print(
        f"Rigid probe force: "
        f"{probe_force:.1f} N"
    )

    print(
        "=" * 72
    )

    print(
        f"Safe samples: "
        f"{len(samples)}"
    )

    print(
        f"Transition rate: "
        f"{transition_rate:.3f}"
    )

    print(
        "MATERIAL_YIELD n: "
        f"{counts[NextTransition.MATERIAL_YIELD.value]}"
    )

    print(
        "SUPPORT_SLIP n: "
        f"{counts[NextTransition.SUPPORT_SLIP.value]}"
    )

    print()

    print(
        "LEGAL       ",
        format_result(
            legal_result
        ),
    )

    print(
        "FORCE_ONLY  ",
        format_result(
            force_result
        ),
    )

    print(
        "MOTION_ONLY ",
        format_result(
            motion_result
        ),
    )

    print(
        "PRIVILEGED  ",
        format_result(
            privileged_result
        ),
    )

    if shuffled_auc is None:
        print(
            "SHUFFLE      AUC=N/A"
        )
    else:
        print(
            "SHUFFLE      "
            f"AUC={shuffled_auc:.3f}"
        )

    print()
    print(
        "Legal observation summaries:"
    )

    feature_names = [
        "Fx",
        "Fy",
        "tip_dx",
        "tip_dy",
    ]

    for label in [
        NextTransition.MATERIAL_YIELD.value,
        NextTransition.SUPPORT_SLIP.value,
    ]:
        summary = summaries[
            label
        ]

        print()
        print(
            f"  {label}"
        )

        if summary is None:
            print(
                "    no samples"
            )

            continue

        for index, name in enumerate(
            feature_names
        ):
            print(
                f"    {name:6} "
                f"mean="
                f"{summary['mean'][index]: .6f} "
                f"std="
                f"{summary['std'][index]: .6f}"
            )


def main():
    print(
        "=== Rigid Baseline Anomaly Audit ==="
    )

    print(
        "Purpose:"
    )

    print(
        "Determine whether above-chance rigid results "
        "reflect real observation information, "
        "finite-sample variance, or implementation leakage."
    )

    for probe_force in PROBE_FORCES:
        run_force(
            probe_force
        )


if __name__ == "__main__":
    main()