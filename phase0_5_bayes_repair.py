import math
import random

import numpy as np

from phase0_5_bayes import (
    sample_bayes_particles,
)
from phase0_5_bayes_ess import (
    ObservationESS,
    summarize_ess,
)
from phase0_5_split import (
    pair_level_train_test_split,
)
from physics import (
    NextTransition,
)
from run_phase0_5_confirmatory import (
    generate_confirmatory_dataset,
    load_manifest,
)


PARTICLE_COUNTS = [
    20_000,
    50_000,
    100_000,
]


def ess_from_log_weights(
    log_weights: np.ndarray,
) -> float:
    maximum = float(
        np.max(
            log_weights
        )
    )

    weights = np.exp(
        log_weights
        - maximum
    )

    numerator = (
        float(
            np.sum(
                weights
            )
        )
        ** 2
    )

    denominator = float(
        np.sum(
            weights
            * weights
        )
    )

    return (
        numerator
        / denominator
    )


def evaluate_particle_count(
    test_records,
    particles,
    manifest,
):
    noise = (
        manifest[
            "noise_model"
        ]
    )

    labels = np.asarray(
        [
            particle.label
            for particle in particles
        ]
    )

    means = np.asarray(
        [
            [
                particle.mean_fx,
                particle.mean_fy,
                particle.mean_tip_dx,
                particle.mean_tip_dy,
            ]
            for particle in particles
        ],
        dtype=float,
    )

    slip_mask = (
        labels
        == NextTransition.SUPPORT_SLIP.value
    )

    yield_mask = (
        labels
        == NextTransition.MATERIAL_YIELD.value
    )

    n_slip = int(
        np.sum(
            slip_mask
        )
    )

    n_yield = int(
        np.sum(
            yield_mask
        )
    )

    std = np.asarray(
        [
            noise[
                "force_noise_std_N"
            ],
            noise[
                "force_noise_std_N"
            ],
            noise[
                "motion_noise_std_m"
            ],
            noise[
                "motion_noise_std_m"
            ],
        ],
        dtype=float,
    )

    normalization_terms = np.log(
        2.0
        * math.pi
        * (
            std
            ** 2
        )
    )

    ess_rows = []

    for record in test_records:
        observation = np.asarray(
            [
                record.fx,
                record.fy,
                record.tip_dx,
                record.tip_dy,
            ],
            dtype=float,
        )

        residual = (
            observation
            - means
        )

        log_likelihoods = (
            -0.5
            * np.sum(
                normalization_terms
                + (
                    residual
                    ** 2
                    / (
                        std
                        ** 2
                    )
                ),
                axis=1,
            )
        )

        slip_logs = (
            log_likelihoods[
                slip_mask
            ]
        )

        yield_logs = (
            log_likelihoods[
                yield_mask
            ]
        )

        slip_ess = (
            ess_from_log_weights(
                slip_logs
            )
        )

        yield_ess = (
            ess_from_log_weights(
                yield_logs
            )
        )

        ess_rows.append(
            ObservationESS(
                pair_id=record.pair_id,
                side=record.side,
                slip_ess=slip_ess,
                yield_ess=yield_ess,
                slip_ess_fraction=(
                    slip_ess
                    / n_slip
                ),
                yield_ess_fraction=(
                    yield_ess
                    / n_yield
                ),
            )
        )

    summary = (
        summarize_ess(
            observations=ess_rows,
            n_particles=len(
                particles
            ),
        )
    )

    criterion = (
        manifest[
            "bayes_ess_criterion"
        ]
    )

    slip_pass = (
        summary.slip_fraction_p05
        >= criterion[
            "minimum_p05_ess_fraction"
        ]
        and summary.slip_fraction_median
        >= criterion[
            "minimum_median_ess_fraction"
        ]
    )

    yield_pass = (
        summary.yield_fraction_p05
        >= criterion[
            "minimum_p05_ess_fraction"
        ]
        and summary.yield_fraction_median
        >= criterion[
            "minimum_median_ess_fraction"
        ]
    )

    return (
        summary,
        slip_pass
        and yield_pass,
    )


def main():
    manifest = (
        load_manifest()
    )

    print(
        "=== Phase 0.5 Bayes repair study ==="
    )

    print(
        "This is a numerical repair diagnostic only."
    )

    print(
        "No frozen scientific parameter is changed."
    )

    print()

    dataset_seed = (
        manifest[
            "confirmatory_dataset"
        ][
            "dataset_seed"
        ]
    )

    split_seed = (
        manifest[
            "confirmatory_dataset"
        ][
            "split_seed"
        ]
    )

    random.seed(
        dataset_seed
    )

    (
        compliant_records,
        _,
        _,
    ) = (
        generate_confirmatory_dataset(
            manifest
        )
    )

    split = (
        pair_level_train_test_split(
            records=compliant_records,
            test_fraction=(
                manifest[
                    "confirmatory_dataset"
                ][
                    "test_fraction"
                ]
            ),
            seed=split_seed,
        )
    )

    max_particles = max(
        PARTICLE_COUNTS
    )

    print(
        "Sampling nested particle bank:",
        max_particles,
    )

    all_particles = (
        sample_bayes_particles(
            n_particles=max_particles,
            seed=(
                manifest[
                    "bayes_reference"
                ][
                    "particle_seed"
                ]
            ),
        )
    )

    print(
        "Official held-out rows:",
        len(
            split.test
        ),
    )

    print()

    print(
        "particles | slip_p05 | slip_median | "
        "yield_p05 | yield_median | frozen_ESS_pass"
    )

    for count in PARTICLE_COUNTS:
        particles = (
            all_particles[
                :count
            ]
        )

        (
            summary,
            passes,
        ) = (
            evaluate_particle_count(
                test_records=split.test,
                particles=particles,
                manifest=manifest,
            )
        )

        print(
            f"{count:9d}"
            f" | {summary.slip_fraction_p05:.5f}"
            f" | {summary.slip_fraction_median:.5f}"
            f" | {summary.yield_fraction_p05:.5f}"
            f" | {summary.yield_fraction_median:.5f}"
            f" | {passes}"
        )

    print()

    print(
        "Frozen ESS thresholds:"
    )

    print(
        "p05 fraction >=",
        manifest[
            "bayes_ess_criterion"
        ][
            "minimum_p05_ess_fraction"
        ],
    )

    print(
        "median fraction >=",
        manifest[
            "bayes_ess_criterion"
        ][
            "minimum_median_ess_fraction"
        ],
    )

    print()

    print(
        "STATUS: exploratory numerical repair study only"
    )


if __name__ == "__main__":
    main()