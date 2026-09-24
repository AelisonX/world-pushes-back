import math

import numpy as np

from phase0_5_bayes import (
    sample_bayes_particles,
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


FEATURE_SETS = {
    "FULL": [
        0,
        1,
        2,
        3,
    ],
    "FORCE_ONLY": [
        0,
        1,
    ],
    "MOTION_ONLY": [
        2,
        3,
    ],
    "TIP_DX_ONLY": [
        2,
    ],
    "TIP_DY_ONLY": [
        3,
    ],
}


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


def summarize_fractions(
    fractions: list[float],
) -> tuple[
    float,
    float,
    float,
]:
    values = np.asarray(
        fractions,
        dtype=float,
    )

    return (
        float(
            np.min(
                values
            )
        ),
        float(
            np.percentile(
                values,
                5,
            )
        ),
        float(
            np.median(
                values
            )
        ),
    )


def main():
    manifest = (
        load_manifest()
    )

    config = (
        manifest[
            "confirmatory_dataset"
        ]
    )

    noise = (
        manifest[
            "noise_model"
        ]
    )

    print(
        "=== Phase 0.5 Bayes ESS channel ablation ==="
    )

    print(
        "Diagnostic only. No frozen scientific parameter is changed."
    )

    print()

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
                config[
                    "test_fraction"
                ]
            ),
            seed=(
                config[
                    "split_seed"
                ]
            ),
        )
    )

    n_particles = (
        manifest[
            "bayes_reference"
        ][
            "particles"
        ]
    )

    particles = (
        sample_bayes_particles(
            n_particles=n_particles,
            seed=(
                manifest[
                    "bayes_reference"
                ][
                    "particle_seed"
                ]
            ),
        )
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

    criterion = (
        manifest[
            "bayes_ess_criterion"
        ]
    )

    print(
        "Particles:",
        n_particles,
    )

    print(
        "Official held-out rows:",
        len(
            split.test
        ),
    )

    print()

    print(
        "channel       | slip_p05 | slip_med | yield_p05 | yield_med | pass"
    )

    for (
        name,
        indices,
    ) in FEATURE_SETS.items():
        slip_fractions = []
        yield_fractions = []

        selected_std = (
            std[
                indices
            ]
        )

        normalization_terms = np.log(
            2.0
            * math.pi
            * (
                selected_std
                ** 2
            )
        )

        selected_means = (
            means[
                :,
                indices,
            ]
        )

        for record in split.test:
            observation_all = np.asarray(
                [
                    record.fx,
                    record.fy,
                    record.tip_dx,
                    record.tip_dy,
                ],
                dtype=float,
            )

            observation = (
                observation_all[
                    indices
                ]
            )

            residual = (
                observation
                - selected_means
            )

            log_likelihoods = (
                -0.5
                * np.sum(
                    normalization_terms
                    + (
                        residual
                        ** 2
                        / (
                            selected_std
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

            slip_fractions.append(
                slip_ess
                / n_slip
            )

            yield_fractions.append(
                yield_ess
                / n_yield
            )

        (
            _,
            slip_p05,
            slip_median,
        ) = (
            summarize_fractions(
                slip_fractions
            )
        )

        (
            _,
            yield_p05,
            yield_median,
        ) = (
            summarize_fractions(
                yield_fractions
            )
        )

        passes = (
            slip_p05
            >= criterion[
                "minimum_p05_ess_fraction"
            ]
            and slip_median
            >= criterion[
                "minimum_median_ess_fraction"
            ]
            and yield_p05
            >= criterion[
                "minimum_p05_ess_fraction"
            ]
            and yield_median
            >= criterion[
                "minimum_median_ess_fraction"
            ]
        )

        print(
            f"{name:13s}"
            f" | {slip_p05:.5f}"
            f" | {slip_median:.5f}"
            f" | {yield_p05:.5f}"
            f" | {yield_median:.5f}"
            f" | {passes}"
        )

    print()

    print(
        "Frozen p05 threshold:",
        criterion[
            "minimum_p05_ess_fraction"
        ],
    )

    print(
        "Frozen median threshold:",
        criterion[
            "minimum_median_ess_fraction"
        ],
    )

    print()

    print(
        "STATUS: exploratory ESS channel diagnostic only"
    )


if __name__ == "__main__":
    main()