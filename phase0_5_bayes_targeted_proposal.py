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


ALPHAS = [
    0.10,
    0.25,
    0.50,
]

TAU_MULTIPLIERS = [
    0.5,
    1.0,
    2.0,
    4.0,
]


def stable_exp_weights(
    log_values: np.ndarray,
) -> np.ndarray:
    maximum = float(
        np.max(
            log_values
        )
    )

    return np.exp(
        log_values
        - maximum
    )


def uniform_ess_fraction(
    log_likelihoods: np.ndarray,
) -> float:
    weights = (
        stable_exp_weights(
            log_likelihoods
        )
    )

    numerator = (
        float(
            np.sum(
                weights
            )
        )
        ** 2
    )

    denominator = (
        len(
            weights
        )
        * float(
            np.sum(
                weights
                * weights
            )
        )
    )

    return (
        numerator
        / denominator
    )


def proposal_ess_fraction(
    log_likelihoods: np.ndarray,
    tip_dx_means: np.ndarray,
    observed_tip_dx: float,
    alpha: float,
    tau: float,
) -> float:
    """
    Compute the asymptotic ESS fraction for a defensive
    tip-dx-targeted importance proposal over the empirical
    class-conditional particle bank.

    Target p is uniform over the empirical prior particles.

    Proposal:
        q = alpha * p
            + (1 - alpha) * targeted_kernel

    This is exploratory only.
    """

    n = len(
        log_likelihoods
    )

    if n <= 0:
        raise ValueError(
            "Particle class is empty"
        )

    if not (
        0.0
        < alpha
        <= 1.0
    ):
        raise ValueError(
            "alpha must be in (0, 1]"
        )

    if tau <= 0.0:
        raise ValueError(
            "tau must be positive"
        )

    prior_probability = (
        1.0
        / n
    )

    dx_residual = (
        tip_dx_means
        - observed_tip_dx
    )

    kernel_log = (
        -0.5
        * (
            dx_residual
            / tau
        )
        ** 2
    )

    kernel = (
        stable_exp_weights(
            kernel_log
        )
    )

    kernel_sum = float(
        np.sum(
            kernel
        )
    )

    if (
        kernel_sum
        <= 0.0
        or not math.isfinite(
            kernel_sum
        )
    ):
        raise RuntimeError(
            "Targeted kernel normalization failed"
        )

    targeted_probability = (
        kernel
        / kernel_sum
    )

    proposal_probability = (
        alpha
        * prior_probability
        + (
            1.0
            - alpha
        )
        * targeted_probability
    )

    likelihood = (
        stable_exp_weights(
            log_likelihoods
        )
    )

    importance_contribution = (
        prior_probability
        * likelihood
        / proposal_probability
    )

    expected_weight = float(
        np.sum(
            proposal_probability
            * importance_contribution
        )
    )

    expected_weight_squared = float(
        np.sum(
            proposal_probability
            * (
                importance_contribution
                ** 2
            )
        )
    )

    if (
        expected_weight_squared
        <= 0.0
    ):
        raise RuntimeError(
            "Invalid proposal ESS denominator"
        )

    return (
        expected_weight
        ** 2
        / expected_weight_squared
    )


def summarize(
    values: list[float],
) -> dict:
    array = np.asarray(
        values,
        dtype=float,
    )

    return {
        "min": float(
            np.min(
                array
            )
        ),
        "p05": float(
            np.percentile(
                array,
                5,
            )
        ),
        "median": float(
            np.median(
                array
            )
        ),
        "mean": float(
            np.mean(
                array
            )
        ),
    }


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

    criterion = (
        manifest[
            "bayes_ess_criterion"
        ]
    )

    print(
        "=== Phase 0.5 Bayes targeted proposal diagnostic ==="
    )

    print(
        "Exploratory numerical diagnostic only."
    )

    print(
        "No frozen scientific parameter is changed."
    )

    print(
        "Any proposal selected here must be preregistered "
        "and validated on new independent data before "
        "confirmatory use."
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

    motion_std = (
        noise[
            "motion_noise_std_m"
        ]
    )

    results = {}

    proposal_names = [
        "UNIFORM"
    ]

    for alpha in ALPHAS:
        for multiplier in TAU_MULTIPLIERS:
            proposal_names.append(
                (
                    f"A{alpha:.2f}_"
                    f"T{multiplier:.1f}"
                )
            )

    for name in proposal_names:
        results[
            name
        ] = {
            "slip": [],
            "yield": [],
        }

    slip_means = (
        means[
            slip_mask
        ]
    )

    yield_means = (
        means[
            yield_mask
        ]
    )

    for record in split.test:
        observation = np.asarray(
            [
                record.fx,
                record.fy,
                record.tip_dx,
                record.tip_dy,
            ],
            dtype=float,
        )

        slip_residual = (
            observation
            - slip_means
        )

        yield_residual = (
            observation
            - yield_means
        )

        slip_logs = (
            -0.5
            * np.sum(
                normalization_terms
                + (
                    slip_residual
                    ** 2
                    / (
                        std
                        ** 2
                    )
                ),
                axis=1,
            )
        )

        yield_logs = (
            -0.5
            * np.sum(
                normalization_terms
                + (
                    yield_residual
                    ** 2
                    / (
                        std
                        ** 2
                    )
                ),
                axis=1,
            )
        )

        results[
            "UNIFORM"
        ][
            "slip"
        ].append(
            uniform_ess_fraction(
                slip_logs
            )
        )

        results[
            "UNIFORM"
        ][
            "yield"
        ].append(
            uniform_ess_fraction(
                yield_logs
            )
        )

        for alpha in ALPHAS:
            for multiplier in TAU_MULTIPLIERS:
                tau = (
                    motion_std
                    * multiplier
                )

                name = (
                    f"A{alpha:.2f}_"
                    f"T{multiplier:.1f}"
                )

                results[
                    name
                ][
                    "slip"
                ].append(
                    proposal_ess_fraction(
                        log_likelihoods=(
                            slip_logs
                        ),
                        tip_dx_means=(
                            slip_means[
                                :,
                                2,
                            ]
                        ),
                        observed_tip_dx=(
                            record.tip_dx
                        ),
                        alpha=alpha,
                        tau=tau,
                    )
                )

                results[
                    name
                ][
                    "yield"
                ].append(
                    proposal_ess_fraction(
                        log_likelihoods=(
                            yield_logs
                        ),
                        tip_dx_means=(
                            yield_means[
                                :,
                                2,
                            ]
                        ),
                        observed_tip_dx=(
                            record.tip_dx
                        ),
                        alpha=alpha,
                        tau=tau,
                    )
                )

    print(
        "particles=",
        n_particles,
    )

    print(
        "held_out_rows=",
        len(
            split.test
        ),
    )

    print(
        "motion_noise_std=",
        motion_std,
    )

    print()

    print(
        "proposal        | slip_p05 | slip_med | "
        "yield_p05 | yield_med | frozen_threshold_pass"
    )

    best_name = None
    best_score = float(
        "-inf"
    )

    for name in proposal_names:
        slip_summary = (
            summarize(
                results[
                    name
                ][
                    "slip"
                ]
            )
        )

        yield_summary = (
            summarize(
                results[
                    name
                ][
                    "yield"
                ]
            )
        )

        passes = (
            slip_summary[
                "p05"
            ]
            >= criterion[
                "minimum_p05_ess_fraction"
            ]
            and slip_summary[
                "median"
            ]
            >= criterion[
                "minimum_median_ess_fraction"
            ]
            and yield_summary[
                "p05"
            ]
            >= criterion[
                "minimum_p05_ess_fraction"
            ]
            and yield_summary[
                "median"
            ]
            >= criterion[
                "minimum_median_ess_fraction"
            ]
        )

        score = min(
            slip_summary[
                "p05"
            ],
            yield_summary[
                "p05"
            ],
        )

        if score > best_score:
            best_score = score
            best_name = name

        print(
            f"{name:15s}"
            f" | {slip_summary['p05']:.5f}"
            f" | {slip_summary['median']:.5f}"
            f" | {yield_summary['p05']:.5f}"
            f" | {yield_summary['median']:.5f}"
            f" | {passes}"
        )

    print()

    print(
        "Best exploratory proposal by worst-class p05:",
        best_name,
    )

    print(
        "Best worst-class p05:",
        round(
            best_score,
            6,
        ),
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
        "IMPORTANT: this diagnostic uses the already observed "
        "confirmatory held-out set."
    )

    print(
        "It may diagnose a better proposal, but it cannot "
        "retroactively validate the failed confirmatory run."
    )

    print()

    print(
        "STATUS: exploratory targeted-proposal diagnostic only"
    )


if __name__ == "__main__":
    main()