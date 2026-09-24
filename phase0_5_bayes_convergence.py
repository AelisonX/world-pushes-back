from dataclasses import dataclass
import random

import numpy as np

from sklearn.metrics import (
    roc_auc_score,
)

from phase0_5_bayes import (
    BayesParticle,
    bayes_support_slip_probability,
    sample_bayes_particles,
)
from phase0_5_controls import (
    generate_matched_control_dataset,
)
from phase0_5_evaluate import (
    labels_to_binary,
)


DEFAULT_PARTICLE_COUNTS = [
    500,
    1_000,
    2_000,
    5_000,
    10_000,
]


@dataclass
class ConvergencePoint:
    n_particles: int
    roc_auc: float
    auc_change: float | None
    mean_posterior_change: float | None
    max_posterior_change: float | None


def score_records_with_particles(
    records,
    particles: list[BayesParticle],
) -> np.ndarray:
    """
    Return Bayes SUPPORT_SLIP probabilities for a fixed
    observation dataset and particle set.
    """

    return np.asarray(
        [
            bayes_support_slip_probability(
                record=record,
                particles=particles,
            )
            for record in records
        ],
        dtype=float,
    )


def run_bayes_convergence_study(
    particle_counts: list[int] | None = None,
    n_test_pairs: int = 100,
    particle_seed: int = 123,
    test_seed: int = 999,
) -> list[
    ConvergencePoint
]:
    """
    Measure Monte Carlo Bayes stability as particle count grows.

    One maximum-size particle sample is generated first.

    Smaller conditions use prefixes of that same sample.

    One fixed independent test dataset is reused at every
    particle count.

    This is an exploratory convergence diagnostic.

    It does not itself define the confirmatory convergence
    criterion.
    """

    if particle_counts is None:
        particle_counts = list(
            DEFAULT_PARTICLE_COUNTS
        )

    if not particle_counts:
        raise ValueError(
            "particle_counts must not be empty"
        )

    if any(
        count <= 0
        for count in particle_counts
    ):
        raise ValueError(
            "particle counts must be positive"
        )

    if any(
        count % 2 != 0
        for count in particle_counts
    ):
        raise ValueError(
            "particle counts must be even"
        )

    if particle_counts != sorted(
        particle_counts
    ):
        raise ValueError(
            "particle_counts must be sorted"
        )

    if len(
        set(
            particle_counts
        )
    ) != len(
        particle_counts
    ):
        raise ValueError(
            "particle_counts must be unique"
        )

    if n_test_pairs <= 0:
        raise ValueError(
            "n_test_pairs must be positive"
        )

    max_particles = max(
        particle_counts
    )

    all_particles = (
        sample_bayes_particles(
            n_particles=max_particles,
            seed=particle_seed,
        )
    )

    random.seed(
        test_seed
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=n_test_pairs,
        )
    )

    records = (
        dataset.compliant_records
    )

    y_true = (
        labels_to_binary(
            [
                record.label
                for record in records
            ]
        )
    )

    results = []

    previous_auc = None
    previous_scores = None

    for count in particle_counts:
        particles = (
            all_particles[
                :count
            ]
        )

        scores = (
            score_records_with_particles(
                records=records,
                particles=particles,
            )
        )

        auc = float(
            roc_auc_score(
                y_true,
                scores,
            )
        )

        if previous_auc is None:
            auc_change = None
            mean_change = None
            max_change = None

        else:
            auc_change = abs(
                auc
                - previous_auc
            )

            posterior_changes = np.abs(
                scores
                - previous_scores
            )

            mean_change = float(
                np.mean(
                    posterior_changes
                )
            )

            max_change = float(
                np.max(
                    posterior_changes
                )
            )

        results.append(
            ConvergencePoint(
                n_particles=count,
                roc_auc=auc,
                auc_change=auc_change,
                mean_posterior_change=(
                    mean_change
                ),
                max_posterior_change=(
                    max_change
                ),
            )
        )

        previous_auc = auc
        previous_scores = scores

    return results


def main():
    results = (
        run_bayes_convergence_study()
    )

    print(
        "=== Phase 0.5 Bayes convergence study ==="
    )

    print(
        "particles | AUC | delta_AUC | "
        "mean_delta_p | max_delta_p"
    )

    for result in results:
        if result.auc_change is None:
            auc_change = "N/A"
            mean_change = "N/A"
            max_change = "N/A"

        else:
            auc_change = (
                f"{result.auc_change:.5f}"
            )

            mean_change = (
                f"{result.mean_posterior_change:.5f}"
            )

            max_change = (
                f"{result.max_posterior_change:.5f}"
            )

        print(
            f"{result.n_particles:9d}"
            f" | {result.roc_auc:.4f}"
            f" | {auc_change:>9s}"
            f" | {mean_change:>12s}"
            f" | {max_change:>11s}"
        )

    print()

    print(
        "STATUS: exploratory convergence diagnostic only"
    )


if __name__ == "__main__":
    main()