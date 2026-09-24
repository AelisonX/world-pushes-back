from dataclasses import dataclass
import math
import random

import numpy as np

from sklearn.metrics import (
    roc_auc_score,
)

from phase0_5_controls import (
    generate_matched_control_dataset,
)
from phase0_5_dataset import (
    DEFAULT_FORCE_NOISE_STD,
    DEFAULT_MOTION_NOISE_STD,
    ObservationNoise,
)
from phase0_5_evaluate import (
    labels_to_binary,
)
from phase0_5_safe_cohort import (
    generate_safe_pair_sample,
)
from physics import (
    NextTransition,
    run_safe_probe,
)


@dataclass
class BayesParticle:
    label: str
    mean_fx: float
    mean_fy: float
    mean_tip_dx: float
    mean_tip_dy: float


@dataclass
class BayesReferenceResult:
    n_particles: int
    n_test_rows: int
    roc_auc: float


def gaussian_log_likelihood(
    observed: float,
    mean: float,
    std: float,
) -> float:
    """
    Log likelihood under a univariate Gaussian.
    """

    if std <= 0:
        raise ValueError(
            "std must be positive"
        )

    variance = (
        std
        * std
    )

    return (
        -0.5
        * (
            math.log(
                2.0
                * math.pi
                * variance
            )
            + (
                (
                    observed
                    - mean
                )
                ** 2
                / variance
            )
        )
    )


def logsumexp(
    values: np.ndarray,
) -> float:
    """
    Stable log(sum(exp(values))).
    """

    if len(
        values
    ) == 0:
        return float(
            "-inf"
        )

    maximum = float(
        np.max(
            values
        )
    )

    if math.isinf(
        maximum
    ):
        return maximum

    return (
        maximum
        + math.log(
            float(
                np.sum(
                    np.exp(
                        values
                        - maximum
                    )
                )
            )
        )
    )


def make_noiseless_particle(
    sample,
    side: str,
) -> BayesParticle:
    """
    Convert one safe twin world into a noiseless simulator
    particle under the official compliant observation model.
    """

    if side == "A":
        params = (
            sample.pair.world_a
        )

        label = (
            sample.pair.label_a.value
        )

    elif side == "B":
        params = (
            sample.pair.world_b
        )

        label = (
            sample.pair.label_b.value
        )

    else:
        raise ValueError(
            "side must be A or B"
        )

    result = run_safe_probe(
        params=params,
        action=sample.action,
        support_model="compliant",
        force_noise_std=0.0,
        motion_noise_std=0.0,
    )

    if not result.still_blocked:
        raise RuntimeError(
            "Bayes particle violated safe-cohort invariant"
        )

    observation = (
        result.observation
    )

    return BayesParticle(
        label=label,
        mean_fx=observation.fx,
        mean_fy=observation.fy,
        mean_tip_dx=observation.tip_dx,
        mean_tip_dy=observation.tip_dy,
    )


def sample_bayes_particles(
    n_particles: int,
    seed: int = 123,
) -> list[
    BayesParticle
]:
    """
    Sample particles from the same safe paired
    identification distribution used by Phase 0.5.

    Because each generated pair contributes one twin from each
    class, the Monte Carlo prior remains class balanced.
    """

    if n_particles <= 0:
        raise ValueError(
            "n_particles must be positive"
        )

    if (
        n_particles
        % 2
        != 0
    ):
        raise ValueError(
            "n_particles must be even"
        )

    random.seed(
        seed
    )

    particles = []

    n_pairs = (
        n_particles
        // 2
    )

    for pair_id in range(
        n_pairs
    ):
        sample = (
            generate_safe_pair_sample(
                pair_id=pair_id,
            )
        )

        particles.append(
            make_noiseless_particle(
                sample=sample,
                side="A",
            )
        )

        particles.append(
            make_noiseless_particle(
                sample=sample,
                side="B",
            )
        )

    return particles


def particle_log_likelihood(
    record,
    particle: BayesParticle,
    force_noise_std: float = DEFAULT_FORCE_NOISE_STD,
    motion_noise_std: float = DEFAULT_MOTION_NOISE_STD,
) -> float:
    """
    Exact Gaussian observation likelihood for one simulator
    particle under the current Phase 0.5 noise model.
    """

    return (
        gaussian_log_likelihood(
            observed=record.fx,
            mean=particle.mean_fx,
            std=force_noise_std,
        )
        + gaussian_log_likelihood(
            observed=record.fy,
            mean=particle.mean_fy,
            std=force_noise_std,
        )
        + gaussian_log_likelihood(
            observed=record.tip_dx,
            mean=particle.mean_tip_dx,
            std=motion_noise_std,
        )
        + gaussian_log_likelihood(
            observed=record.tip_dy,
            mean=particle.mean_tip_dy,
            std=motion_noise_std,
        )
    )


def bayes_support_slip_probability(
    record,
    particles: list[BayesParticle],
    force_noise_std: float = DEFAULT_FORCE_NOISE_STD,
    motion_noise_std: float = DEFAULT_MOTION_NOISE_STD,
) -> float:
    """
    Approximate P(SUPPORT_SLIP | observation)
    by Monte Carlo marginalization over hidden worlds.
    """

    slip_logs = []

    yield_logs = []

    for particle in particles:
        log_likelihood = (
            particle_log_likelihood(
                record=record,
                particle=particle,
                force_noise_std=force_noise_std,
                motion_noise_std=motion_noise_std,
            )
        )

        if (
            particle.label
            == NextTransition.SUPPORT_SLIP.value
        ):
            slip_logs.append(
                log_likelihood
            )

        elif (
            particle.label
            == NextTransition.MATERIAL_YIELD.value
        ):
            yield_logs.append(
                log_likelihood
            )

        else:
            raise ValueError(
                f"Unknown particle label: "
                f"{particle.label}"
            )

    if not slip_logs:
        raise ValueError(
            "No SUPPORT_SLIP particles"
        )

    if not yield_logs:
        raise ValueError(
            "No MATERIAL_YIELD particles"
        )

    slip_log_evidence = (
        logsumexp(
            np.asarray(
                slip_logs,
                dtype=float,
            )
        )
        - math.log(
            len(
                slip_logs
            )
        )
    )

    yield_log_evidence = (
        logsumexp(
            np.asarray(
                yield_logs,
                dtype=float,
            )
        )
        - math.log(
            len(
                yield_logs
            )
        )
    )

    normalization = (
        logsumexp(
            np.asarray(
                [
                    slip_log_evidence,
                    yield_log_evidence,
                ],
                dtype=float,
            )
        )
    )

    return float(
        math.exp(
            slip_log_evidence
            - normalization
        )
    )


def evaluate_bayes_reference(
    n_particles: int = 2000,
    n_test_pairs: int = 100,
    particle_seed: int = 123,
    test_seed: int = 999,
) -> BayesReferenceResult:
    """
    Evaluate the Monte Carlo Bayes reference on an independently
    generated held-out Phase 0.5 dataset.

    This is an approximation only.

    It must pass a convergence study before being treated as a
    credible reference.
    """

    particles = (
        sample_bayes_particles(
            n_particles=n_particles,
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

    scores = np.asarray(
        [
            bayes_support_slip_probability(
                record=record,
                particles=particles,
            )
            for record in records
        ],
        dtype=float,
    )

    y_true = (
        labels_to_binary(
            [
                record.label
                for record in records
            ]
        )
    )

    auc = float(
        roc_auc_score(
            y_true,
            scores,
        )
    )

    return BayesReferenceResult(
        n_particles=n_particles,
        n_test_rows=len(
            records
        ),
        roc_auc=auc,
    )


def main():
    result = (
        evaluate_bayes_reference(
            n_particles=2000,
            n_test_pairs=100,
            particle_seed=123,
            test_seed=999,
        )
    )

    print(
        "=== Phase 0.5 Monte Carlo Bayes reference ==="
    )

    print(
        "particles=",
        result.n_particles,
    )

    print(
        "test_rows=",
        result.n_test_rows,
    )

    print(
        "ROC-AUC=",
        round(
            result.roc_auc,
            4,
        ),
    )

    print(
        "STATUS: exploratory approximation only"
    )


if __name__ == "__main__":
    main()