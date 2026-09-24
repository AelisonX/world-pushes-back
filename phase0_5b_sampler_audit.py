from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

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


AUDIT_MANIFEST_PATH = Path(
    "PHASE0_5B_SAMPLER_AUDIT_MANIFEST.json"
)

RESULTS_PATH = Path(
    "phase0_5b_sampler_audit_results.json"
)


def load_audit_manifest() -> dict:
    with AUDIT_MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    if (
        manifest.get(
            "status"
        )
        != "FROZEN_BEFORE_AUDIT_RUN"
    ):
        raise ValueError(
            "Sampler audit manifest is not frozen"
        )

    if (
        manifest.get(
            "manifest_version"
        )
        != "1.0"
    ):
        raise ValueError(
            "Unexpected sampler audit manifest version"
        )

    return manifest


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                65536
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def current_commit_sha() -> str:
    github_sha = os.environ.get(
        "GITHUB_SHA"
    )

    if github_sha:
        return github_sha

    try:
        return (
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                text=True,
            )
            .strip()
        )

    except Exception:
        return "UNKNOWN"


def make_audit_parent_manifest(
    parent: dict,
    audit: dict,
) -> dict:
    derived = deepcopy(
        parent
    )

    dataset = audit[
        "audit_dataset"
    ]

    derived[
        "confirmatory_dataset"
    ][
        "n_pairs"
    ] = dataset[
        "n_pairs"
    ]

    derived[
        "confirmatory_dataset"
    ][
        "total_rows"
    ] = dataset[
        "total_rows"
    ]

    derived[
        "confirmatory_dataset"
    ][
        "test_fraction"
    ] = dataset[
        "test_fraction"
    ]

    derived[
        "confirmatory_dataset"
    ][
        "dataset_seed"
    ] = dataset[
        "dataset_seed"
    ]

    derived[
        "confirmatory_dataset"
    ][
        "split_seed"
    ] = dataset[
        "split_seed"
    ]

    return derived


def stable_logsumexp(
    values: np.ndarray,
) -> float:
    maximum = float(
        np.max(
            values
        )
    )

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


def proposal_probabilities(
    tip_dx_means: np.ndarray,
    observed_tip_dx: float,
    alpha: float,
    tau: float,
) -> np.ndarray:
    n = len(
        tip_dx_means
    )

    if n <= 0:
        raise ValueError(
            "Particle bank is empty"
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

    residual = (
        tip_dx_means
        - observed_tip_dx
    )

    kernel_log = (
        -0.5
        * (
            residual
            / tau
        )
        ** 2
    )

    maximum = float(
        np.max(
            kernel_log
        )
    )

    kernel = np.exp(
        kernel_log
        - maximum
    )

    kernel_sum = float(
        np.sum(
            kernel
        )
    )

    if (
        kernel_sum <= 0.0
        or not math.isfinite(
            kernel_sum
        )
    ):
        raise RuntimeError(
            "Proposal kernel normalization failed"
        )

    targeted = (
        kernel
        / kernel_sum
    )

    q = (
        alpha
        * prior_probability
        + (
            1.0
            - alpha
        )
        * targeted
    )

    return q


def gaussian_log_likelihoods(
    observation: np.ndarray,
    means: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    normalization_terms = np.log(
        2.0
        * math.pi
        * (
            std
            ** 2
        )
    )

    residual = (
        observation
        - means
    )

    return (
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


def full_bank_log_evidence(
    log_likelihoods: np.ndarray,
) -> float:
    return (
        stable_logsumexp(
            log_likelihoods
        )
        - math.log(
            len(
                log_likelihoods
            )
        )
    )


def posterior_slip_probability(
    slip_log_evidence: float,
    yield_log_evidence: float,
) -> float:
    difference = (
        slip_log_evidence
        - yield_log_evidence
    )

    if difference >= 0.0:
        return (
            1.0
            / (
                1.0
                + math.exp(
                    -difference
                )
            )
        )

    exp_difference = math.exp(
        difference
    )

    return (
        exp_difference
        / (
            1.0
            + exp_difference
        )
    )


def normalized_ess(
    weights: np.ndarray,
) -> float:
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
                ** 2
            )
        )
    )

    if denominator <= 0.0:
        raise RuntimeError(
            "Invalid ESS denominator"
        )

    return (
        numerator
        / denominator
    )


def exact_importance_identity_error(
    log_likelihoods: np.ndarray,
    q: np.ndarray,
) -> float:
    n = len(
        log_likelihoods
    )

    p = (
        1.0
        / n
    )

    maximum = float(
        np.max(
            log_likelihoods
        )
    )

    scaled_likelihood = np.exp(
        log_likelihoods
        - maximum
    )

    reference_scaled = float(
        np.mean(
            scaled_likelihood
        )
    )

    corrected = (
        p
        * scaled_likelihood
        / q
    )

    identity_scaled = float(
        np.sum(
            q
            * corrected
        )
    )

    return abs(
        identity_scaled
        - reference_scaled
    )


def sample_targeted(
    log_likelihoods: np.ndarray,
    q: np.ndarray,
    sample_size: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(
        seed
    )

    n_bank = len(
        log_likelihoods
    )

    p = (
        1.0
        / n_bank
    )

    indices = rng.choice(
        n_bank,
        size=sample_size,
        replace=True,
        p=q,
    )

    sampled_logs = (
        log_likelihoods[
            indices
        ]
    )

    sampled_q = (
        q[
            indices
        ]
    )

    maximum = float(
        np.max(
            log_likelihoods
        )
    )

    scaled_likelihood = np.exp(
        sampled_logs
        - maximum
    )

    weights = (
        p
        * scaled_likelihood
        / sampled_q
    )

    scaled_evidence = float(
        np.mean(
            weights
        )
    )

    if scaled_evidence <= 0.0:
        raise RuntimeError(
            "Targeted sampled evidence is nonpositive"
        )

    log_evidence = (
        maximum
        + math.log(
            scaled_evidence
        )
    )

    return {
        "log_evidence": log_evidence,
        "ess_fraction": normalized_ess(
            weights
        ),
    }


def sample_uniform(
    log_likelihoods: np.ndarray,
    sample_size: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(
        seed
    )

    n_bank = len(
        log_likelihoods
    )

    indices = rng.integers(
        0,
        n_bank,
        size=sample_size,
    )

    sampled_logs = (
        log_likelihoods[
            indices
        ]
    )

    maximum = float(
        np.max(
            log_likelihoods
        )
    )

    weights = np.exp(
        sampled_logs
        - maximum
    )

    scaled_evidence = float(
        np.mean(
            weights
        )
    )

    if scaled_evidence <= 0.0:
        raise RuntimeError(
            "Uniform sampled evidence is nonpositive"
        )

    log_evidence = (
        maximum
        + math.log(
            scaled_evidence
        )
    )

    return {
        "log_evidence": log_evidence,
        "ess_fraction": normalized_ess(
            weights
        ),
    }


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
        "p95": float(
            np.percentile(
                array,
                95,
            )
        ),
        "max": float(
            np.max(
                array
            )
        ),
    }


def relative_error_from_logs(
    sampled_log_evidence: float,
    reference_log_evidence: float,
) -> float:
    difference = (
        sampled_log_evidence
        - reference_log_evidence
    )

    if difference > 700.0:
        return float(
            "inf"
        )

    if difference < -700.0:
        return -1.0

    return (
        math.exp(
            difference
        )
        - 1.0
    )


def validate_manifest_relationship(
    parent: dict,
    audit: dict,
) -> None:
    proposal = audit[
        "proposal"
    ]

    if (
        proposal[
            "uniform_prior_mixture_weight_alpha"
        ]
        != 0.10
    ):
        raise ValueError(
            "Unexpected frozen proposal alpha"
        )

    if (
        proposal[
            "tau_m"
        ]
        != 0.0002
    ):
        raise ValueError(
            "Unexpected frozen proposal tau"
        )

    bank = audit[
        "particle_bank"
    ]

    if (
        bank[
            "total_particles"
        ]
        != (
            2
            * bank[
                "particles_per_class"
            ]
        )
    ):
        raise ValueError(
            "Particle-bank class count mismatch"
        )

    noise = audit[
        "noise_model"
    ]

    if (
        noise[
            "force_noise_std_N"
        ]
        != parent[
            "noise_model"
        ][
            "force_noise_std_N"
        ]
    ):
        raise ValueError(
            "Force noise differs from Phase 0.5"
        )

    if (
        noise[
            "motion_noise_std_m"
        ]
        != parent[
            "noise_model"
        ][
            "motion_noise_std_m"
        ]
    ):
        raise ValueError(
            "Motion noise differs from Phase 0.5"
        )

    if (
        audit[
            "target_definition"
        ][
            "reference_angle_deg"
        ]
        != parent[
            "counterfactual_label"
        ][
            "reference_angle_deg"
        ]
    ):
        raise ValueError(
            "Target angle differs from Phase 0.5"
        )


def run_audit() -> dict:
    audit = (
        load_audit_manifest()
    )

    parent = (
        load_manifest()
    )

    validate_manifest_relationship(
        parent=parent,
        audit=audit,
    )

    generator_manifest = (
        make_audit_parent_manifest(
            parent=parent,
            audit=audit,
        )
    )

    (
        records,
        _,
        _,
    ) = (
        generate_confirmatory_dataset(
            generator_manifest
        )
    )

    dataset_config = audit[
        "audit_dataset"
    ]

    split = (
        pair_level_train_test_split(
            records=records,
            test_fraction=(
                dataset_config[
                    "test_fraction"
                ]
            ),
            seed=(
                dataset_config[
                    "split_seed"
                ]
            ),
        )
    )

    ordered_test = sorted(
        split.test,
        key=lambda record: (
            record.pair_id,
            record.side,
        ),
    )

    evaluation_rows = (
        ordered_test[
            :dataset_config[
                "evaluation_subset_rows"
            ]
        ]
    )

    if (
        len(
            evaluation_rows
        )
        != dataset_config[
            "evaluation_subset_rows"
        ]
    ):
        raise RuntimeError(
            "Evaluation subset is smaller than frozen size"
        )

    bank_config = audit[
        "particle_bank"
    ]

    particles = (
        sample_bayes_particles(
            n_particles=(
                bank_config[
                    "total_particles"
                ]
            ),
            seed=(
                bank_config[
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

    slip_means = means[
        slip_mask
    ]

    yield_means = means[
        yield_mask
    ]

    expected_per_class = (
        bank_config[
            "particles_per_class"
        ]
    )

    if (
        len(
            slip_means
        )
        != expected_per_class
        or len(
            yield_means
        )
        != expected_per_class
    ):
        raise RuntimeError(
            "Particle bank is not class balanced"
        )

    noise = audit[
        "noise_model"
    ]

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

    proposal = audit[
        "proposal"
    ]

    alpha = (
        proposal[
            "uniform_prior_mixture_weight_alpha"
        ]
    )

    tau = (
        proposal[
            "tau_m"
        ]
    )

    sampler = audit[
        "sampler"
    ]

    sample_size = (
        sampler[
            "primary_sample_size_per_class"
        ]
    )

    replicates = (
        sampler[
            "replicates"
        ]
    )

    base_seed = (
        sampler[
            "sampler_seed_base"
        ]
    )

    targeted_ess = {
        "SUPPORT_SLIP": [],
        "MATERIAL_YIELD": [],
    }

    uniform_ess = {
        "SUPPORT_SLIP": [],
        "MATERIAL_YIELD": [],
    }

    targeted_posterior_errors = []
    uniform_posterior_errors = []

    targeted_evidence_relative_errors = {
        "SUPPORT_SLIP": [],
        "MATERIAL_YIELD": [],
    }

    uniform_evidence_relative_errors = {
        "SUPPORT_SLIP": [],
        "MATERIAL_YIELD": [],
    }

    targeted_posteriors_by_observation = [
        []
        for _ in evaluation_rows
    ]

    uniform_posteriors_by_observation = [
        []
        for _ in evaluation_rows
    ]

    maximum_proposal_normalization_error = 0.0
    maximum_identity_error = 0.0

    for (
        observation_index,
        record,
    ) in enumerate(
        evaluation_rows
    ):
        observation = np.asarray(
            [
                record.fx,
                record.fy,
                record.tip_dx,
                record.tip_dy,
            ],
            dtype=float,
        )

        slip_logs = (
            gaussian_log_likelihoods(
                observation=observation,
                means=slip_means,
                std=std,
            )
        )

        yield_logs = (
            gaussian_log_likelihoods(
                observation=observation,
                means=yield_means,
                std=std,
            )
        )

        reference_slip = (
            full_bank_log_evidence(
                slip_logs
            )
        )

        reference_yield = (
            full_bank_log_evidence(
                yield_logs
            )
        )

        reference_posterior = (
            posterior_slip_probability(
                reference_slip,
                reference_yield,
            )
        )

        slip_q = (
            proposal_probabilities(
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

        yield_q = (
            proposal_probabilities(
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

        normalization_error = max(
            abs(
                float(
                    np.sum(
                        slip_q
                    )
                )
                - 1.0
            ),
            abs(
                float(
                    np.sum(
                        yield_q
                    )
                )
                - 1.0
            ),
        )

        maximum_proposal_normalization_error = max(
            maximum_proposal_normalization_error,
            normalization_error,
        )

        identity_error = max(
            exact_importance_identity_error(
                log_likelihoods=slip_logs,
                q=slip_q,
            ),
            exact_importance_identity_error(
                log_likelihoods=yield_logs,
                q=yield_q,
            ),
        )

        maximum_identity_error = max(
            maximum_identity_error,
            identity_error,
        )

        for replicate_index in range(
            replicates
        ):
            seed = (
                base_seed
                + observation_index
                * 100
                + replicate_index
            )

            targeted_slip = (
                sample_targeted(
                    log_likelihoods=(
                        slip_logs
                    ),
                    q=slip_q,
                    sample_size=sample_size,
                    seed=seed,
                )
            )

            targeted_yield = (
                sample_targeted(
                    log_likelihoods=(
                        yield_logs
                    ),
                    q=yield_q,
                    sample_size=sample_size,
                    seed=(
                        seed
                        + 1_000_000
                    ),
                )
            )

            uniform_slip = (
                sample_uniform(
                    log_likelihoods=(
                        slip_logs
                    ),
                    sample_size=sample_size,
                    seed=(
                        seed
                        + 2_000_000
                    ),
                )
            )

            uniform_yield = (
                sample_uniform(
                    log_likelihoods=(
                        yield_logs
                    ),
                    sample_size=sample_size,
                    seed=(
                        seed
                        + 3_000_000
                    ),
                )
            )

            targeted_posterior = (
                posterior_slip_probability(
                    targeted_slip[
                        "log_evidence"
                    ],
                    targeted_yield[
                        "log_evidence"
                    ],
                )
            )

            uniform_posterior = (
                posterior_slip_probability(
                    uniform_slip[
                        "log_evidence"
                    ],
                    uniform_yield[
                        "log_evidence"
                    ],
                )
            )

            targeted_posteriors_by_observation[
                observation_index
            ].append(
                targeted_posterior
            )

            uniform_posteriors_by_observation[
                observation_index
            ].append(
                uniform_posterior
            )

            targeted_posterior_errors.append(
                abs(
                    targeted_posterior
                    - reference_posterior
                )
            )

            uniform_posterior_errors.append(
                abs(
                    uniform_posterior
                    - reference_posterior
                )
            )

            targeted_ess[
                "SUPPORT_SLIP"
            ].append(
                targeted_slip[
                    "ess_fraction"
                ]
            )

            targeted_ess[
                "MATERIAL_YIELD"
            ].append(
                targeted_yield[
                    "ess_fraction"
                ]
            )

            uniform_ess[
                "SUPPORT_SLIP"
            ].append(
                uniform_slip[
                    "ess_fraction"
                ]
            )

            uniform_ess[
                "MATERIAL_YIELD"
            ].append(
                uniform_yield[
                    "ess_fraction"
                ]
            )

            targeted_evidence_relative_errors[
                "SUPPORT_SLIP"
            ].append(
                relative_error_from_logs(
                    targeted_slip[
                        "log_evidence"
                    ],
                    reference_slip,
                )
            )

            targeted_evidence_relative_errors[
                "MATERIAL_YIELD"
            ].append(
                relative_error_from_logs(
                    targeted_yield[
                        "log_evidence"
                    ],
                    reference_yield,
                )
            )

            uniform_evidence_relative_errors[
                "SUPPORT_SLIP"
            ].append(
                relative_error_from_logs(
                    uniform_slip[
                        "log_evidence"
                    ],
                    reference_slip,
                )
            )

            uniform_evidence_relative_errors[
                "MATERIAL_YIELD"
            ].append(
                relative_error_from_logs(
                    uniform_yield[
                        "log_evidence"
                    ],
                    reference_yield,
                )
            )

    targeted_ess_summary = {
        label: summarize(
            values
        )
        for (
            label,
            values,
        ) in targeted_ess.items()
    }

    uniform_ess_summary = {
        label: summarize(
            values
        )
        for (
            label,
            values,
        ) in uniform_ess.items()
    }

    targeted_error_summary = (
        summarize(
            targeted_posterior_errors
        )
    )

    uniform_error_summary = (
        summarize(
            uniform_posterior_errors
        )
    )

    targeted_posterior_sd = [
        float(
            np.std(
                values,
                ddof=1,
            )
        )
        for values in (
            targeted_posteriors_by_observation
        )
    ]

    uniform_posterior_sd = [
        float(
            np.std(
                values,
                ddof=1,
            )
        )
        for values in (
            uniform_posteriors_by_observation
        )
    ]

    targeted_evidence_summary = {
        label: {
            "signed_relative_error": (
                summarize(
                    values
                )
            ),
            "absolute_relative_error": (
                summarize(
                    [
                        abs(
                            value
                        )
                        for value in values
                    ]
                )
            ),
        }
        for (
            label,
            values,
        ) in (
            targeted_evidence_relative_errors.items()
        )
    }

    uniform_evidence_summary = {
        label: {
            "signed_relative_error": (
                summarize(
                    values
                )
            ),
            "absolute_relative_error": (
                summarize(
                    [
                        abs(
                            value
                        )
                        for value in values
                    ]
                )
            ),
        }
        for (
            label,
            values,
        ) in (
            uniform_evidence_relative_errors.items()
        )
    }

    ess_criterion = audit[
        "realized_ess_criterion"
    ]

    targeted_ess_pass = all(
        targeted_ess_summary[
            label
        ][
            "p05"
        ]
        >= ess_criterion[
            "minimum_p05_ess_fraction"
        ]
        and targeted_ess_summary[
            label
        ][
            "median"
        ]
        >= ess_criterion[
            "minimum_median_ess_fraction"
        ]
        for label in [
            "SUPPORT_SLIP",
            "MATERIAL_YIELD",
        ]
    )

    posterior_criterion = audit[
        "posterior_accuracy_criterion"
    ]

    posterior_accuracy_pass = (
        targeted_error_summary[
            "p95"
        ]
        <= posterior_criterion[
            "p95_absolute_posterior_error_max"
        ]
        and targeted_error_summary[
            "max"
        ]
        <= posterior_criterion[
            "maximum_absolute_posterior_error_max"
        ]
    )

    normalization_tolerance = audit[
        "proposal_normalization"
    ][
        "absolute_tolerance"
    ]

    proposal_normalization_pass = (
        maximum_proposal_normalization_error
        <= normalization_tolerance
    )

    identity_tolerance = audit[
        "importance_identity_check"
    ][
        "absolute_tolerance"
    ]

    importance_identity_pass = (
        maximum_identity_error
        <= identity_tolerance
    )

    success = (
        targeted_ess_pass
        and posterior_accuracy_pass
        and proposal_normalization_pass
        and importance_identity_pass
    )

    status = (
        "PHASE0_5B_REALIZED_SAMPLER_AUDIT_PASS"
        if success
        else "PHASE0_5B_REALIZED_SAMPLER_AUDIT_FAIL"
    )

    result = {
        "experiment": audit[
            "experiment"
        ],
        "manifest_version": audit[
            "manifest_version"
        ],
        "manifest_sha256": sha256_file(
            AUDIT_MANIFEST_PATH
        ),
        "code_commit_sha": (
            current_commit_sha()
        ),
        "status": status,
        "historical_record": {
            "phase0_5_attempt_1_remains_invalid": True,
            "phase0_5b_theoretical_validation_retained": True,
            "historical_artifacts_modified": False,
        },
        "dataset": {
            "generated_pairs": (
                dataset_config[
                    "n_pairs"
                ]
            ),
            "generated_rows": len(
                records
            ),
            "test_rows": len(
                split.test
            ),
            "evaluation_rows": len(
                evaluation_rows
            ),
            "dataset_seed": (
                dataset_config[
                    "dataset_seed"
                ]
            ),
            "split_seed": (
                dataset_config[
                    "split_seed"
                ]
            ),
        },
        "particle_bank": {
            "total_particles": len(
                particles
            ),
            "particles_per_class": (
                expected_per_class
            ),
            "particle_seed": (
                bank_config[
                    "particle_seed"
                ]
            ),
        },
        "sampler": {
            "sample_size_per_class": (
                sample_size
            ),
            "replicates": replicates,
            "sampler_seed_base": (
                base_seed
            ),
            "actual_categorical_sampling": True,
        },
        "proposal": {
            "alpha": alpha,
            "tau_m": tau,
            "target": (
                "uniform empirical "
                "class-conditional particle bank"
            ),
        },
        "targeted": {
            "realized_ess": (
                targeted_ess_summary
            ),
            "posterior_absolute_error": (
                targeted_error_summary
            ),
            "posterior_sd_across_replicates": (
                summarize(
                    targeted_posterior_sd
                )
            ),
            "evidence_relative_error": (
                targeted_evidence_summary
            ),
            "ess_pass": (
                targeted_ess_pass
            ),
            "posterior_accuracy_pass": (
                posterior_accuracy_pass
            ),
        },
        "uniform_control": {
            "realized_ess": (
                uniform_ess_summary
            ),
            "posterior_absolute_error": (
                uniform_error_summary
            ),
            "posterior_sd_across_replicates": (
                summarize(
                    uniform_posterior_sd
                )
            ),
            "evidence_relative_error": (
                uniform_evidence_summary
            ),
        },
        "sanity_checks": {
            "maximum_proposal_normalization_error": (
                maximum_proposal_normalization_error
            ),
            "proposal_normalization_pass": (
                proposal_normalization_pass
            ),
            "maximum_importance_identity_error": (
                maximum_identity_error
            ),
            "importance_identity_pass": (
                importance_identity_pass
            ),
        },
        "interpretation": (
            audit[
                "interpretation_if_pass"
            ]
            if success
            else audit[
                "interpretation_if_fail"
            ]
        ),
    }

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    return result


def main():
    result = (
        run_audit()
    )

    print(
        "=== PHASE 0.5B REALIZED SAMPLER AUDIT ==="
    )

    print(
        "status=",
        result[
            "status"
        ],
    )

    print(
        "manifest_sha256=",
        result[
            "manifest_sha256"
        ],
    )

    print(
        "code_commit_sha=",
        result[
            "code_commit_sha"
        ],
    )

    print()

    print(
        "Targeted realized ESS:"
    )

    print(
        json.dumps(
            result[
                "targeted"
            ][
                "realized_ess"
            ],
            indent=2,
            sort_keys=True,
        )
    )

    print()

    print(
        "Uniform realized ESS:"
    )

    print(
        json.dumps(
            result[
                "uniform_control"
            ][
                "realized_ess"
            ],
            indent=2,
            sort_keys=True,
        )
    )

    print()

    print(
        "Targeted posterior error:"
    )

    print(
        json.dumps(
            result[
                "targeted"
            ][
                "posterior_absolute_error"
            ],
            indent=2,
            sort_keys=True,
        )
    )

    print()

    print(
        "Targeted posterior SD:"
    )

    print(
        json.dumps(
            result[
                "targeted"
            ][
                "posterior_sd_across_replicates"
            ],
            indent=2,
            sort_keys=True,
        )
    )

    print()

    print(
        "Proposal normalization pass=",
        result[
            "sanity_checks"
        ][
            "proposal_normalization_pass"
        ],
    )

    print(
        "Importance identity pass=",
        result[
            "sanity_checks"
        ][
            "importance_identity_pass"
        ],
    )

    print(
        "Targeted ESS pass=",
        result[
            "targeted"
        ][
            "ess_pass"
        ],
    )

    print(
        "Posterior accuracy pass=",
        result[
            "targeted"
        ][
            "posterior_accuracy_pass"
        ],
    )

    print()

    print(
        "PHASE 0.5 ATTEMPT 1 REMAINS INVALID"
    )

    print(
        "HISTORICAL PHASE 0.5B ARTIFACTS UNCHANGED"
    )

    print(
        "SAMPLER AUDIT ARTIFACT WRITTEN"
    )


if __name__ == "__main__":
    main()