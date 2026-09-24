from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from sklearn.metrics import roc_auc_score

from phase0_5_bayes import (
    sample_bayes_particles,
)
from phase0_5_evaluate import (
    labels_to_binary,
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


PHASE0_5B_MANIFEST_PATH = Path(
    "PHASE0_5B_MANIFEST.json"
)

RESULTS_PATH = Path(
    "phase0_5b_validation_results.json"
)

RECORDS_PATH = Path(
    "phase0_5b_validation_records.csv"
)


def load_phase0_5b_manifest() -> dict:
    with PHASE0_5B_MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    if (
        manifest[
            "status"
        ]
        != "FROZEN_BEFORE_VALIDATION_RUN"
    ):
        raise ValueError(
            "Phase 0.5b manifest is not frozen"
        )

    if (
        manifest[
            "manifest_version"
        ]
        != "1.0"
    ):
        raise ValueError(
            "Unexpected Phase 0.5b manifest version"
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


def proposal_probabilities(
    tip_dx_means: np.ndarray,
    observed_tip_dx: float,
    alpha: float,
    tau: float,
) -> np.ndarray:
    n = len(
        tip_dx_means
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

    kernel = (
        stable_exp_weights(
            kernel_log
        )
    )

    kernel = (
        kernel
        / float(
            np.sum(
                kernel
            )
        )
    )

    return (
        alpha
        * prior_probability
        + (
            1.0
            - alpha
        )
        * kernel
    )


def targeted_ess_fraction(
    log_likelihoods: np.ndarray,
    tip_dx_means: np.ndarray,
    observed_tip_dx: float,
    alpha: float,
    tau: float,
) -> float:
    n = len(
        log_likelihoods
    )

    prior_probability = (
        1.0
        / n
    )

    q = (
        proposal_probabilities(
            tip_dx_means=tip_dx_means,
            observed_tip_dx=(
                observed_tip_dx
            ),
            alpha=alpha,
            tau=tau,
        )
    )

    likelihood = (
        stable_exp_weights(
            log_likelihoods
        )
    )

    importance_weight = (
        prior_probability
        * likelihood
        / q
    )

    expected_weight = float(
        np.sum(
            q
            * importance_weight
        )
    )

    expected_weight_squared = float(
        np.sum(
            q
            * (
                importance_weight
                ** 2
            )
        )
    )

    return (
        expected_weight
        ** 2
        / expected_weight_squared
    )


def uniform_log_evidence(
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


def targeted_log_evidence(
    log_likelihoods: np.ndarray,
    tip_dx_means: np.ndarray,
    observed_tip_dx: float,
    alpha: float,
    tau: float,
) -> float:
    """
    Importance-corrected evidence under the original
    uniform class-conditional particle prior.

    The proposal changes computational allocation only.
    It does not change the scientific prior.
    """

    n = len(
        log_likelihoods
    )

    prior_probability = (
        1.0
        / n
    )

    q = (
        proposal_probabilities(
            tip_dx_means=tip_dx_means,
            observed_tip_dx=(
                observed_tip_dx
            ),
            alpha=alpha,
            tau=tau,
        )
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

    corrected_weight = (
        prior_probability
        * scaled_likelihood
        / q
    )

    evidence_scaled = float(
        np.sum(
            q
            * corrected_weight
        )
    )

    return (
        maximum
        + math.log(
            evidence_scaled
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


def make_validation_parent_manifest(
    parent: dict,
    validation: dict,
) -> dict:
    """
    Reuse the frozen Phase 0.5 physical generator while
    substituting only the preregistered Phase 0.5b
    independent dataset settings.
    """

    derived = deepcopy(
        parent
    )

    config = (
        validation[
            "validation_dataset"
        ]
    )

    derived[
        "confirmatory_dataset"
    ][
        "n_pairs"
    ] = config[
        "n_pairs"
    ]

    derived[
        "confirmatory_dataset"
    ][
        "total_rows"
    ] = config[
        "total_rows"
    ]

    derived[
        "confirmatory_dataset"
    ][
        "test_fraction"
    ] = config[
        "test_fraction"
    ]

    derived[
        "confirmatory_dataset"
    ][
        "dataset_seed"
    ] = config[
        "dataset_seed"
    ]

    derived[
        "confirmatory_dataset"
    ][
        "split_seed"
    ] = config[
        "split_seed"
    ]

    return derived


def validate_frozen_relationship(
    parent: dict,
    validation: dict,
) -> None:
    proposal = (
        validation[
            "proposal"
        ]
    )

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

    if not (
        proposal[
            "importance_weight_correction_required"
        ]
    ):
        raise ValueError(
            "Importance correction must remain required"
        )

    if (
        validation[
            "probe"
        ][
            "force_N"
        ]
        != parent[
            "probe"
        ][
            "force_N"
        ]
    ):
        raise ValueError(
            "Phase 0.5b probe force differs from parent"
        )

    if (
        validation[
            "probe"
        ][
            "angle_deg"
        ]
        != parent[
            "probe"
        ][
            "angle_deg"
        ]
    ):
        raise ValueError(
            "Phase 0.5b probe angle differs from parent"
        )

    if (
        validation[
            "noise_model"
        ][
            "force_noise_std_N"
        ]
        != parent[
            "noise_model"
        ][
            "force_noise_std_N"
        ]
    ):
        raise ValueError(
            "Force noise changed from parent"
        )

    if (
        validation[
            "noise_model"
        ][
            "motion_noise_std_m"
        ]
        != parent[
            "noise_model"
        ][
            "motion_noise_std_m"
        ]
    ):
        raise ValueError(
            "Motion noise changed from parent"
        )

    if (
        validation[
            "ess_criterion"
        ][
            "minimum_p05_ess_fraction"
        ]
        != parent[
            "bayes_ess_criterion"
        ][
            "minimum_p05_ess_fraction"
        ]
    ):
        raise ValueError(
            "ESS p05 threshold changed"
        )

    if (
        validation[
            "ess_criterion"
        ][
            "minimum_median_ess_fraction"
        ]
        != parent[
            "bayes_ess_criterion"
        ][
            "minimum_median_ess_fraction"
        ]
    ):
        raise ValueError(
            "ESS median threshold changed"
        )


def write_records(
    rows: list[dict],
    split,
) -> None:
    output_rows = []

    for row in rows:
        output = dict(
            row
        )

        if (
            row[
                "pair_id"
            ]
            in split.train_pair_ids
        ):
            output[
                "split"
            ] = "train"

        elif (
            row[
                "pair_id"
            ]
            in split.test_pair_ids
        ):
            output[
                "split"
            ] = "test"

        else:
            raise RuntimeError(
                "Validation pair missing from split"
            )

        output_rows.append(
            output
        )

    with RECORDS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                output_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )


def run_validation() -> dict:
    validation = (
        load_phase0_5b_manifest()
    )

    parent = (
        load_manifest()
    )

    validate_frozen_relationship(
        parent=parent,
        validation=validation,
    )

    generator_manifest = (
        make_validation_parent_manifest(
            parent=parent,
            validation=validation,
        )
    )

    (
        compliant_records,
        _,
        provenance_rows,
    ) = (
        generate_confirmatory_dataset(
            generator_manifest
        )
    )

    dataset_config = (
        validation[
            "validation_dataset"
        ]
    )

    split = (
        pair_level_train_test_split(
            records=compliant_records,
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

    bayes_config = (
        validation[
            "bayes_validation"
        ]
    )

    particles = (
        sample_bayes_particles(
            n_particles=(
                bayes_config[
                    "particles"
                ]
            ),
            seed=(
                bayes_config[
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

    if (
        n_slip
        != n_yield
    ):
        raise RuntimeError(
            "Phase 0.5b particle prior is not balanced"
        )

    slip_means = means[
        slip_mask
    ]

    yield_means = means[
        yield_mask
    ]

    noise = (
        validation[
            "noise_model"
        ]
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

    proposal = (
        validation[
            "proposal"
        ]
    )

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

    uniform_scores = []
    targeted_scores = []

    uniform_slip_ess = []
    uniform_yield_ess = []

    targeted_slip_ess = []
    targeted_yield_ess = []

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

        uniform_slip_log_evidence = (
            uniform_log_evidence(
                slip_logs
            )
        )

        uniform_yield_log_evidence = (
            uniform_log_evidence(
                yield_logs
            )
        )

        targeted_slip_log_evidence = (
            targeted_log_evidence(
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

        targeted_yield_log_evidence = (
            targeted_log_evidence(
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

        uniform_scores.append(
            posterior_slip_probability(
                uniform_slip_log_evidence,
                uniform_yield_log_evidence,
            )
        )

        targeted_scores.append(
            posterior_slip_probability(
                targeted_slip_log_evidence,
                targeted_yield_log_evidence,
            )
        )

        uniform_slip_ess.append(
            uniform_ess_fraction(
                slip_logs
            )
        )

        uniform_yield_ess.append(
            uniform_ess_fraction(
                yield_logs
            )
        )

        targeted_slip_ess.append(
            targeted_ess_fraction(
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

        targeted_yield_ess.append(
            targeted_ess_fraction(
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

    y_true = (
        labels_to_binary(
            [
                record.label
                for record in split.test
            ]
        )
    )

    uniform_auc = float(
        roc_auc_score(
            y_true,
            np.asarray(
                uniform_scores,
                dtype=float,
            ),
        )
    )

    targeted_auc = float(
        roc_auc_score(
            y_true,
            np.asarray(
                targeted_scores,
                dtype=float,
            ),
        )
    )

    uniform_summary = {
        "SUPPORT_SLIP": summarize(
            uniform_slip_ess
        ),
        "MATERIAL_YIELD": summarize(
            uniform_yield_ess
        ),
    }

    targeted_summary = {
        "SUPPORT_SLIP": summarize(
            targeted_slip_ess
        ),
        "MATERIAL_YIELD": summarize(
            targeted_yield_ess
        ),
    }

    criterion = (
        validation[
            "ess_criterion"
        ]
    )

    targeted_pass = (
        targeted_summary[
            "SUPPORT_SLIP"
        ][
            "p05"
        ]
        >= criterion[
            "minimum_p05_ess_fraction"
        ]
        and targeted_summary[
            "SUPPORT_SLIP"
        ][
            "median"
        ]
        >= criterion[
            "minimum_median_ess_fraction"
        ]
        and targeted_summary[
            "MATERIAL_YIELD"
        ][
            "p05"
        ]
        >= criterion[
            "minimum_p05_ess_fraction"
        ]
        and targeted_summary[
            "MATERIAL_YIELD"
        ][
            "median"
        ]
        >= criterion[
            "minimum_median_ess_fraction"
        ]
    )

    maximum_score_difference = float(
        np.max(
            np.abs(
                np.asarray(
                    uniform_scores
                )
                - np.asarray(
                    targeted_scores
                )
            )
        )
    )

    auc_difference = abs(
        targeted_auc
        - uniform_auc
    )

    estimator_consistent = (
        maximum_score_difference
        <= 1e-10
        and auc_difference
        <= 1e-12
    )

    success = (
        targeted_pass
        and estimator_consistent
    )

    status = (
        "PHASE0_5B_VALIDATION_PASS"
        if success
        else "PHASE0_5B_VALIDATION_FAIL"
    )

    write_records(
        rows=provenance_rows,
        split=split,
    )

    result = {
        "experiment": (
            validation[
                "experiment"
            ]
        ),
        "manifest_version": (
            validation[
                "manifest_version"
            ]
        ),
        "manifest_sha256": (
            sha256_file(
                PHASE0_5B_MANIFEST_PATH
            )
        ),
        "code_commit_sha": (
            current_commit_sha()
        ),
        "status": status,
        "validation_dataset": {
            "n_pairs": (
                dataset_config[
                    "n_pairs"
                ]
            ),
            "total_rows": len(
                compliant_records
            ),
            "train_pairs": len(
                split.train_pair_ids
            ),
            "test_pairs": len(
                split.test_pair_ids
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
        "proposal": {
            "alpha": alpha,
            "tau_m": tau,
            "particle_seed": (
                bayes_config[
                    "particle_seed"
                ]
            ),
            "particles": (
                bayes_config[
                    "particles"
                ]
            ),
        },
        "uniform": {
            "roc_auc": (
                uniform_auc
            ),
            "ess": (
                uniform_summary
            ),
        },
        "targeted": {
            "roc_auc": (
                targeted_auc
            ),
            "ess": (
                targeted_summary
            ),
            "ess_pass": (
                targeted_pass
            ),
        },
        "estimator_consistency": {
            "maximum_posterior_difference": (
                maximum_score_difference
            ),
            "auc_difference": (
                auc_difference
            ),
            "pass": (
                estimator_consistent
            ),
        },
        "parent_phase0_5_attempt_1_remains_invalid": true,
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
        run_validation()
    )

    print(
        "=== PHASE 0.5B BAYES REPAIR VALIDATION ==="
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
        "Uniform Bayes AUC=",
        round(
            result[
                "uniform"
            ][
                "roc_auc"
            ],
            6,
        ),
    )

    print(
        "Targeted Bayes AUC=",
        round(
            result[
                "targeted"
            ][
                "roc_auc"
            ],
            6,
        ),
    )

    print()

    print(
        "Uniform ESS="
    )

    print(
        json.dumps(
            result[
                "uniform"
            ][
                "ess"
            ],
            indent=2,
            sort_keys=True,
        )
    )

    print()

    print(
        "Targeted ESS="
    )

    print(
        json.dumps(
            result[
                "targeted"
            ][
                "ess"
            ],
            indent=2,
            sort_keys=True,
        )
    )

    print()

    print(
        "Targeted ESS pass=",
        result[
            "targeted"
        ][
            "ess_pass"
        ],
    )

    print(
        "Estimator consistency pass=",
        result[
            "estimator_consistency"
        ][
            "pass"
        ],
    )

    print(
        "Maximum posterior difference=",
        result[
            "estimator_consistency"
        ][
            "maximum_posterior_difference"
        ],
    )

    print()

    print(
        "PHASE 0.5 ATTEMPT 1 REMAINS INVALID"
    )

    print(
        "PHASE 0.5B VALIDATION ARTIFACTS WRITTEN"
    )


if __name__ == "__main__":
    main()