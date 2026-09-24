from dataclasses import (
    asdict,
    dataclass,
    replace,
)
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess

import numpy as np

from sklearn.metrics import (
    roc_auc_score,
)

from phase0_5_bayes import (
    sample_bayes_particles,
)
from phase0_5_bayes_ess import (
    ObservationESS,
    summarize_ess,
)
from phase0_5_bootstrap import (
    fit_test_scores,
    pair_cluster_bootstrap_auc,
)
from phase0_5_controls import (
    CONTROL_FORCE_ONLY,
    CONTROL_FULL,
    CONTROL_MOTION_ONLY,
    CONTROL_RHO_CANARY,
    CONTROL_RIGID_NULL,
    CONTROL_SHAM,
    make_feature_view,
    observe_pair_with_support_model,
    pair_preserving_shuffled_labels,
)
from phase0_5_dataset import (
    ObservationRecord,
    sample_shared_noise,
)
from phase0_5_evaluate import (
    fnr_at_max_fpr,
    labels_to_binary,
    records_for_pair_ids,
)
from phase0_5_margin import (
    normalized_threshold_margin_from_forces,
)
from phase0_5_safe_cohort import (
    generate_safe_pair_sample,
    true_force_components,
    world_is_still_blocked,
)
from phase0_5_split import (
    pair_level_train_test_split,
)
from phase0_5_worlds import (
    sample_phase0_5_world,
)
from physics import (
    NextTransition,
    predict_next_transition,
)


MANIFEST_PATH = Path(
    "PHASE0_5_MANIFEST.json"
)

RESULTS_PATH = Path(
    "phase0_5_confirmatory_results.json"
)

RECORDS_PATH = Path(
    "phase0_5_confirmatory_records.csv"
)

OPERATIONAL_PATH = Path(
    "phase0_5_operational_prior.json"
)


@dataclass
class ControlResult:
    control: str
    train_pairs: int
    test_pairs: int
    train_rows: int
    test_rows: int
    roc_auc: float
    ci_low: float
    ci_high: float
    fnr_at_fpr_10: float


@dataclass
class BayesResult:
    particles: int
    particles_per_class: int
    test_rows: int
    roc_auc: float
    ess_pass: bool
    ess: dict


def load_manifest(
    path: Path = MANIFEST_PATH,
) -> dict:
    """
    Load the frozen confirmatory manifest.
    """

    with path.open(
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
        != "FROZEN_BEFORE_CONFIRMATORY_RUN"
    ):
        raise ValueError(
            "Manifest is not frozen for confirmatory use"
        )

    return manifest


def manifest_sha256(
    path: Path = MANIFEST_PATH,
) -> str:
    """
    SHA256 of the exact manifest bytes used by the run.
    """

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
    """
    Return the GitHub Actions SHA when available.

    Fall back to local git for reproducible local execution.
    """

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


def validate_manifest(
    manifest: dict,
) -> None:
    """
    Validate critical frozen invariants before touching data.
    """

    if (
        manifest[
            "manifest_version"
        ]
        != "1.1"
    ):
        raise ValueError(
            "Unexpected manifest version"
        )

    if (
        manifest[
            "confirmatory_dataset"
        ][
            "rows_per_pair"
        ]
        != 2
    ):
        raise ValueError(
            "Confirmatory dataset must have two rows per pair"
        )

    if (
        manifest[
            "confirmatory_dataset"
        ][
            "split_unit"
        ]
        != "pair_id"
    ):
        raise ValueError(
            "Confirmatory split must operate by pair_id"
        )

    if (
        manifest[
            "probe"
        ][
            "force_N"
        ]
        != 5.0
    ):
        raise ValueError(
            "Frozen probe force does not match implementation"
        )

    if (
        manifest[
            "probe"
        ][
            "angle_deg"
        ]
        != 45.0
    ):
        raise ValueError(
            "Frozen probe angle does not match implementation"
        )

    if (
        manifest[
            "safe_cohort"
        ][
            "rho_safe_max"
        ]
        >= 1.0
    ):
        raise ValueError(
            "Frozen safe cohort permits clipped support regime"
        )

    if (
        manifest[
            "bayes_reference"
        ][
            "particles"
        ]
        % 2
        != 0
    ):
        raise ValueError(
            "Bayes particle count must be even"
        )

    if not (
        manifest[
            "bayes_convergence_criterion"
        ][
            "validated_before_freeze"
        ]
    ):
        raise ValueError(
            "Bayes convergence was not validated before freeze"
        )

    if not (
        manifest[
            "bayes_ess_criterion"
        ][
            "validated_before_freeze"
        ]
    ):
        raise ValueError(
            "Bayes ESS was not validated before freeze"
        )


def parameter_dict(
    params,
) -> dict:
    """
    Serialize one hidden physical world.
    """

    return {
        "material_yield_strength": (
            params.material_yield_strength
        ),
        "contact_area": (
            params.contact_area
        ),
        "container_mass": (
            params.container_mass
        ),
        "static_friction": (
            params.static_friction
        ),
        "kinetic_friction": (
            params.kinetic_friction
        ),
        "contact_stiffness": (
            params.contact_stiffness
        ),
        "pre_slip_displacement_limit": (
            params.pre_slip_displacement_limit
        ),
        "support_gain": (
            params.support_gain
        ),
    }


def generate_confirmatory_dataset(
    manifest: dict,
) -> tuple[
    list[ObservationRecord],
    list[ObservationRecord],
    list[dict],
]:
    """
    Generate the frozen paired identification dataset.

    Returns:

    - compliant legal observations
    - matched rigid-null observations
    - full reproducibility provenance rows
    """

    config = (
        manifest[
            "confirmatory_dataset"
        ]
    )

    safe_config = (
        manifest[
            "safe_cohort"
        ]
    )

    tie_config = (
        manifest[
            "counterfactual_label"
        ]
    )

    probe_config = (
        manifest[
            "probe"
        ]
    )

    noise_config = (
        manifest[
            "noise_model"
        ]
    )

    random.seed(
        config[
            "dataset_seed"
        ]
    )

    compliant_records = []
    rigid_records = []
    provenance_rows = []

    for pair_id in range(
        config[
            "n_pairs"
        ]
    ):
        from physics import Action

        action = Action(
            force=(
                probe_config[
                    "force_N"
                ]
            ),
            angle_deg=(
                probe_config[
                    "angle_deg"
                ]
            ),
        )

        sample = (
            generate_safe_pair_sample(
                pair_id=pair_id,
                action=action,
                rho_safe_max=(
                    safe_config[
                        "rho_safe_max"
                    ]
                ),
                epsilon_tie=(
                    tie_config[
                        "epsilon_tie"
                    ]
                ),
                max_attempts=(
                    safe_config[
                        "generation_max_attempts"
                    ]
                ),
            )
        )

        noise = (
            sample_shared_noise(
                force_noise_std=(
                    noise_config[
                        "force_noise_std_N"
                    ]
                ),
                motion_noise_std=(
                    noise_config[
                        "motion_noise_std_m"
                    ]
                ),
            )
        )

        (
            compliant_a,
            compliant_b,
        ) = (
            observe_pair_with_support_model(
                sample=sample,
                noise=noise,
                support_model="compliant",
            )
        )

        (
            rigid_a,
            rigid_b,
        ) = (
            observe_pair_with_support_model(
                sample=sample,
                noise=noise,
                support_model="rigid",
            )
        )

        compliant_records.extend(
            [
                compliant_a,
                compliant_b,
            ]
        )

        rigid_records.extend(
            [
                rigid_a,
                rigid_b,
            ]
        )

        (
            true_fx,
            true_fy,
        ) = (
            true_force_components(
                action
            )
        )

        for (
            side,
            params,
            record,
            rigid_record,
        ) in [
            (
                "A",
                sample.pair.world_a,
                compliant_a,
                rigid_a,
            ),
            (
                "B",
                sample.pair.world_b,
                compliant_b,
                rigid_b,
            ),
        ]:
            normal_force = (
                params.container_mass
                * 9.81
                + true_fy
            )

            if normal_force <= 0.0:
                raise RuntimeError(
                    "Confirmatory sample has nonpositive normal force"
                )

            if (
                record.rho
                >= safe_config[
                    "rho_safe_max"
                ]
            ):
                raise RuntimeError(
                    "Confirmatory sample violates rho safety bound"
                )

            if (
                record.rho
                >= 1.0
            ):
                raise RuntimeError(
                    "Confirmatory sample entered clipped support regime"
                )

            if not (
                world_is_still_blocked(
                    params,
                    action,
                )
            ):
                raise RuntimeError(
                    "Confirmatory probe reached a transition"
                )

            row = {
                "pair_id": pair_id,
                "side": side,
                "label": record.label,
                "action_force_N": (
                    action.force
                ),
                "action_angle_deg": (
                    action.angle_deg
                ),
                "true_fx": true_fx,
                "true_fy": true_fy,

                "noise_fx": noise.fx,
                "noise_fy": noise.fy,
                "noise_tip_dx": (
                    noise.tip_dx
                ),
                "noise_tip_dy": (
                    noise.tip_dy
                ),

                "fx": record.fx,
                "fy": record.fy,
                "tip_dx": (
                    record.tip_dx
                ),
                "tip_dy": (
                    record.tip_dy
                ),

                "rigid_fx": (
                    rigid_record.fx
                ),
                "rigid_fy": (
                    rigid_record.fy
                ),
                "rigid_tip_dx": (
                    rigid_record.tip_dx
                ),
                "rigid_tip_dy": (
                    rigid_record.tip_dy
                ),

                "rho": record.rho,
                "transition_margin": (
                    record.margin
                ),
                "normal_force": (
                    normal_force
                ),
            }

            row.update(
                parameter_dict(
                    params
                )
            )

            provenance_rows.append(
                row
            )

    expected_rows = (
        config[
            "n_pairs"
        ]
        * 2
    )

    if (
        len(
            compliant_records
        )
        != expected_rows
    ):
        raise RuntimeError(
            "Unexpected compliant dataset size"
        )

    if (
        len(
            rigid_records
        )
        != expected_rows
    ):
        raise RuntimeError(
            "Unexpected rigid dataset size"
        )

    return (
        compliant_records,
        rigid_records,
        provenance_rows,
    )


def apply_global_label_shuffle(
    records: list[ObservationRecord],
    seed: int,
) -> list[ObservationRecord]:
    """
    Apply the frozen label shuffle once to the complete dataset.

    Splitting happens only after this global shuffled-label
    assignment has been created.
    """

    shuffled_labels = (
        pair_preserving_shuffled_labels(
            records=records,
            seed=seed,
        )
    )

    return [
        replace(
            record,
            label=label,
        )
        for (
            record,
            label,
        ) in zip(
            records,
            shuffled_labels,
        )
    ]


def evaluate_control_records(
    train_records: list[ObservationRecord],
    test_records: list[ObservationRecord],
    control: str,
    display_name: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> ControlResult:
    """
    Fit one frozen control condition and compute mandatory
    held-out metrics.
    """

    train_view = (
        make_feature_view(
            records=train_records,
            control=control,
            seed=bootstrap_seed,
        )
    )

    test_view = (
        make_feature_view(
            records=test_records,
            control=control,
            seed=bootstrap_seed,
        )
    )

    (
        y_test,
        scores,
    ) = fit_test_scores(
        train_view=train_view,
        test_view=test_view,
    )

    auc = float(
        roc_auc_score(
            y_test,
            scores,
        )
    )

    fnr = float(
        fnr_at_max_fpr(
            y_true=y_test,
            scores=scores,
            max_fpr=0.10,
        )
    )

    (
        ci_low,
        ci_high,
    ) = (
        pair_cluster_bootstrap_auc(
            y_true=y_test,
            scores=scores,
            pair_ids=(
                test_view.pair_ids
            ),
            n_bootstrap=(
                bootstrap_replicates
            ),
            seed=bootstrap_seed,
        )
    )

    train_pair_ids = {
        record.pair_id
        for record in train_records
    }

    test_pair_ids = {
        record.pair_id
        for record in test_records
    }

    if not (
        train_pair_ids.isdisjoint(
            test_pair_ids
        )
    ):
        raise RuntimeError(
            "Train/test pair leakage detected"
        )

    return ControlResult(
        control=display_name,
        train_pairs=len(
            train_pair_ids
        ),
        test_pairs=len(
            test_pair_ids
        ),
        train_rows=len(
            train_records
        ),
        test_rows=len(
            test_records
        ),
        roc_auc=auc,
        ci_low=ci_low,
        ci_high=ci_high,
        fnr_at_fpr_10=fnr,
    )


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


def ess_from_log_weights(
    values: np.ndarray,
) -> float:
    maximum = float(
        np.max(
            values
        )
    )

    weights = np.exp(
        values
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


def evaluate_bayes_on_test(
    test_records: list[ObservationRecord],
    manifest: dict,
) -> BayesResult:
    """
    Evaluate the frozen 20k-particle Bayes reference on the
    exact same held-out compliant records as the classifier.

    Likelihood evaluation is vectorized for the official run.
    """

    config = (
        manifest[
            "bayes_reference"
        ]
    )

    noise = (
        manifest[
            "noise_model"
        ]
    )

    n_particles = (
        config[
            "particles"
        ]
    )

    particles = (
        sample_bayes_particles(
            n_particles=n_particles,
            seed=(
                config[
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
            "Bayes particle prior is not class balanced"
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

    normalization_terms = (
        np.log(
            2.0
            * math.pi
            * (
                std
                ** 2
            )
        )
    )

    probabilities = []
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

        slip_log_evidence = (
            stable_logsumexp(
                slip_logs
            )
            - math.log(
                n_slip
            )
        )

        yield_log_evidence = (
            stable_logsumexp(
                yield_logs
            )
            - math.log(
                n_yield
            )
        )

        difference = (
            slip_log_evidence
            - yield_log_evidence
        )

        if difference >= 0.0:
            probability = (
                1.0
                / (
                    1.0
                    + math.exp(
                        -difference
                    )
                )
            )

        else:
            exp_difference = (
                math.exp(
                    difference
                )
            )

            probability = (
                exp_difference
                / (
                    1.0
                    + exp_difference
                )
            )

        probabilities.append(
            probability
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

    y_true = (
        labels_to_binary(
            [
                record.label
                for record in test_records
            ]
        )
    )

    auc = float(
        roc_auc_score(
            y_true,
            np.asarray(
                probabilities,
                dtype=float,
            ),
        )
    )

    ess_summary = (
        summarize_ess(
            observations=ess_rows,
            n_particles=n_particles,
        )
    )

    criterion = (
        manifest[
            "bayes_ess_criterion"
        ]
    )

    slip_pass = (
        ess_summary.slip_fraction_p05
        >= criterion[
            "minimum_p05_ess_fraction"
        ]
        and ess_summary.slip_fraction_median
        >= criterion[
            "minimum_median_ess_fraction"
        ]
    )

    yield_pass = (
        ess_summary.yield_fraction_p05
        >= criterion[
            "minimum_p05_ess_fraction"
        ]
        and ess_summary.yield_fraction_median
        >= criterion[
            "minimum_median_ess_fraction"
        ]
    )

    ess_pass = (
        slip_pass
        and yield_pass
    )

    if not ess_pass:
        raise RuntimeError(
            "Confirmatory Bayes ESS criterion failed"
        )

    return BayesResult(
        particles=n_particles,
        particles_per_class=(
            n_particles
            // 2
        ),
        test_rows=len(
            test_records
        ),
        roc_auc=auc,
        ess_pass=ess_pass,
        ess=asdict(
            ess_summary
        ),
    )


def run_operational_prior(
    manifest: dict,
) -> dict:
    """
    Estimate frozen operational-prior prevalence separately
    from the balanced identification experiment.
    """

    config = (
        manifest[
            "operational_prior"
        ]
    )

    epsilon_tie = (
        manifest[
            "counterfactual_label"
        ][
            "epsilon_tie"
        ]
    )

    random.seed(
        config[
            "seed"
        ]
    )

    counts = {
        NextTransition.MATERIAL_YIELD.value: 0,
        NextTransition.SUPPORT_SLIP.value: 0,
    }

    near_ties = 0

    for _ in range(
        config[
            "n_worlds"
        ]
    ):
        world = (
            sample_phase0_5_world()
        )

        (
            label,
            material_force,
            slip_force,
        ) = (
            predict_next_transition(
                world,
                config[
                    "reference_angle_deg"
                ],
            )
        )

        counts[
            label.value
        ] += 1

        margin = (
            normalized_threshold_margin_from_forces(
                material_force=material_force,
                slip_force=slip_force,
            )
        )

        if margin < epsilon_tie:
            near_ties += 1

    n_worlds = (
        config[
            "n_worlds"
        ]
    )

    return {
        "n_worlds": n_worlds,
        "seed": config[
            "seed"
        ],
        "reference_angle_deg": (
            config[
                "reference_angle_deg"
            ]
        ),
        "counts": counts,
        "prevalence": {
            label: (
                count
                / n_worlds
            )
            for (
                label,
                count,
            ) in counts.items()
        },
        "near_tie_count": (
            near_ties
        ),
        "near_tie_fraction": (
            near_ties
            / n_worlds
        ),
    }


def write_records_csv(
    rows: list[dict],
    split,
    path: Path = RECORDS_PATH,
) -> None:
    """
    Save the complete provenance table.
    """

    train_ids = (
        split.train_pair_ids
    )

    test_ids = (
        split.test_pair_ids
    )

    output_rows = []

    for row in rows:
        row = dict(
            row
        )

        if (
            row[
                "pair_id"
            ]
            in train_ids
        ):
            row[
                "split"
            ] = "train"

        elif (
            row[
                "pair_id"
            ]
            in test_ids
        ):
            row[
                "split"
            ] = "test"

        else:
            raise RuntimeError(
                "Pair missing from train/test assignment"
            )

        output_rows.append(
            row
        )

    with path.open(
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


def run_confirmatory() -> dict:
    """
    Execute the frozen Phase 0.5 confirmatory experiment.
    """

    manifest = (
        load_manifest()
    )

    validate_manifest(
        manifest
    )

    manifest_hash = (
        manifest_sha256()
    )

    commit_sha = (
        current_commit_sha()
    )

    (
        compliant_records,
        rigid_records,
        provenance_rows,
    ) = (
        generate_confirmatory_dataset(
            manifest
        )
    )

    config = (
        manifest[
            "confirmatory_dataset"
        ]
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

    bootstrap = (
        manifest[
            "bootstrap"
        ]
    )

    results = []

    for control in [
        CONTROL_FULL,
        CONTROL_FORCE_ONLY,
        CONTROL_MOTION_ONLY,
        CONTROL_SHAM,
        CONTROL_RHO_CANARY,
    ]:
        results.append(
            evaluate_control_records(
                train_records=split.train,
                test_records=split.test,
                control=control,
                display_name=control,
                bootstrap_replicates=(
                    bootstrap[
                        "replicates"
                    ]
                ),
                bootstrap_seed=(
                    bootstrap[
                        "seed"
                    ]
                ),
            )
        )

    shuffle_seed = (
        manifest[
            "controls"
        ][
            "LABEL_SHUFFLE"
        ][
            "seed"
        ]
    )

    shuffled_records = (
        apply_global_label_shuffle(
            records=compliant_records,
            seed=shuffle_seed,
        )
    )

    shuffled_train = (
        records_for_pair_ids(
            records=shuffled_records,
            pair_ids=(
                split.train_pair_ids
            ),
        )
    )

    shuffled_test = (
        records_for_pair_ids(
            records=shuffled_records,
            pair_ids=(
                split.test_pair_ids
            ),
        )
    )

    results.append(
        evaluate_control_records(
            train_records=(
                shuffled_train
            ),
            test_records=(
                shuffled_test
            ),
            control=CONTROL_FULL,
            display_name=(
                "LABEL_SHUFFLE"
            ),
            bootstrap_replicates=(
                bootstrap[
                    "replicates"
                ]
            ),
            bootstrap_seed=(
                bootstrap[
                    "seed"
                ]
            ),
        )
    )

    rigid_train = (
        records_for_pair_ids(
            records=rigid_records,
            pair_ids=(
                split.train_pair_ids
            ),
        )
    )

    rigid_test = (
        records_for_pair_ids(
            records=rigid_records,
            pair_ids=(
                split.test_pair_ids
            ),
        )
    )

    results.append(
        evaluate_control_records(
            train_records=rigid_train,
            test_records=rigid_test,
            control=CONTROL_RIGID_NULL,
            display_name=(
                CONTROL_RIGID_NULL
            ),
            bootstrap_replicates=(
                bootstrap[
                    "replicates"
                ]
            ),
            bootstrap_seed=(
                bootstrap[
                    "seed"
                ]
            ),
        )
    )

    bayes = (
        evaluate_bayes_on_test(
            test_records=split.test,
            manifest=manifest,
        )
    )

    operational = (
        run_operational_prior(
            manifest
        )
    )

    write_records_csv(
        rows=provenance_rows,
        split=split,
    )

    with OPERATIONAL_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            operational,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    final = {
        "experiment": (
            manifest[
                "experiment"
            ]
        ),
        "manifest_version": (
            manifest[
                "manifest_version"
            ]
        ),
        "manifest_sha256": (
            manifest_hash
        ),
        "code_commit_sha": (
            commit_sha
        ),
        "status": (
            "CONFIRMATORY_RUN_COMPLETE"
        ),
        "confirmatory_dataset": {
            "n_pairs": (
                config[
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
                config[
                    "dataset_seed"
                ]
            ),
            "split_seed": (
                config[
                    "split_seed"
                ]
            ),
        },
        "controls": [
            asdict(
                result
            )
            for result in results
        ],
        "bayes_reference": (
            asdict(
                bayes
            )
        ),
        "operational_prior": (
            operational
        ),
        "artifacts": {
            "results": str(
                RESULTS_PATH
            ),
            "records": str(
                RECORDS_PATH
            ),
            "operational_prior": str(
                OPERATIONAL_PATH
            ),
        },
    }

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            final,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    return final


def main():
    result = (
        run_confirmatory()
    )

    print(
        "=== PHASE 0.5 CONFIRMATORY RUN ==="
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
        "control | AUC | 95% CI | FNR@FPR<=0.10"
    )

    for item in (
        result[
            "controls"
        ]
    ):
        print(
            f"{item['control']:14s}"
            f" | {item['roc_auc']:.4f}"
            f" | ["
            f"{item['ci_low']:.4f}, "
            f"{item['ci_high']:.4f}]"
            f" | {item['fnr_at_fpr_10']:.4f}"
        )

    bayes = (
        result[
            "bayes_reference"
        ]
    )

    print()

    print(
        "Bayes AUC=",
        round(
            bayes[
                "roc_auc"
            ],
            4,
        ),
    )

    print(
        "Bayes ESS pass=",
        bayes[
            "ess_pass"
        ],
    )

    operational = (
        result[
            "operational_prior"
        ]
    )

    print()

    print(
        "Operational prior prevalence="
    )

    print(
        json.dumps(
            operational[
                "prevalence"
            ],
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "Near-tie fraction=",
        round(
            operational[
                "near_tie_fraction"
            ],
            6,
        ),
    )

    print()

    print(
        "CONFIRMATORY ARTIFACTS WRITTEN"
    )


if __name__ == "__main__":
    main()