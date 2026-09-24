from dataclasses import dataclass
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess

import numpy as np

from phase0_5_dataset import (
    add_observation_noise,
    sample_shared_noise,
)
from phase0_5_margin import (
    evaluate_pair_margin,
)
from phase0_5_pairs import (
    IdentificationPair,
    generate_identification_pair,
)
from phase0_5_safe_cohort import (
    true_force_components,
)
from physics import (
    Action,
    GRAVITY,
    NextTransition,
    predict_next_transition,
    run_safe_probe,
    support_load_fraction,
)


MANIFEST_PATH = Path(
    "PHASE1A_FEASIBILITY_MANIFEST.json"
)

RESULTS_PATH = Path(
    "phase1a_feasibility_results.json"
)

PAIRS_PATH = Path(
    "phase1a_feasibility_pairs.csv"
)


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    force_N: float
    angle_deg: float

    def action(self) -> Action:
        return Action(
            force=self.force_N,
            angle_deg=self.angle_deg,
        )


@dataclass
class WorldLegality:
    legal: bool
    reasons: tuple[str, ...]
    rho: float
    normal_force: float
    first_transition_force: float
    transition_margin: float


@dataclass
class PairActionLegality:
    pair_id: int
    action_id: str
    legal: bool
    world_a: WorldLegality
    world_b: WorldLegality
    reasons: tuple[str, ...]
    primary_reason: str | None


@dataclass
class ParticleBank:
    support_slip: np.ndarray
    material_yield: np.ndarray


REASON_TRANSITION = (
    "TRANSITION_BEFORE_OR_AT_PROBE"
)

REASON_RHO = (
    "SUPPORT_RHO_LIMIT"
)

REASON_NORMAL = (
    "NONPOSITIVE_NORMAL_FORCE"
)

REASON_CLIPPED = (
    "CLIPPED_SUPPORT_REGIME"
)


PARAMETERS = [
    "static_friction",
    "material_yield_strength",
    "container_mass",
    "contact_area",
    "contact_stiffness",
    "support_gain",
    "kinetic_friction",
]


def load_manifest() -> dict:
    with MANIFEST_PATH.open(
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
        != "FROZEN_BEFORE_FEASIBILITY_RUN"
    ):
        raise ValueError(
            "Phase 1A feasibility manifest is not frozen"
        )

    if (
        manifest.get(
            "manifest_version"
        )
        != "1.0"
    ):
        raise ValueError(
            "Unexpected Phase 1A manifest version"
        )

    return manifest


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


def action_specs(
    manifest: dict,
) -> list[ActionSpec]:
    return [
        ActionSpec(
            action_id=row[
                "action_id"
            ],
            force_N=float(
                row[
                    "force_N"
                ]
            ),
            angle_deg=float(
                row[
                    "angle_deg"
                ]
            ),
        )
        for row in manifest[
            "action_grid"
        ]
    ]


def percentile_summary(
    values,
) -> dict:
    array = np.asarray(
        list(
            values
        ),
        dtype=float,
    )

    if len(
        array
    ) == 0:
        return {
            "min": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
        }

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


def target_margin_for_world(
    params,
    target_angle_deg: float,
) -> float:
    (
        _,
        material_force,
        slip_force,
    ) = predict_next_transition(
        params,
        target_angle_deg,
    )

    denominator = min(
        material_force,
        slip_force,
    )

    if math.isinf(
        denominator
    ):
        return 0.0

    return (
        abs(
            material_force
            - slip_force
        )
        / denominator
    )


def pair_is_target_valid(
    pair: IdentificationPair,
    target_angle_deg: float,
    epsilon_tie: float,
) -> bool:
    margin = evaluate_pair_margin(
        pair=pair,
        angle_deg=target_angle_deg,
        epsilon_tie=epsilon_tie,
    )

    return (
        margin.separated_pair
    )


def generate_target_valid_pairs(
    n_pairs: int,
    seed: int,
    target_angle_deg: float,
    epsilon_tie: float,
    max_attempts_per_pair: int,
) -> list[IdentificationPair]:
    random.seed(
        seed
    )

    accepted = []
    candidate_id = 0

    max_total_attempts = (
        n_pairs
        * max_attempts_per_pair
    )

    attempts = 0

    while (
        len(
            accepted
        )
        < n_pairs
    ):
        attempts += 1

        if attempts > max_total_attempts:
            raise RuntimeError(
                "Could not generate enough target-valid "
                "Phase 1A identification pairs"
            )

        pair = generate_identification_pair(
            pair_id=candidate_id,
            angle_deg=target_angle_deg,
            max_attempts=max_attempts_per_pair,
        )

        candidate_id += 1

        if not pair_is_target_valid(
            pair=pair,
            target_angle_deg=target_angle_deg,
            epsilon_tie=epsilon_tie,
        ):
            continue

        new_id = len(
            accepted
        )

        pair.pair_id = new_id

        accepted.append(
            pair
        )

    return accepted


def evaluate_world_legality(
    params,
    action: Action,
    rho_safe_max: float,
) -> WorldLegality:
    (
        true_fx,
        true_fy,
    ) = true_force_components(
        action
    )

    normal_force = (
        params.container_mass
        * GRAVITY
        + true_fy
    )

    rho = support_load_fraction(
        params=params,
        true_fx=true_fx,
        true_fy=true_fy,
    )

    (
        _,
        material_force,
        slip_force,
    ) = predict_next_transition(
        params,
        action.angle_deg,
    )

    first_transition_force = min(
        material_force,
        slip_force,
    )

    if (
        math.isinf(
            first_transition_force
        )
    ):
        transition_margin = 1.0

    else:
        transition_margin = (
            first_transition_force
            - action.force
        ) / first_transition_force

    reasons = []

    if (
        action.force
        >= first_transition_force
    ):
        reasons.append(
            REASON_TRANSITION
        )

    if (
        rho
        >= rho_safe_max
    ):
        reasons.append(
            REASON_RHO
        )

    if (
        normal_force
        <= 0.0
    ):
        reasons.append(
            REASON_NORMAL
        )

    if (
        rho
        >= 1.0
    ):
        reasons.append(
            REASON_CLIPPED
        )

    return WorldLegality(
        legal=(
            len(
                reasons
            )
            == 0
        ),
        reasons=tuple(
            reasons
        ),
        rho=float(
            rho
        ),
        normal_force=float(
            normal_force
        ),
        first_transition_force=float(
            first_transition_force
        ),
        transition_margin=float(
            transition_margin
        ),
    )


def evaluate_pair_action(
    pair: IdentificationPair,
    spec: ActionSpec,
    rho_safe_max: float,
    precedence: list[str],
) -> PairActionLegality:
    action = spec.action()

    world_a = evaluate_world_legality(
        params=pair.world_a,
        action=action,
        rho_safe_max=rho_safe_max,
    )

    world_b = evaluate_world_legality(
        params=pair.world_b,
        action=action,
        rho_safe_max=rho_safe_max,
    )

    combined = set(
        world_a.reasons
    ) | set(
        world_b.reasons
    )

    reasons = tuple(
        reason
        for reason in precedence
        if reason in combined
    )

    primary_reason = (
        reasons[
            0
        ]
        if reasons
        else None
    )

    return PairActionLegality(
        pair_id=pair.pair_id,
        action_id=spec.action_id,
        legal=(
            world_a.legal
            and world_b.legal
        ),
        world_a=world_a,
        world_b=world_b,
        reasons=reasons,
        primary_reason=primary_reason,
    )


def exact_reason_key(
    reasons: tuple[str, ...],
) -> str:
    if not reasons:
        return "NONE"

    return "+".join(
        reasons
    )


def standardized_mean_difference(
    original,
    selected,
) -> float:
    original = np.asarray(
        original,
        dtype=float,
    )

    selected = np.asarray(
        selected,
        dtype=float,
    )

    if (
        len(
            original
        )
        < 2
        or len(
            selected
        )
        < 2
    ):
        return float(
            "nan"
        )

    mean_difference = (
        float(
            np.mean(
                selected
            )
        )
        - float(
            np.mean(
                original
            )
        )
    )

    variance_original = float(
        np.var(
            original,
            ddof=1,
        )
    )

    variance_selected = float(
        np.var(
            selected,
            ddof=1,
        )
    )

    pooled = math.sqrt(
        (
            variance_original
            + variance_selected
        )
        / 2.0
    )

    if pooled == 0.0:
        return (
            0.0
            if mean_difference == 0.0
            else math.copysign(
                float(
                    "inf"
                ),
                mean_difference,
            )
        )

    return (
        mean_difference
        / pooled
    )


def parameter_shift_summary(
    original_pairs: list[IdentificationPair],
    universal_pairs: list[IdentificationPair],
) -> dict:
    result = {}

    for parameter in PARAMETERS:
        original_values = []
        universal_values = []

        for pair in original_pairs:
            original_values.extend(
                [
                    getattr(
                        pair.world_a,
                        parameter,
                    ),
                    getattr(
                        pair.world_b,
                        parameter,
                    ),
                ]
            )

        for pair in universal_pairs:
            universal_values.extend(
                [
                    getattr(
                        pair.world_a,
                        parameter,
                    ),
                    getattr(
                        pair.world_b,
                        parameter,
                    ),
                ]
            )

        result[
            parameter
        ] = {
            "original_mean": float(
                np.mean(
                    original_values
                )
            ),
            "universal_mean": float(
                np.mean(
                    universal_values
                )
            )
            if universal_values
            else None,
            "original_median": float(
                np.median(
                    original_values
                )
            ),
            "universal_median": float(
                np.median(
                    universal_values
                )
            )
            if universal_values
            else None,
            "standardized_mean_difference": (
                standardized_mean_difference(
                    original_values,
                    universal_values,
                )
                if universal_values
                else None
            ),
        }

    return result


def gaussian_log_likelihoods(
    observation: np.ndarray,
    means: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    normalization = np.log(
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
            normalization
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

    maximum = float(
        np.max(
            kernel_log
        )
    )

    kernel = np.exp(
        kernel_log
        - maximum
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


def realized_ess_fraction(
    log_likelihoods: np.ndarray,
    q: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> float:
    n_bank = len(
        log_likelihoods
    )

    prior_probability = (
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

    likelihood = np.exp(
        sampled_logs
        - maximum
    )

    weights = (
        prior_probability
        * likelihood
        / sampled_q
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
                ** 2
            )
        )
    )

    if denominator <= 0.0:
        raise RuntimeError(
            "Invalid realized ESS denominator"
        )

    return (
        numerator
        / denominator
    )


def generate_action_legal_pair(
    pair_id: int,
    spec: ActionSpec,
    target_angle_deg: float,
    epsilon_tie: float,
    rho_safe_max: float,
    max_attempts: int,
) -> IdentificationPair:
    precedence = [
        REASON_TRANSITION,
        REASON_RHO,
        REASON_NORMAL,
        REASON_CLIPPED,
    ]

    for _ in range(
        max_attempts
    ):
        pair = generate_identification_pair(
            pair_id=pair_id,
            angle_deg=target_angle_deg,
            max_attempts=max_attempts,
        )

        if not pair_is_target_valid(
            pair=pair,
            target_angle_deg=target_angle_deg,
            epsilon_tie=epsilon_tie,
        ):
            continue

        check = evaluate_pair_action(
            pair=pair,
            spec=spec,
            rho_safe_max=rho_safe_max,
            precedence=precedence,
        )

        if check.legal:
            return pair

    raise RuntimeError(
        f"Could not generate legal pair for "
        f"{spec.action_id}"
    )


def noiseless_action_observation(
    params,
    spec: ActionSpec,
) -> np.ndarray:
    result = run_safe_probe(
        params=params,
        action=spec.action(),
        support_model="compliant",
        force_noise_std=0.0,
        motion_noise_std=0.0,
    )

    if not (
        result.still_blocked
    ):
        raise RuntimeError(
            "Bayes particle crossed transition"
        )

    observation = (
        result.observation
    )

    return np.asarray(
        [
            observation.fx,
            observation.fy,
            observation.tip_dx,
            observation.tip_dy,
        ],
        dtype=float,
    )


def build_action_particle_bank(
    spec: ActionSpec,
    n_particles_per_class: int,
    seed: int,
    target_angle_deg: float,
    epsilon_tie: float,
    rho_safe_max: float,
    max_attempts: int,
) -> ParticleBank:
    random.seed(
        seed
    )

    slip = []
    yield_rows = []

    for pair_id in range(
        n_particles_per_class
    ):
        pair = generate_action_legal_pair(
            pair_id=pair_id,
            spec=spec,
            target_angle_deg=target_angle_deg,
            epsilon_tie=epsilon_tie,
            rho_safe_max=rho_safe_max,
            max_attempts=max_attempts,
        )

        for (
            params,
            label,
        ) in [
            (
                pair.world_a,
                pair.label_a,
            ),
            (
                pair.world_b,
                pair.label_b,
            ),
        ]:
            observation = (
                noiseless_action_observation(
                    params=params,
                    spec=spec,
                )
            )

            if (
                label
                == NextTransition.SUPPORT_SLIP
            ):
                slip.append(
                    observation
                )

            elif (
                label
                == NextTransition.MATERIAL_YIELD
            ):
                yield_rows.append(
                    observation
                )

            else:
                raise RuntimeError(
                    "Unknown target label"
                )

    if (
        len(
            slip
        )
        != n_particles_per_class
        or len(
            yield_rows
        )
        != n_particles_per_class
    ):
        raise RuntimeError(
            "Per-action particle bank is not balanced"
        )

    return ParticleBank(
        support_slip=np.asarray(
            slip,
            dtype=float,
        ),
        material_yield=np.asarray(
            yield_rows,
            dtype=float,
        ),
    )


def generate_action_evaluation_rows(
    spec: ActionSpec,
    n_rows: int,
    seed: int,
    target_angle_deg: float,
    epsilon_tie: float,
    rho_safe_max: float,
    max_attempts: int,
    force_noise_std: float,
    motion_noise_std: float,
) -> list[np.ndarray]:
    if (
        n_rows
        % 2
        != 0
    ):
        raise ValueError(
            "rows_per_action must be even"
        )

    random.seed(
        seed
    )

    rows = []

    for pair_id in range(
        n_rows
        // 2
    ):
        pair = generate_action_legal_pair(
            pair_id=pair_id,
            spec=spec,
            target_angle_deg=target_angle_deg,
            epsilon_tie=epsilon_tie,
            rho_safe_max=rho_safe_max,
            max_attempts=max_attempts,
        )

        shared_noise = sample_shared_noise(
            force_noise_std=force_noise_std,
            motion_noise_std=motion_noise_std,
        )

        for params in [
            pair.world_a,
            pair.world_b,
        ]:
            result = run_safe_probe(
                params=params,
                action=spec.action(),
                support_model="compliant",
                force_noise_std=0.0,
                motion_noise_std=0.0,
            )

            noisy = add_observation_noise(
                result.observation,
                shared_noise,
            )

            rows.append(
                np.asarray(
                    [
                        noisy.fx,
                        noisy.fy,
                        noisy.tip_dx,
                        noisy.tip_dy,
                    ],
                    dtype=float,
                )
            )

    return rows


def evaluate_action_ess(
    spec: ActionSpec,
    action_index: int,
    manifest: dict,
) -> dict:
    gate = manifest[
        "bayes_numerical_gate"
    ]

    target = manifest[
        "target"
    ]

    population = manifest[
        "paired_identification_population"
    ]

    safety = manifest[
        "safety"
    ]

    bank_config = gate[
        "particle_bank"
    ]

    evaluation = gate[
        "evaluation"
    ]

    proposal = gate[
        "proposal"
    ]

    noise = gate[
        "noise_model"
    ]

    particle_seed = (
        bank_config[
            "particle_seed_base"
        ]
        + action_index
    )

    bank = build_action_particle_bank(
        spec=spec,
        n_particles_per_class=(
            bank_config[
                "particles_per_class"
            ]
        ),
        seed=particle_seed,
        target_angle_deg=(
            target[
                "reference_angle_deg"
            ]
        ),
        epsilon_tie=(
            target[
                "epsilon_tie"
            ]
        ),
        rho_safe_max=(
            safety[
                "rho_safe_max"
            ]
        ),
        max_attempts=(
            population[
                "pair_generation_max_attempts"
            ]
        ),
    )

    evaluation_seed = (
        evaluation[
            "evaluation_seed_base"
        ]
        + action_index
    )

    rows = generate_action_evaluation_rows(
        spec=spec,
        n_rows=(
            evaluation[
                "rows_per_action"
            ]
        ),
        seed=evaluation_seed,
        target_angle_deg=(
            target[
                "reference_angle_deg"
            ]
        ),
        epsilon_tie=(
            target[
                "epsilon_tie"
            ]
        ),
        rho_safe_max=(
            safety[
                "rho_safe_max"
            ]
        ),
        max_attempts=(
            population[
                "pair_generation_max_attempts"
            ]
        ),
        force_noise_std=(
            noise[
                "force_noise_std_N"
            ]
        ),
        motion_noise_std=(
            noise[
                "motion_noise_std_m"
            ]
        ),
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

    alpha = (
        proposal[
            "uniform_prior_mixture_weight_alpha"
        ]
    )

    tau = proposal[
        "tau_m"
    ]

    sample_size = (
        evaluation[
            "sampler_sample_size_per_class"
        ]
    )

    replicates = (
        evaluation[
            "sampler_replicates"
        ]
    )

    slip_ess = []
    yield_ess = []

    for (
        observation_index,
        observation,
    ) in enumerate(
        rows
    ):
        slip_logs = (
            gaussian_log_likelihoods(
                observation=observation,
                means=(
                    bank.support_slip
                ),
                std=std,
            )
        )

        yield_logs = (
            gaussian_log_likelihoods(
                observation=observation,
                means=(
                    bank.material_yield
                ),
                std=std,
            )
        )

        slip_q = (
            proposal_probabilities(
                tip_dx_means=(
                    bank.support_slip[
                        :,
                        2
                    ]
                ),
                observed_tip_dx=float(
                    observation[
                        2
                    ]
                ),
                alpha=alpha,
                tau=tau,
            )
        )

        yield_q = (
            proposal_probabilities(
                tip_dx_means=(
                    bank.material_yield[
                        :,
                        2
                    ]
                ),
                observed_tip_dx=float(
                    observation[
                        2
                    ]
                ),
                alpha=alpha,
                tau=tau,
            )
        )

        for replicate_index in range(
            replicates
        ):
            seed = (
                evaluation[
                    "sampler_seed_base"
                ]
                + action_index
                * 100000
                + observation_index
                * 100
                + replicate_index
            )

            rng = np.random.default_rng(
                seed
            )

            slip_ess.append(
                realized_ess_fraction(
                    log_likelihoods=(
                        slip_logs
                    ),
                    q=slip_q,
                    sample_size=sample_size,
                    rng=rng,
                )
            )

            yield_ess.append(
                realized_ess_fraction(
                    log_likelihoods=(
                        yield_logs
                    ),
                    q=yield_q,
                    sample_size=sample_size,
                    rng=rng,
                )
            )

    slip_summary = percentile_summary(
        slip_ess
    )

    yield_summary = percentile_summary(
        yield_ess
    )

    passes = (
        slip_summary[
            "p05"
        ]
        >= gate[
            "minimum_p05_ess_fraction"
        ]
        and slip_summary[
            "median"
        ]
        >= gate[
            "minimum_median_ess_fraction"
        ]
        and yield_summary[
            "p05"
        ]
        >= gate[
            "minimum_p05_ess_fraction"
        ]
        and yield_summary[
            "median"
        ]
        >= gate[
            "minimum_median_ess_fraction"
        ]
    )

    return {
        "particle_seed": particle_seed,
        "evaluation_seed": (
            evaluation_seed
        ),
        "SUPPORT_SLIP": {
            "p05": (
                slip_summary[
                    "p05"
                ]
            ),
            "median": (
                slip_summary[
                    "median"
                ]
            ),
        },
        "MATERIAL_YIELD": {
            "p05": (
                yield_summary[
                    "p05"
                ]
            ),
            "median": (
                yield_summary[
                    "median"
                ]
            ),
        },
        "pass": bool(
            passes
        ),
    }


def write_pair_rows(
    pairs: list[IdentificationPair],
    legality_by_action: dict,
    universal_ids: set[int],
    target_angle_deg: float,
) -> None:
    rows = []

    for pair in pairs:
        for (
            side,
            world,
            label,
        ) in [
            (
                "A",
                pair.world_a,
                pair.label_a.value,
            ),
            (
                "B",
                pair.world_b,
                pair.label_b.value,
            ),
        ]:
            row = {
                "pair_id": pair.pair_id,
                "side": side,
                "label": label,
                "universal_legal": (
                    pair.pair_id
                    in universal_ids
                ),
                "target_margin": (
                    target_margin_for_world(
                        world,
                        target_angle_deg,
                    )
                ),
            }

            for parameter in PARAMETERS:
                row[
                    parameter
                ] = getattr(
                    world,
                    parameter,
                )

            for (
                action_id,
                pair_checks,
            ) in legality_by_action.items():
                check = pair_checks[
                    pair.pair_id
                ]

                world_check = (
                    check.world_a
                    if side == "A"
                    else check.world_b
                )

                row[
                    f"{action_id}_legal"
                ] = world_check.legal

                row[
                    f"{action_id}_rho"
                ] = world_check.rho

                row[
                    f"{action_id}_transition_margin"
                ] = (
                    world_check.transition_margin
                )

                row[
                    f"{action_id}_reasons"
                ] = "|".join(
                    world_check.reasons
                )

            rows.append(
                row
            )

    with PAIRS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def run_feasibility() -> dict:
    manifest = load_manifest()

    specs = action_specs(
        manifest
    )

    target = manifest[
        "target"
    ]

    population = manifest[
        "paired_identification_population"
    ]

    safety = manifest[
        "safety"
    ]

    precedence = manifest[
        "rejection_reasons"
    ][
        "primary_precedence"
    ]

    pairs = generate_target_valid_pairs(
        n_pairs=(
            population[
                "n_target_valid_pairs"
            ]
        ),
        seed=(
            population[
                "dataset_seed"
            ]
        ),
        target_angle_deg=(
            target[
                "reference_angle_deg"
            ]
        ),
        epsilon_tie=(
            target[
                "epsilon_tie"
            ]
        ),
        max_attempts_per_pair=(
            population[
                "pair_generation_max_attempts"
            ]
        ),
    )

    legality_by_action = {}

    action_summaries = {}

    for spec in specs:
        checks = {}

        legal_pair_ids = set()

        legal_world_count = 0

        primary_counts = {
            reason: 0
            for reason in precedence
        }

        overlap_counts = {}

        all_rho = []

        legal_transition_margins = []

        transition_rejected_pairs = 0

        for pair in pairs:
            check = evaluate_pair_action(
                pair=pair,
                spec=spec,
                rho_safe_max=(
                    safety[
                        "rho_safe_max"
                    ]
                ),
                precedence=precedence,
            )

            checks[
                pair.pair_id
            ] = check

            all_rho.extend(
                [
                    check.world_a.rho,
                    check.world_b.rho,
                ]
            )

            if check.world_a.legal:
                legal_world_count += 1
                legal_transition_margins.append(
                    check.world_a.transition_margin
                )

            if check.world_b.legal:
                legal_world_count += 1
                legal_transition_margins.append(
                    check.world_b.transition_margin
                )

            if check.legal:
                legal_pair_ids.add(
                    pair.pair_id
                )

            else:
                if (
                    check.primary_reason
                    is not None
                ):
                    primary_counts[
                        check.primary_reason
                    ] += 1

                key = exact_reason_key(
                    check.reasons
                )

                overlap_counts[
                    key
                ] = (
                    overlap_counts.get(
                        key,
                        0,
                    )
                    + 1
                )

                if (
                    REASON_TRANSITION
                    in check.reasons
                ):
                    transition_rejected_pairs += 1

        legality_by_action[
            spec.action_id
        ] = checks

        action_summaries[
            spec.action_id
        ] = {
            "force_N": spec.force_N,
            "angle_deg": spec.angle_deg,
            "legal_pair_count": len(
                legal_pair_ids
            ),
            "legal_pair_fraction": (
                len(
                    legal_pair_ids
                )
                / len(
                    pairs
                )
            ),
            "legal_world_fraction": (
                legal_world_count
                / (
                    2
                    * len(
                        pairs
                    )
                )
            ),
            "primary_rejection_counts": (
                primary_counts
            ),
            "rejection_overlap_counts": (
                overlap_counts
            ),
            "transition_rejected_pair_fraction": (
                transition_rejected_pairs
                / len(
                    pairs
                )
            ),
            "rho": percentile_summary(
                all_rho
            ),
            "legal_transition_margin": (
                percentile_summary(
                    legal_transition_margins
                )
            ),
            "legal_pair_ids": (
                legal_pair_ids
            ),
        }

    universal_ids = set(
        pair.pair_id
        for pair in pairs
    )

    for spec in specs:
        universal_ids &= (
            action_summaries[
                spec.action_id
            ][
                "legal_pair_ids"
            ]
        )

    universal_fraction = (
        len(
            universal_ids
        )
        / len(
            pairs
        )
    )

    rejected_by_action_count = {
        str(
            count
        ): 0
        for count in range(
            0,
            len(
                specs
            )
            + 1
        )
    }

    for pair in pairs:
        failures = sum(
            not legality_by_action[
                spec.action_id
            ][
                pair.pair_id
            ].legal
            for spec in specs
        )

        rejected_by_action_count[
            str(
                failures
            )
        ] += 1

    leave_one_out = {}

    for omitted in specs:
        ids = set(
            pair.pair_id
            for pair in pairs
        )

        for retained in specs:
            if (
                retained.action_id
                == omitted.action_id
            ):
                continue

            ids &= (
                action_summaries[
                    retained.action_id
                ][
                    "legal_pair_ids"
                ]
            )

        fraction = (
            len(
                ids
            )
            / len(
                pairs
            )
        )

        leave_one_out[
            omitted.action_id
        ] = {
            "fraction_without_action": (
                fraction
            ),
            "absolute_gain": (
                fraction
                - universal_fraction
            ),
        }

    nesting = []

    for left in specs:
        left_ids = (
            action_summaries[
                left.action_id
            ][
                "legal_pair_ids"
            ]
        )

        for right in specs:
            if (
                left.action_id
                == right.action_id
            ):
                continue

            right_ids = (
                action_summaries[
                    right.action_id
                ][
                    "legal_pair_ids"
                ]
            )

            if left_ids <= right_ids:
                nesting.append(
                    {
                        "subset": (
                            left.action_id
                        ),
                        "superset": (
                            right.action_id
                        ),
                        "exact": (
                            left_ids
                            == right_ids
                        ),
                    }
                )

    universal_pairs = [
        pair
        for pair in pairs
        if pair.pair_id
        in universal_ids
    ]

    parameter_shift = (
        parameter_shift_summary(
            original_pairs=pairs,
            universal_pairs=(
                universal_pairs
            ),
        )
    )

    original_target_margins = []

    universal_target_margins = []

    for pair in pairs:
        for world in [
            pair.world_a,
            pair.world_b,
        ]:
            margin = (
                target_margin_for_world(
                    world,
                    target[
                        "reference_angle_deg"
                    ],
                )
            )

            original_target_margins.append(
                margin
            )

            if (
                pair.pair_id
                in universal_ids
            ):
                universal_target_margins.append(
                    margin
                )

    original_margin_summary = (
        percentile_summary(
            original_target_margins
        )
    )

    universal_margin_summary = (
        percentile_summary(
            universal_target_margins
        )
    )

    epsilon = target[
        "epsilon_tie"
    ]

    near_upper = (
        2.0
        * epsilon
    )

    original_near_fraction = (
        float(
            np.mean(
                (
                    np.asarray(
                        original_target_margins
                    )
                    >= epsilon
                )
                & (
                    np.asarray(
                        original_target_margins
                    )
                    < near_upper
                )
            )
        )
    )

    universal_near_fraction = (
        float(
            np.mean(
                (
                    np.asarray(
                        universal_target_margins
                    )
                    >= epsilon
                )
                & (
                    np.asarray(
                        universal_target_margins
                    )
                    < near_upper
                )
            )
        )
        if universal_target_margins
        else None
    )

    label_counts = {
        NextTransition.MATERIAL_YIELD.value: 0,
        NextTransition.SUPPORT_SLIP.value: 0,
    }

    for pair in universal_pairs:
        label_counts[
            pair.label_a.value
        ] += 1

        label_counts[
            pair.label_b.value
        ] += 1

    if universal_pairs:
        expected = len(
            universal_pairs
        )

        if (
            label_counts[
                NextTransition.MATERIAL_YIELD.value
            ]
            != expected
            or label_counts[
                NextTransition.SUPPORT_SLIP.value
            ]
            != expected
        ):
            raise RuntimeError(
                "Universal cohort lost paired class balance"
            )

    bayes_ess = {}

    if manifest[
        "bayes_numerical_gate"
    ][
        "enabled"
    ]:
        for (
            action_index,
            spec,
        ) in enumerate(
            specs
        ):
            print(
                "Bayes ESS gate:",
                spec.action_id,
            )

            bayes_ess[
                spec.action_id
            ] = (
                evaluate_action_ess(
                    spec=spec,
                    action_index=(
                        action_index
                    ),
                    manifest=manifest,
                )
            )

    triggers = manifest[
        "redesign_triggers"
    ]

    triggered = []

    if (
        universal_fraction
        < triggers[
            "universal_legal_pair_fraction_below"
        ]
    ):
        triggered.append(
            "UNIVERSAL_COHORT_TOO_SMALL"
        )

    max_leave_one_out_gain = max(
        row[
            "absolute_gain"
        ]
        for row in (
            leave_one_out.values()
        )
    )

    if (
        max_leave_one_out_gain
        >= triggers[
            "single_action_leave_one_out_gain_at_or_above"
        ]
    ):
        triggered.append(
            "SINGLE_ACTION_BOTTLENECK"
        )

    for (
        action_id,
        summary,
    ) in action_summaries.items():
        if (
            summary[
                "transition_rejected_pair_fraction"
            ]
            >= triggers[
                "transition_before_probe_pair_fraction_at_or_above"
            ]
        ):
            triggered.append(
                (
                    "TRANSITION_BEFORE_PROBE:"
                    + action_id
                )
            )

    for (
        parameter,
        summary,
    ) in parameter_shift.items():
        smd = summary[
            "standardized_mean_difference"
        ]

        if (
            smd is not None
            and abs(
                smd
            )
            >= triggers[
                "absolute_standardized_mean_difference_at_or_above"
            ]
        ):
            triggered.append(
                (
                    "HIDDEN_PARAMETER_SHIFT:"
                    + parameter
                )
            )

    if (
        universal_margin_summary[
            "median"
        ]
        is not None
        and original_margin_summary[
            "median"
        ]
        not in (
            None,
            0.0,
        )
    ):
        relative_shift = abs(
            universal_margin_summary[
                "median"
            ]
            - original_margin_summary[
                "median"
            ]
        ) / abs(
            original_margin_summary[
                "median"
            ]
        )

    else:
        relative_shift = None

    if (
        relative_shift is not None
        and relative_shift
        >= triggers[
            "target_margin_median_relative_shift_at_or_above"
        ]
    ):
        triggered.append(
            "TARGET_MARGIN_DISTORTION"
        )

    for (
        action_id,
        ess,
    ) in bayes_ess.items():
        if not ess[
            "pass"
        ]:
            triggered.append(
                (
                    "BAYES_ESS_FAILURE:"
                    + action_id
                )
            )

    write_pair_rows(
        pairs=pairs,
        legality_by_action=(
            legality_by_action
        ),
        universal_ids=(
            universal_ids
        ),
        target_angle_deg=(
            target[
                "reference_angle_deg"
            ]
        ),
    )

    serializable_actions = {}

    for (
        action_id,
        summary,
    ) in action_summaries.items():
        serializable = dict(
            summary
        )

        del serializable[
            "legal_pair_ids"
        ]

        serializable_actions[
            action_id
        ] = serializable

    result = {
        "experiment": manifest[
            "experiment"
        ],
        "manifest_version": (
            manifest[
                "manifest_version"
            ]
        ),
        "code_commit_sha": (
            current_commit_sha()
        ),
        "status": (
            "PHASE1A_REDESIGN_REQUIRED"
            if triggered
            else "PHASE1A_GRID_FEASIBLE"
        ),
        "forbidden_action_value_outputs_computed": False,
        "population": {
            "target_valid_pairs": len(
                pairs
            ),
            "target_valid_worlds": (
                2
                * len(
                    pairs
                )
            ),
            "dataset_seed": (
                population[
                    "dataset_seed"
                ]
            ),
        },
        "actions": (
            serializable_actions
        ),
        "universal_cohort": {
            "legal_pairs": len(
                universal_ids
            ),
            "legal_worlds": (
                2
                * len(
                    universal_ids
                )
            ),
            "legal_pair_fraction": (
                universal_fraction
            ),
            "rejected_by_number_of_actions": (
                rejected_by_action_count
            ),
        },
        "leave_one_out": (
            leave_one_out
        ),
        "legality_nesting": (
            nesting
        ),
        "hidden_parameter_shift": (
            parameter_shift
        ),
        "target_margin_shift": {
            "near_threshold_definition": (
                f"{epsilon} <= margin < "
                f"{near_upper}"
            ),
            "original": {
                **original_margin_summary,
                "near_threshold_fraction": (
                    original_near_fraction
                ),
            },
            "universal": {
                **universal_margin_summary,
                "near_threshold_fraction": (
                    universal_near_fraction
                ),
            },
            "median_relative_shift": (
                relative_shift
            ),
        },
        "class_balance": {
            "MATERIAL_YIELD": (
                label_counts[
                    NextTransition.MATERIAL_YIELD.value
                ]
            ),
            "SUPPORT_SLIP": (
                label_counts[
                    NextTransition.SUPPORT_SLIP.value
                ]
            ),
            "paired_balance_invariant_pass": (
                label_counts[
                    NextTransition.MATERIAL_YIELD.value
                ]
                == label_counts[
                    NextTransition.SUPPORT_SLIP.value
                ]
            ),
        },
        "bayes_ess_gate": (
            bayes_ess
        ),
        "redesign_triggers": {
            "triggered": (
                triggered
            ),
            "count": len(
                triggered
            ),
        },
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
    result = run_feasibility()

    print()
    print(
        "=== PHASE 1A ACTION-GRID FEASIBILITY ==="
    )

    print(
        "status=",
        result[
            "status"
        ],
    )

    print(
        "target_valid_pairs=",
        result[
            "population"
        ][
            "target_valid_pairs"
        ],
    )

    print(
        "universal_pair_fraction=",
        round(
            result[
                "universal_cohort"
            ][
                "legal_pair_fraction"
            ],
            6,
        ),
    )

    print()

    print(
        "Per-action legal pair fractions:"
    )

    for (
        action_id,
        row,
    ) in result[
        "actions"
    ].items():
        print(
            action_id,
            "force=",
            row[
                "force_N"
            ],
            "angle=",
            row[
                "angle_deg"
            ],
            "legal_pair_fraction=",
            round(
                row[
                    "legal_pair_fraction"
                ],
                6,
            ),
        )

    print()

    print(
        "Leave-one-out gains:"
    )

    for (
        action_id,
        row,
    ) in result[
        "leave_one_out"
    ].items():
        print(
            action_id,
            round(
                row[
                    "absolute_gain"
                ],
                6,
            ),
        )

    print()

    print(
        "Bayes ESS gates:"
    )

    for (
        action_id,
        row,
    ) in result[
        "bayes_ess_gate"
    ].items():
        print(
            action_id,
            "pass=",
            row[
                "pass"
            ],
            "slip_p05=",
            round(
                row[
                    "SUPPORT_SLIP"
                ][
                    "p05"
                ],
                6,
            ),
            "yield_p05=",
            round(
                row[
                    "MATERIAL_YIELD"
                ][
                    "p05"
                ],
                6,
            ),
        )

    print()

    print(
        "Redesign triggers:"
    )

    print(
        json.dumps(
            result[
                "redesign_triggers"
            ],
            indent=2,
        )
    )

    print()

    print(
        "NO AUC, INFORMATION GAIN, OR ACTION "
        "RANKING WAS COMPUTED"
    )

    print(
        "PHASE 1A ARTIFACTS WRITTEN"
    )


if __name__ == "__main__":
    main()