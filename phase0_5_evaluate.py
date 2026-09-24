from dataclasses import dataclass
import random

import numpy as np

from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import (
    make_pipeline,
)
from sklearn.preprocessing import (
    StandardScaler,
)

from phase0_5_controls import (
    CONTROL_FORCE_ONLY,
    CONTROL_FULL,
    CONTROL_LABEL_SHUFFLE,
    CONTROL_MOTION_ONLY,
    CONTROL_RHO_CANARY,
    CONTROL_RIGID_NULL,
    CONTROL_SHAM,
    FeatureView,
    generate_matched_control_dataset,
    make_feature_view,
)
from phase0_5_dataset import (
    ObservationRecord,
)
from phase0_5_split import (
    pair_level_train_test_split,
)
from physics import (
    NextTransition,
)


POSITIVE_LABEL = (
    NextTransition.SUPPORT_SLIP.value
)


@dataclass
class EvaluationResult:
    control: str
    train_rows: int
    test_rows: int
    train_pairs: int
    test_pairs: int
    roc_auc: float
    fnr_at_fpr_10: float


def labels_to_binary(
    labels: list[str],
) -> np.ndarray:
    """
    Encode SUPPORT_SLIP as the positive class.
    """

    allowed = {
        NextTransition.MATERIAL_YIELD.value,
        NextTransition.SUPPORT_SLIP.value,
    }

    unknown = (
        set(labels)
        - allowed
    )

    if unknown:
        raise ValueError(
            f"Unknown labels: {sorted(unknown)}"
        )

    return np.array(
        [
            1
            if label == POSITIVE_LABEL
            else 0
            for label in labels
        ],
        dtype=int,
    )


def records_for_pair_ids(
    records: list[ObservationRecord],
    pair_ids: set[int],
) -> list[ObservationRecord]:
    """
    Select complete records belonging to a set of pair IDs.
    """

    return [
        record
        for record in records
        if record.pair_id in pair_ids
    ]


def fnr_at_max_fpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    max_fpr: float = 0.10,
) -> float:
    """
    Return the minimum false-negative rate achievable while
    keeping false-positive rate at or below max_fpr.

    Lower is better.
    """

    if not (
        0.0
        <= max_fpr
        < 1.0
    ):
        raise ValueError(
            "max_fpr must be between 0 and 1"
        )

    fpr, tpr, _ = roc_curve(
        y_true,
        scores,
    )

    allowed = (
        fpr
        <= max_fpr
    )

    if not np.any(
        allowed
    ):
        return 1.0

    best_tpr = float(
        np.max(
            tpr[
                allowed
            ]
        )
    )

    return (
        1.0
        - best_tpr
    )


def train_and_score(
    train_view: FeatureView,
    test_view: FeatureView,
) -> tuple[
    float,
    float,
]:
    """
    Train a fold-local scaling + logistic regression pipeline
    and evaluate on a held-out pair-level test set.
    """

    x_train = np.asarray(
        train_view.features,
        dtype=float,
    )

    x_test = np.asarray(
        test_view.features,
        dtype=float,
    )

    y_train = labels_to_binary(
        train_view.labels
    )

    y_test = labels_to_binary(
        test_view.labels
    )

    if len(
        np.unique(
            y_train
        )
    ) != 2:
        raise ValueError(
            "Training set must contain both classes"
        )

    if len(
        np.unique(
            y_test
        )
    ) != 2:
        raise ValueError(
            "Test set must contain both classes"
        )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
        ),
    )

    model.fit(
        x_train,
        y_train,
    )

    scores = (
        model.predict_proba(
            x_test
        )[
            :,
            1
        ]
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

    return (
        auc,
        fnr,
    )


def evaluate_control(
    train_records: list[ObservationRecord],
    test_records: list[ObservationRecord],
    control: str,
    seed: int = 42,
) -> EvaluationResult:
    """
    Evaluate one control using already separated pair-level
    train and test records.
    """

    train_view = (
        make_feature_view(
            records=train_records,
            control=control,
            seed=seed,
        )
    )

    test_view = (
        make_feature_view(
            records=test_records,
            control=control,
            seed=seed + 1,
        )
    )

    auc, fnr = (
        train_and_score(
            train_view=train_view,
            test_view=test_view,
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
            "Pair leakage detected between train and test"
        )

    return EvaluationResult(
        control=control,
        train_rows=len(
            train_records
        ),
        test_rows=len(
            test_records
        ),
        train_pairs=len(
            train_pair_ids
        ),
        test_pairs=len(
            test_pair_ids
        ),
        roc_auc=auc,
        fnr_at_fpr_10=fnr,
    )


def evaluate_all_controls(
    n_pairs: int = 1000,
    test_fraction: float = 0.25,
    dataset_seed: int = 42,
    split_seed: int = 42,
) -> list[
    EvaluationResult
]:
    """
    Generate one matched Phase 0.5 dataset and evaluate all
    planned controls using the same pair-level split.

    RIGID_NULL uses the rigid observations from those same
    physical worlds, actions, pair IDs, and noise realizations.
    """

    if n_pairs < 2:
        raise ValueError(
            "n_pairs must be at least 2"
        )

    random.seed(
        dataset_seed
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=n_pairs,
        )
    )

    split = (
        pair_level_train_test_split(
            records=(
                dataset.compliant_records
            ),
            test_fraction=test_fraction,
            seed=split_seed,
        )
    )

    compliant_train = (
        split.train
    )

    compliant_test = (
        split.test
    )

    rigid_train = (
        records_for_pair_ids(
            records=dataset.rigid_records,
            pair_ids=split.train_pair_ids,
        )
    )

    rigid_test = (
        records_for_pair_ids(
            records=dataset.rigid_records,
            pair_ids=split.test_pair_ids,
        )
    )

    results = []

    for control in [
        CONTROL_FULL,
        CONTROL_FORCE_ONLY,
        CONTROL_MOTION_ONLY,
        CONTROL_SHAM,
        CONTROL_LABEL_SHUFFLE,
        CONTROL_RHO_CANARY,
    ]:
        result = (
            evaluate_control(
                train_records=compliant_train,
                test_records=compliant_test,
                control=control,
                seed=split_seed,
            )
        )

        results.append(
            result
        )

    rigid_result = (
        evaluate_control(
            train_records=rigid_train,
            test_records=rigid_test,
            control=CONTROL_RIGID_NULL,
            seed=split_seed,
        )
    )

    results.append(
        rigid_result
    )

    return results


def main():
    results = (
        evaluate_all_controls(
            n_pairs=1000,
            test_fraction=0.25,
            dataset_seed=42,
            split_seed=42,
        )
    )

    print(
        "=== Phase 0.5 initial classifier evaluation ==="
    )

    print(
        "control | train_pairs | test_pairs | "
        "ROC-AUC | FNR@FPR<=0.10"
    )

    for result in results:
        print(
            f"{result.control:14s}"
            f" | {result.train_pairs:4d}"
            f" | {result.test_pairs:4d}"
            f" | {result.roc_auc:.3f}"
            f" | {result.fnr_at_fpr_10:.3f}"
        )


if __name__ == "__main__":
    main()