from dataclasses import dataclass
import random

import numpy as np

from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.metrics import (
    roc_auc_score,
)
from sklearn.pipeline import (
    make_pipeline,
)
from sklearn.preprocessing import (
    StandardScaler,
)

from phase0_5_controls import (
    CONTROL_FULL,
    CONTROL_RIGID_NULL,
    FeatureView,
    generate_matched_control_dataset,
    make_feature_view,
)
from phase0_5_dataset import (
    ObservationRecord,
)
from phase0_5_evaluate import (
    labels_to_binary,
    records_for_pair_ids,
)
from phase0_5_split import (
    pair_level_train_test_split,
)


@dataclass
class BootstrapResult:
    control: str
    point_auc: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    test_pairs: int


def fit_test_scores(
    train_view: FeatureView,
    test_view: FeatureView,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Fit the classifier once on the training set and return
    held-out test labels and predicted probabilities.
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

    return (
        y_test,
        scores,
    )


def pair_cluster_bootstrap_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    pair_ids: list[int],
    n_bootstrap: int = 2000,
    seed: int = 123,
) -> tuple[
    float,
    float,
]:
    """
    Compute a percentile 95% confidence interval for ROC-AUC
    by resampling complete pair clusters with replacement.

    Rows from one twin pair always enter a bootstrap replicate
    together.
    """

    if n_bootstrap <= 0:
        raise ValueError(
            "n_bootstrap must be positive"
        )

    if not (
        len(y_true)
        == len(scores)
        == len(pair_ids)
    ):
        raise ValueError(
            "y_true, scores, and pair_ids must have equal length"
        )

    grouped_indices = {}

    for (
        index,
        pair_id,
    ) in enumerate(
        pair_ids
    ):
        grouped_indices.setdefault(
            pair_id,
            [],
        ).append(
            index
        )

    unique_pair_ids = list(
        grouped_indices.keys()
    )

    if len(
        unique_pair_ids
    ) < 2:
        raise ValueError(
            "At least two pair clusters are required"
        )

    for (
        pair_id,
        indices,
    ) in grouped_indices.items():
        if len(
            indices
        ) != 2:
            raise ValueError(
                f"pair_id={pair_id} "
                "must contain exactly two rows"
            )

    rng = random.Random(
        seed
    )

    bootstrap_aucs = []

    for _ in range(
        n_bootstrap
    ):
        sampled_pair_ids = [
            rng.choice(
                unique_pair_ids
            )
            for _ in range(
                len(
                    unique_pair_ids
                )
            )
        ]

        sampled_indices = []

        for pair_id in sampled_pair_ids:
            sampled_indices.extend(
                grouped_indices[
                    pair_id
                ]
            )

        sampled_y = y_true[
            sampled_indices
        ]

        sampled_scores = scores[
            sampled_indices
        ]

        if len(
            np.unique(
                sampled_y
            )
        ) != 2:
            continue

        auc = float(
            roc_auc_score(
                sampled_y,
                sampled_scores,
            )
        )

        bootstrap_aucs.append(
            auc
        )

    if not bootstrap_aucs:
        raise RuntimeError(
            "No valid bootstrap replicates were produced"
        )

    ci_low = float(
        np.percentile(
            bootstrap_aucs,
            2.5,
        )
    )

    ci_high = float(
        np.percentile(
            bootstrap_aucs,
            97.5,
        )
    )

    return (
        ci_low,
        ci_high,
    )


def evaluate_control_with_bootstrap(
    train_records: list[ObservationRecord],
    test_records: list[ObservationRecord],
    control: str,
    n_bootstrap: int = 2000,
    seed: int = 123,
) -> BootstrapResult:
    """
    Evaluate one control and attach a pair-cluster bootstrap
    95% confidence interval to the held-out ROC-AUC.
    """

    train_view = make_feature_view(
        records=train_records,
        control=control,
        seed=seed,
    )

    test_view = make_feature_view(
        records=test_records,
        control=control,
        seed=seed + 1,
    )

    (
        y_test,
        scores,
    ) = fit_test_scores(
        train_view=train_view,
        test_view=test_view,
    )

    point_auc = float(
        roc_auc_score(
            y_test,
            scores,
        )
    )

    (
        ci_low,
        ci_high,
    ) = pair_cluster_bootstrap_auc(
        y_true=y_test,
        scores=scores,
        pair_ids=test_view.pair_ids,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    return BootstrapResult(
        control=control,
        point_auc=point_auc,
        ci_low=ci_low,
        ci_high=ci_high,
        n_bootstrap=n_bootstrap,
        test_pairs=len(
            set(
                test_view.pair_ids
            )
        ),
    )


def main():
    random.seed(
        42
    )

    dataset = (
        generate_matched_control_dataset(
            n_pairs=1000,
        )
    )

    split = (
        pair_level_train_test_split(
            records=dataset.compliant_records,
            test_fraction=0.25,
            seed=42,
        )
    )

    full_result = (
        evaluate_control_with_bootstrap(
            train_records=split.train,
            test_records=split.test,
            control=CONTROL_FULL,
            n_bootstrap=2000,
            seed=123,
        )
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

    rigid_result = (
        evaluate_control_with_bootstrap(
            train_records=rigid_train,
            test_records=rigid_test,
            control=CONTROL_RIGID_NULL,
            n_bootstrap=2000,
            seed=123,
        )
    )

    print(
        "=== Phase 0.5 pair-cluster bootstrap ==="
    )

    for result in [
        full_result,
        rigid_result,
    ]:
        print(
            result.control,
            f"AUC={result.point_auc:.3f}",
            f"95% CI=["
            f"{result.ci_low:.3f}, "
            f"{result.ci_high:.3f}]",
            f"test_pairs={result.test_pairs}",
            f"bootstrap={result.n_bootstrap}",
        )


if __name__ == "__main__":
    main()