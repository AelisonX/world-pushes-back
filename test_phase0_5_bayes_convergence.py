from phase0_5_bayes_convergence import (
    run_bayes_convergence_study,
)


def test_convergence_study_returns_requested_counts():
    counts = [
        20,
        40,
        80,
    ]

    results = (
        run_bayes_convergence_study(
            particle_counts=counts,
            n_test_pairs=5,
            particle_seed=42,
            test_seed=99,
        )
    )

    assert [
        result.n_particles
        for result in results
    ] == counts


def test_first_convergence_point_has_no_delta():
    results = (
        run_bayes_convergence_study(
            particle_counts=[
                20,
                40,
            ],
            n_test_pairs=5,
            particle_seed=42,
            test_seed=99,
        )
    )

    first = results[0]

    assert (
        first.auc_change
        is None
    )

    assert (
        first.mean_posterior_change
        is None
    )

    assert (
        first.max_posterior_change
        is None
    )


def test_later_convergence_points_have_nonnegative_deltas():
    results = (
        run_bayes_convergence_study(
            particle_counts=[
                20,
                40,
                80,
            ],
            n_test_pairs=5,
            particle_seed=42,
            test_seed=99,
        )
    )

    for result in results[1:]:
        assert (
            result.auc_change
            >= 0.0
        )

        assert (
            result.mean_posterior_change
            >= 0.0
        )

        assert (
            result.max_posterior_change
            >= 0.0
        )


def test_convergence_auc_stays_inside_valid_range():
    results = (
        run_bayes_convergence_study(
            particle_counts=[
                20,
                40,
            ],
            n_test_pairs=5,
            particle_seed=42,
            test_seed=99,
        )
    )

    for result in results:
        assert (
            0.0
            <= result.roc_auc
            <= 1.0
        )


def test_convergence_study_is_reproducible():
    kwargs = dict(
        particle_counts=[
            20,
            40,
        ],
        n_test_pairs=5,
        particle_seed=42,
        test_seed=99,
    )

    result_a = (
        run_bayes_convergence_study(
            **kwargs
        )
    )

    result_b = (
        run_bayes_convergence_study(
            **kwargs
        )
    )

    assert (
        result_a
        == result_b
    )


def test_unsorted_particle_counts_are_rejected():
    try:
        run_bayes_convergence_study(
            particle_counts=[
                40,
                20,
            ],
            n_test_pairs=5,
        )
    except ValueError as error:
        assert (
            "sorted"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected unsorted counts to fail"
        )


def test_duplicate_particle_counts_are_rejected():
    try:
        run_bayes_convergence_study(
            particle_counts=[
                20,
                20,
            ],
            n_test_pairs=5,
        )
    except ValueError as error:
        assert (
            "unique"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected duplicate counts to fail"
        )


def test_odd_particle_count_is_rejected():
    try:
        run_bayes_convergence_study(
            particle_counts=[
                21,
                40,
            ],
            n_test_pairs=5,
        )
    except ValueError as error:
        assert (
            "even"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected odd particle count to fail"
        )


def test_empty_particle_count_list_is_rejected():
    try:
        run_bayes_convergence_study(
            particle_counts=[],
            n_test_pairs=5,
        )
    except ValueError as error:
        assert (
            "must not be empty"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected empty particle counts to fail"
        )