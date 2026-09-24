import random

from phase0_5_pairs import (
    generate_identification_pair,
    shared_nuisance_signature,
)
from physics import (
    NextTransition,
    predict_next_transition,
)


def test_pair_has_opposite_labels():
    random.seed(
        42
    )

    pair = (
        generate_identification_pair(
            pair_id=1,
        )
    )

    assert (
        pair.label_a
        != pair.label_b
    )

    assert {
        pair.label_a,
        pair.label_b,
    } == {
        NextTransition.MATERIAL_YIELD,
        NextTransition.SUPPORT_SLIP,
    }


def test_pair_labels_match_analytic_transition_prediction():
    random.seed(
        42
    )

    pair = (
        generate_identification_pair(
            pair_id=2,
        )
    )

    predicted_a, _, _ = (
        predict_next_transition(
            pair.world_a,
            angle_deg=45.0,
        )
    )

    predicted_b, _, _ = (
        predict_next_transition(
            pair.world_b,
            angle_deg=45.0,
        )
    )

    assert (
        predicted_a
        == pair.label_a
    )

    assert (
        predicted_b
        == pair.label_b
    )


def test_pair_preserves_shared_nuisance_variables():
    random.seed(
        42
    )

    pair = (
        generate_identification_pair(
            pair_id=3,
        )
    )

    assert (
        shared_nuisance_signature(
            pair.world_a
        )
        ==
        shared_nuisance_signature(
            pair.world_b
        )
    )


def test_pair_differs_in_failure_margin_variables():
    random.seed(
        42
    )

    pair = (
        generate_identification_pair(
            pair_id=4,
        )
    )

    assert (
        pair.world_a.static_friction
        != pair.world_b.static_friction
    )

    assert (
        pair.world_a.material_yield_strength
        != pair.world_b.material_yield_strength
    )


def test_pair_id_is_preserved():
    random.seed(
        42
    )

    pair = (
        generate_identification_pair(
            pair_id=12345,
        )
    )

    assert (
        pair.pair_id
        == 12345
    )


def test_a_side_is_not_permanently_tied_to_one_label():
    random.seed(
        42
    )

    labels_seen = set()

    for pair_id in range(
        50
    ):
        pair = (
            generate_identification_pair(
                pair_id=pair_id,
            )
        )

        labels_seen.add(
            pair.label_a
        )

    assert (
        NextTransition.MATERIAL_YIELD
        in labels_seen
    )

    assert (
        NextTransition.SUPPORT_SLIP
        in labels_seen
    )


def test_multiple_pairs_keep_unique_ids():
    random.seed(
        42
    )

    pairs = [
        generate_identification_pair(
            pair_id=pair_id,
        )
        for pair_id in range(
            20
        )
    ]

    pair_ids = [
        pair.pair_id
        for pair in pairs
    ]

    assert (
        len(
            pair_ids
        )
        ==
        len(
            set(
                pair_ids
            )
        )
    )